use anyhow::{Context, Result, ensure};
use clap::{Args, Parser, Subcommand, ValueEnum};
use serde_json::json;
use std::{
    io::{self, Write},
    net::SocketAddr,
    path::{Path, PathBuf},
    time::Instant,
};

mod generation;
mod hub;
mod model;
mod planner;
mod server;
mod tokenizer;

use generation::{Engine, Message, Sampling};
use model::{CachePolicy, ExpertOptions, ModelFile};
use planner::{Mode, Plan, PlanOptions};

#[derive(Parser)]
#[command(version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Stream text from a local GGUF model.
    Run(Generate),
    /// Measure load, prefill, decode and process memory in JSON.
    Benchmark(Generate),
    /// Inspect a GGUF and estimate memory before loading weights.
    Inspect {
        /// Single-file GGUF.
        model: PathBuf,
        #[command(flatten)]
        load: Load,
    },
    /// Serve one local model with an OpenAI-compatible API.
    Serve {
        /// Single-file GGUF, or use --hf and --hf-file to fetch one.
        #[arg(required_unless_present = "hf")]
        model: Option<PathBuf>,
        #[command(flatten)]
        hub: hub::HubOptions,
        #[command(flatten)]
        load: Load,
        #[arg(long, default_value = "127.0.0.1:8080")]
        bind: SocketAddr,
        #[arg(long)]
        model_id: Option<String>,
    },
    /// Show CPU, RAM and NVIDIA telemetry.
    Hardware,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Execution {
    Resident,
    Hybrid,
    DenseStream,
    MoeStream,
}

impl From<Execution> for Mode {
    fn from(value: Execution) -> Self {
        match value {
            Execution::Resident => Self::Resident,
            Execution::Hybrid => Self::Hybrid,
            Execution::DenseStream => Self::DenseStream,
            Execution::MoeStream => Self::MoeStream,
        }
    }
}

#[derive(Args)]
struct Load {
    /// Defaults to the model context or 2048, whichever is smaller.
    #[arg(long)]
    context: Option<usize>,
    /// Rayon workers (defaults to up to four logical cores).
    #[arg(long)]
    threads: Option<usize>,
    #[arg(long, value_enum, default_value = "resident")]
    mode: Execution,
    /// Number of trailing transformer blocks on CUDA. Hybrid can select automatically.
    #[arg(long)]
    gpu_layers: Option<usize>,
    #[arg(long, default_value_t = 0)]
    cuda_device: usize,
    /// Estimated RAM limit in MiB.
    #[arg(long)]
    ram_mib: Option<u64>,
    /// Estimated VRAM limit in MiB.
    #[arg(long)]
    vram_mib: Option<u64>,
    /// CPU MoE expert cache budget in MiB.
    #[arg(long, default_value_t = 0)]
    expert_cache_mib: u64,
    /// Eviction policy for the CPU expert cache.
    #[arg(long, value_enum, default_value = "lru")]
    expert_cache_policy: CachePolicy,
    /// Matching Hugging Face tokenizer.json.
    #[arg(long)]
    tokenizer: Option<PathBuf>,
    /// Optional MiniJinja-compatible chat template file.
    #[arg(long)]
    chat_template: Option<PathBuf>,
}

#[derive(Args)]
struct Generate {
    /// Single-file GGUF.
    model: PathBuf,
    #[command(flatten)]
    load: Load,
    #[arg(long, default_value = "Once upon a time")]
    prompt: String,
    /// Treat the prompt as a single user chat message.
    #[arg(long)]
    chat: bool,
    #[arg(long, default_value_t = 32)]
    max_tokens: usize,
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,
    #[arg(long, default_value_t = 0.95)]
    top_p: f64,
    #[arg(long)]
    top_k: Option<usize>,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long)]
    stop: Vec<String>,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Hardware => print_json(&json!({
            "hardware": planner::hardware(), "cuda_build": cfg!(feature = "cuda"),
        })),
        Command::Inspect { model, load } => {
            let (model, context, plan, _) = prepare(&model, &load)?;
            let mut formats = std::collections::BTreeMap::new();
            for info in model.gguf.tensor_infos.values() {
                *formats
                    .entry(format!("{:?}", info.ggml_dtype))
                    .or_insert(0usize) += 1;
            }
            print_json(&json!({
                "file_bytes": model.file_bytes, "model": model.config, "tensor_formats": formats,
                "context": context, "plan": plan,
            }))
        }
        Command::Run(args) => generate(args, false),
        Command::Benchmark(args) => generate(args, true),
        Command::Serve {
            model,
            hub,
            load,
            bind,
            model_id,
        } => {
            ensure!(bind.ip().is_loopback(), "bind address must be loopback");
            let path = hub.resolve(model)?;
            let id = model_id.unwrap_or_else(|| model_id_from(&path));
            ensure!(!id.trim().is_empty(), "model ID must not be empty");
            eprintln!("Loading {id} from {}", path.display());
            let (engine, _, _) = open_engine(&path, &load)?;
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()?
                .block_on(server::serve(engine, bind, id))
        }
    }
}

fn model_id_from(path: &Path) -> String {
    path.file_stem()
        .and_then(|p| p.to_str())
        .unwrap_or("oryndral")
        .to_owned()
}

fn bytes(mib: u64) -> Result<u64> {
    mib.checked_mul(1024 * 1024)
        .context("MiB budget is too large")
}

fn prepare(path: &Path, load: &Load) -> Result<(ModelFile, usize, Plan, ExpertOptions)> {
    let file = ModelFile::open(path)?;
    let context = load.context.unwrap_or(file.config.context.min(2048));
    let mode = Mode::from(load.mode);
    ensure!(
        (load.expert_cache_mib == 0 && load.expert_cache_policy == CachePolicy::Lru)
            || mode == Mode::MoeStream,
        "expert cache options require moe-stream mode"
    );
    let experts = ExpertOptions {
        capacity: bytes(load.expert_cache_mib)?,
        policy: load.expert_cache_policy,
    };
    let memory = file.memory(context, experts.capacity)?;
    let ram_budget = load.ram_mib.map(bytes).transpose()?;
    let vram_budget = match (load.vram_mib, mode, load.gpu_layers) {
        (Some(mib), _, _) => Some(bytes(mib)?),
        (None, Mode::Hybrid, None) if cfg!(feature = "cuda") => {
            let gpus = planner::hardware().nvidia_gpus;
            ensure!(
                gpus.len() <= 1 && std::env::var_os("CUDA_VISIBLE_DEVICES").is_none(),
                "automatic VRAM detection requires one visible GPU; set --vram-mib"
            );
            gpus.into_iter()
                .find(|g| g.index == load.cuda_device)
                .and_then(|g| g.available_bytes)
                .map(|free| free.saturating_sub(256 * 1024 * 1024))
        }
        _ => None,
    };
    let plan = planner::plan(
        &memory,
        PlanOptions {
            mode,
            gpu_layers: load.gpu_layers,
            ram_budget,
            vram_budget,
        },
    )?;
    Ok((file, context, plan, experts))
}

fn threads(load: &Load) -> usize {
    load.threads.unwrap_or_else(|| {
        std::thread::available_parallelism().map_or(1, |n| usize::from(n).min(4))
    })
}

fn open_engine(path: &Path, load: &Load) -> Result<(Engine, serde_json::Value, f64)> {
    let start = Instant::now();
    let hybrid = Mode::from(load.mode) == Mode::Hybrid && load.gpu_layers != Some(0);
    ensure!(
        cfg!(feature = "cuda") || (!hybrid && load.gpu_layers.unwrap_or(0) == 0),
        "this executable was built for CPU inference. Build with --features cuda for GPU offload"
    );
    let threads = threads(load);
    ensure!((1..=1024).contains(&threads), "threads must be in 1..=1024");
    // Candle also reads this at GEMM call sites, so set it before spawning threads.
    unsafe {
        std::env::set_var("RAYON_NUM_THREADS", threads.to_string());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .context("initialize CPU workers")?;
    let (file, context, plan, experts) = prepare(path, load)?;
    ensure!(
        !hybrid || plan.gpu_layers > 0,
        "the hybrid plan placed zero blocks on CUDA. Check free VRAM or set --gpu-layers explicitly"
    );
    eprintln!(
        "Placing {} transformer blocks on CPU and {} on CUDA",
        plan.cpu_layers, plan.gpu_layers
    );
    let details = json!({ "context": context, "threads": threads, "file_bytes": file.file_bytes, "plan": plan, "experts": experts });
    let engine = Engine::load(
        file,
        plan,
        context,
        load.cuda_device,
        experts,
        load.tokenizer.as_deref(),
        load.chat_template.as_deref(),
    )?;
    Ok((engine, details, start.elapsed().as_secs_f64()))
}

fn generate(args: Generate, benchmark: bool) -> Result<()> {
    let sampling = Sampling {
        max_tokens: args.max_tokens,
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        seed: args.seed,
        stop: args.stop,
    };
    sampling.validate()?;
    let (mut engine, details, load_seconds) = open_engine(&args.model, &args.load)?;
    let prompt = if args.chat {
        engine.chat_prompt(&[Message {
            role: "user".to_owned(),
            content: args.prompt,
        }])?
    } else {
        args.prompt
    };
    let load_io = engine.load_io();
    let before_io = benchmark.then(planner::process_io);
    let stdout = io::stdout();
    let mut output = stdout.lock();
    let mut error = None;
    let result = engine.generate(&prompt, &sampling, |text| {
        if benchmark || text.is_empty() {
            return true;
        }
        match write!(output, "{text}").and_then(|_| output.flush()) {
            Ok(()) => true,
            Err(err) => {
                error = Some(err);
                false
            }
        }
    })?;
    if let Some(err) = error {
        return Err(err.into());
    }
    if let Some(before) = before_io {
        let after = planner::process_io();
        let delta = |a: Option<u64>, b: Option<u64>| b.zip(a).map(|(b, a)| b.saturating_sub(a));
        let model_read_rate =
            result.metrics.io.weight_read_bytes as f64 / result.metrics.elapsed_seconds;
        drop(output);
        return print_json(&json!({
            "configuration": details, "prompt": prompt, "sampling": sampling,
            "load_seconds": load_seconds, "load_io": load_io, "generation": result,
            "process_memory": planner::process_memory(),
            "generation_process_io": {
                "read_transfer_bytes": delta(before.read_transfer_bytes, after.read_transfer_bytes),
                "storage_read_bytes": delta(before.storage_read_bytes, after.storage_read_bytes)
            },
            "generation_model_read_bytes_per_second": model_read_rate,
        }));
    }
    writeln!(output)?;
    Ok(())
}

fn print_json(value: &serde_json::Value) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}
