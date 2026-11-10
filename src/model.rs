use crate::planner::{Mode, ModelMemory, Plan};
use anyhow::{Context, Result, bail, ensure};
use candle::quantized::gguf_file::{Content as Gguf, TensorInfo, Value};
use candle::quantized::{GgmlDType, QMatMul};
use candle::{DType, Device, Module, Tensor};
use serde::Serialize;
use std::sync::Arc;
use std::{collections::HashMap, fs::File, path::Path};

pub const PREFILL_BATCH: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Architecture {
    Llama,
    Qwen2,
}

impl Architecture {
    fn prefix(self) -> &'static str {
        match self {
            Self::Llama => "llama",
            Self::Qwen2 => "qwen2",
        }
    }
}

#[derive(Debug, Serialize)]
pub struct Config {
    pub architecture: Architecture,
    pub blocks: usize,
    pub context: usize,
    pub width: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub vocab: usize,
    pub experts: usize,
    pub experts_used: usize,
    rope_base: f64,
    epsilon: f32,
    sliding_window: Option<usize>,
}

impl Config {
    fn head_dim(&self) -> usize {
        self.width / self.heads
    }
}

pub struct ModelFile {
    pub gguf: Gguf,
    pub config: Config,
    pub file_bytes: u64,
    file: File,
    read_bytes: u64,
}

impl ModelFile {
    pub fn open(path: &Path) -> Result<Self> {
        let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let file_bytes = file.metadata()?.len();
        let gguf = Gguf::read(&mut file).context("read GGUF header")?;
        let meta = &gguf.metadata;
        let architecture = match meta
            .get("general.architecture")
            .context("missing GGUF general.architecture")?
            .to_string()?
            .as_str()
        {
            "llama" => Architecture::Llama,
            "qwen2" => Architecture::Qwen2,
            name => bail!(
                "unsupported GGUF architecture {name:?}. Supported architectures are llama and qwen2"
            ),
        };
        let key = |suffix: &str| format!("{}.{suffix}", architecture.prefix());
        let optional = |suffix: &str| get_usize(meta, &key(suffix));
        let required =
            |suffix: &str| optional(suffix)?.with_context(|| format!("missing {}", key(suffix)));
        ensure!(
            get_usize(meta, "split.count")?.unwrap_or(1) <= 1,
            "split GGUF is unsupported"
        );
        ensure!(
            meta.get(&key("rope.scaling.type"))
                .map(Value::to_string)
                .transpose()?
                .is_none_or(|v| v == "none")
                && meta
                    .get(&key("rope.scaling.factor"))
                    .map(number)
                    .transpose()?
                    .is_none_or(|v| v == 1.0)
                && meta
                    .get(&key("rope.scale_linear"))
                    .map(number)
                    .transpose()?
                    .is_none_or(|v| v == 1.0)
                && !gguf.tensor_infos.contains_key("rope_freqs.weight"),
            "this model requires an unsupported RoPE scaling variant"
        );
        let heads = required("attention.head_count")?;
        let vocab = gguf
            .tensor_infos
            .get("token_embd.weight")
            .context("missing token_embd.weight")?
            .shape
            .dims()
            .first()
            .copied()
            .unwrap_or(0);
        let cfg = Config {
            architecture,
            blocks: required("block_count")?,
            context: required("context_length")?,
            width: required("embedding_length")?,
            heads,
            kv_heads: optional("attention.head_count_kv")?.unwrap_or(heads),
            vocab,
            experts: optional("expert_count")?.unwrap_or(0),
            experts_used: optional("expert_used_count")?.unwrap_or(0),
            rope_base: meta
                .get(&key("rope.freq_base"))
                .map(number)
                .transpose()?
                .unwrap_or(10_000.),
            epsilon: number(
                meta.get(&key("attention.layer_norm_rms_epsilon"))
                    .with_context(|| {
                        format!("missing {}", key("attention.layer_norm_rms_epsilon"))
                    })?,
            )? as f32,
            sliding_window: optional("attention.sliding_window")?.filter(|v| *v != 0),
        };
        ensure!(
            (1..=512).contains(&cfg.blocks)
                && (1..=131_072).contains(&cfg.context)
                && (1..=32_768).contains(&cfg.width)
                && (2..=1_000_000).contains(&cfg.vocab)
                && heads > 0
                && cfg.kv_heads > 0
                && cfg.width.is_multiple_of(heads)
                && heads.is_multiple_of(cfg.kv_heads)
                && cfg.head_dim().is_multiple_of(2)
                && cfg.rope_base.is_finite()
                && cfg.rope_base > 0.
                && cfg.epsilon.is_finite()
                && cfg.epsilon > 0.,
            "invalid {} dimensions or constants",
            architecture.prefix()
        );
        for suffix in [
            "rope.dimension_count",
            "attention.key_length",
            "attention.value_length",
        ] {
            ensure!(
                optional(suffix)?.is_none_or(|n| n == cfg.head_dim()),
                "{} must match the attention head dimension",
                key(suffix)
            );
        }
        ensure!(
            architecture == Architecture::Llama || (cfg.experts == 0 && cfg.experts_used == 0),
            "Qwen2 support requires a dense model"
        );
        ensure!(
            architecture == Architecture::Llama || cfg.sliding_window.is_none(),
            "Qwen2 support requires full attention"
        );
        ensure!(
            cfg.experts == 0 && cfg.experts_used == 0
                || cfg.experts <= 256 && (1..=cfg.experts).contains(&cfg.experts_used),
            "invalid expert counts"
        );
        let model = Self {
            gguf,
            config: cfg,
            file_bytes,
            file,
            read_bytes: 0,
        };
        model.check_layout()?;
        Ok(model)
    }

    fn check_layout(&self) -> Result<()> {
        let c = &self.config;
        let kv_width = c.kv_heads * c.head_dim();
        let expect = |name: &str, dims: &[usize]| -> Result<()> {
            let t = self.info(name)?;
            ensure!(
                t.shape.dims() == dims,
                "{name}: expected shape {dims:?}, found {:?}",
                t.shape
            );
            ensure!(
                dims.last()
                    .is_some_and(|n| n.is_multiple_of(t.ggml_dtype.block_size())),
                "{name}: rows do not align with the quantization block size"
            );
            Ok(())
        };
        for (name, t) in &self.gguf.tensor_infos {
            let end = self
                .gguf
                .tensor_data_offset
                .checked_add(t.offset)
                .and_then(|start| start.checked_add(tensor_bytes(t).ok()?));
            ensure!(
                end.is_some_and(|end| end <= self.file_bytes),
                "{name}: data outside GGUF file"
            );
            let bias_supported = c.architecture == Architecture::Qwen2
                && (name == "output.bias"
                    || name
                        .strip_prefix("blk.")
                        .and_then(|name| name.split_once('.'))
                        .is_some_and(|(index, tensor)| {
                            index
                                .parse::<usize>()
                                .is_ok_and(|i| i < c.blocks && i.to_string() == index)
                                && matches!(tensor, "attn_q.bias" | "attn_k.bias" | "attn_v.bias")
                        }));
            ensure!(
                (!name.ends_with(".bias") || bias_supported)
                    && !name.contains("ffn_gate_shexp")
                    && !name.contains("attn_q_norm")
                    && !name.contains("attn_k_norm"),
                "{name}: unsupported {} tensor",
                c.architecture.prefix()
            );
            ensure!(
                !bias_supported || is_float(t.ggml_dtype),
                "{name}: expected an unquantized bias"
            );
        }
        expect("token_embd.weight", &[c.vocab, c.width])?;
        expect(self.output_name(), &[c.vocab, c.width])?;
        expect("output_norm.weight", &[c.width])?;
        if self.gguf.tensor_infos.contains_key("output.bias") {
            expect("output.bias", &[c.vocab])?;
        }
        for i in 0..c.blocks {
            let p = format!("blk.{i}.");
            let e = |name: &str, dims: &[usize]| expect(&format!("{p}{name}"), dims);
            for name in ["attn_norm.weight", "ffn_norm.weight"] {
                e(name, &[c.width])?;
            }
            for (stem, width) in [
                ("attn_q", c.width),
                ("attn_k", kv_width),
                ("attn_v", kv_width),
            ] {
                e(&format!("{stem}.weight"), &[width, c.width])?;
                let bias = format!("{stem}.bias");
                if self.gguf.tensor_infos.contains_key(&format!("{p}{bias}")) {
                    e(&bias, &[width])?;
                }
            }
            e("attn_output.weight", &[c.width, c.width])?;
            let gate = if c.experts == 0 {
                "ffn_gate.weight"
            } else {
                "ffn_gate_exps.weight"
            };
            let gate_dims = self.info(&format!("{p}{gate}"))?.shape.dims();
            let hidden = *gate_dims
                .iter()
                .rev()
                .nth(1)
                .context("invalid FFN dimensions")?;
            ensure!((1..=131_072).contains(&hidden), "invalid FFN width");
            if c.experts == 0 {
                e(gate, &[hidden, c.width])?;
                e("ffn_up.weight", &[hidden, c.width])?;
                e("ffn_down.weight", &[c.width, hidden])?;
            } else {
                e("ffn_gate_inp.weight", &[c.experts, c.width])?;
                e(gate, &[c.experts, hidden, c.width])?;
                e("ffn_up_exps.weight", &[c.experts, hidden, c.width])?;
                e("ffn_down_exps.weight", &[c.experts, c.width, hidden])?;
            }
        }
        Ok(())
    }

    fn output_name(&self) -> &'static str {
        if self.gguf.tensor_infos.contains_key("output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        }
    }

    fn info(&self, name: &str) -> Result<&TensorInfo> {
        self.gguf
            .tensor_infos
            .get(name)
            .with_context(|| format!("missing GGUF tensor {name}"))
    }

    pub fn memory(&self, context: usize, cache: u64) -> Result<ModelMemory> {
        let c = &self.config;
        ensure!(
            (1..=c.context).contains(&context),
            "context must be in 1..={}",
            c.context
        );
        let mut layers = vec![0u64; c.blocks];
        let mut max_tensor = 0u64;
        let mut max_expert = 0u64;
        let mut ffn_width = 0u64;
        for (name, t) in &self.gguf.tensor_infos {
            let bytes = tensor_bytes(t)?;
            if name.ends_with(".ffn_gate.weight") || name.ends_with(".ffn_gate_exps.weight") {
                ffn_width =
                    ffn_width.max(t.shape.dims().iter().rev().nth(1).copied().unwrap_or(0) as u64);
            }
            if let Some(layer) = name
                .strip_prefix("blk.")
                .and_then(|n| n.split('.').next())
                .and_then(|n| n.parse::<usize>().ok())
                .and_then(|i| layers.get_mut(i))
            {
                if c.experts != 0 && name.ends_with("_exps.weight") {
                    max_expert = max_expert.max(runtime_bytes(t)? / c.experts as u64);
                    continue;
                }
                *layer = layer
                    .checked_add(runtime_bytes(t)?)
                    .context("layer size overflow")?;
            }
            max_tensor = max_tensor.max(bytes);
        }
        let output_bytes = if self.output_name() == "token_embd.weight" {
            0
        } else {
            runtime_bytes(self.info(self.output_name())?)?
        };
        let output_bias_bytes = self
            .gguf
            .tensor_infos
            .get("output.bias")
            .map(runtime_bytes)
            .transpose()?
            .unwrap_or(0);
        let non_layer_bytes = runtime_bytes(self.info("token_embd.weight")?)?
            .checked_add(output_bytes)
            .and_then(|b| b.checked_add(output_bias_bytes))
            .and_then(|b| b.checked_add(c.width as u64 * 4))
            .context("model size overflow")?;
        let kv_per_layer_bytes = 2 * context as u64 * c.kv_heads as u64 * c.head_dim() as u64 * 4;
        let batch = context.min(PREFILL_BATCH) as u64;
        let attention = 4 * batch * context as u64 * (3 * c.heads as u64 + 1);
        let rope = context as u64 * c.head_dim() as u64 * 4;
        let activation = 4 * batch * (c.width as u64 * 16 + ffn_width * 3);
        let scratch = attention + rope + activation + kv_per_layer_bytes;
        let ram_scratch_bytes = scratch
            .checked_add(max_tensor)
            .and_then(|v| v.checked_add(cache))
            .and_then(|v| v.checked_add(max_expert * 6))
            .context("RAM estimate overflow")?;
        Ok(ModelMemory {
            non_layer_bytes,
            layers,
            kv_per_layer_bytes,
            ram_scratch_bytes,
            vram_scratch_bytes: scratch,
        })
    }

    fn matrix(&mut self, name: &str, device: &Device) -> Result<QMatMul> {
        as_matrix(self.quant(name, device)?)
    }

    fn vector(&mut self, name: &str, device: &Device) -> Result<Tensor> {
        Ok(self.quant(name, device)?.dequantize(device)?)
    }

    fn bias(&mut self, name: &str, device: &Device) -> Result<Option<Tensor>> {
        self.gguf
            .tensor_infos
            .contains_key(name)
            .then(|| self.vector(name, device))
            .transpose()
    }

    fn quant(&mut self, name: &str, device: &Device) -> Result<candle::quantized::QTensor> {
        let size = tensor_bytes(self.info(name)?)?;
        let value = self
            .gguf
            .tensor(&mut self.file, name, device)
            .with_context(|| format!("read GGUF tensor {name}"))?;
        self.read_bytes = self.read_bytes.saturating_add(size);
        Ok(value)
    }

    fn expert_matrix(&mut self, name: &str, expert: usize) -> Result<(QMatMul, u64)> {
        let t = self.info(name)?;
        let bytes = tensor_bytes(t)? / self.config.experts as u64;
        let t = TensorInfo {
            offset: t.offset + bytes * expert as u64,
            shape: candle::Shape::from(t.shape.dims()[1..].to_vec()),
            ggml_dtype: t.ggml_dtype,
        };
        let size = runtime_bytes(&t)?;
        let qt = t
            .read(&mut self.file, self.gguf.tensor_data_offset, &Device::Cpu)
            .with_context(|| format!("read {name}, expert {expert}"))?;
        self.read_bytes = self.read_bytes.saturating_add(bytes);
        Ok((as_matrix(qt)?, size))
    }
}

pub fn get_usize(meta: &HashMap<String, Value>, key: &str) -> Result<Option<usize>> {
    meta.get(key)
        .map(|v| -> Result<usize> { Ok(usize::try_from(v.to_u64()?)?) })
        .transpose()
        .with_context(|| format!("invalid {key}"))
}

fn number(value: &Value) -> Result<f64> {
    Ok(match value {
        Value::F32(v) => f64::from(*v),
        Value::F64(v) => *v,
        _ => bail!("expected floating-point GGUF metadata"),
    })
}

fn elements(t: &TensorInfo) -> Result<u64> {
    t.shape.dims().iter().try_fold(1u64, |n, d| {
        ensure!(*d > 0, "empty GGUF tensor");
        n.checked_mul(*d as u64)
            .context("GGUF tensor shape overflow")
    })
}

fn tensor_bytes(t: &TensorInfo) -> Result<u64> {
    let count = elements(t)?;
    let block = t.ggml_dtype.block_size() as u64;
    ensure!(count.is_multiple_of(block), "misaligned GGUF tensor");
    (count / block)
        .checked_mul(t.ggml_dtype.type_size() as u64)
        .context("GGUF byte size overflow")
}

/// Float tensors are held as f32 at run time. Quantized tensors keep their GGUF encoding.
fn is_float(dtype: GgmlDType) -> bool {
    matches!(dtype, GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16)
}

fn runtime_bytes(t: &TensorInfo) -> Result<u64> {
    if is_float(t.ggml_dtype) {
        elements(t)?
            .checked_mul(4)
            .context("tensor byte size overflow")
    } else {
        tensor_bytes(t)
    }
}

fn as_matrix(t: candle::quantized::QTensor) -> Result<QMatMul> {
    Ok(if is_float(t.dtype()) {
        QMatMul::Tensor(t.dequantize(&t.device())?)
    } else {
        QMatMul::QTensor(Arc::new(t))
    })
}

#[derive(Default)]
struct Kv {
    keys: Option<Tensor>,
    values: Option<Tensor>,
}

struct Rope {
    cos: Tensor,
    sin: Tensor,
}

impl Rope {
    fn new(cfg: &Config, context: usize, device: &Device) -> Result<Self> {
        let inv = (0..cfg.head_dim())
            .step_by(2)
            .map(|i| cfg.rope_base.powf(-(i as f64) / cfg.head_dim() as f64) as f32)
            .collect::<Vec<_>>();
        let inv = Tensor::new(inv.as_slice(), device)?.unsqueeze(0)?;
        let pos = Tensor::arange(0u32, context as u32, device)?
            .to_dtype(DType::F32)?
            .unsqueeze(1)?;
        let angles = pos.matmul(&inv)?;
        Ok(Self {
            cos: angles.cos()?,
            sin: angles.sin()?,
        })
    }
}

struct Mlp {
    gate: QMatMul,
    down: QMatMul,
    up: QMatMul,
}

impl Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self
            .down
            .forward(&(self.gate.forward(x)?.silu()? * self.up.forward(x)?)?)?)
    }
}

enum FeedForward {
    Dense(Mlp),
    Experts(QMatMul),
}

struct Projection {
    weight: QMatMul,
    bias: Option<Tensor>,
}

impl Projection {
    fn load(src: &mut ModelFile, name: &str, device: &Device) -> Result<Self> {
        Ok(Self {
            weight: src.matrix(&format!("{name}.weight"), device)?,
            bias: src.bias(&format!("{name}.bias"), device)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        add_bias(self.weight.forward(x)?, self.bias.as_ref())
    }
}

fn add_bias(tensor: Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    match bias {
        Some(bias) => Ok(tensor.broadcast_add(bias)?),
        None => Ok(tensor),
    }
}

struct Layer {
    q: Projection,
    k: Projection,
    v: Projection,
    output: QMatMul,
    attn_norm: Tensor,
    ffn_norm: Tensor,
    ff: FeedForward,
}

impl Layer {
    fn load(src: &mut ModelFile, index: usize, device: &Device) -> Result<Self> {
        let prefix = format!("blk.{index}.");
        let name = |suffix| format!("{prefix}{suffix}");
        let ff = if src.config.experts == 0 {
            FeedForward::Dense(Mlp {
                gate: src.matrix(&name("ffn_gate.weight"), device)?,
                down: src.matrix(&name("ffn_down.weight"), device)?,
                up: src.matrix(&name("ffn_up.weight"), device)?,
            })
        } else {
            FeedForward::Experts(src.matrix(&name("ffn_gate_inp.weight"), device)?)
        };
        Ok(Self {
            q: Projection::load(src, &name("attn_q"), device)?,
            k: Projection::load(src, &name("attn_k"), device)?,
            v: Projection::load(src, &name("attn_v"), device)?,
            output: src.matrix(&name("attn_output.weight"), device)?,
            attn_norm: src.vector(&name("attn_norm.weight"), device)?,
            ffn_norm: src.vector(&name("ffn_norm.weight"), device)?,
            ff,
        })
    }

    fn attention(&self, x: &Tensor, state: &Step<'_>, kv: &mut Kv) -> Result<Tensor> {
        let c = state.cfg;
        let (seq, dim) = x.dims2()?;
        let z = candle_nn::ops::rms_norm(x, &self.attn_norm, c.epsilon)?;
        let project = |mat: &Projection, heads| -> Result<Tensor> {
            Ok(mat
                .forward(&z)?
                .reshape((1, seq, heads, c.head_dim()))?
                .transpose(1, 2)?
                .contiguous()?)
        };
        let cos = state.rope.cos.narrow(0, state.pos, seq)?;
        let sin = state.rope.sin.narrow(0, state.pos, seq)?;
        let rotate = match c.architecture {
            Architecture::Llama => candle_nn::rotary_emb::rope_i,
            Architecture::Qwen2 => candle_nn::rotary_emb::rope,
        };
        let q = rotate(&project(&self.q, c.heads)?, &cos, &sin)?;
        let mut k = rotate(&project(&self.k, c.kv_heads)?, &cos, &sin)?;
        let mut v = project(&self.v, c.kv_heads)?;
        if let (Some(keys), Some(values)) = (&kv.keys, &kv.values) {
            k = Tensor::cat(&[keys, &k], 2)?;
            v = Tensor::cat(&[values, &v], 2)?;
        }
        kv.keys = Some(k.clone());
        kv.values = Some(v.clone());
        let total = state.pos + seq;
        // Batch the query heads by their shared KV head without expanding the KV cache.
        let q = q.reshape((c.kv_heads, c.heads / c.kv_heads * seq, c.head_dim()))?;
        let k = k.reshape((c.kv_heads, total, c.head_dim()))?;
        let v = v.reshape((c.kv_heads, total, c.head_dim()))?;
        let mut scores = (q.matmul(&k.transpose(1, 2)?)? * (c.head_dim() as f64).sqrt().recip())?
            .reshape((c.heads, seq, total))?;
        if let Some(mask) = state.mask {
            scores = scores.broadcast_add(mask)?;
        }
        let weights = candle_nn::ops::softmax_last_dim(&scores)?.reshape((
            c.kv_heads,
            c.heads / c.kv_heads * seq,
            total,
        ))?;
        let attn = weights
            .matmul(&v)?
            .reshape((c.heads, seq, c.head_dim()))?
            .transpose(0, 1)?
            .reshape((seq, dim))?;
        Ok((x + self.output.forward(&attn)?)?)
    }
}

struct Step<'a> {
    cfg: &'a Config,
    pos: usize,
    rope: &'a Rope,
    mask: Option<&'a Tensor>,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct IoStats {
    pub weight_read_bytes: u64,
    pub expert_hits: u64,
    pub expert_misses: u64,
}

#[derive(Default)]
struct ExpertCache {
    entries: HashMap<(usize, usize), CachedExpert>,
    used: u64,
    options: ExpertOptions,
    tick: u64,
    hits: u64,
    misses: u64,
}

struct CachedExpert {
    mlp: Mlp,
    bytes: u64,
    last_use: u64,
    uses: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, clap::ValueEnum, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum CachePolicy {
    #[default]
    Lru,
    Lfu,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct ExpertOptions {
    pub capacity: u64,
    pub policy: CachePolicy,
}

impl ExpertCache {
    fn apply(
        &mut self,
        src: &mut ModelFile,
        layer: usize,
        expert: usize,
        x: &Tensor,
    ) -> Result<Tensor> {
        self.tick += 1;
        let key = (layer, expert);
        if let Some(entry) = self.entries.get_mut(&key) {
            self.hits += 1;
            entry.last_use = self.tick;
            entry.uses += 1;
            return entry.mlp.forward(x);
        }
        self.misses += 1;
        let p = format!("blk.{layer}.ffn");
        let (gate, g) = src.expert_matrix(&format!("{p}_gate_exps.weight"), expert)?;
        let (down, d) = src.expert_matrix(&format!("{p}_down_exps.weight"), expert)?;
        let (up, u) = src.expert_matrix(&format!("{p}_up_exps.weight"), expert)?;
        let size = g + d + u;
        let mlp = Mlp { gate, down, up };
        let output = mlp.forward(x)?;
        if size <= self.options.capacity {
            while size > self.options.capacity - self.used {
                let oldest = *self
                    .entries
                    .iter()
                    .min_by_key(|(_, entry)| {
                        let frequency = match self.options.policy {
                            CachePolicy::Lru => 0,
                            CachePolicy::Lfu => entry.uses,
                        };
                        (frequency, entry.last_use)
                    })
                    .context("expert cache accounting")?
                    .0;
                let evicted = self.entries.remove(&oldest).context("expert cache entry")?;
                self.used -= evicted.bytes;
            }
            self.entries.insert(
                key,
                CachedExpert {
                    mlp,
                    bytes: size,
                    last_use: self.tick,
                    uses: 1,
                },
            );
            self.used += size;
        }
        Ok(output)
    }
}

fn top_experts(logits: &[f32], count: usize) -> Vec<(usize, f64)> {
    let mut scores = logits.iter().copied().enumerate().collect::<Vec<_>>();
    scores.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let max = scores[0].1;
    let mut selected = scores[..count]
        .iter()
        .map(|(i, score)| (*i, f64::from(score - max).exp()))
        .collect::<Vec<_>>();
    let sum = selected.iter().map(|(_, w)| *w).sum::<f64>();
    for (_, weight) in &mut selected {
        *weight /= sum;
    }
    selected
}

pub struct Model {
    src: ModelFile,
    layers: Vec<Layer>,
    embedding: QMatMul,
    norm: Tensor,
    output: QMatMul,
    output_bias: Option<Tensor>,
    rope: Rope,
    gpu: Option<(Device, Rope)>,
    plan: Plan,
    cache: ExpertCache,
}

pub struct Session {
    kv: Vec<Kv>,
    pos: usize,
}

impl Model {
    pub fn load(
        mut src: ModelFile,
        plan: Plan,
        context: usize,
        gpu_index: usize,
        experts: ExpertOptions,
    ) -> Result<Self> {
        ensure!(
            (src.config.experts > 0) == (plan.mode == Mode::MoeStream),
            "expert-packed Mixtral models require moe-stream mode, which requires a Mixtral model"
        );
        let cpu = &Device::Cpu;
        let gpu = if plan.gpu_layers > 0 {
            ensure!(
                cfg!(feature = "cuda"),
                "build with --features cuda to offload layers"
            );
            let dev = Device::new_cuda(gpu_index).context("initialize CUDA")?;
            let rope = Rope::new(&src.config, context, &dev)?;
            Some((dev, rope))
        } else {
            None
        };
        let embedding = src.matrix("token_embd.weight", cpu)?;
        let norm = src.vector("output_norm.weight", cpu)?;
        let output = if src.output_name() == "token_embd.weight" {
            embedding.clone()
        } else {
            src.matrix(src.output_name(), cpu)?
        };
        let output_bias = src.bias("output.bias", cpu)?;
        let mut layers = Vec::new();
        if plan.mode != Mode::DenseStream {
            for idx in 0..src.config.blocks {
                let dev = gpu
                    .as_ref()
                    .filter(|_| idx >= plan.cpu_layers)
                    .map_or(cpu, |(d, _)| d);
                layers.push(Layer::load(&mut src, idx, dev)?);
            }
        }
        let rope = Rope::new(&src.config, context, cpu)?;
        Ok(Self {
            src,
            layers,
            embedding,
            norm,
            output,
            output_bias,
            rope,
            gpu,
            plan,
            cache: ExpertCache {
                options: experts,
                ..Default::default()
            },
        })
    }

    pub fn session(&self) -> Session {
        Session {
            kv: (0..self.src.config.blocks).map(|_| Kv::default()).collect(),
            pos: 0,
        }
    }

    pub fn stats(&self) -> IoStats {
        IoStats {
            weight_read_bytes: self.src.read_bytes,
            expert_hits: self.cache.hits,
            expert_misses: self.cache.misses,
        }
    }

    pub fn forward(
        &mut self,
        ids: &[u32],
        session: &mut Session,
        logits: bool,
    ) -> Result<Option<Vec<f32>>> {
        let c = &self.src.config;
        let seq = ids.len();
        let total = session.pos + seq;
        let mask = if seq > 1 || c.sliding_window.is_some_and(|w| total > w) {
            let mut values = Vec::with_capacity(seq * total);
            for i in session.pos..total {
                values.extend((0..total).map(|j| {
                    if j > i || c.sliding_window.is_some_and(|w| i - j >= w) {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                }));
            }
            Some(Tensor::from_vec(values, (seq, total), &Device::Cpu)?)
        } else {
            None
        };
        let gpu_mask = match (&self.gpu, &mask) {
            (Some((dev, _)), Some(mask)) => Some(mask.to_device(dev)?),
            _ => None,
        };
        let mut x = self.embedding.embedding(&Tensor::new(ids, &Device::Cpu)?)?;
        for index in 0..c.blocks {
            let (device, rope, mask) = match &self.gpu {
                Some((d, r)) if index >= self.plan.cpu_layers => (d, r, gpu_mask.as_ref()),
                _ => (&Device::Cpu, &self.rope, mask.as_ref()),
            };
            let stream;
            let layer = if self.plan.mode == Mode::DenseStream {
                stream = Layer::load(&mut self.src, index, device)?;
                &stream
            } else {
                &self.layers[index]
            };
            x = x.to_device(device)?;
            let step = Step {
                cfg: &self.src.config,
                pos: session.pos,
                rope,
                mask,
            };
            x = layer.attention(&x, &step, &mut session.kv[index])?;
            let norm = candle_nn::ops::rms_norm(&x, &layer.ffn_norm, self.src.config.epsilon)?;
            let ff = match &layer.ff {
                FeedForward::Dense(mlp) => mlp.forward(&norm)?,
                FeedForward::Experts(router) => {
                    let scores = router.forward(&norm)?.to_vec2::<f32>()?;
                    let mut rows = Vec::with_capacity(seq);
                    for (i, logits) in scores.iter().enumerate() {
                        ensure!(
                            logits.iter().all(|v| v.is_finite()),
                            "non-finite expert router logits"
                        );
                        let routes = top_experts(logits, self.src.config.experts_used);
                        let row = norm.narrow(0, i, 1)?;
                        let mut out = Tensor::zeros_like(&row)?;
                        for (expert, weight) in routes {
                            let expert_out =
                                self.cache.apply(&mut self.src, index, expert, &row)?;
                            out = (out + (expert_out * weight)?)?;
                        }
                        rows.push(out);
                    }
                    Tensor::cat(&rows, 0)?
                }
            };
            x = (&x + ff)?;
        }
        session.pos = total;
        if !logits {
            return Ok(None);
        }
        let last = x.narrow(0, seq - 1, 1)?.to_device(&Device::Cpu)?;
        let last = candle_nn::ops::rms_norm(&last, &self.norm, self.src.config.epsilon)?;
        let output = add_bias(self.output.forward(&last)?, self.output_bias.as_ref())?;
        Ok(Some(output.squeeze(0)?.to_vec1::<f32>()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routing_selects_and_normalizes_only_the_top_experts() {
        let routes = top_experts(&[1., 8., 6., 0.], 2);
        assert_eq!([routes[0].0, routes[1].0], [1, 2]);
        assert!((routes[0].1 - 1. / (1. + (-2f64).exp())).abs() < 1e-10);
        assert!((routes.iter().map(|p| p.1).sum::<f64>() - 1.).abs() < 1e-10);
    }

    #[test]
    #[ignore = "requires the pinned TinyStories Q4_0 GGUF"]
    fn streamed_and_resident_logits_agree() {
        let path = std::env::var_os("ORYNDRAL_TEST_GGUF").expect("pinned GGUF path");
        let load = |mode| {
            let file = ModelFile::open(Path::new(&path)).unwrap();
            let memory = file.memory(128, 0).unwrap();
            let options = crate::planner::PlanOptions {
                mode,
                ..Default::default()
            };
            let plan = crate::planner::plan(&memory, options).unwrap();
            Model::load(file, plan, 128, 0, ExpertOptions::default()).unwrap()
        };
        let (mut resident, mut streamed) = (load(Mode::Resident), load(Mode::DenseStream));
        let (mut a, mut b) = (resident.session(), streamed.session());
        for ids in [&[1, 9038, 2501, 263, 931][..], &[29892], &[727]] {
            let a = resident.forward(ids, &mut a, true).unwrap().unwrap();
            let b = streamed.forward(ids, &mut b, true).unwrap().unwrap();
            assert_eq!(a.len(), b.len());
            let max = a
                .iter()
                .zip(&b)
                .map(|(a, b)| (a - b).abs())
                .fold(0., f32::max);
            assert!(
                max <= 1e-5 && a.iter().chain(&b).all(|v| v.is_finite()),
                "max logit delta {max}"
            );
        }
    }
}
