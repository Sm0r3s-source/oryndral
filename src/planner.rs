use anyhow::{Context, Result, ensure};
use serde::Serialize;
use std::process::Command;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Mode {
    Resident,
    Hybrid,
    DenseStream,
    MoeStream,
}

/// Estimates for a single active sequence at the configured context length.
pub struct ModelMemory {
    pub non_layer_bytes: u64,
    pub layers: Vec<u64>,
    pub kv_per_layer_bytes: u64,
    pub ram_scratch_bytes: u64,
    pub vram_scratch_bytes: u64,
}

pub struct PlanOptions {
    pub mode: Mode,
    /// Explicit contiguous suffix length. In hybrid mode `None` uses the VRAM budget.
    pub gpu_layers: Option<usize>,
    pub ram_budget: Option<u64>,
    pub vram_budget: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct Plan {
    pub mode: Mode,
    pub cpu_layers: usize,
    pub gpu_layers: usize,
    pub ram_bytes: u64,
    pub vram_bytes: u64,
    pub ram_budget: Option<u64>,
    pub vram_budget: Option<u64>,
}

/// Estimate peak active placement, then enforce any supplied limits.
/// Allocator overhead, filesystem cache and competing workloads are outside these estimates.
pub fn plan(model: &ModelMemory, options: PlanOptions) -> Result<Plan> {
    let layer_count = model.layers.len();
    ensure!(layer_count != 0, "model has no transformer layers");
    let streaming = matches!(options.mode, Mode::DenseStream | Mode::MoeStream);
    let gpu_layers = match (options.mode, options.gpu_layers, options.vram_budget) {
        (Mode::Hybrid, None, Some(budget)) => {
            let mut bytes = model.vram_scratch_bytes;
            let mut count = 0;
            for &layer in model.layers.iter().rev() {
                bytes = add(add(bytes, layer)?, model.kv_per_layer_bytes)?;
                if bytes > budget {
                    break;
                }
                count += 1;
            }
            count
        }
        (_, count, _) => count.unwrap_or(0),
    };
    ensure!(
        gpu_layers <= layer_count,
        "requested {gpu_layers} GPU layers, but the model has {layer_count} layers"
    );
    ensure!(
        !streaming || gpu_layers == 0,
        "streaming mode currently requires CPU execution"
    );
    ensure!(
        options.mode != Mode::Resident || gpu_layers == 0 || gpu_layers == layer_count,
        "use hybrid mode to split transformer layers between CPU and CUDA"
    );
    let cpu_layers = layer_count - gpu_layers;
    let (cpu_weights, gpu_weights) = model.layers.split_at(cpu_layers);
    let cpu_weight_bytes = if options.mode == Mode::DenseStream {
        cpu_weights.iter().copied().max().unwrap_or(0)
    } else {
        sum(cpu_weights)?
    };
    let ram_bytes = sum(&[
        model.non_layer_bytes,
        cpu_weight_bytes,
        kv_bytes(model, cpu_layers)?,
        model.ram_scratch_bytes,
    ])?;
    let vram_bytes = if gpu_layers == 0 {
        0
    } else {
        sum(&[
            sum(gpu_weights)?,
            kv_bytes(model, gpu_layers)?,
            model.vram_scratch_bytes,
        ])?
    };
    check_budget("RAM", ram_bytes, options.ram_budget)?;
    check_budget("VRAM", vram_bytes, options.vram_budget)?;
    Ok(Plan {
        mode: options.mode,
        cpu_layers,
        gpu_layers,
        ram_bytes,
        vram_bytes,
        ram_budget: options.ram_budget,
        vram_budget: options.vram_budget,
    })
}

fn add(left: u64, right: u64) -> Result<u64> {
    left.checked_add(right).context("memory estimate overflow")
}

fn sum(values: &[u64]) -> Result<u64> {
    values.iter().copied().try_fold(0, add)
}

fn kv_bytes(model: &ModelMemory, layers: usize) -> Result<u64> {
    model
        .kv_per_layer_bytes
        .checked_mul(layers as u64)
        .context("KV cache estimate overflow")
}

fn check_budget(resource: &str, required: u64, budget: Option<u64>) -> Result<()> {
    if let Some(limit) = budget {
        ensure!(
            required <= limit,
            "estimated {resource} need is {required} bytes, above the {limit}-byte budget"
        );
    }
    Ok(())
}

#[derive(Debug, Serialize)]
pub struct Hardware {
    pub os: &'static str,
    pub arch: &'static str,
    pub logical_threads: usize,
    pub avx2: bool,
    pub ram_total_bytes: Option<u64>,
    pub ram_available_bytes: Option<u64>,
    pub nvidia_gpus: Vec<NvidiaGpu>,
}

#[derive(Debug, Serialize)]
pub struct NvidiaGpu {
    /// This is NVIDIA's physical index, which can differ from a visible CUDA ordinal.
    pub index: usize,
    pub name: String,
    pub total_bytes: Option<u64>,
    pub available_bytes: Option<u64>,
    pub compute_capability: Option<String>,
}

pub fn hardware() -> Hardware {
    let (ram_total_bytes, ram_available_bytes) = ram();
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let avx2 = std::is_x86_feature_detected!("avx2");
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    let avx2 = false;
    Hardware {
        os: std::env::consts::OS,
        arch: std::env::consts::ARCH,
        logical_threads: std::thread::available_parallelism().map_or(1, usize::from),
        avx2,
        ram_total_bytes,
        ram_available_bytes,
        nvidia_gpus: nvidia_gpus(),
    }
}

fn nvidia_gpus() -> Vec<NvidiaGpu> {
    let result = Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,memory.total,memory.free,compute_cap",
            "--format=csv,noheader,nounits",
        ])
        .output();
    match result {
        Ok(output) if output.status.success() => {
            parse_gpu_rows(&String::from_utf8_lossy(&output.stdout))
        }
        _ => Vec::new(),
    }
}

fn parse_gpu_rows(input: &str) -> Vec<NvidiaGpu> {
    input
        .lines()
        .filter_map(|line| {
            let (index, rest) = line.split_once(',')?;
            let mut columns = rest.rsplitn(4, ',').map(str::trim);
            let capability = columns.next()?;
            let available_bytes = mib(columns.next()?);
            let total_bytes = mib(columns.next()?);
            let name = columns.next()?.trim_matches('"').to_owned();
            Some(NvidiaGpu {
                index: index.trim().parse().ok()?,
                name,
                total_bytes,
                available_bytes,
                compute_capability: capability.split_once('.').map(|_| capability.to_owned()),
            })
        })
        .collect()
}

fn mib(value: &str) -> Option<u64> {
    value.parse::<u64>().ok()?.checked_mul(1024 * 1024)
}

#[derive(Debug, Default, Serialize)]
pub struct ProcessMemory {
    pub rss_bytes: Option<u64>,
    pub peak_rss_bytes: Option<u64>,
}

#[derive(Debug, Default, Serialize)]
pub struct ProcessIo {
    /// Cumulative bytes attributed to process read operations, including cached reads.
    pub read_transfer_bytes: Option<u64>,
    /// Cumulative bytes attributed to storage reads where the OS exposes this counter.
    pub storage_read_bytes: Option<u64>,
}

#[cfg(target_os = "linux")]
fn proc_value(contents: &str, key: &str) -> Option<u64> {
    let line = contents.lines().find_map(|line| line.strip_prefix(key))?;
    let mut fields = line.split_whitespace();
    let value = fields.next()?.parse::<u64>().ok()?;
    if fields.next() == Some("kB") {
        value.checked_mul(1024)
    } else {
        Some(value)
    }
}

#[cfg(target_os = "linux")]
fn ram() -> (Option<u64>, Option<u64>) {
    let info = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    (
        proc_value(&info, "MemTotal:"),
        proc_value(&info, "MemAvailable:"),
    )
}

#[cfg(target_os = "linux")]
pub fn process_memory() -> ProcessMemory {
    let info = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    ProcessMemory {
        rss_bytes: proc_value(&info, "VmRSS:"),
        peak_rss_bytes: proc_value(&info, "VmHWM:"),
    }
}

#[cfg(target_os = "linux")]
pub fn process_io() -> ProcessIo {
    let info = std::fs::read_to_string("/proc/self/io").unwrap_or_default();
    ProcessIo {
        read_transfer_bytes: proc_value(&info, "rchar:"),
        storage_read_bytes: proc_value(&info, "read_bytes:"),
    }
}

#[cfg(windows)]
use windows::ram;
#[cfg(windows)]
pub use windows::{process_io, process_memory};

#[cfg(windows)]
mod windows {
    use super::{ProcessIo, ProcessMemory};
    use std::ffi::c_void;
    use std::mem::size_of;

    #[repr(C)]
    #[derive(Default)]
    struct MemoryStatus {
        length: u32,
        load: u32,
        total_physical: u64,
        available_physical: u64,
        total_page_file: u64,
        available_page_file: u64,
        total_virtual: u64,
        available_virtual: u64,
        available_extended_virtual: u64,
    }

    #[repr(C)]
    #[derive(Default)]
    struct MemoryCounters {
        length: u32,
        page_faults: u32,
        peak_working_set: usize,
        working_set: usize,
        peak_paged_pool: usize,
        paged_pool: usize,
        peak_nonpaged_pool: usize,
        nonpaged_pool: usize,
        page_file: usize,
        peak_page_file: usize,
    }

    #[repr(C)]
    #[derive(Default)]
    struct IoCounters {
        read_operations: u64,
        write_operations: u64,
        other_operations: u64,
        read_transfer: u64,
        write_transfer: u64,
        other_transfer: u64,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(status: *mut MemoryStatus) -> i32;
        fn GetCurrentProcess() -> *mut c_void;
        fn K32GetProcessMemoryInfo(
            process: *mut c_void,
            counters: *mut MemoryCounters,
            length: u32,
        ) -> i32;
        fn GetProcessIoCounters(process: *mut c_void, counters: *mut IoCounters) -> i32;
    }

    pub(super) fn ram() -> (Option<u64>, Option<u64>) {
        let mut status = MemoryStatus {
            length: size_of::<MemoryStatus>() as u32,
            ..Default::default()
        };
        // SAFETY: the writable C-layout buffer includes its required byte length.
        if unsafe { GlobalMemoryStatusEx(&mut status) } == 0 {
            (None, None)
        } else {
            (Some(status.total_physical), Some(status.available_physical))
        }
    }

    pub fn process_memory() -> ProcessMemory {
        let length = size_of::<MemoryCounters>() as u32;
        let mut counters = MemoryCounters {
            length,
            ..Default::default()
        };
        // SAFETY: the pseudo-handle identifies this process and the buffer is writable.
        if unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, length) } == 0 {
            ProcessMemory::default()
        } else {
            ProcessMemory {
                rss_bytes: Some(counters.working_set as u64),
                peak_rss_bytes: Some(counters.peak_working_set as u64),
            }
        }
    }

    pub fn process_io() -> ProcessIo {
        let mut counters = IoCounters::default();
        // SAFETY: the pseudo-handle identifies this process and the buffer is writable.
        if unsafe { GetProcessIoCounters(GetCurrentProcess(), &mut counters) } == 0 {
            ProcessIo::default()
        } else {
            ProcessIo {
                read_transfer_bytes: Some(counters.read_transfer),
                storage_read_bytes: None,
            }
        }
    }
}

#[cfg(not(any(windows, target_os = "linux")))]
fn ram() -> (Option<u64>, Option<u64>) {
    (None, None)
}

#[cfg(not(any(windows, target_os = "linux")))]
pub fn process_memory() -> ProcessMemory {
    ProcessMemory::default()
}

#[cfg(not(any(windows, target_os = "linux")))]
pub fn process_io() -> ProcessIo {
    ProcessIo::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> ModelMemory {
        ModelMemory {
            non_layer_bytes: 40,
            layers: vec![50, 100, 80],
            kv_per_layer_bytes: 5,
            ram_scratch_bytes: 10,
            vram_scratch_bytes: 15,
        }
    }

    fn options(mode: Mode) -> PlanOptions {
        PlanOptions {
            mode,
            gpu_layers: None,
            ram_budget: None,
            vram_budget: None,
        }
    }

    #[test]
    fn resident_and_streaming_estimates_account_for_all_kv() {
        let resident = plan(&model(), options(Mode::Resident)).unwrap();
        assert_eq!((resident.ram_bytes, resident.vram_bytes), (295, 0));
        let streamed = plan(
            &model(),
            PlanOptions {
                ram_budget: Some(165),
                ..options(Mode::DenseStream)
            },
        )
        .unwrap();
        assert_eq!((streamed.ram_bytes, streamed.vram_bytes), (165, 0));
        let moe = plan(&model(), options(Mode::MoeStream)).unwrap();
        assert_eq!((moe.ram_bytes, moe.vram_bytes), (295, 0));
    }

    #[test]
    fn auto_hybrid_fits_a_contiguous_suffix_with_kv_and_scratch() {
        let planned = plan(
            &model(),
            PlanOptions {
                vram_budget: Some(204),
                ..options(Mode::Hybrid)
            },
        )
        .unwrap();
        assert_eq!((planned.cpu_layers, planned.gpu_layers), (2, 1));
        assert_eq!((planned.ram_bytes, planned.vram_bytes), (210, 100));
    }

    #[test]
    fn explicit_placement_rejects_over_budget_and_invalid_modes() {
        for mode in [Mode::Resident, Mode::DenseStream, Mode::MoeStream] {
            assert!(
                plan(
                    &model(),
                    PlanOptions {
                        gpu_layers: Some(1),
                        ..options(mode)
                    },
                )
                .is_err()
            );
        }
        assert!(
            plan(
                &model(),
                PlanOptions {
                    gpu_layers: Some(1),
                    vram_budget: Some(99),
                    ..options(Mode::Hybrid)
                },
            )
            .is_err()
        );
        assert!(
            plan(
                &model(),
                PlanOptions {
                    ram_budget: Some(294),
                    ..options(Mode::Resident)
                },
            )
            .is_err()
        );
        let gpu_resident = plan(
            &model(),
            PlanOptions {
                gpu_layers: Some(3),
                ..options(Mode::Resident)
            },
        )
        .unwrap();
        assert_eq!((gpu_resident.ram_bytes, gpu_resident.vram_bytes), (50, 260));
    }

    #[test]
    fn malformed_gpu_rows_and_unavailable_fields_are_handled() {
        let gpus =
            parse_gpu_rows("junk\n0, RTX 4050, 6141, 5325, 8.9\n1, GPU, [N/A], [N/A], [N/A]");
        assert_eq!(gpus.len(), 2);
        assert_eq!(gpus[0].available_bytes, Some(5325 * 1024 * 1024));
        assert_eq!(gpus[0].compute_capability.as_deref(), Some("8.9"));
        assert_eq!(gpus[1].total_bytes, None);
    }
}
