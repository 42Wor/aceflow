//! Default compute device: WGPU first, CPU SIMD only when no adapter exists.

use std::fmt;
use std::sync::{Mutex, OnceLock};

/// Compute placement for tensors and modules.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Device {
    /// WebGPU adapter index (typically `0`).
    Wgpu(usize),
    /// Host AVX2/NEON SIMD fallback.
    Cpu,
}

impl Device {
    pub fn wgpu() -> Self {
        Self::Wgpu(0)
    }

    pub fn is_wgpu(self) -> bool {
        matches!(self, Self::Wgpu(_))
    }

    pub fn kind(self) -> &'static str {
        match self {
            Self::Wgpu(_) => "wgpu",
            Self::Cpu => "cpu",
        }
    }

    /// Map an unavailable WGPU request onto CPU.
    pub fn resolve(self) -> Self {
        match self {
            Self::Wgpu(_) if !wgpu_available() => Self::Cpu,
            other => other,
        }
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Wgpu(index) => write!(f, "wgpu:{index}"),
            Self::Cpu => write!(f, "cpu"),
        }
    }
}

impl std::str::FromStr for Device {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        parse_device_str(value)
    }
}

static DETECTED: OnceLock<Device> = OnceLock::new();
static OVERRIDE: Mutex<Option<Device>> = Mutex::new(None);

/// Process-wide default. Prefers `Device::Wgpu(0)` when a Vulkan/DX12/Metal
/// adapter is present; otherwise CPU with a one-time informational log.
pub fn default_device() -> Device {
    if let Ok(guard) = OVERRIDE.lock() {
        if let Some(device) = *guard {
            return device.resolve();
        }
    }
    *DETECTED.get_or_init(detect_default_device)
}

/// Override the process default (`"wgpu"` / `"cpu"` from Python).
pub fn set_default_device(device: Device) {
    if let Ok(mut guard) = OVERRIDE.lock() {
        *guard = Some(device);
    }
}

pub fn parse_device_str(value: &str) -> std::result::Result<Device, String> {
    let trimmed = value.trim().to_ascii_lowercase();
    if trimmed == "cpu" {
        return Ok(Device::Cpu);
    }
    if trimmed == "wgpu" || trimmed == "gpu" {
        return Ok(Device::Wgpu(0));
    }
    if let Some(rest) = trimmed.strip_prefix("wgpu:") {
        let index = rest
            .parse::<usize>()
            .map_err(|_| format!("invalid device '{value}'"))?;
        return Ok(Device::Wgpu(index));
    }
    Err(format!(
        "unknown device '{value}'; expected 'cpu', 'wgpu', or 'wgpu:<index>'"
    ))
}

pub fn promote_device(left: Device, right: Device) -> Device {
    match (left.resolve(), right.resolve()) {
        (Device::Wgpu(index), _) | (_, Device::Wgpu(index)) => Device::Wgpu(index),
        _ => Device::Cpu,
    }
}

pub fn wgpu_available() -> bool {
    #[cfg(feature = "wgpu-runtime")]
    {
        crate::wgpu::context::probe()
    }
    #[cfg(not(feature = "wgpu-runtime"))]
    {
        false
    }
}

fn detect_default_device() -> Device {
    #[cfg(feature = "wgpu-runtime")]
    {
        if wgpu_available() {
            return Device::Wgpu(0);
        }
        eprintln!(
            "mallow-core: no compatible WGPU adapter (Vulkan/DX12/Metal); falling back to CPU SIMD"
        );
        return Device::Cpu;
    }
    #[cfg(not(feature = "wgpu-runtime"))]
    {
        Device::Cpu
    }
}
