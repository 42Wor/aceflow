//! Backend storage tags for [`crate::tensor::Tensor`].
//!
//! Host `f32` values live on [`crate::tensor::Tensor::data`] so existing
//! field access (mallow-whisper) stays valid. This enum records whether that
//! buffer is the compute source (CPU SIMD) or a synchronized mirror of WGPU.

use crate::device::Device;

/// CPU contiguous buffer metadata. The `Vec<f32>` itself is `Tensor.data`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CpuStorage;

/// GPU placement. Host `Tensor.data` is kept in sync after every kernel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WgpuStorage {
    pub adapter: usize,
}

impl Default for WgpuStorage {
    fn default() -> Self {
        Self { adapter: 0 }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Storage {
    Cpu(CpuStorage),
    Wgpu(WgpuStorage),
}

impl Storage {
    pub fn from_device(device: Device) -> Self {
        match device.resolve() {
            Device::Cpu => Self::Cpu(CpuStorage),
            Device::Wgpu(adapter) => Self::Wgpu(WgpuStorage { adapter }),
        }
    }

    pub fn device(&self) -> Device {
        match self {
            Self::Cpu(_) => Device::Cpu,
            Self::Wgpu(storage) => Device::Wgpu(storage.adapter),
        }
    }

    pub fn is_wgpu(&self) -> bool {
        matches!(self, Self::Wgpu(_))
    }
}
