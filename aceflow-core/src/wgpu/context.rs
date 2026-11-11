//! Lazy process-wide WGPU instance, adapter, device, queue, and pipeline cache.

use std::sync::{Arc, OnceLock};

use super::pipeline::PipelineCache;

pub struct WgpuRuntime {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub adapter_name: String,
    pub pipelines: PipelineCache,
}

pub struct WgpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub adapter_name: String,
}

impl Default for WgpuContext {
    fn default() -> Self {
        Self::new()
    }
}

impl WgpuContext {
    pub fn new() -> Self {
        match try_runtime() {
            Some(runtime) => Self {
                device: runtime.device.clone(),
                queue: runtime.queue.clone(),
                adapter_name: runtime.adapter_name.clone(),
            },
            None => panic!("Failed to find suitable GPU adapter"),
        }
    }
}

static RUNTIME: OnceLock<Option<WgpuRuntime>> = OnceLock::new();

/// Returns `true` when a Vulkan, DX12, or Metal adapter can be initialized.
pub fn probe() -> bool {
    try_runtime().is_some()
}

pub fn try_runtime() -> Option<&'static WgpuRuntime> {
    RUNTIME.get_or_init(init_runtime).as_ref()
}

pub fn runtime() -> Option<&'static WgpuRuntime> {
    try_runtime()
}

fn init_runtime() -> Option<WgpuRuntime> {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN
                | wgpu::Backends::DX12
                | wgpu::Backends::METAL
                | wgpu::Backends::BROWSER_WEBGPU,
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await?;
        let backend = adapter.get_info().backend;
        match backend {
            wgpu::Backend::Vulkan
            | wgpu::Backend::Dx12
            | wgpu::Backend::Metal
            | wgpu::Backend::BrowserWebGpu => {}
            _ => return None,
        }
        let adapter_name = adapter.get_info().name.clone();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .ok()?;
        let pipelines = PipelineCache::new(&device);
        Some(WgpuRuntime {
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter_name,
            pipelines,
        })
    })
}
