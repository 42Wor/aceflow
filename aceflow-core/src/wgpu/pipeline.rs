use std::borrow::Cow;
use std::collections::HashMap;

/// Cached compute pipeline plus its inferred bind-group layout.
pub struct CachedPipeline {
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl CachedPipeline {
    fn from_wgsl(
        device: &wgpu::Device,
        label: &str,
        source: &'static str,
        entry_point: &str,
    ) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(source)),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: None,
            module: &module,
            entry_point,
            compilation_options: Default::default(),
        });
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        Self {
            pipeline,
            bind_group_layout,
        }
    }
}

/// Precompiled compute pipelines keyed by shader name (`"gemm"`, `"rmsnorm"`, ...).
pub struct PipelineCache {
    pipelines: HashMap<&'static str, CachedPipeline>,
}

impl PipelineCache {
    pub fn new(device: &wgpu::Device) -> Self {
        let mut pipelines = HashMap::new();
        let specs: &[(&str, &str, &str, &str)] = &[
            (
                "gemm",
                "mallow.gemm",
                include_str!("shaders/gemm.wgsl"),
                "gemm_main",
            ),
            (
                "conv1d",
                "mallow.conv1d",
                include_str!("shaders/conv1d.wgsl"),
                "conv1d_main",
            ),
            (
                "layernorm",
                "mallow.layernorm",
                include_str!("shaders/layernorm.wgsl"),
                "layernorm_main",
            ),
            (
                "softmax",
                "mallow.softmax",
                include_str!("shaders/softmax.wgsl"),
                "softmax_rows",
            ),
            (
                "attn_av",
                "mallow.attn_av",
                include_str!("shaders/attention.wgsl"),
                "attn_av_main",
            ),
            (
                "act",
                "mallow.act",
                include_str!("shaders/activations.wgsl"),
                "act_main",
            ),
            (
                "binary",
                "mallow.binary",
                include_str!("shaders/binary.wgsl"),
                "binary_main",
            ),
            (
                "retention_step",
                "mallow.retention_step",
                include_str!("shaders/retention.wgsl"),
                "retention_step",
            ),
            (
                "retention",
                "mallow.retention",
                include_str!("shaders/retention_recurrent.wgsl"),
                "retention_recurrent",
            ),
        ];
        for (key, label, source, entry) in specs {
            pipelines.insert(*key, CachedPipeline::from_wgsl(device, label, source, entry));
        }
        Self { pipelines }
    }

    pub fn get(&self, key: &str) -> Option<&CachedPipeline> {
        self.pipelines.get(key)
    }
}

/// Retention pipeline used by [`crate::wgpu::WgpuAceEngine`].
pub struct WgpuPipeline {
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl WgpuPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let cached = CachedPipeline::from_wgsl(
            device,
            "RetentionPipeline",
            include_str!("shaders/retention_recurrent.wgsl"),
            "retention_recurrent",
        );
        Self {
            pipeline: cached.pipeline,
            bind_group_layout: cached.bind_group_layout,
        }
    }
}
