use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UniformConfig {
    pub num_heads: u32,
    pub head_dim: u32,
    pub scale: f32,
    pub _pad: u32,
}

pub struct WgpuBuffers {
    pub num_heads: u32,
    pub head_dim: u32,
    pub num_layers: u32,
    pub state_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    pub gamma_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    pub config_buffer: wgpu::Buffer,
    pub qkv_buf: wgpu::Buffer,
    pub out_buf: wgpu::Buffer,
    pub staging_buf: wgpu::Buffer,
    pub bind_groups: Vec<wgpu::BindGroup>,
}

pub type WgpuState = WgpuBuffers;

impl WgpuBuffers {
    pub fn new(
        device: &wgpu::Device,
        bind_group_layout: &wgpu::BindGroupLayout,
        num_heads: usize,
        head_dim: usize,
        num_layers: usize,
    ) -> Self {
        let gamma: Vec<f32> = (0..num_heads)
            .map(|head| 1.0 - 2.0f32.powf(-5.0 - head as f32))
            .collect();
        let gamma_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GammaBuffer"),
            contents: bytemuck::cast_slice(&gamma),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let layer_state_size =
            (num_heads * head_dim * head_dim * std::mem::size_of::<f32>()) as u64;
        let state_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("StateBuffer"),
            size: layer_state_size * num_layers as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let config = UniformConfig {
            num_heads: num_heads as u32,
            head_dim: head_dim as u32,
            scale: 1.0 / (head_dim as f32).sqrt(),
            _pad: 0,
        };
        let config_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ConfigBuffer"),
            contents: bytemuck::bytes_of(&config),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let d_size = num_heads * head_dim;
        let byte_size = (d_size * std::mem::size_of::<f32>()) as u64;
        let qkv_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("QKV"),
            size: byte_size * 3,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging"),
            size: byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut bind_groups = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            let offset = layer as u64 * layer_state_size;
            bind_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("RetentionBindGroup_{layer}")),
                layout: bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: config_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: gamma_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: qkv_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &state_buffer,
                            offset,
                            size: std::num::NonZeroU64::new(layer_state_size),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: out_buf.as_entire_binding(),
                    },
                ],
            }));
        }

        Self {
            num_heads: num_heads as u32,
            head_dim: head_dim as u32,
            num_layers: num_layers as u32,
            state_buffer,
            gamma_buffer,
            config_buffer,
            qkv_buf,
            out_buf,
            staging_buf,
            bind_groups,
        }
    }
}
