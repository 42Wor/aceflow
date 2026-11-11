//! Generic WGPU tensor kernels. Pipelines are resolved from the process cache.

use wgpu::util::DeviceExt;

use super::context::{runtime, WgpuRuntime};

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct U4 {
    a: u32,
    b: u32,
    c: u32,
    d: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct NormParams {
    rows: u32,
    dim: u32,
    eps: f32,
    kind: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ConvParams {
    c_out: u32,
    c_in: u32,
    kernel: u32,
    t_in: u32,
    t_out: u32,
    stride: u32,
    padding: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ActParams {
    n: u32,
    op: u32,
    scalar: f32,
    _pad: u32,
}

fn storage_init(runtime: &WgpuRuntime, data: &[f32]) -> wgpu::Buffer {
    if data.is_empty() {
        return runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mallow.empty_storage"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
    }
    runtime
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mallow.storage"),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        })
}

fn storage_empty(runtime: &WgpuRuntime, count: usize) -> wgpu::Buffer {
    runtime.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mallow.storage_out"),
        size: (count.max(1) * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn uniform_buf(runtime: &WgpuRuntime, bytes: &[u8]) -> wgpu::Buffer {
    let mut padded = bytes.to_vec();
    while padded.len() % 16 != 0 {
        padded.push(0);
    }
    runtime
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mallow.uniform"),
            contents: &padded,
            usage: wgpu::BufferUsages::UNIFORM,
        })
}

fn bind_group(
    runtime: &WgpuRuntime,
    key: &str,
    buffers: &[wgpu::BindGroupEntry<'_>],
) -> Option<wgpu::BindGroup> {
    let pipeline = runtime.pipelines.get(key)?;
    Some(
        runtime
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(key),
                layout: &pipeline.bind_group_layout,
                entries: buffers,
            }),
    )
}

fn submit_read(
    runtime: &WgpuRuntime,
    key: &str,
    bind_group: &wgpu::BindGroup,
    output: &wgpu::Buffer,
    count: usize,
    workgroups: [u32; 3],
) -> Option<Vec<f32>> {
    let pipeline = runtime.pipelines.get(key)?;
    let mut encoder = runtime
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mallow.dispatch"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(key),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(workgroups[0].max(1), workgroups[1].max(1), workgroups[2].max(1));
    }
    if count == 0 {
        runtime.queue.submit(Some(encoder.finish()));
        return Some(Vec::new());
    }
    let size = (count * 4) as u64;
    let staging = runtime.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mallow.staging"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(output, 0, &staging, 0, size);
    runtime.queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    runtime.device.poll(wgpu::Maintain::Wait);
    receiver.recv().ok()?.ok()?;
    let mapped = slice.get_mapped_range();
    let values: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    staging.unmap();
    Some(values)
}

pub fn gemm(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    transpose_b: bool,
) -> Option<Vec<f32>> {
    let runtime = runtime()?;
    if m == 0 || n == 0 {
        return Some(vec![0.0; m * n]);
    }
    let a_buf = storage_init(runtime, a);
    let b_buf = storage_init(runtime, b);
    let c_buf = storage_empty(runtime, m * n);
    let dims = U4 {
        a: m as u32,
        b: k as u32,
        c: n as u32,
        d: u32::from(transpose_b),
    };
    let ubo = uniform_buf(runtime, bytemuck::bytes_of(&dims));
    let group = bind_group(
        runtime,
        "gemm",
        &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: c_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: ubo.as_entire_binding(),
            },
        ],
    )?;
    submit_read(
        runtime,
        "gemm",
        &group,
        &c_buf,
        m * n,
        [((n as u32) + 15) / 16, ((m as u32) + 15) / 16, 1],
    )
}

pub fn binary(lhs: &[f32], rhs: &[f32], op: u32) -> Option<Vec<f32>> {
    let runtime = runtime()?;
    let n = lhs.len();
    let a_buf = storage_init(runtime, lhs);
    let b_buf = storage_init(runtime, rhs);
    let c_buf = storage_empty(runtime, n);
    let params = ActParams {
        n: n as u32,
        op,
        scalar: 0.0,
        _pad: 0,
    };
    let ubo = uniform_buf(runtime, bytemuck::bytes_of(&params));
    let group = bind_group(
        runtime,
        "binary",
        &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: c_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: ubo.as_entire_binding(),
            },
        ],
    )?;
    submit_read(
        runtime,
        "binary",
        &group,
        &c_buf,
        n,
        [((n as u32) + 255) / 256, 1, 1],
    )
}

pub fn unary(values: &[f32], op: u32, scalar: f32) -> Option<Vec<f32>> {
    let runtime = runtime()?;
    let n = values.len();
    let x_buf = storage_init(runtime, values);
    let y_buf = storage_empty(runtime, n);
    let params = ActParams {
        n: n as u32,
        op,
        scalar,
        _pad: 0,
    };
    let ubo = uniform_buf(runtime, bytemuck::bytes_of(&params));
    let group = bind_group(
        runtime,
        "act",
        &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: x_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: y_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: ubo.as_entire_binding(),
            },
        ],
    )?;
    submit_read(
        runtime,
        "act",
        &group,
        &y_buf,
        n,
        [((n as u32) + 255) / 256, 1, 1],
    )
}

pub fn softmax_rows(values: &[f32], rows: usize, dim: usize) -> Option<Vec<f32>> {
    let runtime = runtime()?;
    let buf = storage_init(runtime, values);
    let params = U4 {
        a: rows as u32,
        b: dim as u32,
        c: 0,
        d: 0,
    };
    let ubo = uniform_buf(runtime, bytemuck::bytes_of(&params));
    let group = bind_group(
        runtime,
        "softmax",
        &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: ubo.as_entire_binding(),
            },
        ],
    )?;
    submit_read(
        runtime,
        "softmax",
        &group,
        &buf,
        values.len(),
        [((rows as u32) + 63) / 64, 1, 1],
    )
}

pub fn normalize(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    rows: usize,
    dim: usize,
    eps: f32,
    rms: bool,
) -> Option<Vec<f32>> {
    let runtime = runtime()?;
    let x_buf = storage_init(runtime, x);
    let w_buf = storage_init(runtime, weight);
    let b_buf = storage_init(runtime, bias);
    let y_buf = storage_empty(runtime, x.len());
    let params = NormParams {
        rows: rows as u32,
        dim: dim as u32,
        eps,
        kind: u32::from(rms),
    };
    let ubo = uniform_buf(runtime, bytemuck::bytes_of(&params));
    let group = bind_group(
        runtime,
        "layernorm",
        &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: x_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: w_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: b_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: y_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: ubo.as_entire_binding(),
            },
        ],
    )?;
    submit_read(
        runtime,
        "layernorm",
        &group,
        &y_buf,
        x.len(),
        [((rows as u32) + 63) / 64, 1, 1],
    )
}

pub fn conv1d(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    c_out: usize,
    c_in: usize,
    kernel: usize,
    t_in: usize,
    t_out: usize,
    stride: usize,
    padding: usize,
) -> Option<Vec<f32>> {
    let runtime = runtime()?;
    let dummy = vec![0.0f32; c_out.max(1)];
    let bias_data = bias.unwrap_or(&dummy);
    let x_buf = storage_init(runtime, input);
    let w_buf = storage_init(runtime, weight);
    let b_buf = storage_init(runtime, bias_data);
    let y_buf = storage_empty(runtime, c_out * t_out);
    let params = ConvParams {
        c_out: c_out as u32,
        c_in: c_in as u32,
        kernel: kernel as u32,
        t_in: t_in as u32,
        t_out: t_out as u32,
        stride: stride as u32,
        padding: padding as u32,
        _pad: 0,
    };
    let ubo = uniform_buf(runtime, bytemuck::bytes_of(&params));
    let group = bind_group(
        runtime,
        "conv1d",
        &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: x_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: w_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: b_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: y_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: ubo.as_entire_binding(),
            },
        ],
    )?;
    submit_read(
        runtime,
        "conv1d",
        &group,
        &y_buf,
        c_out * t_out,
        [((t_out as u32) + 15) / 16, ((c_out as u32) + 15) / 16, 1],
    )
}

pub fn attn_av(scores: &[f32], values: &[f32], tq: usize, tk: usize, dim: usize) -> Option<Vec<f32>> {
    let runtime = runtime()?;
    let s_buf = storage_init(runtime, scores);
    let v_buf = storage_init(runtime, values);
    let o_buf = storage_empty(runtime, tq * dim);
    let params = U4 {
        a: tq as u32,
        b: tk as u32,
        c: dim as u32,
        d: 0,
    };
    let ubo = uniform_buf(runtime, bytemuck::bytes_of(&params));
    let group = bind_group(
        runtime,
        "attn_av",
        &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: s_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: v_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: o_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: ubo.as_entire_binding(),
            },
        ],
    )?;
    submit_read(
        runtime,
        "attn_av",
        &group,
        &o_buf,
        tq * dim,
        [((dim as u32) + 15) / 16, ((tq as u32) + 15) / 16, 1],
    )
}
