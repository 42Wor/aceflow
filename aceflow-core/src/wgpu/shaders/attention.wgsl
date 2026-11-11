struct AttnParams {
    tq: u32,
    tk: u32,
    dim: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> scores: array<f32>;
@group(0) @binding(1) var<storage, read_write> values: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: AttnParams;

@compute @workgroup_size(16, 16)
fn attn_av_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    if (row >= params.tq || col >= params.dim) {
        return;
    }
    var acc = 0.0;
    let tk = params.tk;
    for (var t = 0u; t < tk; t = t + 1u) {
        acc = acc + scores[row * tk + t] * values[t * params.dim + col];
    }
    output[row * params.dim + col] = acc;
}
