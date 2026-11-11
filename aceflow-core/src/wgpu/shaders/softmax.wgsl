struct SoftmaxParams {
    rows: u32,
    dim: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read_write> values: array<f32>;
@group(0) @binding(1) var<uniform> params: SoftmaxParams;

@compute @workgroup_size(64)
fn softmax_rows(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= params.rows) {
        return;
    }
    let dim = params.dim;
    let base = row * dim;
    var max_v = values[base];
    for (var j = 1u; j < dim; j = j + 1u) {
        max_v = max(max_v, values[base + j]);
    }
    var sum = 0.0;
    for (var j = 0u; j < dim; j = j + 1u) {
        let e = exp(values[base + j] - max_v);
        values[base + j] = e;
        sum = sum + e;
    }
    let inv = 1.0 / sum;
    for (var j = 0u; j < dim; j = j + 1u) {
        values[base + j] = values[base + j] * inv;
    }
}
