struct NormParams {
    rows: u32,
    dim: u32,
    eps: f32,
    kind: u32,
}

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> params: NormParams;

@compute @workgroup_size(64)
fn layernorm_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= params.rows) {
        return;
    }
    let dim = params.dim;
    let base = row * dim;

    if (params.kind == 0u) {
        var mean = 0.0;
        for (var j = 0u; j < dim; j = j + 1u) {
            mean = mean + x[base + j];
        }
        mean = mean / f32(dim);
        var var_acc = 0.0;
        for (var j = 0u; j < dim; j = j + 1u) {
            let c = x[base + j] - mean;
            var_acc = var_acc + c * c;
        }
        let inv = inverseSqrt(var_acc / f32(dim) + params.eps);
        for (var j = 0u; j < dim; j = j + 1u) {
            y[base + j] = (x[base + j] - mean) * inv * weight[j] + bias[j];
        }
    } else {
        var mean_sq = 0.0;
        for (var j = 0u; j < dim; j = j + 1u) {
            let v = x[base + j];
            mean_sq = mean_sq + v * v;
        }
        let inv = inverseSqrt(mean_sq / f32(dim) + params.eps);
        for (var j = 0u; j < dim; j = j + 1u) {
            y[base + j] = x[base + j] * inv * weight[j];
        }
    }
}
