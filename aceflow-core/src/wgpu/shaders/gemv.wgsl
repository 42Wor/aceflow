struct GemvConfig {
    m: u32,
    k_vec4: u32,
}

@group(0) @binding(0) var<uniform> cfg: GemvConfig;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> weights: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn gemv_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if (row >= cfg.m) {
        return;
    }

    var sum = 0.0;
    let row_offset = row * cfg.k_vec4;
    for (var j = 0u; j < cfg.k_vec4; j = j + 1u) {
        let xv = x[j];
        let wv = weights[row_offset + j];
        sum = sum + dot(xv, wv);
    }
    output[row] = sum;
}
