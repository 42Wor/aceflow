struct SwigluConfig {
    ffn_dim: u32,
}

@group(0) @binding(0) var<uniform> cfg: SwigluConfig;
@group(0) @binding(1) var<storage, read> gate_up: array<f32>;
@group(0) @binding(2) var<storage, read_write> hidden: array<f32>;

@compute @workgroup_size(64)
fn swiglu_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= cfg.ffn_dim) {
        return;
    }

    let g = gate_up[idx];
    let u = gate_up[cfg.ffn_dim + idx];
    let silu_g = g / (1.0 + exp(-g));
    hidden[idx] = silu_g * u;
}
