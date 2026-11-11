struct RetainParams {
    heads: u32,
    dim: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read_write> qkv: array<f32>;
@group(0) @binding(1) var<storage, read_write> state: array<f32>;
@group(0) @binding(2) var<storage, read_write> gamma: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> rp: RetainParams;

@compute @workgroup_size(64)
fn retention_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let h = gid.x;
    if (h >= rp.heads) {
        return;
    }
    let d = rp.dim;
    let q_off = h * d;
    let k_off = rp.heads * d + h * d;
    let v_off = 2u * rp.heads * d + h * d;
    let state_off = h * d * d;
    let decay = gamma[h];
    for (var col = 0u; col < d; col = col + 1u) {
        var acc = 0.0;
        for (var row = 0u; row < d; row = row + 1u) {
            let idx = state_off + row * d + col;
            let updated = decay * state[idx] + qkv[k_off + row] * qkv[v_off + col];
            state[idx] = updated;
            acc = acc + qkv[q_off + row] * updated;
        }
        out[h * d + col] = acc;
    }
}
