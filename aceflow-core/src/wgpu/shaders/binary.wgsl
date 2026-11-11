struct BinParams {
    n: u32,
    op: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read_write> lhs: array<f32>;
@group(0) @binding(1) var<storage, read_write> rhs: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: BinParams;

@compute @workgroup_size(256)
fn binary_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) {
        return;
    }
    if (params.op == 0u) {
        out[i] = lhs[i] + rhs[i];
    } else {
        out[i] = lhs[i] * rhs[i];
    }
}
