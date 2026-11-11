struct ActParams {
    n: u32,
    op: u32,
    scalar: f32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> params: ActParams;

fn gelu_erf(v: f32) -> f32 {
    let k = 0.7071067811865476;
    let ax = abs(v * k);
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let yv = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-ax * ax);
    let erf = select(-yv, yv, v >= 0.0);
    return 0.5 * v * (1.0 + erf);
}

@compute @workgroup_size(64)
fn act_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x * 4u;
    if (i >= params.n) {
        return;
    }
    let remaining = min(4u, params.n - i);
    for (var k = 0u; k < remaining; k = k + 1u) {
        let v = x[i + k];
        var o = v;
        if (params.op == 0u) {
            o = gelu_erf(v);
        } else if (params.op == 1u) {
            o = v / (1.0 + exp(-v));
        } else if (params.op == 2u) {
            o = v * params.scalar;
        }
        y[i + k] = o;
    }
}
