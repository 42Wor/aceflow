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

@group(0) @binding(0) var<storage, read_write> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: ConvParams;

@compute @workgroup_size(16, 16)
fn conv1d_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let oc = gid.y;
    let t = gid.x;
    if (oc >= params.c_out || t >= params.t_out) {
        return;
    }
    var acc = 0.0;
    if (arrayLength(&bias) > oc) {
        acc = bias[oc];
    }
    for (var ic = 0u; ic < params.c_in; ic = ic + 1u) {
        for (var ki = 0u; ki < params.kernel; ki = ki + 1u) {
            let input_t = i32(t * params.stride + ki) - i32(params.padding);
            if (input_t >= 0 && input_t < i32(params.t_in)) {
                let w = weight[(oc * params.c_in + ic) * params.kernel + ki];
                let x = input[ic * params.t_in + u32(input_t)];
                acc = acc + w * x;
            }
        }
    }
    output[oc * params.t_out + t] = acc;
}
