struct UniformConfig {
    num_heads: u32,
    head_dim: u32,
    scale: f32,
    _pad: u32,
};

@group(0) @binding(0) var<uniform> cfg: UniformConfig;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> qkv: array<f32>;
@group(0) @binding(3) var<storage, read_write> state: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> output: array<vec4<f32>>;

var<workgroup> shared_acc: array<array<vec4<f32>, 64>, 4>;

@compute @workgroup_size(64, 4, 1)
fn retention_recurrent(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let head = global_id.z;
    let i = global_id.x;
    let j_vec = global_id.y;
    let tx = local_id.x;
    let ty = local_id.y;

    let vec_dim = cfg.head_dim / 4u;
    let total_dim = cfg.num_heads * cfg.head_dim;

    if (head >= cfg.num_heads || i >= cfg.head_dim || j_vec >= vec_dim) {
        return;
    }

    let g = gamma[head];
    let k_val = qkv[total_dim + head * cfg.head_dim + i];
    let v_base = 2u * total_dim + (head * cfg.head_dim + j_vec * 4u);
    let v_vec = vec4<f32>(
        qkv[v_base],
        qkv[v_base + 1u],
        qkv[v_base + 2u],
        qkv[v_base + 3u],
    );

    let state_idx = (head * cfg.head_dim * vec_dim) + (i * vec_dim + j_vec);
    let prev_s = state[state_idx];
    let kv = v_vec * (k_val * cfg.scale);
    let new_s = prev_s * g + kv;
    state[state_idx] = new_s;

    let q_val = qkv[head * cfg.head_dim + i];
    shared_acc[ty][tx] = new_s * q_val;
    workgroupBarrier();

    for (var step: u32 = 32u; step > 0u; step = step >> 1u) {
        if (tx < step && tx + step < cfg.head_dim) {
            shared_acc[ty][tx] = shared_acc[ty][tx] + shared_acc[ty][tx + step];
        }
        workgroupBarrier();
    }

    if (tx == 0u) {
        output[head * vec_dim + j_vec] = shared_acc[ty][0];
    }
}
