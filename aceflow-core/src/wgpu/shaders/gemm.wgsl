struct Dims {
    M: u32,
    K: u32,
    N: u32,
    flags: u32,
}

@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

const TILE: u32 = 16u;

var<workgroup> tile_a: array<array<f32, 16>, 16>;
var<workgroup> tile_b: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn gemm_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = gid.y;
    let col = gid.x;
    var acc = 0.0;
    let tiles = (dims.K + TILE - 1u) / TILE;
    let transpose_b = (dims.flags & 1u) == 1u;

    for (var t = 0u; t < tiles; t = t + 1u) {
        let a_col = t * TILE + lid.x;
        let b_row = t * TILE + lid.y;
        if (row < dims.M && a_col < dims.K) {
            tile_a[lid.y][lid.x] = A[row * dims.K + a_col];
        } else {
            tile_a[lid.y][lid.x] = 0.0;
        }
        if (b_row < dims.K && col < dims.N) {
            if (transpose_b) {
                tile_b[lid.y][lid.x] = B[col * dims.K + b_row];
            } else {
                tile_b[lid.y][lid.x] = B[b_row * dims.N + col];
            }
        } else {
            tile_b[lid.y][lid.x] = 0.0;
        }
        workgroupBarrier();
        for (var k = 0u; k < TILE; k = k + 1u) {
            acc = acc + tile_a[lid.y][k] * tile_b[k][lid.x];
        }
        workgroupBarrier();
    }

    if (row < dims.M && col < dims.N) {
        C[row * dims.N + col] = acc;
    }
}
