
## Python → C/Rust Migration Strategy

AceFlow no longer ships C extensions. The former OpenMP kernels in
`aceflow/core/c_core/` were rewritten in `mallow-core` (SIMD GEMV + Rayon)
and are imported as `mallow_core.rnn_forward` and
`mallow_core.bahdanau_attention_forward`.

| Python Function | File | Language | Backend |
|----------------|------|----------|---------|
| `RNNLayer.forward()` (eval / no-grad) | `layers.py` | **Rust** (`mallow-core`) | AVX2/FMA or NEON GEMV |
| `BahdanauAttention.forward()` (eval / no-grad) | `attention.py` | **Rust** (`mallow-core`) | SIMD GEMV + axpy |
| `RNNLayer.forward()` (training) | `layers.py` | **PyTorch** | autograd |
| `beam_search()` | `model.py` | **Rust** (`aceflow-core`, optional) | beam search helper |

Training still uses PyTorch so gradients flow. Inference (`model.eval()` +
`torch.no_grad()`) uses the native `mallow-core` kernels packed from the
live PyTorch weights.
