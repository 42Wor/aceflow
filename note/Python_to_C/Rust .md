
## Python → C/Rust Migration Strategy

| Python Function | File | Language | Reason | Priority | Expected Speedup | 
|----------------|------|----------|---------|----------|------------------|
| **C IMPLEMENTATIONS** | | | | | |
| `RNNLayer.forward()` | `layers.py` | **C** | Direct hardware access, existing BLAS libraries | 🔴 HIGH | 3-5x 🟢 |
| `BahdanauAttention.forward()` | `attention.py` | **C** | Matrix operations, GPU integration | 🔴 HIGH | 5-8x 🟢|
| | | | | | |
| **RUST IMPLEMENTATIONS** | | | | | |
| `beam_search()` | `model.py` | **Rust** | Memory safety, complex data structures | 🔴 HIGH | 10x 🟢  |
