
## Python → C/Rust Migration Strategy

| Python Function | File | Language | Reason | Priority | Expected Speedup | 
|----------------|------|----------|---------|----------|------------------|
| **C IMPLEMENTATIONS** | | | | | |
| `RNNLayer.forward()` | `layers.py` | **C** | Direct hardware access, existing BLAS libraries | 🔴 HIGH | 3-5x |🟢 🟢 |
| `BahdanauAttention.forward()` | `attention.py` | **C** | Matrix operations, GPU integration | 🔴 HIGH | 5-8x |
| `_compute_metrics()` | `seq2seq_trainer.py` | **C** | Numerical computations, SIMD optimization | 🟡 MEDIUM | 2-3x |
| `Vocabulary.encode_word()` | `vocabulary.py` | **C** | Hash table operations, memory efficiency | 🟡 MEDIUM | 3-5x |
| | | | | | |
| **RUST IMPLEMENTATIONS** | | | | | |
| `beam_search()` | `model.py` | **Rust** | Memory safety, complex data structures | 🔴 HIGH | 10x 🟢  |
| `Tokenizer.encode()` | `tokenizer.py` | **Rust** | String processing, Unicode handling | 🟡 MEDIUM | 2-4x |
| `Preprocessor.process()` | `preprocessor.py` | **Rust** | Text processing, regex performance | LOW | 2-3x |

