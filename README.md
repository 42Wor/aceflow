# AceFlow

High-performance, dual-tier monolithic deep learning framework written in Rust and Python.

<div align="center">

![AceFlow](https://img.shields.io/badge/AceFlow-Core-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![Rust](https://img.shields.io/badge/Rust-2021%20Edition-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

</div>

---

## Architecture

`AceFlow` is structured as a dual-tier framework:
1. **Engine Tier (`aceflow-core` in Rust):**
   - High-throughput SIMD recurrence kernels (AVX2 / NEON)
   - Multi-head and additive attention kernels
   - WebGPU compute shaders (WGSL) for GPU acceleration
   - 80-channel log-mel filterbanks and STFT power audio DSP
   - SIMD Byte-level BPE tokenizer
   - Native PyO3 bridge (`aceflow._native`)
2. **Framework Tier (`aceflow` in Python):**
   - `af.Tensor` & `af.Device`: High-level tensor memory and hardware abstractions (CPU, WebGPU)
   - `af.nn`: PyTorch-compatible modules (`Module`, `Parameter`, `Linear`, `RMSNorm`, `LayerNorm`, `SwiGLU`, `MultiScaleRetention`)
   - `af.ops`: Functional operators (`matmul`, `rms_norm`, `swiglu`, `gelu`, `silu`, `softmax`)
   - `af.audio`: Audio DSP pipelines (`mel_filters`, `log_mel_spectrogram`, `pad_or_trim`)
   - `af.tokenization`: Fast `AutoTokenizer`
   - `af.trainers`: Sequence trainers, early stopping, and metric tracking harnesses
   - `af.io`: Zero-copy MTF (Maaz Tensor Format) memory-mapped reader

---

## Quickstart

```python
import numpy as np
import aceflow as af

# 1. Device and Tensor allocation
dev = af.Device.cpu()
x = af.Tensor.zeros([2, 4], device=dev)
print(x)

# 2. Neural Network Layers
linear = af.nn.Linear(4, 8)
out = linear(x)
print("Linear output shape:", out.shape)

# 3. Multi-Scale Retention (Dual-Form O(1) step inference & O(N) parallel training)
retention = af.nn.MultiScaleRetention(d_model=64, num_heads=4)
seq = af.Tensor(np.random.randn(2, 10, 64).astype(np.float32))
ret_out = retention(seq)
print("Retention output shape:", ret_out.shape)

# 4. Audio DSP (Log-Mel Filterbanks)
mel = af.audio.mel_filters(n_mels=80, n_fft=400, sample_rate=16000)
print("Mel filterbank shape:", mel.shape)
```

---

## Building from Source

Prerequisites:
- Python >= 3.8
- Rust toolchain (cargo, rustc >= 1.70)
- `maturin` >= 1.4

```bash
cd AceFlow
pip install maturin
maturin develop --release
```

Run test suite:
```bash
python -m unittest discover tests
```

---

## License

MIT License.
