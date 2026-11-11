# AceFlow Documentation

Welcome to the documentation for **AceFlow**, a high-performance deep learning framework combining a bare-metal Rust engine (`aceflow-core`) with a PyTorch-like Python interface (`aceflow`).

## Packages & Modules

- **`aceflow.core`**: `Tensor` and `Device` abstractions connecting to native buffers.
- **`aceflow.nn`**: Neural network layers (`Linear`, `RMSNorm`, `LayerNorm`, `SwiGLU`, `MultiScaleRetention`).
- **`aceflow.ops`**: Functional tensor operations (`matmul`, `rms_norm`, `swiglu`, `gelu`, `silu`, `softmax`).
- **`aceflow.audio`**: Native log-mel filterbanks and spectrogram processing.
- **`aceflow.tokenization`**: High-throughput SIMD Byte-level BPE tokenizer.
- **`aceflow.trainers`**: Telemetry, callbacks, early stopping, and training harnesses.
- **`aceflow.io`**: MTF (Maaz Tensor Format) zero-copy tensor loader.

## Hardware Support

AceFlow supports:
- **AVX2 / NEON CPU**: Multi-threaded SIMD execution via Rayon.
- **WebGPU (WGPU)**: Portable GPU acceleration using native WGSL shaders.

