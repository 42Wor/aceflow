# AceFlow — Seq2Seq Model Library

<div align="center">

![AceFlow](https://img.shields.io/badge/AceFlow-Seq2Seq-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

A Python library for building and training Sequence-to-Sequence models with attention.  
**Inference** RNN and attention kernels run in `mallow-core` (Rust / SIMD).  
**Training** stays on PyTorch autograd.

</div>

---

## Features

- **Multiple RNN types** — LSTM, GRU, vanilla RNN, bidirectional variants
- **Attention mechanisms** — Bahdanau (additive) and Luong (dot / general / concat)
- **Native inference** — eval-mode forward pass uses AVX2/NEON kernels via `mallow_core`
- **Beam search** — `mallow_core.beam_search_rust` (falls back to `aceflow_core` then pure Python)
- **Custom model format** — save/load in `.ace` format
- **AutoTokenizer** — `mallow_core.AutoTokenizer` handles `aceflow-word` schema

---

## Quick example

```python
from aceflow import Seq2SeqModel
from aceflow.utils import Tokenizer

model = Seq2SeqModel(
    src_vocab_size=1000,
    tgt_vocab_size=1000,
    hidden_size=256,
    rnn_type="lstm",
    use_attention=True,
)

model.save("model.ace")
loaded = Seq2SeqModel.load("model.ace")
```

---

## Installation

```console
# 1. build and install mallow-core first
cd ../mallow-core
maturin develop --release
cd ../AceFlow

# 2. editable install
pip install -e .
```

No C compiler required.

---

## Architecture

```
aceflow/
├── core/
│   ├── model.py        Seq2SeqModel — beam search via mallow_core
│   ├── layers.py       RNNLayer    — training on PyTorch, eval via mallow_core
│   ├── attention.py    BahdanauAttention / LuongAttention
│   └── native.py       weight packing + mallow_core kernel calls
└── utils/
    └── tokenizer/
        └── tokenizer.py   backed by mallow_core.AutoTokenizer
```

### Import priority for beam search (`model.py`)

```
mallow_core  →  aceflow_core (legacy shim)  →  pure-Python fallback
```

---

## Tests

```console
python -m pytest tests/ -q
# 25 passed, 7 subtests passed
```

---

## License

MIT — see [LICENSE](LICENSE).

---

<div align="center">Made with ❤️ by Maaz Waheed</div>
