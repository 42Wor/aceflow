# AceFlow Seq2Seq

A modern Sequence-to-Sequence library built from scratch with custom .ace file format support.

## Features

- **Pure Python Implementation**: No PyTorch/TensorFlow dependencies
- **Modern Architecture**: Bidirectional GRU encoder with attention
- **Custom File Format**: Save and load models in .ace format
- **Training Utilities**: Complete training loop with tqdm progress bars
- **Flexible Tokenization**: Built-in tokenizer with special tokens support
- **Metrics**: BLEU score and accuracy calculation

## Installation

```bash
pip install numpy tqdm
pip install .