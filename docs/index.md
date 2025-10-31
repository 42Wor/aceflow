
# AceFlow Documentation

<div align="center">

![AceFlow Banner](https://via.placeholder.com/800x200/2D3748/FFFFFF?text=AceFlow+-+Seq2Seq+Models+Made+Simple)
*Building sequence-to-sequence models has never been easier*

</div>

## 🎯 What is AceFlow?

AceFlow is a powerful, flexible Python library for building and training **Sequence-to-Sequence models** with attention mechanisms. Whether you're working on machine translation, text summarization, chatbots, or any other sequence generation task, AceFlow provides the tools you need with an intuitive API.

### Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **Multiple RNN Types** | LSTM, GRU, RNN, and bidirectional variants |
| 🔍 **Attention Mechanisms** | Bahdanau and Luong-style attention |
| 💾 **Custom Model Format** | Save/load models in `.ace` format |
| 🛠️ **Advanced Tokenization** | Flexible preprocessing pipelines |
| 📊 **Training Utilities** | Comprehensive training with metrics |
| 🚀 **Production Ready** | Easy deployment and inference |

## 🚀 Quick Start

### Installation

```bash
pip install torch tqdm h5py pyyaml contractions
```

### Your First Model in 5 Minutes

```python
from aceflow import Seq2SeqModel
from aceflow.utils import Tokenizer

# Prepare data
english = ["hello world", "how are you"]
french = ["bonjour le monde", "comment allez vous"]

# Initialize tokenizers
src_tokenizer = Tokenizer(name="english")
tgt_tokenizer = Tokenizer(name="french")
src_tokenizer.fit(english)
tgt_tokenizer.fit(french)

# Create and train model
model = Seq2SeqModel(
    src_vocab_size=len(src_tokenizer),
    tgt_vocab_size=len(tgt_tokenizer),
    hidden_size=128,
    use_attention=True
)

# Train and save
model.save("my_model.ace")
```

## 📚 Documentation Structure

### Getting Started
- **[Installation Guide](installation.md)** - Set up AceFlow and dependencies
- **[Quick Start](quickstart.md)** - Build your first model in minutes
- **[Examples](examples/)** - Practical examples for common use cases

### Core Concepts
- **[Tokenizers Guide](guides/tokenizers.md)** - Text preprocessing and vocabulary management
- **[Models Guide](guides/models.md)** - Model architecture and configuration
- **[Training Guide](guides/training.md)** - Effective model training strategies
- **[Inference Guide](guides/inference.md)** - Deploying and using trained models

### API Reference
- **[Seq2SeqModel](api/seq2seqmodel.md)** - Main model class and methods
- **[Tokenizer](api/tokenizer.md)** - Text tokenization utilities
- **[Trainer](api/trainer.md)** - Model training and evaluation
- **[Utilities](api/utilities.md)** - Data loading and serialization

## 🎯 Use Cases

AceFlow is perfect for:

### 🤖 Machine Translation
```python
model = Seq2SeqModel(
    src_vocab_size=src_vocab,
    tgt_vocab_size=tgt_vocab,
    rnn_type='bilstm',  # Bidirectional for context
    use_attention=True
)
```

### 📝 Text Summarization
```python
model = Seq2SeqModel(
    src_vocab_size=src_vocab,
    tgt_vocab_size=tgt_vocab,
    max_length=200,  # Longer sequences for documents
    use_attention=True
)
```

### 💬 Chatbots
```python
model = Seq2SeqModel(
    src_vocab_size=src_vocab,
    tgt_vocab_size=tgt_vocab,
    rnn_type='gru',  # Faster inference
    teacher_forcing_ratio=0.7  # Better for conversation
)
```

## 🔧 Key Components

### Model Architecture
AceFlow implements the encoder-decoder architecture with attention:

```
Input Sequence → Encoder → Context Vector → Decoder → Output Sequence
                         ↘ Attention ↗
```

### Supported RNN Types

| Type | Best For | Pros |
|------|----------|------|
| **LSTM** | General purpose | Excellent long-term memory |
| **GRU** | Speed-critical tasks | Faster training, good performance |
| **BiLSTM** | Context-aware tasks | Bidirectional context |
| **RNN** | Simple sequences | Fast, lightweight |

### Attention Mechanisms

- **Bahdanau (Additive)**: `attention_method='concat'`
- **Luong (Multiplicative)**: `attention_method='general'` or `'dot'`

## 📊 Performance Tips

### Model Configuration
```python
# For high accuracy
model = Seq2SeqModel(
    hidden_size=512,    # Larger hidden states
    num_layers=3,       # More layers
    rnn_type='bilstm',  # Bidirectional
    dropout=0.2         # Regularization
)

# For faster training
model = Seq2SeqModel(
    hidden_size=256,    # Smaller hidden states
    num_layers=2,       # Fewer layers  
    rnn_type='gru',     # Faster RNN type
    dropout=0.1         # Less regularization
)
```

### Training Optimization
```python
from aceflow.trainers import Trainer

trainer = Trainer(
    model=model,
    learning_rate=0.001,
    device='cuda'  # Use GPU if available
)
```

## 🛠️ Advanced Features

### Custom Preprocessing
```python
from aceflow.utils import Preprocessor

# Create custom pipeline
preprocessor = Preprocessor(language="english")
preprocessor.add_step(lambda x: x.replace('&', 'and'))
tokenizer.preprocessor = preprocessor
```

### Beam Search Inference
```python
# Better quality translations
output_sequence = model.beam_search(
    input_tensor, 
    beam_width=5,    # Explore more possibilities
    max_length=50
)
```

### Transfer Learning
```python
# Load pre-trained model
pretrained = Seq2SeqModel.load("pretrained.ace")

# Fine-tune on new data
pretrained.fit(new_data_loader, epochs=10)
```

## 📈 Examples Gallery

### [Machine Translation](examples/translation.md)
Build English-to-French translation system with attention visualization.

### [Text Summarization](examples/summarization.md)
Create a model that summarizes long documents into concise summaries.

### [Chatbot Development](examples/chatbot.md)
Develop a conversational AI that can handle multi-turn dialogues.

### [Custom Task](examples/custom.md)
Implement your own sequence-to-sequence task with custom preprocessing.

## ❓ Getting Help

### Common Issues

**Q: My model isn't learning**
```python
# Try these fixes:
model = Seq2SeqModel(
    teacher_forcing_ratio=0.5,  # Balance teacher forcing
    use_attention=True,         # Enable attention
    dropout=0.2                 # Add regularization
)
```

**Q: Training is too slow**
```python
# Optimization tips:
trainer = Trainer(model, device='cuda')  # Use GPU
# Reduce batch size if memory limited
# Use GRU instead of LSTM for speed
```

**Q: Poor translation quality**
```python
# Quality improvements:
output = model.beam_search(input, beam_width=10)  # Better search
# Increase model capacity (hidden_size, num_layers)
# Use more training data
```

### Support Channels

- 📚 **Documentation**: This site!
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/42Wor/aceflow/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/42Wor/aceflow/discussions)
- 📧 **Email**: wwor4287@gmail.com , maaz.waheed@mbktechstudio.com



## 🔗 Useful Links

- **GitHub Repository**: [github.com/42Wor/aceflow](https://github.com/42Wor/aceflow)
- **Issue Tracker**: [GitHub Issues](https://github.com/42Wor/aceflow/issues)
- **Releases**: [GitHub Releases](https://github.com/42Wor/aceflow/releases)
- **PyPI Package**: [pypi.org/project/aceflow](https://pypi.org/project/aceflow) *(coming soon)*

## 🎓 Learn More

### Background Reading
- [Sequence-to-Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) - Original Seq2Seq paper
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) - Attention mechanism
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) - LSTM explanation

### Related Projects
- [Hugging Face Transformers](https://github.com/huggingface/transformers) - Transformer-based models
- [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) - Another Seq2Seq library
- [Fairseq](https://github.com/facebookresearch/fairseq) - Facebook's sequence modeling toolkit

---

<div align="center">

**Ready to start building?** Check out the [Quick Start Guide](quickstart.md)!

*Last updated: Version 1.4.0 | 2025*

</div>
