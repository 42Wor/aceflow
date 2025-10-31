
# Seq2SeqModel API Reference

The main model class for sequence-to-sequence tasks with attention mechanisms.

## Overview

`Seq2SeqModel` implements the encoder-decoder architecture with optional attention mechanisms. It supports multiple RNN types and is designed for tasks like machine translation, text summarization, and chatbots.

## Class Definition

```python
class Seq2SeqModel(
    src_vocab_size: int,
    tgt_vocab_size: int,
    hidden_size: int = 256,
    num_layers: int = 2,
    dropout: float = 0.1,
    rnn_type: str = 'lstm',
    use_attention: bool = True,
    teacher_forcing_ratio: float = 0.5,
    max_length: int = 50,
    bidirectional: bool = False,
    attention_method: str = 'concat',
    embedding_dim: Optional[int] = None
)
```

## Parameters

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `src_vocab_size` | `int` | **Required** | Size of source vocabulary |
| `tgt_vocab_size` | `int` | **Required** | Size of target vocabulary |
| `hidden_size` | `int` | `256` | Size of hidden states in RNN layers |
| `num_layers` | `int` | `2` | Number of RNN layers in encoder and decoder |
| `dropout` | `float` | `0.1` | Dropout probability for regularization |

### Architecture Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rnn_type` | `str` | `'lstm'` | Type of RNN: `'rnn'`, `'lstm'`, `'gru'`, `'bilstm'`, `'bigru'` |
| `use_attention` | `bool` | `True` | Whether to use attention mechanism |
| `bidirectional` | `bool` | `False` | Use bidirectional encoder |
| `attention_method` | `str` | `'concat'` | Attention type: `'concat'`, `'general'`, `'dot'` |

### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `teacher_forcing_ratio` | `float` | `0.5` | Probability of using teacher forcing |
| `max_length` | `int` | `50` | Maximum sequence length for generation |

### Advanced Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedding_dim` | `int` | `None` | Custom embedding dimension (defaults to hidden_size) |

## RNN Types

### Supported RNN Types

| Type | Description | Use Cases |
|------|-------------|-----------|
| `'rnn'` | Basic RNN cell | Simple tasks, computational constraints |
| `'lstm'` | Long Short-Term Memory | Most general purpose, long sequences |
| `'gru'` | Gated Recurrent Unit | Balance of performance and speed |
| `'bilstm'` | Bidirectional LSTM | Context-aware tasks (translation) |
| `'bigru'` | Bidirectional GRU | Faster bidirectional processing |

### RNN Type Selection Guide

```python
# For translation with context
model = Seq2SeqModel(..., rnn_type='bilstm', bidirectional=True)

# For chatbot with fast inference
model = Seq2SeqModel(..., rnn_type='gru')

# For simple sequence generation
model = Seq2SeqModel(..., rnn_type='lstm')

# When computation is limited
model = Seq2SeqModel(..., rnn_type='rnn')
```

## Attention Mechanisms

### Supported Attention Types

| Method | Description | Formula |
|--------|-------------|---------|
| `'concat'` | Bahdanau-style (additive) | `score = v^T * tanh(W1*h_enc + W2*h_dec)` |
| `'general'` | Luong-style (general) | `score = h_dec^T * W * h_enc` |
| `'dot'` | Luong-style (dot) | `score = h_dec^T * h_enc` |

### Attention Configuration

```python
# Bahdanau attention (default)
model = Seq2SeqModel(..., attention_method='concat')

# Luong general attention
model = Seq2SeqModel(..., attention_method='general')

# Luong dot product attention
model = Seq2SeqModel(..., attention_method='dot')
```

## Methods

### Forward Pass

```python
forward(
    src: torch.Tensor,
    tgt: Optional[torch.Tensor] = None,
    teacher_forcing_ratio: Optional[float] = None
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
```

**Parameters:**
- `src`: Source sequence tensor of shape `(batch_size, src_seq_len)`
- `tgt`: Target sequence tensor of shape `(batch_size, tgt_seq_len)` (optional for inference)
- `teacher_forcing_ratio`: Override default teacher forcing ratio

**Returns:**
- With attention: `(decoder_outputs, attention_weights)`
- Without attention: `decoder_outputs`

**Example:**
```python
# Training forward pass
src = torch.tensor([[1, 2, 3, 4, 0]])  # batch_size=1, seq_len=5
tgt = torch.tensor([[1, 5, 6, 7, 2]])  # batch_size=1, seq_len=5

# With attention
outputs, attention_weights = model(src, tgt)
# outputs: (1, 5, tgt_vocab_size), attention_weights: (1, 5, 5)

# Without attention
outputs = model(src, tgt)
# outputs: (1, 5, tgt_vocab_size)

# Inference (no target)
outputs = model(src)  # Uses max_length for generation
```

### Encoding

```python
encode(src: torch.Tensor) -> Tuple[torch.Tensor, Any]
```

Encode source sequences into hidden representations.

**Parameters:**
- `src`: Source sequence tensor of shape `(batch_size, src_seq_len)`

**Returns:**
- `encoder_outputs`: Encoder outputs of shape `(batch_size, src_seq_len, hidden_size * num_directions)`
- `encoder_hidden`: Final encoder hidden state

**Example:**
```python
src = torch.tensor([[1, 2, 3, 0, 0]])
encoder_outputs, encoder_hidden = model.encode(src)

print(encoder_outputs.shape)  # (1, 5, 512) for hidden_size=256, bidirectional=True
```

### Decoding

```python
decode(
    decoder_input: torch.Tensor,
    decoder_hidden: Any,
    encoder_outputs: torch.Tensor
) -> Union[Tuple[torch.Tensor, Any], Tuple[torch.Tensor, Any, torch.Tensor]]
```

Perform single decoding step.

**Parameters:**
- `decoder_input`: Input for current decoding step, shape `(batch_size, 1)`
- `decoder_hidden`: Previous decoder hidden state
- `encoder_outputs`: Encoder outputs for attention

**Returns:**
- With attention: `(output, hidden, attention_weights)`
- Without attention: `(output, hidden)`

**Example:**
```python
# Single decoding step
decoder_input = torch.tensor([[1]])  # Start token
encoder_outputs, encoder_hidden = model.encode(src)
decoder_hidden = model._init_decoder_hidden(encoder_hidden)

# With attention
output, hidden, attn = model.decode(decoder_input, decoder_hidden, encoder_outputs)
# Without attention
output, hidden = model.decode(decoder_input, decoder_hidden, encoder_outputs)
```

### Beam Search

```python
beam_search(
    src: torch.Tensor,
    beam_width: int = 5,
    max_length: int = 50
) -> List[int]
```

Generate sequence using beam search.

**Parameters:**
- `src`: Source sequence tensor of shape `(1, src_seq_len)`
- `beam_width`: Number of beams to keep
- `max_length`: Maximum generation length

**Returns:**
- `List[int]`: Generated sequence indices

**Example:**
```python
src = torch.tensor([[1, 2, 3, 4, 0]])  # Single sequence
output_sequence = model.beam_search(src, beam_width=5, max_length=20)

print(output_sequence)  # [1, 5, 6, 7, 8, 2]  # Start and end tokens included
```

### Model Information

```python
get_rnn_info() -> Dict[str, Any]
```

Get model configuration and statistics.

**Returns:**
- Dictionary with model information

**Example:**
```python
info = model.get_rnn_info()
print(info)
# {
#     'rnn_type': 'bilstm',
#     'hidden_size': 256,
#     'num_layers': 2,
#     'bidirectional': True,
#     'has_attention': True,
#     'attention_method': 'concat',
#     'total_parameters': 1250432
# }
```

### Save and Load

```python
save(filepath: str) -> None
```

Save model to `.ace` format.

**Parameters:**
- `filepath`: Path to save model (e.g., `"model.ace"`)

```python
@classmethod
load(filepath: str) -> 'Seq2SeqModel'
```

Load model from `.ace` format.

**Parameters:**
- `filepath`: Path to saved model

**Returns:**
- Loaded `Seq2SeqModel` instance

**Example:**
```python
# Save model
model.save("translation_model.ace")

# Load model
loaded_model = Seq2SeqModel.load("translation_model.ace")
```

## Usage Examples

### Basic Model Creation

```python
from aceflow import Seq2SeqModel

# Basic model for translation
model = Seq2SeqModel(
    src_vocab_size=10000,
    tgt_vocab_size=15000,
    hidden_size=512,
    num_layers=2,
    rnn_type='lstm',
    use_attention=True
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Advanced Configuration

```python
# High-performance translation model
model = Seq2SeqModel(
    src_vocab_size=50000,
    tgt_vocab_size=50000,
    hidden_size=1024,
    num_layers=4,
    dropout=0.2,
    rnn_type='bilstm',
    bidirectional=True,
    use_attention=True,
    attention_method='general',
    teacher_forcing_ratio=0.7,
    max_length=100
)
```

### Custom Embeddings

```python
# Use different embedding dimensions
model = Seq2SeqModel(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    hidden_size=256,
    embedding_dim=512  # Larger embeddings with smaller hidden size
)
```

### Training Loop Integration

```python
import torch
import torch.nn as nn
from aceflow import Seq2SeqModel

# Initialize model
model = Seq2SeqModel(
    src_vocab_size=1000,
    tgt_vocab_size=1000,
    hidden_size=256
)

# Training setup
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training batch
def train_step(src, tgt):
    optimizer.zero_grad()
    
    # Forward pass with teacher forcing
    output = model(src, tgt, teacher_forcing_ratio=0.5)
    
    # Calculate loss (ignore first token)
    loss = criterion(output[:, 1:].reshape(-1, output.size(-1)), 
                    tgt[:, 1:].reshape(-1))
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return loss.item()
```

### Inference Pipeline

```python
def translate_sequence(model, src_sequence, src_tokenizer, tgt_tokenizer):
    """Translate a sequence using the model"""
    # Encode source
    encoded_src = src_tokenizer.encode(src_sequence, return_tensors='list')
    src_tensor = torch.tensor([encoded_src])
    
    # Generate translation
    with torch.no_grad():
        output_sequence = model.beam_search(src_tensor, beam_width=5)
        translation = tgt_tokenizer.decode(output_sequence)
    
    return translation

# Usage
translation = translate_sequence(
    model, 
    "hello world", 
    src_tokenizer, 
    tgt_tokenizer
)
print(f"Translation: {translation}")
```

## Model Architecture Details

### Encoder Structure
- Embedding layer: `(vocab_size, hidden_size)`
- RNN layers: `num_layers` of specified `rnn_type`
- Dropout between layers if `num_layers > 1`
- Bidirectional if specified

### Decoder Structure
- Embedding layer: `(vocab_size, hidden_size)`
- Attention mechanism (if enabled)
- RNN layers matching encoder configuration
- Output projection to target vocabulary

### Attention Implementation
When `use_attention=True`, the decoder uses:
- **Context vector** from encoder outputs
- **Attention weights** over source sequence
- **Combined input** of embedding and context

## Best Practices

### Model Size Selection

```python
# Small model (fast training)
small_model = Seq2SeqModel(
    hidden_size=128,
    num_layers=1,
    rnn_type='gru'
)

# Medium model (balanced)
medium_model = Seq2SeqModel(
    hidden_size=256, 
    num_layers=2,
    rnn_type='lstm'
)

# Large model (high quality)
large_model = Seq2SeqModel(
    hidden_size=512,
    num_layers=3, 
    rnn_type='bilstm',
    bidirectional=True
)
```

### Regularization

```python
# Prevent overfitting
model = Seq2SeqModel(
    dropout=0.2,  # Increased dropout
    teacher_forcing_ratio=0.5,  # Balance teacher forcing
    num_layers=2  # Avoid too many layers for small datasets
)
```

### Vocabulary Considerations

```python
# Match vocabulary sizes to your data
model = Seq2SeqModel(
    src_vocab_size=len(src_tokenizer),  # Use actual tokenizer size
    tgt_vocab_size=len(tgt_tokenizer)   # Not estimated values
)
```

## Troubleshooting

### Common Issues

**1. Out of Memory**
```python
# Reduce model size
model = Seq2SeqModel(
    hidden_size=256,  # Instead of 512
    num_layers=2,     # Instead of 3
    rnn_type='lstm'   # Instead of bilstm
)
```

**2. Poor Convergence**
```python
# Adjust training parameters
model = Seq2SeqModel(
    teacher_forcing_ratio=0.8,  # More teacher forcing initially
    dropout=0.1,               # Less regularization
    use_attention=True         # Ensure attention is enabled
)
```

**3. Slow Inference**
```python
# Use faster RNN types and smaller beam width
model = Seq2SeqModel(rnn_type='gru')
output_sequence = model.beam_search(src, beam_width=3)  # Smaller beam
```

## See Also

- [Tokenizer API](tokenizer.md) - Text preprocessing and tokenization
- [Trainer API](trainer.md) - Model training utilities
- [Training Guide](../guides/training.md) - Best practices for training

```