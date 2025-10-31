
# Models Guide

Complete guide to AceFlow's Seq2Seq model architecture, configuration, and customization.

## Overview

AceFlow provides a flexible Sequence-to-Sequence model with attention mechanisms, supporting multiple RNN types and bidirectional encoding.

## Basic Model Creation

```python
from aceflow import Seq2SeqModel

# Minimal configuration
model = Seq2SeqModel(
    src_vocab_size=1000,    # Source vocabulary size
    tgt_vocab_size=1000,    # Target vocabulary size
    hidden_size=256         # Hidden state dimension
)

# Full configuration
model = Seq2SeqModel(
    src_vocab_size=5000,
    tgt_vocab_size=5000,
    hidden_size=512,
    num_layers=3,
    dropout=0.2,
    rnn_type='lstm',
    use_attention=True,
    teacher_forcing_ratio=0.5,
    max_length=100,
    bidirectional=True,
    attention_method='concat',
    embedding_dim=300
)
```

## Model Architecture

### Encoder-Decoder with Attention

```
Input Sequence → Encoder (RNN/LSTM/GRU) → Encoder States
                                     ↘
                                      Attention Mechanism  
                                     ↗
Output Sequence ← Decoder (RNN/LSTM/GRU) ← Context Vector
```

### Components

1. **Encoder**: Processes input sequence, can be bidirectional
2. **Decoder**: Generates output sequence, uses encoder states
3. **Attention**: Aligns decoder with relevant encoder states
4. **Embeddings**: Word representations for source and target

## RNN Types

AceFlow supports multiple RNN architectures:

### 1. Basic RNN
```python
model = Seq2SeqModel(
    rnn_type='rnn',  # Simple recurrent network
    hidden_size=256
)
```
**Use Case**: Simple tasks, short sequences, computational constraints

### 2. LSTM (Long Short-Term Memory)
```python
model = Seq2SeqModel(
    rnn_type='lstm',  # Default - best for most tasks
    hidden_size=256
)
```
**Use Case**: General purpose, excellent long-term memory

### 3. GRU (Gated Recurrent Unit)
```python
model = Seq2SeqModel(
    rnn_type='gru',  # Balance of performance and speed
    hidden_size=256
)
```
**Use Case**: Faster training, good performance

### 4. Bidirectional RNNs
```python
model = Seq2SeqModel(
    rnn_type='bilstm',  # or 'bigru', 'birnn'
    hidden_size=256,
    bidirectional=True  # Automatically set for bi- types
)
```
**Use Case**: When context from both directions is important

## Attention Mechanisms

### 1. Bahdanau (Additive) Attention
```python
model = Seq2SeqModel(
    use_attention=True,
    attention_method='concat'  # Default - Bahdanau style
)
```

### 2. Luong (Multiplicative) Attention
```python
model = Seq2SeqModel(
    use_attention=True,
    attention_method='general'  # Luong general attention
)

# Or dot product attention
model = Seq2SeqModel(
    use_attention=True,
    attention_method='dot'  # Luong dot attention
)
```

## Parameter Guidelines

### Hidden Size
```python
# Small model (fast training)
model = Seq2SeqModel(hidden_size=128)

# Medium model (good balance)
model = Seq2SeqModel(hidden_size=256)  # Recommended

# Large model (high capacity)
model = Seq2SeqModel(hidden_size=512)
```

### Number of Layers
```python
# Single layer (simpler)
model = Seq2SeqModel(num_layers=1)

# Multiple layers (more capacity)
model = Seq2SeqModel(num_layers=2)  # Recommended

# Deep model (complex patterns)
model = Seq2SeqModel(num_layers=3)
```

### Dropout
```python
# No regularization
model = Seq2SeqModel(dropout=0.0)

# Light regularization
model = Seq2SeqModel(dropout=0.1)  # Recommended

# Strong regularization
model = Seq2SeqModel(dropout=0.3)
```

## Task-Specific Configurations

### Machine Translation
```python
translation_model = Seq2SeqModel(
    src_vocab_size=30000,
    tgt_vocab_size=30000,
    hidden_size=512,
    num_layers=3,
    rnn_type='bilstm',
    use_attention=True,
    dropout=0.2,
    bidirectional=True,
    max_length=60
)
```

### Text Summarization
```python
summarization_model = Seq2SeqModel(
    src_vocab_size=50000,    # Larger source vocabulary
    tgt_vocab_size=20000,    # Smaller target vocabulary
    hidden_size=512,
    num_layers=2,
    rnn_type='lstm',
    use_attention=True,
    dropout=0.3,             # More regularization
    max_length=200           # Longer source sequences
)
```

### Chatbot/Dialogue System
```python
chatbot_model = Seq2SeqModel(
    src_vocab_size=20000,
    tgt_vocab_size=20000,
    hidden_size=256,
    num_layers=2,
    rnn_type='gru',          # Faster inference
    use_attention=True,
    teacher_forcing_ratio=0.7,  # More teacher forcing
    dropout=0.1,
    max_length=30            # Shorter conversations
)
```

## Model Information and Analysis

### Get Model Information
```python
model = Seq2SeqModel(
    src_vocab_size=5000,
    tgt_vocab_size=5000,
    hidden_size=256
)

# Basic info
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Detailed info
info = model.get_rnn_info()
print("Model Configuration:")
for key, value in info.items():
    print(f"  {key}: {value}")
```

### Parameter Count Estimation
```python
def estimate_parameters(src_vocab, tgt_vocab, hidden_size, num_layers, bidirectional=False):
    # Embedding parameters
    embed_params = src_vocab * hidden_size + tgt_vocab * hidden_size
    
    # LSTM parameters (4 gates × hidden_size × (hidden_size + hidden_size))
    lstm_params_per_layer = 4 * hidden_size * (hidden_size + hidden_size)
    lstm_params = lstm_params_per_layer * num_layers
    
    # Bidirectional multiplier
    bidir_multiplier = 2 if bidirectional else 1
    
    # Output layer
    output_params = hidden_size * bidir_multiplier * tgt_vocab
    
    total = embed_params + (lstm_params * bidir_multiplier) + output_params
    return total

params = estimate_parameters(5000, 5000, 256, 2, bidirectional=True)
print(f"Estimated parameters: {params:,}")
```

## Custom Model Architectures

### Custom Encoder
```python
from aceflow.core.layers import Encoder
import torch.nn as nn

class CustomEncoder(Encoder):
    def __init__(self, vocab_size, hidden_size, num_layers=2, dropout=0.1, 
                 rnn_type='lstm', bidirectional=False, custom_param=0.5):
        super().__init__(vocab_size, hidden_size, num_layers, dropout, rnn_type, bidirectional)
        
        # Add custom layers
        self.custom_layer = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.custom_param = custom_param
    
    def forward(self, x, hidden=None):
        # Standard encoder forward
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.rnn(embedded, hidden)
        
        # Custom processing
        output = self.custom_layer(output)
        output = self.layer_norm(output)
        
        return output, hidden
```

### Custom Seq2Seq Model
```python
from aceflow.core.model import Seq2SeqModel
import torch.nn as nn

class CustomSeq2SeqModel(Seq2SeqModel):
    def __init__(self, *args, **kwargs):
        # Extract custom parameters
        self.custom_feature_size = kwargs.pop('custom_feature_size', 64)
        self.use_custom_attention = kwargs.pop('use_custom_attention', False)
        
        super().__init__(*args, **kwargs)
        
        # Add custom components
        if self.use_custom_attention:
            self.custom_attention = nn.MultiheadAttention(
                self.hidden_size, num_heads=8
            )
        
        self.feature_projection = nn.Linear(
            self.hidden_size, self.custom_feature_size
        )
    
    def forward(self, src, tgt=None, teacher_forcing_ratio=None, custom_features=None):
        # Custom preprocessing
        if custom_features is not None:
            src = self.feature_projection(src)
        
        # Call parent forward
        outputs = super().forward(src, tgt, teacher_forcing_ratio)
        
        # Custom postprocessing
        if self.use_custom_attention and isinstance(outputs, tuple):
            output, attention_weights = outputs
            # Apply custom attention
            output, _ = self.custom_attention(output, output, output)
            outputs = (output, attention_weights)
        
        return outputs

# Use custom model
custom_model = CustomSeq2SeqModel(
    src_vocab_size=5000,
    tgt_vocab_size=5000,
    hidden_size=256,
    custom_feature_size=128,
    use_custom_attention=True
)
```

## Model Initialization

### Custom Weight Initialization
```python
def init_weights(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            if 'embedding' in name:
                # Embedding initialization
                nn.init.normal_(param, mean=0, std=0.1)
            elif 'rnn' in name:
                # RNN weight initialization
                if len(param.shape) >= 2:
                    nn.init.orthogonal_(param)
            elif 'linear' in name:
                # Linear layer initialization
                nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            # Bias initialization
            nn.init.constant_(param, 0.0)

model = Seq2SeqModel(src_vocab_size=5000, tgt_vocab_size=5000, hidden_size=256)
init_weights(model)
```

### Pretrained Embeddings
```python
def load_pretrained_embeddings(embedding_layer, word2idx, embedding_file):
    # Load pre-trained embeddings (GloVe, Word2Vec, etc.)
    pretrained_embeddings = {}
    with open(embedding_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = torch.tensor([float(val) for val in values[1:]])
            pretrained_embeddings[word] = vector
    
    # Initialize embedding layer
    for word, idx in word2idx.items():
        if word in pretrained_embeddings:
            embedding_layer.weight.data[idx] = pretrained_embeddings[word]
        elif word in ['<pad>', '<start>', '<end>', '<unk>']:
            # Special tokens - keep random initialization
            continue
        else:
            # Random initialization for OOV words
            nn.init.normal_(embedding_layer.weight.data[idx], mean=0, std=0.1)

# Usage
model = Seq2SeqModel(src_vocab_size=5000, tgt_vocab_size=5000, hidden_size=300)
load_pretrained_embeddings(
    model.encoder.embedding,
    src_tokenizer.vocab.word2idx,
    'glove.6B.300d.txt'
)
```

## Model Serialization

### Saving Models
```python
# Save model
model.save("translation_model.ace")

# Save with metadata
import datetime
model.config['training_date'] = datetime.datetime.now().isoformat()
model.config['dataset'] = 'english-french'
model.save("translation_model_with_metadata.ace")
```

### Loading Models
```python
# Load model
loaded_model = Seq2SeqModel.load("translation_model.ace")

# Check loaded configuration
print(loaded_model.config)

# Continue training or use for inference
```

### Model Conversion
```python
# Convert to different RNN type
def convert_rnn_type(model, new_rnn_type):
    """Convert model to use different RNN type"""
    model.config['rnn_type'] = new_rnn_type
    # Note: This requires reinitializing RNN layers
    # In practice, you'd create a new model and copy compatible weights
    return model
```

## Performance Optimization

### Gradient Clipping
```python
# During training
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Mixed Precision Training
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

def mixed_precision_step(model, input, target):
    with autocast():
        output = model(input, target)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Model Pruning
```python
# Prune small weights
def prune_model(model, pruning_threshold=0.01):
    for name, param in model.named_parameters():
        if 'weight' in name:
            # Mask small weights
            mask = torch.abs(param) > pruning_threshold
            param.data *= mask.float()
```

## Troubleshooting

### Common Model Issues

**1. Exploding Gradients**
```python
# Solution: Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Or reduce learning rate
trainer = Trainer(model, learning_rate=0.0001)
```

**2. Vanishing Gradients**
```python
# Solution: Use LSTM/GRU instead of RNN
model = Seq2SeqModel(rnn_type='lstm')  # Instead of 'rnn'

# Or use skip connections
class ResidualSeq2SeqModel(Seq2SeqModel):
    # Implement residual connections
    pass
```

**3. Overfitting**
```python
# Solution: Increase dropout
model = Seq2SeqModel(dropout=0.3)  # Instead of 0.1

# Or add more regularization
model = Seq2SeqModel(
    dropout=0.2,
    num_layers=2  # Instead of 3
)
```

**4. Underfitting**
```python
# Solution: Increase model capacity
model = Seq2SeqModel(
    hidden_size=512,  # Instead of 256
    num_layers=3,     # Instead of 2
    rnn_type='bilstm' # Instead of 'lstm'
)
```

## Best Practices

### 1. Start Simple
```python
# Begin with simple configuration
simple_model = Seq2SeqModel(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    hidden_size=256,
    num_layers=2,
    rnn_type='lstm',
    use_attention=True
)
```

### 2. Scale Gradually
```python
# If simple model works, scale up
advanced_model = Seq2SeqModel(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    hidden_size=512,
    num_layers=3,
    rnn_type='bilstm',
    use_attention=True,
    dropout=0.2,
    bidirectional=True
)
```

### 3. Monitor Model Health
```python
def model_health_check(model, dataloader):
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            output = model(batch['src'])
            # Check for NaN/inf
            if torch.isnan(output).any() or torch.isinf(output).any():
                print("Warning: Model output contains NaN/inf")
                return False
    return True
```

## Next Steps

- Learn about [Model Training](training.md)
- Explore [Inference Techniques](inference.md)
- Check [API Reference](../api/seq2seqmodel.md) for complete documentation
- See [Examples](../examples/) for practical implementations
```

This comprehensive models guide covers everything from basic usage to advanced customization, helping users understand and effectively use AceFlow's Seq2Seq model architecture.