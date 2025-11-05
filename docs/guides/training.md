
# Training Guide

This guide shows you how to train sequence-to-sequence models with AceFlow.

## Quick Start

### Basic Training

```python
import torch
from aceflow import Seq2SeqModel
from aceflow.utils import Tokenizer, create_data_loader
from aceflow.trainers import Seq2SeqTrainer

# 1. Prepare your data
english_sentences = ["hello", "how are you", "good morning"]
french_sentences = ["bonjour", "comment allez vous", "bonjour"]

# 2. Create tokenizers
src_tokenizer = Tokenizer(name="english")
tgt_tokenizer = Tokenizer(name="french")
src_tokenizer.fit(english_sentences)
tgt_tokenizer.fit(french_sentences)

# 3. Create data loaders
train_loader = create_data_loader(
    english_sentences, french_sentences, 
    src_tokenizer, tgt_tokenizer, 
    batch_size=2, max_length=10
)

# 4. Initialize model
model = Seq2SeqModel(
    src_vocab_size=len(src_tokenizer),
    tgt_vocab_size=len(tgt_tokenizer),
    hidden_size=128,
    use_attention=True
)

# 5. Create trainer and train
trainer = Seq2SeqTrainer(model, learning_rate=0.001)
history = trainer.train(
    train_loader, train_loader,  # Using same data for train/val
    epochs=10,
    save_path="my_model.ace"
)

# 6. Save tokenizers
src_tokenizer.save("english_tokenizer")
tgt_tokenizer.save("french_tokenizer")
```

## Complete Example

### Data Preparation

```python
# Your parallel dataset
source_texts = [
    "hello world",
    "how are you", 
    "what is your name",
    "i love machine learning"
]

target_texts = [
    "bonjour le monde",
    "comment allez vous",
    "quel est votre nom", 
    "j aime l apprentissage automatique"
]

# Split into train/validation
train_src = source_texts[:3]
train_tgt = target_texts[:3]
val_src = source_texts[3:]
val_tgt = target_texts[3:]
```

### Model Configuration

```python
model = Seq2SeqModel(
    src_vocab_size=5000,      # Should match your source vocabulary
    tgt_vocab_size=5000,      # Should match your target vocabulary  
    hidden_size=256,          # Size of hidden layers
    num_layers=2,             # Number of RNN layers
    dropout=0.1,              # Dropout for regularization
    use_attention=True,       # Use attention mechanism
    rnn_type='lstm',          # 'lstm', 'gru', or 'rnn'
    bidirectional=False       # Use bidirectional encoder
)
```

### Advanced Training

```python
from aceflow.trainers import ModelCheckpoint, EarlyStopping

# Advanced trainer configuration
trainer = Seq2SeqTrainer(
    model=model,
    learning_rate=0.001,
    early_stopping_patience=5,
    gradient_clip=1.0,
    teacher_forcing_ratio=0.5
)

# Callbacks for better training
callbacks = [
    ModelCheckpoint("best_model.ace", monitor='val_loss'),
    EarlyStopping(patience=5, monitor='val_loss')
]

# Train with validation
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader, 
    epochs=50,
    save_path="final_model.ace",
    callbacks=callbacks
)
```

## Training Output

You'll see a clean training progress table:

```
==========================================================================================
                             TRAINING PROGRESS
==========================================================================================
+----------+------------+-----------+----------+---------+----------+--------------+
|  Epoch   | Train Loss | Train Acc | Val Loss | Val Acc |    LR    |    Status    |
+----------+------------+-----------+----------+---------+----------+--------------+
|   1/50   |   2.1345   |  0.2543   |  1.9876  |  0.3125 | 1.00e-03 |    [BEST]    |
|   2/50   |   1.8765   |  0.4123   |  1.7654  |  0.4375 | 1.00e-03 |    [BEST]    |
```

## Loading and Using Trained Models

```python
# Load trained model
model = Seq2SeqModel.load("final_model.ace")

# Load tokenizers  
src_tokenizer = Tokenizer.load("english_tokenizer")
tgt_tokenizer = Tokenizer.load("french_tokenizer")

# Translate new text
def translate(text):
    encoded = src_tokenizer.encode(text, return_tensors='list')
    input_tensor = torch.tensor([encoded], dtype=torch.long)
    
    with torch.no_grad():
        output_sequence = model.beam_search(input_tensor, beam_width=5)
        translation = tgt_tokenizer.decode(output_sequence)
    
    return translation

print(translate("hello world"))  # Output: "bonjour le monde"
```

## Training Tips

### 1. **Vocabulary Size**
- Keep vocabulary reasonable (5,000-50,000 words)
- Use `min_freq` to filter rare words
```python
tokenizer.fit(texts, max_vocab_size=10000, min_freq=2)
```

### 2. **Sequence Length**
- Use appropriate max length for your data
- Longer sequences need more memory
```python
create_data_loader(..., max_length=50)  # Adjust based on your data
```

### 3. **Batch Size**
- Larger batches = faster training but more memory
- Start with 32-64 and adjust based on GPU memory

### 4. **Learning Rate**
- Start with 0.001 for Adam optimizer
- Use learning rate scheduling if loss plateaus

### 5. **Early Stopping**
- Prevents overfitting
- Typical patience: 5-10 epochs
```python
EarlyStopping(patience=7, monitor='val_loss')
```

## Common Issues

### Out of Memory
- Reduce batch size
- Use shorter sequences
- Enable gradient checkpointing

### Slow Training
- Use GPU if available
- Increase batch size
- Use mixed precision training
```python
Seq2SeqTrainer(..., use_amp=True)  # Requires CUDA
```

### Poor Results
- Check data quality and preprocessing
- Adjust model capacity (hidden_size, num_layers)
- Tune hyperparameters (learning_rate, dropout)



Need help? [Open an issue](https://github.com/42Wor/aceflow/issues) on GitHub!
```

This training guide provides:

1. **Quick Start** - Get training in minutes
2. **Complete Examples** - From data prep to inference
3. **Visual Examples** - See what the output looks like
4. **Practical Tips** - Best practices for good results
5. **Troubleshooting** - Common issues and solutions
6. **Clean Formatting** - Easy to read and follow

The guide is simple but comprehensive, perfect for users who want to start training models quickly without getting overwhelmed.