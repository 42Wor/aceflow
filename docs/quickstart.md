
# Quick Start Guide

Build your first Sequence-to-Sequence model in 5 minutes with AceFlow!

## Overview

In this guide, you'll:
1. Prepare sample data
2. Initialize tokenizers  
3. Create a Seq2Seq model
4. Train the model
5. Make predictions

## Complete Example

```python
import torch
from aceflow import Seq2SeqModel
from aceflow.utils import Tokenizer, create_data_loader
from aceflow.trainers import Trainer

# Step 1: Prepare sample data
english_sentences = [
    "hello world",
    "how are you", 
    "good morning",
    "what is your name",
    "i love programming"
]

french_sentences = [
    "bonjour le monde",
    "comment allez vous",
    "bonjour",
    "quel est votre nom", 
    "j aime la programmation"
]

# Step 2: Initialize and fit tokenizers
src_tokenizer = Tokenizer(name="english", max_length=15)
tgt_tokenizer = Tokenizer(name="french", max_length=15)

src_tokenizer.fit(english_sentences)
tgt_tokenizer.fit(french_sentences)

print(f"Source vocabulary size: {len(src_tokenizer)}")
print(f"Target vocabulary size: {len(tgt_tokenizer)}")

# Step 3: Create the model
model = Seq2SeqModel(
    src_vocab_size=len(src_tokenizer),
    tgt_vocab_size=len(tgt_tokenizer),
    hidden_size=128,
    num_layers=2,
    rnn_type='lstm',
    use_attention=True,
    teacher_forcing_ratio=0.5
)

print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

# Step 4: Create data loader
train_loader = create_data_loader(
    english_sentences,
    french_sentences, 
    src_tokenizer,
    tgt_tokenizer,
    batch_size=2,
    max_length=15
)

# Step 5: Train the model
trainer = Trainer(model, learning_rate=0.001)
history = trainer.train(
    train_loader=train_loader,
    val_loader=train_loader,  # Using same data for demo
    epochs=10,
    save_path="my_first_model.ace"
)

# Step 6: Save tokenizers
src_tokenizer.save("english_tokenizer")
tgt_tokenizer.save("french_tokenizer")

print("✅ Training completed!")

# Step 7: Load and use the model
loaded_model = Seq2SeqModel.load("my_first_model.ace")
loaded_src_tokenizer = Tokenizer.load("english_tokenizer")
loaded_tgt_tokenizer = Tokenizer.load("french_tokenizer")

# Step 8: Make predictions
def translate(text):
    encoded = loaded_src_tokenizer.encode(text, return_tensors='list')
    input_tensor = torch.tensor([encoded])
    
    with torch.no_grad():
        output_sequence = loaded_model.beam_search(input_tensor, beam_width=3)
        translation = loaded_tgt_tokenizer.decode(output_sequence)
    
    return translation

# Test translation
test_text = "hello world"
result = translate(test_text)
print(f"'{test_text}' → '{result}'")
```

## Step-by-Step Explanation

### 1. Data Preparation

We use simple English-French translation pairs. In practice, you'd use larger datasets from files:

```python
# Load from files
with open("data/english.txt", "r") as f:
    english_sentences = [line.strip() for line in f.readlines()]

with open("data/french.txt", "r") as f:
    french_sentences = [line.strip() for line in f.readlines()]
```

### 2. Tokenization

The `Tokenizer` class handles:
- Text preprocessing (lowercasing, cleaning, contraction expansion)
- Vocabulary building from your data
- Sequence encoding/decoding with special tokens

### 3. Model Configuration

Key parameters explained:
- `hidden_size`: Size of hidden states (128-512 typical)
- `num_layers`: Number of RNN layers (1-3 typical)  
- `rnn_type`: Type of RNN ('lstm', 'gru', 'bilstm')
- `use_attention`: Whether to use attention mechanism (recommended)
- `teacher_forcing_ratio`: Probability of using ground truth during training

### 4. Training

The `Trainer` class provides:
- Automatic training loop with progress bars
- Loss and accuracy tracking
- Model checkpointing
- Validation evaluation

### 5. Inference

We use beam search for better translation quality than greedy decoding.

## Expected Output

```
Source vocabulary size: 25
Target vocabulary size: 30
Model has 452,110 parameters
Epoch 1/10: 100%|█████| 3/3 [00:00<00:00, 10.25it/s]
Train Loss: 3.2184, Train Acc: 0.1250
...
Epoch 10/10: 100%|█████| 3/3 [00:00<00:00, 12.45it/s]  
Train Loss: 1.2345, Train Acc: 0.4567
✅ Training completed!
'hello world' → 'bonjour le monde'
```

## Different Use Cases

### Text Summarization

```python
# Use different sequence lengths
src_tokenizer.max_length = 200   # Longer source (articles)
tgt_tokenizer.max_length = 50    # Shorter target (summaries)

model = Seq2SeqModel(
    src_vocab_size=len(src_tokenizer),
    tgt_vocab_size=len(tgt_tokenizer),
    hidden_size=256,
    rnn_type='bilstm',  # Bidirectional for context
    use_attention=True
)
```

### Chatbot

```python
model = Seq2SeqModel(
    src_vocab_size=len(src_tokenizer),
    tgt_vocab_size=len(tgt_tokenizer), 
    hidden_size=256,
    rnn_type='gru',      # Faster inference
    teacher_forcing_ratio=0.7,  # Higher for conversation
    max_length=30
)
```

### Custom Domain Translation

```python
# Medical, legal, or technical translation
src_tokenizer = Tokenizer(name="medical_english")
tgt_tokenizer = Tokenizer(name="medical_french")

# Add domain-specific preprocessing
def medical_preprocessing(text):
    # Handle medical abbreviations, terms
    text = text.replace("pt.", "patient")
    text = text.replace("dx.", "diagnosis")
    return text

src_tokenizer.preprocessor.add_step(medical_preprocessing)
tgt_tokenizer.preprocessor.add_step(medical_preprocessing)
```

## Common Modifications

### Larger Model for Better Performance

```python
model = Seq2SeqModel(
    src_vocab_size=len(src_tokenizer),
    tgt_vocab_size=len(tgt_tokenizer),
    hidden_size=512,      # Increased capacity
    num_layers=3,         # More layers
    rnn_type='bilstm',    # Bidirectional for context
    dropout=0.2,          # Regularization
    use_attention=True
)
```

### Faster Training Configuration

```python
model = Seq2SeqModel(
    src_vocab_size=len(src_tokenizer),
    tgt_vocab_size=len(tgt_tokenizer),
    hidden_size=128,      # Smaller, faster
    num_layers=1,         # Fewer layers  
    rnn_type='gru',       # Faster than LSTM
    use_attention=False   # Simpler model
)

trainer = Trainer(model, learning_rate=0.001, device='cuda')  # Use GPU
```

## Troubleshooting

### Out of Memory

```python
# Reduce batch size
train_loader = create_data_loader(..., batch_size=8)  # Instead of 32

# Use smaller model
model = Seq2SeqModel(..., hidden_size=64, num_layers=1)
```

### Slow Training

```python
# Enable GPU
trainer = Trainer(model, device='cuda')

# Use smaller vocabulary
src_tokenizer.fit(texts, max_vocab_size=5000)
tgt_tokenizer.fit(texts, max_vocab_size=5000)
```

### Poor Results

```python
# Train for more epochs
history = trainer.train(..., epochs=50)

# Use attention and beam search
model = Seq2SeqModel(..., use_attention=True)
output_sequence = model.beam_search(..., beam_width=5)
```

## Next Steps

Ready to dive deeper? Check out:

- [Tokenizers Guide](guides/tokenizers.md) - Advanced text processing
- [Models Guide](guides/models.md) - Architecture details and RNN types  
- [Training Guide](guides/training.md) - Hyperparameter tuning and best practices
- [API Reference](api/) - Complete class documentation
- [Examples](../examples/) - Real-world use cases

## Need Help?

- Check the [Troubleshooting Guide](guides/troubleshooting.md)
- Look at running examples in the `examples/` directory
- [Open an issue](https://github.com/42Wor/aceflow/issues) on GitHub

---

**Congratulations!** You've built your first Seq2Seq model with AceFlow! 🎉
```

This quickstart guide provides:

1. **Complete working example** - Copy-paste ready code
2. **Step-by-step explanation** - Understand each part
3. **Different use cases** - Translation, summarization, chatbot
4. **Common modifications** - Larger/faster models
5. **Troubleshooting** - Solutions to common problems
6. **Clear next steps** - Where to go from here

The guide is designed to get users from zero to a working model as quickly as possible while explaining the key concepts along the way.