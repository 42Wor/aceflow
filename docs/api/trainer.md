
# Training Guide

This guide covers how to train sequence-to-sequence models using AceFlow's training system, from basic setups to advanced configurations.

## Table of Contents

- [Quick Start](#quick-start)
- [Basic Training](#basic-training)
- [Advanced Training](#advanced-training)
- [Training Configuration](#training-configuration)
- [Monitoring & Visualization](#monitoring--visualization)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

## Quick Start

### Minimal Training Example

```python
import torch
from aceflow import Seq2SeqModel
from aceflow.utils import Tokenizer, create_data_loader
from aceflow.trainers import Seq2SeqTrainer

# 1. Prepare your data
english_sentences = ["hello world", "how are you", "good morning"]
french_sentences = ["bonjour le monde", "comment allez vous", "bonjour"]

# 2. Initialize tokenizers
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

# 5. Train model
trainer = Seq2SeqTrainer(model, learning_rate=0.001)
history = trainer.train(
    train_loader, train_loader,  # Using same data for train/val in this example
    epochs=10,
    save_path="my_first_model.ace"
)

print("Training completed!")
```

## Basic Training

### Step 1: Prepare Your Data

```python
# Sample translation dataset
train_english = [
    "hello world", "how are you", "good morning", "what is your name",
    "i love programming", "the weather is nice", "see you later",
    "thank you", "have a nice day", "where is the station"
]

train_french = [
    "bonjour le monde", "comment allez vous", "bonjour", "quel est votre nom",
    "j aime la programmation", "le temps est agreable", "a plus tard",
    "merci", "passez une bonne journee", "ou est la gare"
]

# Validation split
val_english = train_english[:2]  # First 2 samples for validation
val_french = train_french[:2]
train_english = train_english[2:]  # Remaining for training
train_french = train_french[2:]
```

### Step 2: Initialize Tokenizers

```python
from aceflow.utils import Tokenizer

# Configure tokenizers
src_tokenizer = Tokenizer(
    name="english_tokenizer",
    language="english",
    max_length=15,    # Based on your data analysis
    padding="post",
    truncation="post"
)

tgt_tokenizer = Tokenizer(
    name="french_tokenizer", 
    language="french",
    max_length=20,    # Target sequences might be longer
    padding="post",
    truncation="post"
)

# Build vocabulary
src_tokenizer.fit(train_english + val_english)
tgt_tokenizer.fit(train_french + val_french)

print(f"Source vocabulary: {len(src_tokenizer)} tokens")
print(f"Target vocabulary: {len(tgt_tokenizer)} tokens")
```

### Step 3: Create Data Loaders

```python
from aceflow.utils import create_data_loader

train_loader = create_data_loader(
    train_english, train_french,
    src_tokenizer, tgt_tokenizer,
    batch_size=4,
    max_length=15,
    shuffle=True
)

val_loader = create_data_loader(
    val_english, val_french,
    src_tokenizer, tgt_tokenizer,
    batch_size=4,
    max_length=15,
    shuffle=False  # No need to shuffle validation data
)
```

### Step 4: Configure Your Model

```python
from aceflow import Seq2SeqModel

model = Seq2SeqModel(
    src_vocab_size=len(src_tokenizer),
    tgt_vocab_size=len(tgt_tokenizer),
    hidden_size=256,           # Larger for more complex tasks
    num_layers=2,              # Deeper networks for complex patterns
    dropout=0.1,               # Regularization
    rnn_type='lstm',           # Options: 'lstm', 'gru', 'rnn'
    use_attention=True,        # Highly recommended for translation
    bidirectional=False,       # Set to True for better encoding
    teacher_forcing_ratio=0.5, # Balance between teacher forcing and free running
    max_length=50              # Maximum generation length
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Step 5: Initialize Trainer

```python
from aceflow.trainers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    model=model,
    learning_rate=0.001,
    device='auto',              # Automatically use GPU if available
    early_stopping_patience=5,  # Stop if no improvement for 5 epochs
    gradient_clip=1.0,          # Prevent gradient explosion
    use_amp=False,              # Set to True for faster training on supported GPUs
    teacher_forcing_ratio=0.5   # Can override model's setting
)
```

### Step 6: Train the Model

```python
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=50,
    save_path="translation_model.ace",
    eval_every=1,               # Validate after every epoch
    teacher_forcing_ratio=0.5   # Can be adjusted here too
)

print("Training completed successfully!")
```

## Advanced Training

### Using Callbacks for Enhanced Training

```python
from aceflow.trainers import ModelCheckpoint, EarlyStopping, ProgressLogger

# Configure callbacks
callbacks = [
    ModelCheckpoint(
        "models/best_model.ace",
        monitor='val_loss',      # Monitor validation loss
        save_best_only=True,     # Only save when validation loss improves
        mode='min'               # Lower is better for loss
    ),
    EarlyStopping(
        patience=10,             # Stop after 10 epochs without improvement
        monitor='val_loss',
        min_delta=0.001,         # Minimum change to qualify as improvement
        mode='min'
    ),
    ProgressLogger()             # Print progress after each epoch
]

# Train with callbacks
history = trainer.train(
    train_loader, val_loader,
    epochs=100,
    save_path="models/final_model.ace",
    callbacks=callbacks
)
```

### Custom Training Loop

For maximum control, you can create a custom training loop:

```python
from aceflow.trainers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(model, learning_rate=0.001)

best_val_loss = float('inf')
for epoch in range(100):
    # Custom training logic
    train_loss, train_acc = trainer.train_epoch(train_loader)
    val_loss, val_acc = trainer.validate_epoch(val_loader)
    
    print(f"Epoch {epoch+1}: "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Custom early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        model.save(f"models/best_epoch_{epoch+1}.ace")
        print(f"New best model saved!")
    
    # Custom learning rate scheduling
    if epoch > 0 and epoch % 10 == 0:
        current_lr = trainer.get_learning_rate()
        new_lr = current_lr * 0.5
        trainer.set_learning_rate(new_lr)
        print(f"Learning rate decreased to {new_lr:.2e}")
```

### Curriculum Learning

Gradually increase difficulty during training:

```python
def curriculum_training():
    trainer = Seq2SeqTrainer(model, learning_rate=0.001)
    
    # Phase 1: Easy examples (short sequences)
    easy_indices = [i for i, text in enumerate(train_english) if len(text.split()) <= 5]
    easy_english = [train_english[i] for i in easy_indices]
    easy_french = [train_french[i] for i in easy_indices]
    
    easy_loader = create_data_loader(
        easy_english, easy_french, src_tokenizer, tgt_tokenizer,
        batch_size=8, max_length=10
    )
    
    print("Phase 1: Training on easy examples...")
    trainer.train(easy_loader, val_loader, epochs=20, save_path="phase1_model.ace")
    
    # Phase 2: All examples
    full_loader = create_data_loader(
        train_english, train_french, src_tokenizer, tgt_tokenizer,
        batch_size=8, max_length=20
    )
    
    print("Phase 2: Training on all examples...")
    trainer.train(full_loader, val_loader, epochs=50, save_path="phase2_model.ace")
```

## Training Configuration

### Model Architecture Choices

```python
# Small model (fast training, less capacity)
small_model = Seq2SeqModel(
    src_vocab_size=5000,
    tgt_vocab_size=5000,
    hidden_size=128,
    num_layers=1,
    use_attention=True
)

# Medium model (balanced)
medium_model = Seq2SeqModel(
    src_vocab_size=10000,
    tgt_vocab_size=10000,
    hidden_size=256,
    num_layers=2,
    use_attention=True,
    dropout=0.1
)

# Large model (high capacity, slower training)
large_model = Seq2SeqModel(
    src_vocab_size=30000,
    tgt_vocab_size=30000,
    hidden_size=512,
    num_layers=3,
    use_attention=True,
    bidirectional=True,
    dropout=0.2
)
```

### Optimizer and Learning Rate

```python
from torch.optim import Adam, AdamW
from aceflow.trainers import Seq2SeqTrainer

# Different optimizers
trainer_adam = Seq2SeqTrainer(
    model=model,
    optimizer=Adam(model.parameters(), lr=0.001, weight_decay=1e-5),
    learning_rate=0.001  # Will be overridden by optimizer's lr
)

trainer_adamw = Seq2SeqTrainer(
    model=model,
    optimizer=AdamW(model.parameters(), lr=0.001, weight_decay=0.01),
    learning_rate=0.001
)
```

### Handling Different RNN Types

```python
# Compare different RNN architectures
rnn_types = ['rnn', 'gru', 'lstm', 'bilstm']

for rnn_type in rnn_types:
    print(f"\nTraining with {rnn_type.upper()}...")
    
    model = Seq2SeqModel(
        src_vocab_size=len(src_tokenizer),
        tgt_vocab_size=len(tgt_tokenizer),
        hidden_size=128,
        rnn_type=rnn_type,
        use_attention=True,
        bidirectional=rnn_type.startswith('bi')
    )
    
    trainer = Seq2SeqTrainer(model, learning_rate=0.001)
    history = trainer.train(
        train_loader, val_loader,
        epochs=10,
        save_path=f"models/model_{rnn_type}.ace"
    )
```

## Monitoring & Visualization

### Tracking Training Progress

```python
# After training, analyze results
best_epoch = trainer.get_best_epoch()
best_val_loss = min(history['val_loss'])
final_train_acc = history['train_accuracy'][-1]
final_val_acc = history['val_accuracy'][-1]

print(f"Best epoch: {best_epoch + 1}")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Final training accuracy: {final_train_acc:.4f}")
print(f"Final validation accuracy: {final_val_acc:.4f}")

# Save training history
trainer.save_training_history("training_history.json")
```

### Visualization

```python
import matplotlib.pyplot as plt

def plot_training_curves(history):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    ax1.plot(history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    ax1.plot(history['val_loss'], label='Val Loss', color='red', linewidth=2)
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(history['train_accuracy'], label='Train Accuracy', color='green', linewidth=2)
    ax2.plot(history['val_accuracy'], label='Val Accuracy', color='orange', linewidth=2)
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Learning rate
    if 'learning_rates' in history:
        ax3.plot(history['learning_rates'], label='Learning Rate', color='purple', linewidth=2)
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Epoch times
    if 'epoch_times' in history:
        ax4.plot(history['epoch_times'], label='Epoch Time', color='brown', linewidth=2)
        ax4.set_title('Epoch Training Time')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Time (seconds)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

# Use the plotting function
plot_training_curves(history)
```

### Real-time Monitoring

```python
class TrainingMonitor:
    def __init__(self):
        self.epoch_losses = []
        self.epoch_accuracies = []
    
    def on_epoch_end(self, epoch, train_metrics, val_metrics, trainer=None):
        self.epoch_losses.append({
            'epoch': epoch,
            'train_loss': train_metrics.get('loss', 0),
            'val_loss': val_metrics.get('loss', 0)
        })
        self.epoch_accuracies.append({
            'epoch': epoch, 
            'train_acc': train_metrics.get('accuracy', 0),
            'val_acc': val_metrics.get('accuracy', 0)
        })
        
        # Print custom progress
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: "
                  f"Train Loss: {train_metrics.get('loss', 0):.4f}, "
                  f"Val Loss: {val_metrics.get('loss', 0):.4f}, "
                  f"LR: {trainer.get_learning_rate():.2e}")

# Use custom monitor
monitor = TrainingMonitor()
history = trainer.train(
    train_loader, val_loader,
    epochs=50,
    callbacks=[monitor]
)
```

## Troubleshooting

### Common Training Issues

#### 1. Loss Not Decreasing

```python
# Solution: Adjust learning rate or model capacity
trainer = Seq2SeqTrainer(
    model=model,
    learning_rate=0.0001,  # Try lower learning rate
    gradient_clip=1.0      # Prevent gradient explosion
)

# Or use a larger model
larger_model = Seq2SeqModel(
    src_vocab_size=len(src_tokenizer),
    tgt_vocab_size=len(tgt_tokenizer),
    hidden_size=512,  # Increased capacity
    num_layers=3,     # More layers
    dropout=0.3       # More regularization
)
```

#### 2. Overfitting

```python
# Solutions for overfitting
model = Seq2SeqModel(
    # ... other parameters ...
    dropout=0.3,                    # Increase dropout
    teacher_forcing_ratio=0.8       # More teacher forcing
)

trainer = Seq2SeqTrainer(
    model=model,
    early_stopping_patience=5,      # Stop early
    gradient_clip=1.0
)

# Use more aggressive data augmentation
# Increase dataset size
# Use smaller model
```

#### 3. Memory Issues

```python
# Reduce batch size
train_loader = create_data_loader(..., batch_size=16)  # Instead of 32

# Use gradient accumulation
def train_with_gradient_accumulation(trainer, dataloader, accumulation_steps=4):
    trainer.optimizer.zero_grad()
    
    for i, batch in enumerate(dataloader):
        # Forward pass
        loss = compute_loss(batch)
        loss = loss / accumulation_steps  # Normalize loss
        
        # Backward pass
        loss.backward()
        
        # Step only after accumulation_steps
        if (i + 1) % accumulation_steps == 0:
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()
    
    # Don't forget the remaining batches
    if len(dataloader) % accumulation_steps != 0:
        trainer.optimizer.step()
        trainer.optimizer.zero_grad()
```

#### 4. Slow Training

```python
# Enable mixed precision (if supported)
trainer = Seq2SeqTrainer(
    model=model,
    use_amp=True,  # Automatic Mixed Precision
    device='cuda'  # Use GPU
)

# Use larger batches
train_loader = create_data_loader(..., batch_size=64)

# Reduce model complexity
smaller_model = Seq2SeqModel(
    hidden_size=128,  # Instead of 256
    num_layers=1,     # Instead of 2
    # ... other parameters
)
```

### Debugging Tools

```python
def debug_training_batch(trainer, dataloader):
    """Debug the first batch to check data and model output"""
    batch = next(iter(dataloader))
    
    print("Batch info:")
    print(f"  Source shape: {batch['src'].shape}")
    print(f"  Target shape: {batch['tgt'].shape}")
    print(f"  Source non-pad tokens: {(batch['src'] != 0).sum().item()}")
    print(f"  Target non-pad tokens: {(batch['tgt'] != 0).sum().item()}")
    
    # Test forward pass
    trainer.model.eval()
    with torch.no_grad():
        outputs = trainer.model(batch['src'], batch['tgt'])
        if isinstance(outputs, tuple):
            outputs = outputs[0]  # Unpack if attention model
        
        print(f"  Output shape: {outputs.shape}")
        print(f"  Output range: [{outputs.min().item():.3f}, {outputs.max().item():.3f}]")
    
    return batch

# Run debug
debug_batch = debug_training_batch(trainer, train_loader)
```

## Best Practices

### 1. Data Preparation

```python
def prepare_training_data(src_texts, tgt_texts, train_ratio=0.8):
    """Prepare and validate training data"""
    assert len(src_texts) == len(tgt_texts), "Source and target must have same length"
    
    # Filter empty sequences
    valid_indices = [
        i for i, (src, tgt) in enumerate(zip(src_texts, tgt_texts))
        if src.strip() and tgt.strip()
    ]
    
    src_texts = [src_texts[i] for i in valid_indices]
    tgt_texts = [tgt_texts[i] for i in valid_indices]
    
    # Split data
    split_idx = int(len(src_texts) * train_ratio)
    train_src, val_src = src_texts[:split_idx], src_texts[split_idx:]
    train_tgt, val_tgt = tgt_texts[:split_idx], tgt_texts[split_idx:]
    
    print(f"Training samples: {len(train_src)}")
    print(f"Validation samples: {len(val_src)}")
    
    return train_src, train_tgt, val_src, val_tgt
```

### 2. Hyperparameter Tuning

```python
def hyperparameter_search():
    """Simple hyperparameter search"""
    best_score = float('inf')
    best_params = {}
    
    for hidden_size in [128, 256, 512]:
        for learning_rate in [0.001, 0.0005, 0.0001]:
            for num_layers in [1, 2, 3]:
                print(f"Testing: hidden_size={hidden_size}, lr={learning_rate}, layers={num_layers}")
                
                model = Seq2SeqModel(
                    src_vocab_size=len(src_tokenizer),
                    tgt_vocab_size=len(tgt_tokenizer),
                    hidden_size=hidden_size,
                    num_layers=num_layers
                )
                
                trainer = Seq2SeqTrainer(model, learning_rate=learning_rate)
                
                # Quick training for evaluation
                history = trainer.train(train_loader, val_loader, epochs=5)
                final_val_loss = history['val_loss'][-1]
                
                if final_val_loss < best_score:
                    best_score = final_val_loss
                    best_params = {
                        'hidden_size': hidden_size,
                        'learning_rate': learning_rate,
                        'num_layers': num_layers
                    }
    
    print(f"Best parameters: {best_params}")
    print(f"Best validation loss: {best_score:.4f}")
    return best_params
```

### 3. Model Checkpointing Strategy

```python
import os
from datetime import datetime

def create_checkpoint_strategy(base_path="models"):
    """Create organized checkpoint strategy"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = f"{base_path}/run_{timestamp}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            f"{checkpoint_dir}/best_loss.ace",
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        ),
        ModelCheckpoint(
            f"{checkpoint_dir}/best_accuracy.ace", 
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]
    
    return checkpoint_dir, callbacks

# Use in training
checkpoint_dir, callbacks = create_checkpoint_strategy()
history = trainer.train(
    train_loader, val_loader,
    epochs=100,
    save_path=f"{checkpoint_dir}/final_model.ace",
    callbacks=callbacks
)
```

### 4. Complete Training Pipeline

```python
def complete_training_pipeline(src_texts, tgt_texts, model_name="translation_model"):
    """Complete training pipeline from data to deployed model"""
    
    # 1. Data preparation
    train_src, train_tgt, val_src, val_tgt = prepare_training_data(src_texts, tgt_texts)
    
    # 2. Tokenizer setup
    src_tokenizer = Tokenizer(name=f"{model_name}_src")
    tgt_tokenizer = Tokenizer(name=f"{model_name}_tgt")
    src_tokenizer.fit(train_src + val_src)
    tgt_tokenizer.fit(train_tgt + val_tgt)
    
    # 3. Data loaders
    train_loader = create_data_loader(train_src, train_tgt, src_tokenizer, tgt_tokenizer)
    val_loader = create_data_loader(val_src, val_tgt, src_tokenizer, tgt_tokenizer)
    
    # 4. Model configuration
    model = Seq2SeqModel(
        src_vocab_size=len(src_tokenizer),
        tgt_vocab_size=len(tgt_tokenizer),
        hidden_size=256,
        num_layers=2,
        use_attention=True
    )
    
    # 5. Training with checkpoints
    checkpoint_dir, callbacks = create_checkpoint_strategy()
    
    trainer = Seq2SeqTrainer(
        model=model,
        learning_rate=0.001,
        early_stopping_patience=10
    )
    
    # 6. Train model
    history = trainer.train(
        train_loader, val_loader,
        epochs=100,
        save_path=f"{checkpoint_dir}/{model_name}.ace",
        callbacks=callbacks
    )
    
    # 7. Save artifacts
    src_tokenizer.save(f"{checkpoint_dir}/src_tokenizer")
    tgt_tokenizer.save(f"{checkpoint_dir}/tgt_tokenizer")
    trainer.save_training_history(f"{checkpoint_dir}/training_history.json")
    
    # 8. Create deployment package
    deployment_files = {
        'model': f"{checkpoint_dir}/best_loss.ace",
        'src_tokenizer': f"{checkpoint_dir}/src_tokenizer",
        'tgt_tokenizer': f"{checkpoint_dir}/tgt_tokenizer", 
        'config': f"{checkpoint_dir}/training_history.json"
    }
    
    print("Training pipeline completed!")
    print(f"Model and artifacts saved to: {checkpoint_dir}")
    
    return deployment_files

# Run complete pipeline
deployment_files = complete_training_pipeline(english_texts, french_texts)
```

This comprehensive training guide covers everything from basic usage to advanced techniques, troubleshooting, and best practices. The examples provided should help you successfully train sequence-to-sequence models for various tasks.

For more specific use cases or advanced techniques, check the [examples directory](../examples/) and [API reference](../api/).
```