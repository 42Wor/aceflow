
# Trainer API Reference

The `Trainer` class provides comprehensive utilities for training and evaluating Seq2Seq models.

## Overview

```python
from aceflow.trainers import Trainer
```

The Trainer handles:
- Training loop with progress tracking
- Validation and metrics calculation
- Model checkpointing
- Learning rate scheduling
- GPU/CPU device management

## Class Definition

```python
Trainer(
    model,
    learning_rate=0.001,
    device='auto'
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `Seq2SeqModel` | **Required** | The model to train |
| `learning_rate` | `float` | `0.001` | Learning rate for optimizer |
| `device` | `str` | `'auto'` | Device: `'auto'`, `'cuda'`, `'cpu'` |

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `model` | `Seq2SeqModel` | The model being trained |
| `optimizer` | `torch.optim.Adam` | Adam optimizer |
| `criterion` | `nn.CrossEntropyLoss` | Loss function (ignores padding) |
| `device` | `torch.device` | Training device |
| `history` | `dict` | Training history metrics |

## Methods

### train

```python
train(
    train_loader,
    val_loader,
    epochs=10,
    save_path=None,
    teacher_forcing_ratio=0.5,
    eval_every=1
)
```

Main training method that runs the complete training loop.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_loader` | `DataLoader` | **Required** | Training data loader |
| `val_loader` | `DataLoader` | **Required** | Validation data loader |
| `epochs` | `int` | `10` | Number of training epochs |
| `save_path` | `str` | `None` | Path to save model checkpoints |
| `teacher_forcing_ratio` | `float` | `0.5` | Probability of using teacher forcing |
| `eval_every` | `int` | `1` | Validate every N epochs |

**Returns:**
- `history` (`dict`): Training history containing:
  - `train_loss`: List of training losses
  - `val_loss`: List of validation losses  
  - `train_accuracy`: List of training accuracies
  - `val_accuracy`: List of validation accuracies

**Example:**
```python
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=50,
    save_path="models/checkpoint.ace",
    teacher_forcing_ratio=0.5,
    eval_every=2
)
```

### train_epoch

```python
train_epoch(dataloader, teacher_forcing_ratio=0.5)
```

Train the model for a single epoch.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataloader` | `DataLoader` | **Required** | Training data loader |
| `teacher_forcing_ratio` | `float` | `0.5` | Teacher forcing probability |

**Returns:**
- `avg_loss` (`float`): Average loss for the epoch
- `accuracy` (`float`): Accuracy for the epoch

**Example:**
```python
train_loss, train_acc = trainer.train_epoch(
    dataloader=train_loader,
    teacher_forcing_ratio=0.7
)
```

### validate_epoch

```python
validate_epoch(dataloader)
```

Validate the model for a single epoch.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataloader` | `DataLoader` | **Required** | Validation data loader |

**Returns:**
- `avg_loss` (`float`): Average validation loss
- `accuracy` (`float`): Validation accuracy

**Example:**
```python
val_loss, val_acc = trainer.validate_epoch(val_loader)
```

### save_training_history

```python
save_training_history(filepath)
```

Save training history to a JSON file.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filepath` | `str` | **Required** | Path to save history JSON |

**Example:**
```python
trainer.save_training_history("training_history.json")
```

### load_training_history

```python
load_training_history(filepath)
```

Load training history from a JSON file.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filepath` | `str` | **Required** | Path to load history JSON |

**Example:**
```python
trainer.load_training_history("training_history.json")
```

## Usage Examples

### Basic Training

```python
from aceflow.trainers import Trainer

# Initialize trainer
trainer = Trainer(
    model=model,
    learning_rate=0.001,
    device='auto'  # Automatically use GPU if available
)

# Train model
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    save_path="best_model.ace"
)
```

### Custom Training Loop

```python
# Manual training loop with custom logic
for epoch in range(100):
    # Custom learning rate scheduling
    if epoch == 50:
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = 0.0001
    
    # Train one epoch
    train_loss, train_acc = trainer.train_epoch(
        train_loader, 
        teacher_forcing_ratio=0.5
    )
    
    # Validate every 5 epochs
    if epoch % 5 == 0:
        val_loss, val_acc = trainer.validate_epoch(val_loader)
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            model.save("best_model.ace")
            best_val_loss = val_loss
```

### Monitoring Training

```python
import matplotlib.pyplot as plt

# Train model
history = trainer.train(...)

# Plot training curves
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss Curves')

plt.subplot(1, 2, 2)
plt.plot(history['train_accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Accuracy Curves')

plt.tight_layout()
plt.savefig('training_curves.png')
```

### Advanced Configuration

```python
# Custom optimizer and loss
import torch.optim as optim
import torch.nn as nn

trainer = Trainer(model, learning_rate=0.001)

# Replace optimizer
trainer.optimizer = optim.AdamW(
    model.parameters(), 
    lr=0.001, 
    weight_decay=0.01
)

# Replace loss function
trainer.criterion = nn.CrossEntropyLoss(
    ignore_index=0,  # Ignore padding
    label_smoothing=0.1  # Label smoothing
)

# Add learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(
    trainer.optimizer, 
    step_size=10, 
    gamma=0.5
)

# Use in training loop
for epoch in range(100):
    train_loss, train_acc = trainer.train_epoch(train_loader)
    scheduler.step()  # Update learning rate
```

## Training Configuration

### Recommended Hyperparameters

| Scenario | Learning Rate | Batch Size | Teacher Forcing | Epochs |
|----------|---------------|------------|-----------------|---------|
| Small Dataset | 0.001 | 16-32 | 0.7-0.9 | 50-100 |
| Medium Dataset | 0.0005 | 32-64 | 0.5-0.7 | 100-200 |
| Large Dataset | 0.0001 | 64-128 | 0.3-0.5 | 200-500 |

### Device Management

```python
# Automatic device selection
trainer = Trainer(model, device='auto')  # Uses CUDA if available

# Force CPU
trainer = Trainer(model, device='cpu')

# Force GPU
trainer = Trainer(model, device='cuda')

# Specific GPU
trainer = Trainer(model, device='cuda:0')
```

### Gradient Clipping

```python
# Add gradient clipping to training
def train_epoch_with_clipping(dataloader, max_norm=1.0):
    trainer.model.train()
    total_loss = 0
    
    for batch in dataloader:
        src = batch['src'].to(trainer.device)
        tgt = batch['tgt'].to(trainer.device)
        
        trainer.optimizer.zero_grad()
        output = trainer.model(src, tgt)
        
        # Calculate loss
        output = output[:, :-1].contiguous()
        tgt = tgt[:, 1:].contiguous()
        loss = trainer.criterion(output.view(-1, output.size(-1)), tgt.view(-1))
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            trainer.model.parameters(), 
            max_norm=max_norm
        )
        
        trainer.optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
```

## Error Handling

### Common Exceptions

```python
try:
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100
    )
except RuntimeError as e:
    if "out of memory" in str(e):
        print("GPU out of memory. Try reducing batch size.")
        # Reduce batch size and try again
        train_loader = create_data_loader(..., batch_size=16)
    else:
        raise e
```

### Recovery from Checkpoints

```python
# Save checkpoint with additional info
checkpoint = {
    'model_state': trainer.model.state_dict(),
    'optimizer_state': trainer.optimizer.state_dict(),
    'history': trainer.history,
    'epoch': epoch
}
torch.save(checkpoint, "checkpoint.pth")

# Load checkpoint
checkpoint = torch.load("checkpoint.pth")
trainer.model.load_state_dict(checkpoint['model_state'])
trainer.optimizer.load_state_dict(checkpoint['optimizer_state'])
trainer.history = checkpoint['history']
start_epoch = checkpoint['epoch']
```

## Best Practices

### 1. Early Stopping

```python
best_val_loss = float('inf')
patience = 10
patience_counter = 0

for epoch in range(100):
    train_loss, train_acc = trainer.train_epoch(train_loader)
    val_loss, val_acc = trainer.validate_epoch(val_loader)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        model.save("best_model.ace")
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break
```

### 2. Learning Rate Scheduling

```python
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(
    trainer.optimizer, 
    mode='min', 
    factor=0.5, 
    patience=5
)

for epoch in range(100):
    train_loss, train_acc = trainer.train_epoch(train_loader)
    val_loss, val_acc = trainer.validate_epoch(val_loader)
    
    scheduler.step(val_loss)  # Update LR based on validation loss
    
    current_lr = trainer.optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch}: LR = {current_lr:.6f}")
```

### 3. Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

def train_epoch_mixed_precision(dataloader):
    trainer.model.train()
    total_loss = 0
    
    for batch in dataloader:
        src = batch['src'].to(trainer.device)
        tgt = batch['tgt'].to(trainer.device)
        
        trainer.optimizer.zero_grad()
        
        with autocast():
            output = trainer.model(src, tgt)
            output = output[:, :-1].contiguous()
            tgt = tgt[:, 1:].contiguous()
            loss = trainer.criterion(output.view(-1, output.size(-1)), tgt.view(-1))
        
        scaler.scale(loss).backward()
        scaler.step(trainer.optimizer)
        scaler.update()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
```

## Related Resources

- [Seq2SeqModel API](seq2seqmodel.md) - Model architecture reference
- [Training Guide](../guides/training.md) - Best practices for training
- [Examples](../examples/) - Practical training examples
- [Troubleshooting](../guides/troubleshooting.md) - Common issues and solutions
```