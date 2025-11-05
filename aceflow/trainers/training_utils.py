import torch
import numpy as np
from typing import Dict, List, Optional
import json
import matplotlib.pyplot as plt

def plot_training_history(history: Dict, save_path: Optional[str] = None):
    """Plot training history"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    if history.get('train_loss'):
        ax1.plot(history['train_loss'], label='Train Loss', color='blue')
    if history.get('val_loss'):
        ax1.plot(history['val_loss'], label='Val Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    if history.get('train_accuracy'):
        ax2.plot(history['train_accuracy'], label='Train Accuracy', color='blue')
    if history.get('val_accuracy'):
        ax2.plot(history['val_accuracy'], label='Val Accuracy', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Learning rate plot
    if history.get('learning_rates'):
        ax3.plot(history['learning_rates'], label='Learning Rate', color='green')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True)
    
    # Epoch time plot
    if history.get('epoch_times'):
        ax4.plot(history['epoch_times'], label='Epoch Time', color='purple')
        ax4.set_title('Epoch Training Time')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Time (seconds)')
        ax4.legend()
        ax4.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def save_training_report(history: Dict, config: Dict, save_path: str):
    """Save comprehensive training report"""
    report = {
        'config': config,
        'history': history,
        'best_epoch': np.argmin(history['val_loss']) if history['val_loss'] else 0,
        'best_val_loss': min(history['val_loss']) if history['val_loss'] else 0,
        'final_train_loss': history['train_loss'][-1] if history['train_loss'] else 0,
        'final_val_loss': history['val_loss'][-1] if history['val_loss'] else 0,
    }
    
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)

def count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size(model: torch.nn.Module) -> str:
    """Get model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_mb = (param_size + buffer_size) / 1024**2
    return f"{size_mb:.2f} MB"

def setup_mixed_precision():
    """Setup mixed precision training if available"""
    try:
        from torch.cuda.amp import GradScaler, autocast
        return True, GradScaler()
    except ImportError:
        return False, None