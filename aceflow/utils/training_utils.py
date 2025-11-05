import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
import os

def convert_to_serializable(obj: Any) -> Any:
    """Convert NumPy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    else:
        return obj

def save_training_report(history: Dict, config: Dict, save_path: str):
    """Save comprehensive training report with proper JSON serialization"""
    
    # Convert all data to JSON-serializable types
    history_serializable = convert_to_serializable(history)
    config_serializable = convert_to_serializable(config)
    
    # Calculate summary statistics
    val_losses = history_serializable.get('val_loss', [])
    train_losses = history_serializable.get('train_loss', [])
    val_accuracies = history_serializable.get('val_accuracy', [])
    train_accuracies = history_serializable.get('train_accuracy', [])
    epoch_times = history_serializable.get('epoch_times', [])
    
    report = {
        'config': config_serializable,
        'history_summary': {
            'best_epoch': np.argmin(val_losses) if val_losses else 0,
            'best_val_loss': min(val_losses) if val_losses else float('inf'),
            'final_train_loss': train_losses[-1] if train_losses else 0,
            'final_val_loss': val_losses[-1] if val_losses else 0,
            'final_train_acc': train_accuracies[-1] if train_accuracies else 0,
            'final_val_acc': val_accuracies[-1] if val_accuracies else 0,
            'min_train_loss': min(train_losses) if train_losses else 0,
            'min_val_loss': min(val_losses) if val_losses else 0,
            'max_train_acc': max(train_accuracies) if train_accuracies else 0,
            'max_val_acc': max(val_accuracies) if val_accuracies else 0,
        },
        'training_stats': {
            'total_epochs': len(train_losses),
            'total_training_time': sum(epoch_times) if epoch_times else 0,
            'avg_epoch_time': sum(epoch_times) / len(epoch_times) if epoch_times else 0,
            'final_learning_rate': history_serializable.get('learning_rates', [0])[-1] if history_serializable.get('learning_rates') else 0,
        }
    }
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    # Convert the report to serializable format
    report_serializable = convert_to_serializable(report)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(report_serializable, f, indent=2, ensure_ascii=False)

def plot_training_history(history: Dict, save_path: Optional[str] = None, show: bool = True):
    """Plot training history with error handling"""
    try:
        # Convert history to serializable for plotting
        history_serializable = convert_to_serializable(history)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        if history_serializable.get('train_loss'):
            ax1.plot(history_serializable['train_loss'], label='Train Loss', color='blue', linewidth=2)
        if history_serializable.get('val_loss'):
            ax1.plot(history_serializable['val_loss'], label='Val Loss', color='red', linewidth=2)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        if history_serializable.get('train_accuracy'):
            ax2.plot(history_serializable['train_accuracy'], label='Train Accuracy', color='blue', linewidth=2)
        if history_serializable.get('val_accuracy'):
            ax2.plot(history_serializable['val_accuracy'], label='Val Accuracy', color='red', linewidth=2)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Learning rate plot
        if history_serializable.get('learning_rates'):
            ax3.plot(history_serializable['learning_rates'], label='Learning Rate', color='green', linewidth=2)
            ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.set_yscale('log')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No Learning Rate Data', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        
        # Epoch time plot
        if history_serializable.get('epoch_times'):
            ax4.plot(history_serializable['epoch_times'], label='Epoch Time', color='purple', linewidth=2)
            ax4.set_title('Epoch Training Time', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Time (seconds)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No Epoch Time Data', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Epoch Training Time', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ Training plots saved to: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
            
    except Exception as e:
        print(f"⚠️ Could not generate training plots: {e}")
        if show:
            # Create a simple text-based alternative
            print("\n" + "="*50)
            print("TRAINING SUMMARY (Text Version)")
            print("="*50)
            if history.get('train_loss'):
                print(f"Final Train Loss: {history['train_loss'][-1]:.4f}")
            if history.get('val_loss'):
                print(f"Final Val Loss: {history['val_loss'][-1]:.4f}")
            if history.get('train_accuracy'):
                print(f"Final Train Acc: {history['train_accuracy'][-1]:.4f}")
            if history.get('val_accuracy'):
                print(f"Final Val Acc: {history['val_accuracy'][-1]:.4f}")
            print("="*50)

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