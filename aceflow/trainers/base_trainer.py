import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer
import os
import json
import numpy as np
import time
from typing import Dict, List, Optional, Callable, Any
from .callback import Callback, CallbackHandler
from .metrics import MetricTracker

# Import tqdm with proper handling
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Import termcolor with fallback
try:
    from termcolor import colored
    TERMCOLOR_AVAILABLE = True
except ImportError:
    TERMCOLOR_AVAILABLE = False
    # Fallback colored function
    def colored(text, color=None, on_color=None, attrs=None):
        return text

class BaseTrainer:
    """
    Base trainer class with common training functionality
    """
    
    def __init__(self, 
                 model: nn.Module,
                 optimizer: Optional[Optimizer] = None,
                 criterion: Optional[nn.Module] = None,
                 learning_rate: float = 0.001,
                 device: str = 'auto',
                 early_stopping_patience: Optional[int] = None,
                 early_stopping_min_delta: float = 0.001,
                 gradient_clip: float = 1.0,
                 use_amp: bool = False):
        
        self.model = model
        self.device = self._setup_device(device)
        self.model.to(self.device)
        
        # Training components
        self.optimizer = optimizer or Adam(model.parameters(), lr=learning_rate)
        self.criterion = criterion
        self.gradient_clip = gradient_clip
        
        # Mixed precision training - only enable if CUDA is available
        self.use_amp = use_amp and torch.cuda.is_available()
        if self.use_amp:
            try:
                self.scaler = torch.amp.GradScaler('cuda')
            except (AttributeError, RuntimeError):
                # Fallback for older PyTorch versions or CPU
                self.use_amp = False
                self.scaler = None
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_accuracy': [], 'val_accuracy': [],
            'learning_rates': [], 'epoch_times': []
        }
        
        # Early stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.early_stopping_counter = 0
        self.best_val_loss = float('inf')
        self.early_stop = False
        
        # Callbacks and metrics
        self.callbacks = CallbackHandler()
        self.metrics = MetricTracker()
        
        # Table formatting
        self.table_headers = ["Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc", "LR", "Status"]
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup training device"""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        return torch.device(device)
    
    def _safe_print(self, text: str):
        """Safely print text without breaking tqdm progress bars"""
        if TQDM_AVAILABLE:
            tqdm.write(text)
        else:
            print(text)
    
    def print_table_header(self):
        """Print the table header"""
        headers = self.table_headers
        header_str = (f"| {headers[0]:^8} | {headers[1]:^10} | {headers[2]:^9} | "
                      f"{headers[3]:^8} | {headers[4]:^7} | {headers[5]:^8} | {headers[6]:^12} |")
        separator = "+" + "+".join(["-"*10, "-"*12, "-"*11, "-"*10, "-"*9, "-"*10, "-"*14]) + "+"
        
        self._safe_print("\n" + "="*90)
        self._safe_print(" " * 30 + "TRAINING PROGRESS")
        self._safe_print("="*90)
        self._safe_print(separator)
        self._safe_print(header_str)
        self._safe_print(separator)
    
    def _get_table_row_str(self, epoch: int, total_epochs: int, metrics: Dict, status: str) -> str:
        """Return a formatted table row string"""
        epoch_str = f"{epoch+1}/{total_epochs}"
        train_loss_str = f"{metrics.get('train_loss', 0):.4f}"
        train_acc_str = f"{metrics.get('train_accuracy', 0):.4f}"
        val_loss_str = f"{metrics.get('val_loss', 0):.4f}"
        val_acc_str = f"{metrics.get('val_accuracy', 0):.4f}"
        lr_str = f"{metrics.get('learning_rate', 0):.2e}"

        # Apply simple formatting
        if status == "Saved Best":
            status_display = "[BEST]"
        elif status == "Early Stop":
            status_display = "[STOP]"
        elif status == "Final":
            status_display = "[FINAL]"
        else:
            status_display = ""

        return (f"| {epoch_str:^8} | {train_loss_str:^10} | {train_acc_str:^9} | "
                f"{val_loss_str:^8} | {val_acc_str:^7} | {lr_str:^8} | {status_display:^12} |")
    
    def check_early_stopping(self, val_loss: float) -> tuple:
        """Check if training should stop early based on validation loss"""
        if self.early_stopping_patience is None:
            return False, "-"
        
        improvement = self.best_val_loss - val_loss
        if improvement > self.early_stopping_min_delta:
            self.best_val_loss = val_loss
            self.early_stopping_counter = 0
            return False, "Saved Best"
        else:
            self.early_stopping_counter += 1
            if self.early_stopping_counter >= self.early_stopping_patience:
                self.early_stop = True
                return True, "Early Stop"
            return False, "-"
    
    def train(self, 
              train_loader, 
              val_loader, 
              epochs: int = 10, 
              save_path: Optional[str] = None,
              eval_every: int = 1,
              callbacks: Optional[List[Callback]] = None) -> Dict:
        
        # Setup callbacks
        if callbacks:
            for callback in callbacks:
                self.callbacks.add_callback(callback)
        
        # Training initialization
        self.callbacks.on_train_begin()
        
        self._safe_print(f"Starting training on {self.device}")
        self._safe_print(f"Model has {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
        if self.early_stopping_patience:
            self._safe_print(f"Early stopping: {self.early_stopping_patience} epochs patience")
        
        if self.use_amp:
            self._safe_print("Mixed precision training: Enabled")
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

        self.print_table_header()

        # Create progress bar with proper configuration
        if TQDM_AVAILABLE:
            pbar = tqdm(
                total=epochs, 
                desc="Training Progress", 
                unit="epoch",
                bar_format='{l_bar}{bar:20}{r_bar}',
                position=0,
                leave=True
            )
        else:
            self._safe_print(f"Training for {epochs} epochs...")
        
        for epoch in range(epochs):
            self.epoch = epoch
            epoch_start_time = time.time()
            
            # Training phase
            self.callbacks.on_epoch_begin(epoch)
            train_metrics = self.train_epoch(train_loader, f"Epoch {epoch+1}/{epochs}")
            
            # Validation phase
            if (epoch + 1) % eval_every == 0:
                val_metrics = self.validate_epoch(val_loader, "Validation")
                
                # Update history
                for k, v in train_metrics.items():
                    self.history[f'train_{k}'].append(v)
                for k, v in val_metrics.items():
                    self.history[f'val_{k}'].append(v)
                
                # Check early stopping
                val_loss = val_metrics.get('loss', 0)
                if self.early_stopping_patience:
                    should_stop, status = self.check_early_stopping(val_loss)
                else:
                    should_stop = False
                    status = "Saved Best" if val_loss < self.best_val_loss else "-"
                    if status == "Saved Best":
                        self.best_val_loss = val_loss
            else:
                val_metrics = {}
                status = "-"
                should_stop = False
            
            # Update learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            self.history['epoch_times'].append(epoch_time)
            
            # Prepare metrics for display
            display_metrics = {
                'train_loss': train_metrics.get('loss', 0),
                'train_accuracy': train_metrics.get('accuracy', 0),
                'val_loss': val_metrics.get('loss', 0),
                'val_accuracy': val_metrics.get('accuracy', 0),
                'learning_rate': current_lr
            }
            
            # Final epoch status
            if epoch == epochs - 1:
                status = "Final"
            
            # Print progress row using safe print (tqdm.write)
            row_str = self._get_table_row_str(epoch, epochs, display_metrics, status)
            self._safe_print(row_str)
            
            # Update progress bar if available
            if TQDM_AVAILABLE:
                pbar.update(1)
                pbar.set_postfix({
                    'loss': f"{display_metrics['train_loss']:.4f}",
                    'val_acc': f"{display_metrics['val_accuracy']:.4f}",
                    'lr': f"{current_lr:.2e}"
                })
            
            # Save best model
            if save_path and status == "Saved Best":
                best_path = save_path.replace('.ace', '_best.ace') if save_path.endswith('.ace') else save_path + '_best.ace'
                if hasattr(self.model, 'save'):
                    self.model.save(best_path)
                    self._safe_print(f"✓ Best model saved to {best_path}")
            
            # Callbacks
            self.callbacks.on_epoch_end(epoch, train_metrics, val_metrics)
            
            # Early stopping
            if should_stop:
                self._safe_print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
                self._safe_print(f"Best validation loss: {self.best_val_loss:.4f}")
                break
        
        # Close progress bar
        if TQDM_AVAILABLE:
            pbar.close()
        
        # Save final model
        if save_path and not self.early_stop:
            final_path = save_path.replace('.ace', '_final.ace') if save_path.endswith('.ace') else save_path + '_final.ace'
            if hasattr(self.model, 'save'):
                self.model.save(final_path)
                self._safe_print(f"✓ Final model saved to {final_path}")
        
        # Print table footer
        separator = "+" + "+".join(["-"*10, "-"*12, "-"*11, "-"*10, "-"*9, "-"*10, "-"*14]) + "+"
        self._safe_print(separator)
        
        self.callbacks.on_train_end()
        return self.history
    
    def save_training_history(self, filepath: str):
        """Save training history to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        self._safe_print(f"✓ Training history saved to {filepath}")
    
    def load_training_history(self, filepath: str):
        """Load training history from JSON file"""
        with open(filepath, 'r') as f:
            self.history = json.load(f)
    
    def get_best_epoch(self) -> int:
        """Return the epoch with the best validation loss"""
        if not self.history['val_loss']:
            return -1
        return np.argmin(self.history['val_loss'])
    
    def get_learning_rate(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']
    
    def set_learning_rate(self, lr: float):
        """Set learning rate for all parameter groups"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    # Abstract methods to be implemented by subclasses
    def train_epoch(self, dataloader, desc: str = "Training") -> Dict[str, float]:
        """Train for one epoch - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement train_epoch")
    
    def validate_epoch(self, dataloader, desc: str = "Validation") -> Dict[str, float]:
        """Validate for one epoch - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement validate_epoch")
    
    def compute_loss(self, outputs, targets, **kwargs) -> torch.Tensor:
        """Compute loss - can be overridden by subclasses"""
        if self.criterion is None:
            raise ValueError("Criterion must be provided or compute_loss must be implemented")
        return self.criterion(outputs, targets)
    
    def backward_pass(self, loss: torch.Tensor):
        """Perform backward pass with optional mixed precision"""
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(loss).backward()
            if self.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
        else:
            loss.backward()
    
    def optimizer_step(self):
        """Perform optimizer step with optional mixed precision"""
        if self.use_amp and self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
    
    def clip_gradients(self):
        """Clip gradients if specified"""
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)