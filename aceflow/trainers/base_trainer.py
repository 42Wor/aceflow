import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer
import os
import json
import numpy as np
import time
from typing import Dict, List, Optional
from .callback import Callback, CallbackHandler
from .metrics import MetricTracker

# Check for TQDM
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
    def colored(text, color=None, attrs=None): return text

class BaseTrainer:
    """
    Base trainer class with simplified, clean table output.
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
        
        # Mixed precision
        self.use_amp = use_amp and torch.cuda.is_available()
        if self.use_amp:
            try:
                self.scaler = torch.amp.GradScaler('cuda')
            except (AttributeError, RuntimeError):
                self.use_amp = False
                self.scaler = None
        
        # State
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
        
        self.callbacks = CallbackHandler()
    
    def _setup_device(self, device: str) -> torch.device:
        if device == 'auto':
            if torch.cuda.is_available(): return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): return torch.device('mps')
            return torch.device('cpu')
        return torch.device(device)
    
    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------
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
            
    def get_learning_rate(self) -> float:
        return self.optimizer.param_groups[0]['lr']
    
    def set_learning_rate(self, lr: float):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    # -------------------------------------------------------------------------
    # History & Saving Methods (Restored)
    # -------------------------------------------------------------------------
    def save_training_history(self, filepath: str):
        """Save training history to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"✓ Training history saved to {filepath}")
    
    def load_training_history(self, filepath: str):
        """Load training history from JSON file"""
        with open(filepath, 'r') as f:
            self.history = json.load(f)
    
    def get_best_epoch(self) -> int:
        """Return the epoch with the best validation loss"""
        if not self.history['val_loss']:
            return -1
        return np.argmin(self.history['val_loss'])

    # -------------------------------------------------------------------------
    # Output Formatting
    # -------------------------------------------------------------------------
    def print_table_header(self):
        print("\n" + "-"*75)
        print(f"{'Epoch':^9} | {'Trn Loss':^10} | {'Val Loss':^10} | {'Val Acc':^8} | {'Time':^6} | {'Status'}")
        print("-"*75)
    
    def _print_table_row(self, epoch, total_epochs, metrics, duration, status):
        epoch_str = f"{epoch+1}/{total_epochs}"
        t_loss = f"{metrics.get('train_loss', 0):.4f}"
        v_loss = f"{metrics.get('val_loss', 0):.4f}"
        v_acc = f"{metrics.get('val_accuracy', 0):.4f}"
        
        if duration > 60:
            time_str = f"{int(duration // 60)}m {int(duration % 60)}s"
        else:
            time_str = f"{int(duration)}s"

        if status == "Best":
            status_str = colored("★ Best", 'green', attrs=['bold'])
        elif status == "Stop":
            status_str = colored("! Stop", 'red', attrs=['bold'])
        else:
            status_str = ""

        print(f"{epoch_str:^9} | {t_loss:^10} | {v_loss:^10} | {v_acc:^8} | {time_str:^6} | {status_str}")

    def check_early_stopping(self, val_loss: float) -> tuple:
        if self.early_stopping_patience is None:
            return False, ""
        
        if self.best_val_loss - val_loss > self.early_stopping_min_delta:
            self.best_val_loss = val_loss
            self.early_stopping_counter = 0
            return False, "Best"
        
        self.early_stopping_counter += 1
        if self.early_stopping_counter >= self.early_stopping_patience:
            self.early_stop = True
            return True, "Stop"
        return False, ""
    
    # -------------------------------------------------------------------------
    # Main Train Loop
    # -------------------------------------------------------------------------
    def train(self, 
              train_loader, 
              val_loader, 
              epochs: int = 10, 
              save_path: Optional[str] = None,
              eval_every: int = 1,
              callbacks: Optional[List[Callback]] = None) -> Dict:
        
        # Register Callbacks
        if callbacks:
            for callback in callbacks:
                self.callbacks.add_callback(callback)
        
        self.callbacks.on_train_begin()
        print(f"Device: {self.device} | Model Params: {sum(p.numel() for p in self.model.parameters()):,}")
        
        self.print_table_header()

        for epoch in range(epochs):
            self.epoch = epoch
            start_time = time.time()
            
            # --- Training ---
            self.callbacks.on_epoch_begin(epoch)
            train_metrics = self.train_epoch(train_loader)
            
            # --- Validation ---
            if (epoch + 1) % eval_every == 0:
                val_metrics = self.validate_epoch(val_loader)
                
                val_loss = val_metrics.get('loss', 0)
                status = ""
                should_stop = False
                
                # Check for best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    status = "Best"
                    if save_path:
                        self._save_model(save_path, suffix="_best")
                
                # Check early stopping
                if self.early_stopping_patience:
                    should_stop, es_status = self.check_early_stopping(val_loss)
                    if es_status: status = es_status

                # Update history
                self._update_history(train_metrics, val_metrics)
            else:
                val_metrics = {}
                status = ""
                should_stop = False
            
            # Calculate duration
            duration = time.time() - start_time
            self.history['epoch_times'].append(duration)
            
            # Combine metrics for display
            display_metrics = {
                'train_loss': train_metrics.get('loss', 0),
                'val_loss': val_metrics.get('loss', 0),
                'val_accuracy': val_metrics.get('accuracy', 0)
            }
            
            # --- Print Row ---
            self._print_table_row(epoch, epochs, display_metrics, duration, status)
            
            self.callbacks.on_epoch_end(epoch, train_metrics, val_metrics)
            
            if should_stop:
                print(f"\nEarly stopping triggered. Best Loss: {self.best_val_loss:.4f}")
                break
        
        # Save final
        if save_path and not self.early_stop:
            self._save_model(save_path, suffix="_final")
            
        print("-" * 75 + "\n")
        self.callbacks.on_train_end()
        return self.history

    def _save_model(self, path, suffix=""):
        p = path.replace('.ace', f'{suffix}.ace') if path.endswith('.ace') else path + f'{suffix}.ace'
        if hasattr(self.model, 'save'):
            self.model.save(p)
        else:
            torch.save(self.model.state_dict(), p)

    def _update_history(self, t_metrics, v_metrics):
        for k, v in t_metrics.items(): self.history.setdefault(f'train_{k}', []).append(v)
        for k, v in v_metrics.items(): self.history.setdefault(f'val_{k}', []).append(v)
        self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

    # Abstract methods
    def train_epoch(self, dataloader) -> Dict[str, float]:
        raise NotImplementedError("Implement train_epoch")
    
    def validate_epoch(self, dataloader) -> Dict[str, float]:
        raise NotImplementedError("Implement validate_epoch")