from typing import Dict, List, Any, Optional
import torch
import os
import json

class Callback:
    """Base callback class"""
    
    def on_train_begin(self, trainer=None):
        pass
    
    def on_train_end(self, trainer=None):
        pass
    
    def on_epoch_begin(self, epoch, trainer=None):
        pass
    
    def on_epoch_end(self, epoch, train_metrics, val_metrics, trainer=None):
        pass
    
    def on_batch_begin(self, batch, trainer=None):
        pass
    
    def on_batch_end(self, batch, metrics, trainer=None):
        pass

class CallbackHandler:
    """Manages multiple callbacks"""
    
    def __init__(self):
        self.callbacks = []
    
    def add_callback(self, callback: Callback):
        self.callbacks.append(callback)
    
    def on_train_begin(self, trainer=None):
        for callback in self.callbacks:
            callback.on_train_begin(trainer)
    
    def on_train_end(self, trainer=None):
        for callback in self.callbacks:
            callback.on_train_end(trainer)
    
    def on_epoch_begin(self, epoch, trainer=None):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, trainer)
    
    def on_epoch_end(self, epoch, train_metrics, val_metrics, trainer=None):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, train_metrics, val_metrics, trainer)
    
    def on_batch_begin(self, batch, trainer=None):
        for callback in self.callbacks:
            callback.on_batch_begin(batch, trainer)
    
    def on_batch_end(self, batch, metrics, trainer=None):
        for callback in self.callbacks:
            callback.on_batch_end(batch, metrics, trainer)

class ModelCheckpoint(Callback):
    """Save model checkpoints"""
    
    def __init__(self, filepath, monitor='val_loss', save_best_only=True, mode='min'):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    def on_epoch_end(self, epoch, train_metrics, val_metrics, trainer=None):
        if not trainer or not hasattr(trainer.model, 'save'):
            return
        
        current_value = val_metrics.get(self.monitor, float('inf'))
        
        if self.mode == 'min':
            is_best = current_value < self.best_value
        else:
            is_best = current_value > self.best_value
        
        if is_best:
            self.best_value = current_value
            
            if self.save_best_only:
                # Save as best model
                trainer.model.save(self.filepath)
            else:
                # Save with epoch number
                epoch_path = self.filepath.replace('.ace', f'_epoch_{epoch+1}.ace')
                trainer.model.save(epoch_path)

class LearningRateScheduler(Callback):
    """Learning rate scheduler callback"""
    
    def __init__(self, scheduler):
        self.scheduler = scheduler
    
    def on_epoch_end(self, epoch, train_metrics, val_metrics, trainer=None):
        if self.scheduler:
            self.scheduler.step()

class EarlyStopping(Callback):
    """Early stopping callback"""
    
    def __init__(self, patience=10, min_delta=0, monitor='val_loss', mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.counter = 0
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.early_stop = False
    
    def on_epoch_end(self, epoch, train_metrics, val_metrics, trainer=None):
        if not trainer:
            return
        
        current_value = val_metrics.get(self.monitor, float('inf'))
        
        if self.mode == 'min':
            improvement = self.best_value - current_value
        else:
            improvement = current_value - self.best_value
        
        if improvement > self.min_delta:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                trainer.early_stop = True
                print(f"Early stopping triggered after {epoch + 1} epochs")

class ProgressLogger(Callback):
    """Log training progress"""
    
    def on_epoch_end(self, epoch, train_metrics, val_metrics, trainer=None):
        if trainer:
            print(f"Epoch {epoch + 1}: "
                  f"Train Loss: {train_metrics.get('loss', 0):.4f}, "
                  f"Val Loss: {val_metrics.get('loss', 0):.4f}, "
                  f"LR: {trainer.get_learning_rate():.2e}")