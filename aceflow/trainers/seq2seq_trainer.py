import torch
import torch.nn as nn
from typing import Dict, Optional
from .base_trainer import BaseTrainer, TQDM_AVAILABLE

class Seq2SeqTrainer(BaseTrainer):
    """
    Specialized trainer for Sequence-to-Sequence models
    """
    
    def __init__(self, 
                 model: nn.Module,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 criterion: Optional[nn.Module] = None,
                 learning_rate: float = 0.001,
                 device: str = 'auto',
                 early_stopping_patience: Optional[int] = None,
                 early_stopping_min_delta: float = 0.001,
                 gradient_clip: float = 1.0,
                 use_amp: bool = False,
                 teacher_forcing_ratio: float = 0.5):
        
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            learning_rate=learning_rate,
            device=device,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            gradient_clip=gradient_clip,
            use_amp=use_amp
        )
        
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.criterion = criterion or nn.CrossEntropyLoss(ignore_index=0)
    
    def train_epoch(self, dataloader, desc: str = "Training") -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        
        # Use tqdm for batch progress if available
        if TQDM_AVAILABLE:
            from tqdm import tqdm
            batch_iterator = tqdm(dataloader, desc=desc, leave=False)
        else:
            batch_iterator = dataloader
        
        for batch in batch_iterator:
            # Move data to device
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with optional mixed precision
            if self.use_amp:
                with torch.amp.autocast(device_type=str(self.device)):
                    outputs = self._forward_pass(src, tgt, self.teacher_forcing_ratio)
                    loss, correct, tokens = self._compute_metrics(outputs, tgt)
            else:
                outputs = self._forward_pass(src, tgt, self.teacher_forcing_ratio)
                loss, correct, tokens = self._compute_metrics(outputs, tgt)
            
            # Backward pass
            self.backward_pass(loss)
            self.clip_gradients()
            self.optimizer_step()
            
            # Update metrics
            total_loss += loss.item()
            total_correct += correct
            total_tokens += tokens
            
            # Update batch progress bar
            if TQDM_AVAILABLE:
                batch_iterator.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{(correct / tokens if tokens > 0 else 0):.4f}'
                })
            
            self.global_step += 1
        
        # Close batch progress bar
        if TQDM_AVAILABLE:
            batch_iterator.close()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def validate_epoch(self, dataloader, desc: str = "Validation") -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        
        # Use tqdm for batch progress if available
        if TQDM_AVAILABLE:
            from tqdm import tqdm
            batch_iterator = tqdm(dataloader, desc=desc, leave=False)
        else:
            batch_iterator = dataloader
        
        with torch.no_grad():
            for batch in batch_iterator:
                src = batch['src'].to(self.device)
                tgt = batch['tgt'].to(self.device)
                
                if self.use_amp:
                    with torch.amp.autocast(device_type=str(self.device)):
                        outputs = self._forward_pass(src, tgt, teacher_forcing_ratio=0)
                        loss, correct, tokens = self._compute_metrics(outputs, tgt)
                else:
                    outputs = self._forward_pass(src, tgt, teacher_forcing_ratio=0)
                    loss, correct, tokens = self._compute_metrics(outputs, tgt)
                
                total_loss += loss.item()
                total_correct += correct
                total_tokens += tokens
                
                # Update batch progress bar
                if TQDM_AVAILABLE:
                    batch_iterator.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{(correct / tokens if tokens > 0 else 0):.4f}'
                    })
        
        # Close batch progress bar
        if TQDM_AVAILABLE:
            batch_iterator.close()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def _forward_pass(self, src, tgt, teacher_forcing_ratio: float):
        """Perform forward pass through the model"""
        if hasattr(self.model, 'use_attention') and self.model.use_attention:
            return self.model(src, tgt, teacher_forcing_ratio)
        else:
            return self.model(src, tgt, teacher_forcing_ratio)
    
    def _compute_metrics(self, outputs, targets):
        """Compute loss and accuracy metrics"""
        if isinstance(outputs, tuple):
            # Handle attention models that return (outputs, attention_weights)
            outputs = outputs[0]
        
        # Prepare sequences for loss computation
        outputs = outputs[:, :-1].contiguous()
        targets = targets[:, 1:].contiguous()
        
        # Compute loss
        loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        
        # Compute accuracy
        _, predicted = outputs.max(2)
        mask = targets != 0  # Ignore padding
        correct = ((predicted == targets) & mask).sum().item()
        tokens = mask.sum().item()
        
        return loss, correct, tokens
    
    def set_teacher_forcing_ratio(self, ratio: float):
        """Set teacher forcing ratio"""
        self.teacher_forcing_ratio = max(0.0, min(1.0, ratio))