import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import time
import os
from tqdm import tqdm
from termcolor import cprint
class Trainer:
    def __init__(self, model, learning_rate=0.001, device='auto'):
        self.model = model
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Training components
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
        
        # Track best validation loss
        self.best_val_loss = float('inf')
    
    def train_epoch(self, dataloader, teacher_forcing_ratio=0.5):
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.model.use_attention:
                output, _ = self.model(src, tgt, teacher_forcing_ratio)
            else:
                output = self.model(src, tgt, teacher_forcing_ratio)
            
            # Calculate loss
            output = output[:, :-1].contiguous()
            tgt = tgt[:, 1:].contiguous()
            loss = self.criterion(output.view(-1, output.size(-1)), tgt.view(-1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Calculate accuracy
            _, predicted = output.max(2)
            mask = tgt != 0
            correct = (predicted == tgt) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        
        return avg_loss, accuracy
    
    def validate_epoch(self, dataloader):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                src = batch['src'].to(self.device)
                tgt = batch['tgt'].to(self.device)
                
                # Forward pass
                if self.model.use_attention:
                    output, _ = self.model(src, tgt, teacher_forcing_ratio=0)
                else:
                    output = self.model(src, tgt, teacher_forcing_ratio=0)
                
                # Calculate loss
                output = output[:, :-1].contiguous()
                tgt = tgt[:, 1:].contiguous()
                loss = self.criterion(output.view(-1, output.size(-1)), tgt.view(-1))
                
                # Calculate accuracy
                _, predicted = output.max(2)
                mask = tgt != 0
                correct = (predicted == tgt) & mask
                total_correct += correct.sum().item()
                total_tokens += mask.sum().item()
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs=10, save_path=None, 
            teacher_forcing_ratio=0.5, eval_every=1):
        
        print(f"Starting training on {self.device}")
        print(f"Model has {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
        # Create directory for save_path if it doesn't exist
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, teacher_forcing_ratio)
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_acc)
            
            # Validation
            if (epoch + 1) % eval_every == 0:
                val_loss, val_acc = self.validate_epoch(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_acc)
                
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                
                # Save best model
                if save_path and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_save_path = save_path.replace('.ace', '_best.ace') if save_path.endswith('.ace') else save_path + '_best.ace'
                    self.model.save(best_save_path)
                    # ALSO SAVE THE BASE MODEL
                    self.model.save(save_path)
                    print(f"Best model saved to {best_save_path}")
                    print(f"Base model saved to {save_path}")
            else:
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            
            # Save regular checkpoint
            if save_path:
                # Save with epoch number
                checkpoint_path = save_path.replace('.ace', f'_epoch_{epoch+1}.ace') if save_path.endswith('.ace') else save_path + f'_epoch_{epoch+1}.ace'
                cprint(f"{checkpoint_path}", 'white', 'on_red', attrs=['bold'])
                self.model.save(checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
                
                # Also save latest model
                latest_path = save_path.replace('.ace', '_latest.ace') if save_path.endswith('.ace') else save_path + '_latest.ace'
                cprint(f"{latest_path}", 'white', 'on_red', attrs=['bold'])
                self.model.save(latest_path)
        
        # Save final model
        if save_path:
            final_path = save_path.replace('.ace', '_final.ace') if save_path.endswith('.ace') else save_path + '_final.ace'
            self.model.save(final_path)
            print(f"Final model saved to {final_path}")
            return self.history
        
    def save_training_history(self, filepath):
        """Save training history to JSON file"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load_training_history(self, filepath):
        """Load training history from JSON file"""
        import json
        with open(filepath, 'r') as f:
            self.history = json.load(f)