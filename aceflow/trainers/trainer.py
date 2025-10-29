import torch
import torch.nn as nn
from torch.optim import Adam
import os
from tqdm import tqdm
from termcolor import colored
import json

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
        
        # Table headers
        self.table_headers = ["Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc", "Status"]
    
    def train_epoch(self, dataloader, teacher_forcing_ratio=0.5):
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        
        for batch in dataloader:
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if hasattr(self.model, 'use_attention') and self.model.use_attention:
                output, _ = self.model(src, tgt, teacher_forcing_ratio)
            else:
                output = self.model(src, tgt, teacher_forcing_ratio)
            
            output = output[:, :-1].contiguous()
            tgt = tgt[:, 1:].contiguous()
            loss = self.criterion(output.view(-1, output.size(-1)), tgt.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
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
            for batch in dataloader:
                src = batch['src'].to(self.device)
                tgt = batch['tgt'].to(self.device)
                
                if hasattr(self.model, 'use_attention') and self.model.use_attention:
                    output, _ = self.model(src, tgt, teacher_forcing_ratio=0)
                else:
                    output = self.model(src, tgt, teacher_forcing_ratio=0)
                
                output = output[:, :-1].contiguous()
                tgt = tgt[:, 1:].contiguous()
                loss = self.criterion(output.view(-1, output.size(-1)), tgt.view(-1))
                
                _, predicted = output.max(2)
                mask = tgt != 0
                correct = (predicted == tgt) & mask
                total_correct += correct.sum().item()
                total_tokens += mask.sum().item()
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        
        return avg_loss, accuracy
    
    def print_table_header(self):
        """Print the table header"""
        headers = self.table_headers
        header_str = (f"| {headers[0]:<8} | {headers[1]:<10} | {headers[2]:<9} | "
                      f"{headers[3]:<8} | {headers[4]:<7} | {headers[5]:<12} |")
        separator = "|-" + "-|-".join(["-"*8, "-"*10, "-"*9, "-"*8, "-"*7, "-"*12]) + "-|"
        
        print(colored("\n" + "="*80, 'cyan'))
        print(colored(" " * 29 + "EPOCH SUMMARY TABLE", 'cyan', attrs=['bold']))
        print(colored("="*80, 'cyan'))
        print(colored(header_str, 'white', attrs=['bold']))
        print(colored(separator, 'white'))
    
    def _get_table_row_str(self, epoch, total_epochs, train_loss, train_acc, val_loss, val_acc, status):
        """Return a formatted table row string with colors"""
        epoch_str = f"{epoch+1}/{total_epochs}"
        train_loss_str = f"{train_loss:.4f}"
        train_acc_str = f"{train_acc:.4f}"
        val_loss_str = f"{val_loss:.4f}"
        val_acc_str = f"{val_acc:.4f}"

        # Pad the plain strings first
        epoch_padded = f"{epoch_str:<8}"
        train_loss_padded = f"{train_loss_str:<10}"
        train_acc_padded = f"{train_acc_str:<9}"
        val_loss_padded = f"{val_loss_str:<8}"
        val_acc_padded = f"{val_acc_str:<7}"
        
        # Apply colors after padding
        epoch_colored = colored(epoch_padded, 'white', attrs=['bold'])
        train_loss_colored = colored(train_loss_padded, 'yellow')
        train_acc_colored = colored(train_acc_padded, 'green')
        val_loss_colored = colored(val_loss_padded, 'yellow')
        val_acc_colored = colored(val_acc_padded, 'green')

        if status == "Saved Best":
            status_colored = colored(f"{status:<12}", 'white', 'on_green', attrs=['bold'])
        elif status == "Final":
            status_colored = colored(f"{status:<12}", 'white', 'on_blue', attrs=['bold'])
        else:
            status_colored = f"{status:<12}"

        return (f"| {epoch_colored} | {train_loss_colored} | {train_acc_colored} | "
                f"{val_loss_colored} | {val_acc_colored} | {status_colored} |")

    def train(self, train_loader, val_loader, epochs=10, save_path=None, 
              teacher_forcing_ratio=0.5, eval_every=1):
        
        print(colored(f"Starting training on {self.device}", 'green', attrs=['bold']))
        print(colored(f"Model has {sum(p.numel() for p in self.model.parameters()):,} parameters", 'green'))
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

        self.print_table_header()

        with tqdm(total=epochs,
                  desc=colored("Total Training Progress", 'yellow', attrs=['bold']),
                  bar_format="{l_bar}%s{bar}%s{r_bar}" % (colored('|', 'yellow'), colored('|', 'yellow')),
                  colour='green',
                  leave=True) as pbar:
            
            for epoch in range(epochs):
                train_loss, train_acc = self.train_epoch(train_loader, teacher_forcing_ratio)
                self.history['train_loss'].append(train_loss)
                self.history['train_accuracy'].append(train_acc)
                
                if (epoch + 1) % eval_every == 0:
                    val_loss, val_acc = self.validate_epoch(val_loader)
                    self.history['val_loss'].append(val_loss)
                    self.history['val_accuracy'].append(val_acc)
                else:
                    val_loss = self.history['val_loss'][-1] if self.history['val_loss'] else float('nan')
                    val_acc = self.history['val_accuracy'][-1] if self.history['val_accuracy'] else float('nan')
                
                status = "-"
                is_best = False
                if save_path and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    status = "Saved Best"
                    is_best = True
                
                if epoch == epochs - 1:
                    status = "Final"

                # **FIX**: Use tqdm.write to print the row without breaking the bar
                row_str = self._get_table_row_str(
                    epoch, epochs, train_loss, train_acc, val_loss, val_acc, status
                )
                tqdm.write(row_str)

                pbar.set_postfix({
                    'train_loss': f'{train_loss:.4f}',
                    'val_acc': f'{val_acc:.4f}'
                })
                pbar.update(1)
                
                if save_path:
                    if is_best:
                        best_save_path = save_path.replace('.ace', '_best.ace') if save_path.endswith('.ace') else save_path + '_best.ace'
                        if hasattr(self.model, 'save'): self.model.save(best_save_path)
                    
                    latest_path = save_path.replace('.ace', '_latest.ace') if save_path.endswith('.ace') else save_path + '_latest.ace'
                    if hasattr(self.model, 'save'): self.model.save(latest_path)
        
        if save_path:
            final_path = save_path.replace('.ace', '_final.ace') if save_path.endswith('.ace') else save_path + '_final.ace'
            if hasattr(self.model, 'save'): 
                self.model.save(final_path)
                print(colored(f"\nFinal model saved to {final_path}", 'green', attrs=['bold']))
        
        return self.history
    
    def save_training_history(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load_training_history(self, filepath):
        with open(filepath, 'r') as f:
            self.history = json.load(f)