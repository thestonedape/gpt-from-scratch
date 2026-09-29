import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent))

from model.decoder import GPTDecoder
from model.config import ModelConfig
from training.dataset import get_dataloader
from training.optimizer import get_optimizer, get_scheduler
from training.train_step import train_step, eval_step
from training.checkpoint import save_checkpoint, load_checkpoint


class Trainer:
    def __init__(
        self,
        model: GPTDecoder,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer,
        scheduler,
        device: torch.device,
        checkpoint_dir: str = "checkpoints",
        log_interval: int = 100,
        eval_interval: int = 1000,
        save_interval: int = 5000,
        max_grad_norm: float = 1.0,
        config=None,
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.max_grad_norm = max_grad_norm
        self.config = config
        
        self.current_epoch = 0
        self.global_step = 0
        
    def train(self, num_epochs: int):
        self.model.train()
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_loss = 0.0
            
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch_idx, batch in enumerate(pbar):
                loss_val, loss = train_step(self.model, batch, self.device)
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                if self.scheduler:
                    self.scheduler.step()
                
                epoch_loss += loss_val
                self.global_step += 1
                
                if self.global_step % self.log_interval == 0:
                    avg_loss = epoch_loss / (batch_idx + 1)
                    lr = self.optimizer.param_groups[0]['lr']
                    pbar.set_postfix({'loss': f'{loss_val:.4f}', 'avg_loss': f'{avg_loss:.4f}', 'lr': f'{lr:.2e}'})
                
                if self.global_step % self.eval_interval == 0:
                    val_loss = self.evaluate()
                    print(f"\nStep {self.global_step} - Validation Loss: {val_loss:.4f}")
                    self.model.train()
                
                if self.global_step % self.save_interval == 0:
                    save_checkpoint(
                        self.model, self.optimizer, self.scheduler,
                        epoch, self.global_step, loss_val, self.checkpoint_dir,
                        config=self.config
                    )
            
            avg_epoch_loss = epoch_loss / len(self.train_dataloader)
            print(f"\nEpoch {epoch+1} completed - Average Loss: {avg_epoch_loss:.4f}")
            
            save_checkpoint(
                self.model, self.optimizer, self.scheduler,
                epoch, self.global_step, avg_epoch_loss, self.checkpoint_dir,
                filename=f"epoch_{epoch+1}.pt",
                config=self.config
            )
            print(f"Saved checkpoint: epoch_{epoch+1}.pt")
    
    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                loss = eval_step(self.model, batch, self.device)
                total_loss += loss
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def load_checkpoint(self, checkpoint_path: str):
        epoch, step, loss = load_checkpoint(
            checkpoint_path, self.model, self.optimizer, self.scheduler
        )
        self.current_epoch = epoch
        self.global_step = step
        print(f"Loaded checkpoint from epoch {epoch}, step {step}, loss {loss:.4f}")
