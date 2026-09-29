import torch
import torch.nn as nn
from typing import Tuple


def train_step(model: nn.Module, batch: Tuple[torch.Tensor, torch.Tensor], device: torch.device) -> Tuple[float, torch.Tensor]:
    input_ids, labels = batch
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    
    logits, loss = model(input_ids, labels)
    
    return loss.item(), loss


def eval_step(model: nn.Module, batch: Tuple[torch.Tensor, torch.Tensor], device: torch.device) -> float:
    input_ids, labels = batch
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    
    with torch.no_grad():
        logits, loss = model(input_ids, labels)
    
    return loss.item()
