import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


def get_optimizer(model, learning_rate: float, weight_decay: float, betas: tuple = (0.9, 0.95)):
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)
    
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    optimizer = AdamW(optim_groups, lr=learning_rate, betas=betas)
    return optimizer


def get_scheduler(optimizer, num_training_steps: int, num_warmup_steps: int):
    scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps - num_warmup_steps)
    return scheduler
