import torch
import torch.nn as nn
from typing import Tuple, Optional


class KVCache:
    def __init__(self, batch_size: int, max_seq_len: int, n_heads: int, head_dim: int, device: torch.device):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.device = device
        
        self.k_cache = torch.zeros(
            batch_size, n_heads, max_seq_len, head_dim, device=device
        )
        self.v_cache = torch.zeros(
            batch_size, n_heads, max_seq_len, head_dim, device=device
        )
        self.current_len = 0
    
    def update(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, n_heads, seq_len, head_dim = k.shape
        
        self.k_cache[:, :, self.current_len:self.current_len + seq_len, :] = k
        self.v_cache[:, :, self.current_len:self.current_len + seq_len, :] = v
        
        self.current_len += seq_len
        
        k_full = self.k_cache[:, :, :self.current_len, :]
        v_full = self.v_cache[:, :, :self.current_len, :]
        
        return k_full, v_full
    
    def reset(self):
        self.current_len = 0
        self.k_cache.zero_()
        self.v_cache.zero_()
