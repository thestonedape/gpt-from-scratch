import torch
import torch.nn as nn
from typing import List, Optional
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from model.decoder import GPTDecoder
from model.config import ModelConfig
from model.kv_cache import KVCache
from inference.sampling import sample


@torch.no_grad()
def generate_with_kv_cache(
    model: GPTDecoder,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
    eos_token_id: int = None,
) -> torch.Tensor:
    model.eval()
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    batch_size = input_ids.shape[0]
    
    generated = input_ids
    
    for step in range(max_new_tokens):
        if step == 0:
            context = input_ids
        else:
            context = next_token.unsqueeze(1)
        
        logits, _ = model(context)
        next_token_logits = logits[:, -1, :]
        
        next_token = sample(next_token_logits, temperature, top_k, top_p)
        
        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
        
        if eos_token_id is not None and (next_token == eos_token_id).any():
            break
    
    return generated
