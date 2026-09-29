import torch
import torch.nn.functional as F


def top_k_sampling(logits: torch.Tensor, k: int) -> torch.Tensor:
    top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
    probs = F.softmax(top_k_logits, dim=-1)
    next_token_idx = torch.multinomial(probs, num_samples=1)
    next_token = torch.gather(top_k_indices, -1, next_token_idx)
    return next_token.squeeze(-1)


def top_p_sampling(logits: torch.Tensor, p: float) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(probs, dim=-1)
    
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    sorted_logits[sorted_indices_to_remove] = float('-inf')
    probs = F.softmax(sorted_logits, dim=-1)
    next_token_idx = torch.multinomial(probs, num_samples=1)
    next_token = torch.gather(sorted_indices, -1, next_token_idx)
    return next_token.squeeze(-1)


def temperature_sampling(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token.squeeze(-1)


def greedy_sampling(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=-1)


def sample(logits: torch.Tensor, temperature: float = 1.0, top_k: int = None, top_p: float = None) -> torch.Tensor:
    if temperature == 0:
        return greedy_sampling(logits)
    
    logits = logits / temperature
    
    if top_k is not None:
        return top_k_sampling(logits, top_k)
    
    if top_p is not None:
        return top_p_sampling(logits, top_p)
    
    return temperature_sampling(logits, 1.0)
