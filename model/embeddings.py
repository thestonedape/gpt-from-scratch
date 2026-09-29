import torch
import torch.nn as nn
from .config import ModelConfig


class TokenEmbedding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.d_model = config.d_model
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * (self.d_model ** 0.5)


class PositionalEmbedding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.max_seq_len, config.d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        return self.embedding(positions)


class Embeddings(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.token_embedding = TokenEmbedding(config)
        self.position_embedding = PositionalEmbedding(config)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(x)
        return self.dropout(token_emb + pos_emb)
