from dataclasses import dataclass
import json


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_ff: int = 3072
    max_seq_len: int = 1024
    dropout: float = 0.1
    attn_dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    use_bias: bool = True
    
    def __post_init__(self):
        assert self.d_model % self.n_heads == 0
        assert self.d_ff > 0 and self.n_layers > 0
        assert 0 <= self.dropout <= 1 and 0 <= self.attn_dropout <= 1
    
    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads
    
    @classmethod
    def from_json(cls, json_path: str) -> 'ModelConfig':
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_json(self, json_path: str):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)

def get_micro_config() -> ModelConfig:
    """~1M parameters - optimal for 17-20M tokens (Chinchilla ratio)"""
    return ModelConfig(
        vocab_size=16000, d_model=32, n_layers=2, n_heads=2, d_ff=128, max_seq_len=256
    )


def get_tiny_config() -> ModelConfig:
    return ModelConfig(
        d_model=128, n_layers=2, n_heads=2, d_ff=512, max_seq_len=256
    )


def get_small_config() -> ModelConfig:
    return ModelConfig(
        d_model=512, n_layers=8, n_heads=8, d_ff=2048
    )


def get_medium_config() -> ModelConfig:
    return ModelConfig(
        d_model=768, n_layers=12, n_heads=12, d_ff=3072
    )


def get_config_by_name(name: str) -> ModelConfig:
    configs = {
        'micro': get_micro_config,
        'tiny': get_tiny_config,
        'small': get_small_config,
        'medium': get_medium_config
    }
    if name not in configs:
        raise ValueError(f"Unknown config: {name}")
    return configs[name]()
