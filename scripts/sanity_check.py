import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from model.config import get_tiny_config
from model.decoder import GPTDecoder


def sanity_check():
    print("Running sanity check...")
    
    config = get_tiny_config()
    print(f"\nConfig: {config}")
    
    model = GPTDecoder(config)
    print(f"\nModel parameters: {model.get_num_params():,}")
    
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print(f"\nInput shape: {input_ids.shape}")
    
    model.eval()
    with torch.no_grad():
        logits, loss = model(input_ids, labels)
    
    print(f"Output logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    expected_shape = (batch_size, seq_len, config.vocab_size)
    assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"
    
    print("\n✓ Sanity check passed!")


if __name__ == "__main__":
    sanity_check()
