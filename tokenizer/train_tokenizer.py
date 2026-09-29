import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tqdm import tqdm
import argparse


def train_tokenizer(
    input_files: list,
    output_path: str,
    vocab_size: int = 50257,
    min_frequency: int = 2,
    special_tokens: list = None
):
    if special_tokens is None:
        special_tokens = ["<|endoftext|>", "<|padding|>", "<|unk|>"]
    
    print(f"Initializing BPE tokenizer...")
    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = Whitespace()
    
    print(f"Training tokenizer on {len(input_files)} file(s)...")
    print(f"Target vocab size: {vocab_size:,}")
    
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True
    )
    
    tokenizer.train(files=input_files, trainer=trainer)
    
    print("Setting up post-processor...")
    tokenizer.post_processor = TemplateProcessing(
        single="$A <|endoftext|>",
        special_tokens=[("<|endoftext|>", tokenizer.token_to_id("<|endoftext|>"))]
    )
    
    print(f"Saving tokenizer to {output_path}...")
    tokenizer.save(output_path)
    print(f"✓ Tokenizer saved!")
    print(f"✓ Vocab size: {tokenizer.get_vocab_size():,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True, help="Input text files")
    parser.add_argument("--output", required=True, help="Output tokenizer path")
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--min_frequency", type=int, default=2)
    args = parser.parse_args()
    
    train_tokenizer(args.input, args.output, args.vocab_size, args.min_frequency)
