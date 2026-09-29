import numpy as np
from pathlib import Path
from tokenizers import Tokenizer
import argparse
from tqdm import tqdm


def preprocess_data(input_path: str, output_dir: str, tokenizer_path: str, split_ratio: float = 0.9):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    print(f"Loading data from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Tokenizing {len(text):,} characters...")
    encoded = tokenizer.encode(text)
    token_ids = np.array(encoded.ids, dtype=np.uint16)
    
    print(f"Total tokens: {len(token_ids):,}")
    
    print("Splitting data...")
    split_idx = int(len(token_ids) * split_ratio)
    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]
    
    train_path = output_dir / "train.bin"
    val_path = output_dir / "val.bin"
    
    print(f"Saving train data ({len(train_ids):,} tokens)...")
    with tqdm(total=len(train_ids), unit='tokens', desc='Writing train') as pbar:
        train_ids.tofile(train_path)
        pbar.update(len(train_ids))
    
    print(f"Saving validation data ({len(val_ids):,} tokens)...")
    with tqdm(total=len(val_ids), unit='tokens', desc='Writing val') as pbar:
        val_ids.tofile(val_path)
        pbar.update(len(val_ids))
    
    print(f"\nDone!")
    print(f"Train: {train_path}")
    print(f"Val: {val_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input text file")
    parser.add_argument("--output_dir", required=True, help="Output directory for processed data")
    parser.add_argument("--tokenizer", required=True, help="Path to trained tokenizer")
    parser.add_argument("--split_ratio", type=float, default=0.9, help="Train/val split ratio")
    args = parser.parse_args()
    
    preprocess_data(args.input, args.output_dir, args.tokenizer, args.split_ratio)
