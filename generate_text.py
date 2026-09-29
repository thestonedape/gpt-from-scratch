import torch
import argparse
from tokenizers import Tokenizer

from model.config import ModelConfig
from model.decoder import GPTDecoder
from inference.generate import generate
from training.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--model_size", type=str, default="tiny", choices=["micro", "tiny", "small", "medium"], help="Model size if config not in checkpoint")
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Loading tokenizer from {args.tokenizer}...")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    
    print(f"Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    if 'config' in checkpoint:
        config = checkpoint['config']
        print(f"Using config from checkpoint")
    else:
        from model.config import get_config_by_name
        config = get_config_by_name(args.model_size)
        print(f"No config in checkpoint, using {args.model_size} config")
    
    model = GPTDecoder(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"\nPrompt: {args.prompt}")
    print("-" * 50)
    
    encoded = tokenizer.encode(args.prompt)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long)
    
    output_ids = generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    
    output_text = tokenizer.decode(output_ids[0].tolist())
    print(output_text)
    print("-" * 50)


if __name__ == "__main__":
    main()
