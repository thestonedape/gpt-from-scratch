import torch
import argparse
from pathlib import Path

from model.config import get_config_by_name, ModelConfig
from model.decoder import GPTDecoder
from training.dataset import get_dataloader
from training.optimizer import get_optimizer, get_scheduler
from training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, default="tiny", choices=["micro", "tiny", "small", "medium"])
    parser.add_argument("--train_data", type=str, default="data/processed/train.bin")
    parser.add_argument("--val_data", type=str, default="data/processed/val.bin")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    config = get_config_by_name(args.model_size)
    print(f"\nModel config:\n{config}")
    
    model = GPTDecoder(config)
    print(f"\nModel parameters: {model.get_num_params():,}")
    
    train_loader = get_dataloader(
        args.train_data, 
        config.max_seq_len, 
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_loader = get_dataloader(
        args.val_data,
        config.max_seq_len,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    optimizer = get_optimizer(model, args.learning_rate, args.weight_decay)
    
    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = min(500, total_steps // 10)
    scheduler = get_scheduler(optimizer, total_steps, warmup_steps)
    
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=100,
        eval_interval=1000,
        save_interval=5000,
    )
    
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    print(f"\nStarting training for {args.num_epochs} epochs...")
    trainer.train(args.num_epochs)
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
