import torch
from tokenizers import Tokenizer
from model.config import get_tiny_config
from model.decoder import GPTDecoder
from training.dataset import get_dataloader
from training.optimizer import get_optimizer
from training.checkpoint import save_checkpoint
from inference.generate import generate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

config = get_tiny_config()
model = GPTDecoder(config).to(device)

print("Training for 1 epoch to test pipeline...")
train_loader = get_dataloader("data/processed/train.bin", config.max_seq_len, batch_size=16, num_workers=0)
optimizer = get_optimizer(model, 3e-4, 0.1)

model.train()
for i, (input_ids, labels) in enumerate(train_loader):
    if i >= 10:  # Just 10 batches for testing
        break
    input_ids, labels = input_ids.to(device), labels.to(device)
    logits, loss = model(input_ids, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Batch {i+1}/10 - Loss: {loss.item():.4f}")

print("\nSaving checkpoint...")
save_checkpoint(model, optimizer, None, 0, 10, loss.item(), "checkpoints", "test_checkpoint.pt")

print("\nLoading tokenizer...")
tokenizer = Tokenizer.from_file("data/tokenizer/tokenizer.json")

print("\nGenerating text...")
model.eval()
prompt = "Once upon a time"
encoded = tokenizer.encode(prompt)
input_ids = torch.tensor([encoded.ids], dtype=torch.long)

output_ids = generate(
    model=model,
    input_ids=input_ids,
    max_new_tokens=50,
    temperature=0.8,
)

output_text = tokenizer.decode(output_ids[0].tolist())
print("\n" + "="*50)
print("Generated text:")
print("="*50)
print(output_text)
print("="*50)
print("\n✓ Pipeline test successful!")
