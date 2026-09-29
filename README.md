# GPT from Scratch

A compact PyTorch implementation of a GPT-style decoder-only Transformer.
Includes causal self-attention, decoder blocks, BPE tokenizer training, dataset
preprocessing, checkpointed training, and text generation with sampling and
optional KV caching. Model presets range from micro to medium.

This repository publishes source code and configuration. Local training corpora,
processed datasets, tokenizers, and trained checkpoints are not included. No model
quality benchmark is claimed.

## Setup

Python 3.10 or later is recommended.

```sh
python -m venv .venv
# Activate the environment for your shell, then:
pip install -r requirements.txt
python scripts/sanity_check.py
```

## Prepare your data

Use a UTF-8 text corpus you are permitted to train on. Create `data/raw` and
`data/tokenizer`, and place your text at `data/raw/corpus.txt`.

```sh
python tokenizer/train_tokenizer.py --input data/raw/corpus.txt --output data/tokenizer/tokenizer.json --vocab_size 50257
python scripts/preprocess.py --input data/raw/corpus.txt --output_dir data/processed --tokenizer data/tokenizer/tokenizer.json
```

Keep the tokenizer vocabulary compatible with the selected model preset in
`model/config.py`. Preprocessing uses a sequential 90/10 text split by default.

## Train and generate

```sh
python train.py --model_size tiny --train_data data/processed/train.bin --val_data data/processed/val.bin --num_epochs 5
python generate_text.py --checkpoint checkpoints/best.pt --tokenizer data/tokenizer/tokenizer.json --prompt "Once upon a time"
```

Replace the checkpoint path with the actual file produced by your run. Generation
loads a serialized PyTorch checkpoint; use only checkpoints you trust.

## Structure

- `model/`: configuration, embeddings, attention, decoder, and KV cache
- `training/`: datasets, optimizer, trainer, and checkpoint helpers
- `inference/`: sampling and generation
- `tokenizer/`: BPE tokenizer training
- `scripts/`: data download, preprocessing, and model sanity check
- `configs/`: model configuration examples

`test_pipeline.py` is a local training smoke script that needs prepared data and
writes a checkpoint. It is not an isolated unit-test suite.
