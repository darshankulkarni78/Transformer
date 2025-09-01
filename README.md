# Seq2Seq Transformer Text Generation

This repository contains a PyTorch implementation of a sequence-to-sequence transformer model for text generation trained on the WikiText-2 dataset using the GPT-2 tokenizer.

## Features

- Custom Transformer encoder-decoder architecture
- Training with teacher forcing and label smoothing
- Autoregressive text generation with temperature and top-k sampling
- Repetition penalty to reduce token repetitions

## Setup

Install dependencies:

```
pip install torch transformers datasets
```

## Usage

### Training

Train the model with:

```
train_loop(model, train_loader, val_loader, epochs=10, lr=3e-4, device=device)
```

### Generation

Generate text from a prompt:

```
prompt = "In the middle of the desert, a lone traveler found"
output = generate_text(model, tokenizer, prompt, max_len=100, temperature=1.5, top_k=50, device=device)
print(output)
```

## Notes

- Tune temperature (0.7–2.0) and top-k sampling for best output quality.
- Use GPU if available for faster training and generation.
