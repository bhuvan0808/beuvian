# Training Guide

## Training Pipeline Overview

```
Step 1: Prepare Data       →  corpus.txt (raw text)
Step 2: Train Tokenizer    →  tokenizer.json (BPE vocabulary)
Step 3: Tokenize to Binary →  train.bin + val.bin (fast-loading binary)
Step 4: Train Model        →  ckpt_best.pt (trained weights)
Step 5: Run Inference      →  generated text
Step 6: Deploy API         →  POST /generate endpoint
```

## Step 1: Data Preparation

### Small run (WikiText-103)

```bash
python scripts/prepare_data.py --max_size_mb 50
```

| Flag | Default | Description |
|------|---------|-------------|
| `--output` | `data/processed/corpus.txt` | Output path |
| `--max_size_mb` | `150` | Stop at this file size |
| `--min_length` | `100` | Minimum line length (chars) |

### Production (C4 streaming)

See [scaling.md](scaling.md) for streaming with `prepare_stream.py` and `prepare_parallel.py`.

## Step 2: Train the Tokenizer

```bash
python scripts/train_hf_tokenizer.py --corpus data/processed/corpus.txt --vocab_size 8000
```

| Flag | Default | Description |
|------|---------|-------------|
| `--corpus` | `data/processed/corpus.txt` | Input text |
| `--output` | `tokenizer/tokenizer.json` | Output tokenizer |
| `--vocab_size` | `4000` | BPE vocabulary size |

**Vocab size guidelines:**

| Model Size | Recommended Vocab |
|-----------|------------------|
| < 20M params | 8,000 |
| 20M–125M | 32,000 |
| 125M–1B | 50,000 |
| 1B+ | 64,000–100,000 |

## Step 3: Tokenize to Binary

```bash
python scripts/tokenize_corpus.py \
    --corpus data/processed/corpus.txt \
    --tokenizer tokenizer/tokenizer.json \
    --output_dir data/processed \
    --val_ratio 0.01
```

Output: `train.bin` and `val.bin` — memory-mapped uint16 arrays.

## Step 4: Train the Model

```bash
python training/train.py --config configs/train_125m.yaml --compile --seed 42
```

| Flag | Description |
|------|-------------|
| `--config` | Path to YAML config file |
| `--compile` | Enable torch.compile (1.3–1.8x speedup, PyTorch 2.0+) |
| `--seed` | Random seed for reproducibility (default: 42) |

## Config File Explained

```yaml
model:
  vocab_size: 32000         # Must match tokenizer vocab size
  d_model: 768              # Embedding dimension (width of the model)
  n_layers: 12              # Number of transformer blocks (depth)
  n_heads: 12               # Attention heads (d_model must be divisible by this)
  max_seq_len: 1024         # Context window in tokens
  dropout: 0.0              # 0.0 for pre-training at scale, 0.1 for small runs
  bias: false               # No bias terms (LLaMA/PaLM style)
  gradient_checkpointing: false  # true saves VRAM but ~20% slower

training:
  batch_size: 64            # Sequences per GPU per step
  gradient_accumulation_steps: 2  # Effective batch = batch_size × this
  max_iters: 15000          # Total training steps
  lr: 0.0006                # Peak learning rate
  min_lr: 0.00006           # Minimum LR (10% of peak, after cosine decay)
  warmup_iters: 500         # Linear warmup steps
  weight_decay: 0.1         # AdamW weight decay
  beta1: 0.9                # AdamW beta1
  beta2: 0.95               # AdamW beta2
  grad_clip: 1.0            # Max gradient norm (0 to disable)
  eval_interval: 500        # Evaluate every N steps
  eval_iters: 50            # Batches per evaluation
  log_interval: 10          # Print loss every N steps
  checkpoint_dir: "checkpoints"

data:
  data_dir: "data/processed"  # Directory containing train.bin and val.bin
```

**Effective batch size** = `batch_size` × `gradient_accumulation_steps` × `max_seq_len`

Example: 64 × 2 × 1024 = **131,072 tokens per iteration**

**Total tokens processed** = effective batch × max_iters = 131,072 × 15,000 ≈ **2B tokens**

## Understanding Training Logs

```
iter   100: loss 6.7658, lr 1.21e-04, 348,170 tok/s, MFU 26.0%, grad_norm 0.67
```

| Field | Meaning | Healthy Range |
|-------|---------|--------------|
| `loss` | Cross-entropy loss (lower = better) | Should decrease over time |
| `lr` | Current learning rate | Increases during warmup, then cosine decays |
| `tok/s` | Tokens processed per second | Higher = faster training |
| `MFU` | Model FLOPs Utilization (% of GPU peak used) | 20–50% typical for medium models |
| `grad_norm` | Gradient magnitude after clipping | 0.1–2.0 is healthy, >5.0 is concerning |

```
step  5000: train loss 3.50 (ppl 33.1), val loss 3.62 (ppl 37.3)
  -> new best val loss! saved to checkpoints/ckpt_best.pt
```

| Field | Meaning |
|-------|---------|
| `train loss / ppl` | Performance on training data |
| `val loss / ppl` | Performance on held-out data (this is the real metric) |
| `new best val loss` | Model improved — checkpoint saved |

**Perplexity (PPL)** = exp(loss). "On average, the model is confused between PPL possible next words."

## Checkpointing

### What's in a checkpoint:

```python
checkpoint = {
    'model': model.state_dict(),        # All model weights
    'optimizer': optimizer.state_dict(), # Optimizer state (momentum, etc.)
    'scaler': scaler.state_dict(),       # Gradient scaler state
    'model_args': config_dict,           # Model configuration
    'iter_num': iter_num,                # Training step number
    'best_val_loss': best_val_loss,      # Best validation loss so far
}
```

### Two checkpoints saved:
- `ckpt_best.pt` — Best validation loss ever seen (use this for inference)
- `ckpt.pt` — Latest checkpoint (use this to resume training)

### Resume training:
Load the checkpoint and continue from `iter_num`. The optimizer and scaler states are restored automatically.

## torch.compile

Add `--compile` to enable PyTorch's graph compiler:

```bash
python training/train.py --config configs/train_125m.yaml --compile
```

| Aspect | Without compile | With compile |
|--------|----------------|-------------|
| First iteration | Instant | ~30s (one-time compilation) |
| Steady state | Baseline | 1.3–1.8x faster |
| Memory | Baseline | ~10% more (for compiled graphs) |
| Compatibility | All PyTorch | PyTorch 2.0+ Linux recommended |

The `_orig_mod.` prefix in checkpoint keys is from torch.compile — `generate.py` handles this automatically.

## CPU vs GPU Training

| Aspect | CPU | GPU |
|--------|-----|-----|
| Speed | ~30K tok/s | ~300K+ tok/s |
| Precision | float16 | bfloat16 (preferred on Ampere+) |
| Max practical model | ~10M params | 100M+ params |
| Use case | Testing, validation | Actual training |

The code auto-detects the device. No code changes needed.

## Monitoring Training

### Healthy signs:
- Loss decreases consistently over time
- Train and val loss stay close (gap < 0.3 is healthy)
- Grad norm stays between 0.1–2.0
- No NaN/Inf warnings

### When to stop:
- Val loss stops improving for several eval intervals
- Val loss starts increasing while train loss decreases (overfitting)
- You've processed enough tokens for your model size

### Overfitting detection:

| Gap (val_loss - train_loss) | Status |
|----------------------------|--------|
| < 0.1 | Healthy — may need more training |
| 0.1 – 0.3 | Healthy |
| 0.3 – 0.5 | Moderate — consider more data |
| > 0.5 | Overfitting — need more data or fewer steps |

Our 125M model achieved a gap of **0.03** — very healthy, meaning we could train even longer with more data.
