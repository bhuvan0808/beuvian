# Scaling to Production

## Scaling Data: Streaming from C4

For production training, you need billions of tokens. The C4 (Colossal Clean Crawled Corpus) dataset streams directly from HuggingFace — no full download required.

### Single-stream (simple)

```bash
python scripts/prepare_stream.py \
    --dataset "allenai/c4" \
    --vocab_size 32000 \
    --target_tokens 2000000000 \
    --tokenizer_out tokenizer/tokenizer_32k.json \
    --output_dir data/processed
```

This streams text → tokenizes in memory → writes binary chunks to disk. No raw text stored.

Speed: ~0.37M tokens/sec → ~90 min for 2B tokens.

### Parallel streaming (8x faster)

```bash
# Step 1: Train tokenizer first (uses prepare_stream.py or reuse existing)
# Step 2: Launch 8 parallel workers
python scripts/prepare_parallel.py \
    --dataset "allenai/c4" \
    --tokenizer "tokenizer/tokenizer_32k.json" \
    --num_workers 8 \
    --target_tokens 2000000000 \
    --output_dir data/processed
```

How it works:
1. Each worker streams a different portion of C4 simultaneously
2. Each tokenizes independently and writes its own chunk file
3. Main process monitors progress every 15 seconds
4. When all workers finish, chunks are merged into `train.bin` + `val.bin`
5. Chunk files are cleaned up

Speed: **~1.5M tokens/sec → ~22 min for 2B tokens** (3.6x faster than single stream)

### Data output sizes

| Tokens | Binary size (uint16) | Disk needed |
|--------|---------------------|-------------|
| 100M | 200 MB | Small runs |
| 500M | 1 GB | Medium |
| 2B | 4 GB | Our 125M training |
| 10B | 20 GB | Large 350M+ model |
| 50B | 100 GB | 1B+ model |

## Scaling Model Size

### Configuration examples

| Config | d_model | n_layers | n_heads | vocab | context | Approx Params |
|--------|---------|----------|---------|-------|---------|--------------|
| **Tiny (test)** | 128 | 4 | 4 | 8K | 128 | ~2.8M |
| **Small (validation)** | 384 | 6 | 6 | 8K | 512 | ~13.7M |
| **Medium (GPT-2 Small)** | 768 | 12 | 12 | 32K | 1024 | ~109.5M |
| **Large (GPT-2 Medium)** | 1024 | 24 | 16 | 50K | 2048 | ~350M |
| **XL (GPT-2 Large)** | 1536 | 24 | 16 | 50K | 2048 | ~770M |
| **1B (Pythia-1B)** | 2048 | 16 | 16 | 50K | 2048 | ~1B |

### Scaling rules of thumb
- **Params ~ d_model^2 × n_layers** (roughly)
- **d_model must be divisible by n_heads**
- **head_dim = d_model / n_heads** should be 64 or 128
- **FFN hidden dim** is auto-calculated as `round_to_256(4 * d_model * 2/3)`

## Batch Size Optimization

To find the maximum batch size for your GPU:

```python
import torch
from model.config import BUVNConfig
from model.model import BUVNModel

config = BUVNConfig(vocab_size=32000, d_model=768, n_layers=12,
                     n_heads=12, max_seq_len=1024, dropout=0.0, bias=False)
model = BUVNModel(config).cuda().bfloat16()

for bs in [16, 32, 48, 64, 96, 128]:
    torch.cuda.reset_peak_memory_stats()
    try:
        x = torch.randint(0, 32000, (bs, 1024), device='cuda')
        y = torch.randint(0, 32000, (bs, 1024), device='cuda')
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            _, loss = model(x, y)
            loss.backward()
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f'  batch_size={bs}: {peak:.1f} GB  OK')
        model.zero_grad(); torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        print(f'  batch_size={bs}: OOM')
        break
```

**Our H100 results:** batch_size=96 fits (79.8 GB). We used 64 to leave headroom for torch.compile.

## Hardware Recommendations

| Setup | Hardware | Best For | Max Model |
|-------|----------|----------|-----------|
| CPU | Any CPU, 8+ GB RAM | Testing, learning | ~10M params |
| Entry GPU | RTX 3060/4060 (12 GB) | Small training | ~30M params |
| Mid GPU | RTX 3090/4090 (24 GB) | Medium training | ~125M params |
| Cloud GPU | A100 40 GB | Production training | ~350M params |
| Cloud GPU | A100 80 GB / H100 96 GB | Large scale | ~1B params |
| Multi-GPU | 4× A100/H100 with DDP | Production | 1B+ params |

## Cost Estimates

| Model Size | Tokens | GPU | Time | Approx Cost |
|-----------|--------|-----|------|-------------|
| 10M (test) | 300M | Any GPU | 5 min | ~$0.30 |
| **109M (done)** | **2B** | **H100** | **~2 hrs** | **~$8** |
| 350M | 10B | H100 | ~30 hrs | ~$120 |
| 770M | 40B | H100 | ~5 days | ~$500 |
| 1B | 100B | 4× H100 | ~1 week | ~$2,000 |

*Costs based on cloud GPU pricing of ~$3-4/hr for H100.*

## The BUVN-2.0 Training Run (What We Actually Did)

### Data Preparation (22 minutes)

```bash
# Trained 32K BPE tokenizer on 100K streamed C4 samples (14 seconds)
# Then launched 8 parallel workers to stream and tokenize
python scripts/prepare_parallel.py \
    --dataset "allenai/c4" \
    --tokenizer "tokenizer/tokenizer_32k.json" \
    --num_workers 8 \
    --target_tokens 2000000000
```

- **8 workers** streaming simultaneously at 1.48M tok/s combined
- **2.00B tokens** collected in **22.6 minutes**
- Output: `train.bin` (3.8 GB) + `val.bin` (20 MB)

### Model Training (~2 hours)

```bash
python training/train.py --config configs/train_125m.yaml --compile --seed 42
```

Config: batch_size=64, grad_accum=2, 15K steps, lr=6e-4, cosine decay

| Metric | Value |
|--------|-------|
| Throughput | ~320K tok/s |
| MFU | 24% |
| Peak VRAM | ~53 GB |
| Total tokens processed | ~2B |
| Training time | ~2 hours |

### Results

| Metric | Value |
|--------|-------|
| Val Perplexity | **29.19** |
| Train Perplexity | 28.33 |
| Overfit Gap | 0.03 (healthy) |
| Top-1 Accuracy | 37.88% |
| Top-5 Accuracy | 60.34% |
| Leaderboard Rank | **#8 / 11 (beat GPT-2 Small!)** |

### Total wall-clock time: ~2.5 hours (data prep + training)
### Total cost: ~$8 on cloud H100
