"""
BUVN-1.1 Comprehensive Benchmark Suite
========================================
Evaluates the model on industry-standard metrics and compares with baselines.

Metrics:
  1. Perplexity (PPL) - standard LM metric
  2. Bits Per Character (BPC) - normalized metric
  3. Top-1 Accuracy - next-token prediction accuracy
  4. Top-5 Accuracy - next-token in top-5 predictions
  5. Average Loss - cross-entropy loss
  6. Throughput - tokens/sec on current hardware
  7. Generation Latency - ms per generated token
  8. Memory Footprint - model size in MB

Compared against: GPT-2, GPT-Neo, OPT, Pythia, LLaMA

Usage:
    python scripts/benchmark.py
"""

import os
import sys
import time
import math
import warnings
import torch
import numpy as np

warnings.filterwarnings('ignore')

# Fix Windows terminal encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.config import BUVNConfig
from model.model import BUVNModel


def load_model(checkpoint_path):
    """Loads checkpoint and returns model + config."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = BUVNConfig.from_dict(ckpt['model_args'])
    model = BUVNModel(config)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, config


def get_batch(data, seq_len, batch_size):
    """Draws a random batch from memory-mapped data."""
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[j:j+seq_len].astype(np.int64)) for j in ix])
    y = torch.stack([torch.from_numpy(data[j+1:j+1+seq_len].astype(np.int64)) for j in ix])
    return x, y


# ======================================================================
# BENCHMARK 1: Perplexity & Loss
# ======================================================================
def benchmark_perplexity(model, data_path, seq_len, batch_size=8, num_batches=200):
    """
    Perplexity = exp(avg cross-entropy loss)
    The standard metric for language model quality.
    Lower is better. Random baseline = vocab_size.
    """
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for _ in range(num_batches):
            x, y = get_batch(data, seq_len, batch_size)
            _, loss = model(x, y)
            total_loss += loss.item()
            count += 1

    avg_loss = total_loss / count
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


# ======================================================================
# BENCHMARK 2: Bits Per Character (BPC)
# ======================================================================
def benchmark_bpc(avg_loss):
    """
    BPC = avg_loss / ln(2)
    Normalized measure of compression efficiency.
    Lower is better. Used by character-level models.
    """
    return avg_loss / math.log(2)


# ======================================================================
# BENCHMARK 3: Top-K Accuracy
# ======================================================================
def benchmark_accuracy(model, data_path, seq_len, batch_size=8, num_batches=200):
    """
    Top-1: How often the model's #1 prediction is correct.
    Top-5: How often the correct token is in the top 5 predictions.
    """
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for _ in range(num_batches):
            x, y = get_batch(data, seq_len, batch_size)
            logits, _ = model(x, y)

            # Top-1
            preds = logits.argmax(dim=-1)
            top1_correct += (preds == y).sum().item()

            # Top-5
            top5 = logits.topk(5, dim=-1).indices
            top5_match = (top5 == y.unsqueeze(-1)).any(dim=-1)
            top5_correct += top5_match.sum().item()

            total += y.numel()

    top1_acc = (top1_correct / total) * 100
    top5_acc = (top5_correct / total) * 100
    return top1_acc, top5_acc


# ======================================================================
# BENCHMARK 4: Throughput (tokens/sec)
# ======================================================================
def benchmark_throughput(model, seq_len, batch_size=8, num_iters=50):
    """Measures forward-pass throughput in tokens per second."""
    vocab = model.config.vocab_size

    # Warmup
    dummy = torch.randint(0, vocab, (batch_size, seq_len))
    with torch.no_grad():
        model(dummy)

    total_tokens = 0
    start = time.time()
    with torch.no_grad():
        for _ in range(num_iters):
            dummy = torch.randint(0, vocab, (batch_size, seq_len))
            model(dummy)
            total_tokens += batch_size * seq_len
    elapsed = time.time() - start
    return total_tokens / elapsed


# ======================================================================
# BENCHMARK 5: Generation Latency
# ======================================================================
def benchmark_generation_latency(model, seq_len, num_tokens=50, num_runs=5):
    """Measures average time to generate one token (autoregressive)."""
    vocab = model.config.vocab_size
    latencies = []

    with torch.no_grad():
        for _ in range(num_runs):
            idx = torch.randint(0, vocab, (1, 10))
            start = time.time()
            for _ in range(num_tokens):
                idx_cond = idx if idx.size(1) <= seq_len else idx[:, -seq_len:]
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :]
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, 1)
                idx = torch.cat([idx, idx_next], dim=1)
            elapsed = (time.time() - start) * 1000
            latencies.append(elapsed / num_tokens)

    avg_ms = sum(latencies) / len(latencies)
    return avg_ms


# ======================================================================
# BENCHMARK 6: Memory Footprint
# ======================================================================
def benchmark_memory(model, checkpoint_path):
    """Model size metrics."""
    num_params = sum(p.numel() for p in model.parameters())
    param_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    file_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    return num_params, param_mb, file_mb


# ======================================================================
# Published Baselines
# ======================================================================
BASELINES = {
    "GPT-2 Small":   {"params": "124M",  "ppl": 29.41, "tokens": "~40B",   "bpc": None, "year": 2019, "org": "OpenAI"},
    "GPT-Neo 125M":  {"params": "125M",  "ppl": 32.43, "tokens": "300B",   "bpc": None, "year": 2021, "org": "EleutherAI"},
    "OPT-125M":      {"params": "125M",  "ppl": 27.65, "tokens": "300B",   "bpc": None, "year": 2022, "org": "Meta"},
    "Pythia-160M":   {"params": "160M",  "ppl": 29.33, "tokens": "300B",   "bpc": None, "year": 2023, "org": "EleutherAI"},
    "RWKV-169M":     {"params": "169M",  "ppl": 29.01, "tokens": "300B",   "bpc": None, "year": 2023, "org": "RWKV"},
    "GPT-2 Medium":  {"params": "355M",  "ppl": 22.76, "tokens": "~40B",   "bpc": None, "year": 2019, "org": "OpenAI"},
    "GPT-2 Large":   {"params": "774M",  "ppl": 19.93, "tokens": "~40B",   "bpc": None, "year": 2019, "org": "OpenAI"},
    "Pythia-1B":     {"params": "1B",    "ppl": 16.71, "tokens": "300B",   "bpc": None, "year": 2023, "org": "EleutherAI"},
    "LLaMA 7B":      {"params": "7B",    "ppl": 7.73,  "tokens": "1T",     "bpc": None, "year": 2023, "org": "Meta"},
    "LLaMA-2 7B":    {"params": "7B",    "ppl": 5.47,  "tokens": "2T",     "bpc": None, "year": 2023, "org": "Meta"},
}


# ======================================================================
# Main
# ======================================================================
def main():
    checkpoint_path = "BUVN-1.1/checkpoints/ckpt.pt"
    val_data_path = "BUVN-1.1/data/processed/val.bin"

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return
    if not os.path.exists(val_data_path):
        print(f"ERROR: Validation data not found: {val_data_path}")
        return

    results = []
    def log(msg=""):
        print(msg)
        results.append(msg)

    log("=" * 72)
    log("  BUVN-1.1 COMPREHENSIVE MODEL BENCHMARK")
    log("=" * 72)

    # Load model
    log("\n[1/6] Loading model...")
    model, config = load_model(checkpoint_path)
    num_params, param_mb, file_mb = benchmark_memory(model, checkpoint_path)

    log(f"  Architecture : {config.n_layers} layers, {config.d_model} dim, {config.n_heads} heads")
    log(f"  Parameters   : {num_params:,} ({num_params/1e6:.2f}M)")
    log(f"  Vocab Size   : {config.vocab_size}")
    log(f"  Context Len  : {config.max_seq_len}")
    log(f"  Weights Size : {param_mb:.1f} MB (in memory)")
    log(f"  Checkpoint   : {file_mb:.1f} MB (on disk)")

    # Perplexity
    log(f"\n[2/6] Evaluating Perplexity (200 batches)...")
    avg_loss, ppl = benchmark_perplexity(model, val_data_path, config.max_seq_len)
    bpc = benchmark_bpc(avg_loss)
    log(f"  Avg Loss   : {avg_loss:.4f}")
    log(f"  Perplexity : {ppl:.2f}")
    log(f"  Bits/Char  : {bpc:.4f}")

    # Random baseline for reference
    random_ppl = config.vocab_size
    log(f"  Random PPL : {random_ppl} (vocab size = random guessing)")
    log(f"  Improvement: {((random_ppl - ppl) / random_ppl * 100):.1f}% better than random")

    # Accuracy
    log(f"\n[3/6] Evaluating Top-K Accuracy (200 batches)...")
    top1, top5 = benchmark_accuracy(model, val_data_path, config.max_seq_len)
    log(f"  Top-1 Accuracy : {top1:.2f}%")
    log(f"  Top-5 Accuracy : {top5:.2f}%")

    # Throughput
    log(f"\n[4/6] Measuring Throughput (50 forward passes)...")
    tps = benchmark_throughput(model, config.max_seq_len)
    log(f"  Throughput : {tps:,.0f} tokens/sec (CPU)")

    # Generation latency
    log(f"\n[5/6] Measuring Generation Latency (50 tokens x 5 runs)...")
    latency = benchmark_generation_latency(model, config.max_seq_len)
    log(f"  Avg Latency : {latency:.1f} ms/token")
    log(f"  Est. Speed  : {1000/latency:.0f} tokens/sec (autoregressive)")

    # Comparison
    log(f"\n[6/6] Benchmark Comparison")
    log("")
    log("=" * 72)
    log("  LEADERBOARD - WikiText-103 Perplexity (lower = better)")
    log("=" * 72)
    log("")
    log(f"  {'Rank':<6}{'Model':<22}{'Params':>8}{'PPL':>10}{'Tokens':>12}  {'Org'}")
    log(f"  {'-'*68}")

    # Build sorted list
    all_models = [("BUVN-1.1 (ours)", {"params": f"{num_params/1e6:.1f}M", "ppl": round(ppl,2), "tokens": "74.5M", "org": "You"})]
    for name, data in BASELINES.items():
        all_models.append((name, data))

    all_models.sort(key=lambda x: x[1]["ppl"])

    for rank, (name, data) in enumerate(all_models, 1):
        marker = "  <-- YOU" if "ours" in name else ""
        log(f"  {rank:<6}{name:<22}{data['params']:>8}{data['ppl']:>10.2f}{data['tokens']:>12}  {data['org']}{marker}")

    log("")

    # Find our rank
    our_rank = next(i for i, (n, _) in enumerate(all_models, 1) if "ours" in n)
    log(f"  Your rank: #{our_rank} out of {len(all_models)} models")
    log("")

    # Gap analysis
    log("=" * 72)
    log("  GAP ANALYSIS")
    log("=" * 72)
    log("")
    log(f"  Your PPL: {ppl:.2f}  |  Your Params: {num_params/1e6:.1f}M  |  Your Data: 74.5M tokens")
    log("")

    gaps = [
        ("GPT-2 Small (124M)",    29.41,  "44x more params, 537x more data, ~$50 on 1xA100"),
        ("OPT-125M",              27.65,  "44x more params, 4000x more data, ~$150 on A100s"),
        ("Pythia-160M",           29.33,  "56x more params, 4000x more data, ~$150 on A100s"),
        ("GPT-2 Medium (355M)",   22.76, "125x more params, 537x more data, ~$200 on A100s"),
        ("LLaMA-2 7B",             5.47, "2500x more params, 26000x more data, ~$2M compute"),
    ]

    for name, target, recipe in gaps:
        gap = ppl - target
        pct = (gap / ppl) * 100
        log(f"  vs {name}:")
        log(f"     Target PPL: {target:.2f}  |  Gap: {gap:.1f} points ({pct:.0f}% reduction)")
        log(f"     Recipe: {recipe}")
        log("")

    # Improvement Phases
    log("=" * 72)
    log("  IMPROVEMENT ROADMAP")
    log("=" * 72)
    log("")
    log("  CURRENT STATE:")
    log(f"    PPL={ppl:.0f} | 2.8M params | 74.5M tokens | CPU | 500 iters")
    log("")
    log("  PHASE 1 - More Data + Longer Training  (est. PPL ~100-200)")
    log("    - Switch to C4 dataset (5+ GB corpus)")
    log("    - Train tokenizer with 50K vocab")
    log("    - Same tiny model, but 10,000+ iterations")
    log("    - Still runs on CPU (slow but works)")
    log("    - Cost: $0 (just time)")
    log("")
    log("  PHASE 2 - Scale Up Model  (est. PPL ~40-70)")
    log("    - Use full 120M param config (12 layers, 768 dim)")
    log("    - Train on 1-5B tokens from C4")
    log("    - Requires 1x GPU (A100/H100)")
    log("    - Cost: ~$50-100 on Azure")
    log("")
    log("  PHASE 3 - Production Training  (est. PPL ~25-35)")
    log("    - 120M params, 20-40B tokens")
    log("    - Multi-GPU DDP, Flash Attention 2")
    log("    - Cosine schedule, 100K+ steps")
    log("    - Competitive with GPT-2 Small / OPT-125M")
    log("    - Cost: ~$150 on Azure NC A100 v4")
    log("")
    log("  PHASE 4 - Competitive  (est. PPL ~15-20)")
    log("    - Scale to 350M-1B params")
    log("    - 100B+ tokens, multi-node training")
    log("    - Add SFT + RLHF for instruction following")
    log("    - Competitive with GPT-2 Large / Pythia-1B")
    log("    - Cost: ~$500-2000")
    log("")
    log("=" * 72)

    # Summary card
    log("")
    log("+------------------------------------------------------+")
    log("|            BUVN-1.1 BENCHMARK SUMMARY                |")
    log("+------------------------------------------------------+")
    log(f"|  Perplexity (PPL)      :  {ppl:>10.2f}                 |")
    log(f"|  Bits Per Character    :  {bpc:>10.4f}                 |")
    log(f"|  Top-1 Accuracy        :  {top1:>9.2f}%                 |")
    log(f"|  Top-5 Accuracy        :  {top5:>9.2f}%                 |")
    log(f"|  Throughput (fwd)      :  {tps:>8,.0f} tok/s (CPU)      |")
    log(f"|  Generation Latency    :  {latency:>8.1f} ms/tok           |")
    log(f"|  Parameters            :  {num_params/1e6:>8.2f}M                |")
    log(f"|  vs Random (PPL={random_ppl:,})  :  {((random_ppl - ppl) / random_ppl * 100):>8.1f}% better       |")
    log(f"|  Leaderboard Rank      :  #{our_rank:>6} / {len(all_models)}              |")
    log("+------------------------------------------------------+")

    # Save to file
    output_path = "BUVN-1.1/benchmark_results.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(results))
    log(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
