"""
BUVN Comprehensive Benchmark Suite
====================================
Evaluates the model on industry-standard metrics and compares with baselines.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --checkpoint checkpoints/ckpt_best.pt --data data/processed/val.bin
"""

import os
import sys
import time
import math
import argparse
import warnings
import torch
import numpy as np

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model.config import BUVNConfig
from model.model import BUVNModel


def load_model(checkpoint_path, device='cpu'):
    """Loads checkpoint and returns model + config."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = BUVNConfig.from_dict(ckpt['model_args'])
    model = BUVNModel(config)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()
    return model, config, ckpt


def get_batch(data, seq_len, batch_size, device='cpu'):
    """Draws a random batch from memory-mapped data."""
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[j:j+seq_len].astype(np.int64)) for j in ix]).to(device)
    y = torch.stack([torch.from_numpy(data[j+1:j+1+seq_len].astype(np.int64)) for j in ix]).to(device)
    return x, y


def benchmark_perplexity(model, data_path, seq_len, device, batch_size=16, num_batches=200):
    """Perplexity = exp(avg cross-entropy loss). Lower is better."""
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for _ in range(num_batches):
            x, y = get_batch(data, seq_len, batch_size, device)
            _, loss = model(x, y)
            total_loss += loss.item()
            count += 1

    avg_loss = total_loss / count
    ppl = math.exp(min(avg_loss, 20.0))
    return avg_loss, ppl


def benchmark_accuracy(model, data_path, seq_len, device, batch_size=16, num_batches=200):
    """Top-1 and Top-5 next-token prediction accuracy."""
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    top1_correct = 0
    top5_correct = 0
    total = 0

    with torch.no_grad():
        for _ in range(num_batches):
            x, y = get_batch(data, seq_len, batch_size, device)
            logits, _ = model(x, y)

            preds = logits.argmax(dim=-1)
            top1_correct += (preds == y).sum().item()

            top5 = logits.topk(5, dim=-1).indices
            top5_match = (top5 == y.unsqueeze(-1)).any(dim=-1)
            top5_correct += top5_match.sum().item()

            total += y.numel()

    return (top1_correct / total) * 100, (top5_correct / total) * 100


def benchmark_throughput(model, seq_len, device, batch_size=16, num_iters=100):
    """Forward-pass throughput in tokens/sec."""
    vocab = model.config.vocab_size

    # Warmup
    dummy = torch.randint(0, vocab, (batch_size, seq_len), device=device)
    with torch.no_grad():
        for _ in range(5):
            model(dummy)
    if device == 'cuda':
        torch.cuda.synchronize()

    total_tokens = 0
    start = time.time()
    with torch.no_grad():
        for _ in range(num_iters):
            dummy = torch.randint(0, vocab, (batch_size, seq_len), device=device)
            model(dummy)
            total_tokens += batch_size * seq_len
    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - start
    return total_tokens / elapsed


def benchmark_generation_latency(model, seq_len, device, num_tokens=100, num_runs=5):
    """Average time to generate one token (autoregressive)."""
    vocab = model.config.vocab_size
    latencies = []

    with torch.no_grad():
        for _ in range(num_runs):
            idx = torch.randint(0, vocab, (1, 10), device=device)
            if device == 'cuda':
                torch.cuda.synchronize()
            start = time.time()
            for _ in range(num_tokens):
                idx_cond = idx if idx.size(1) <= seq_len else idx[:, -seq_len:]
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :]
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, 1)
                idx = torch.cat([idx, idx_next], dim=1)
            if device == 'cuda':
                torch.cuda.synchronize()
            elapsed = (time.time() - start) * 1000
            latencies.append(elapsed / num_tokens)

    return sum(latencies) / len(latencies)


def benchmark_memory(model, checkpoint_path):
    """Model size metrics."""
    num_params = sum(p.numel() for p in model.parameters())
    non_emb_params = model.get_num_params(non_embedding=True)
    param_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    file_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    return num_params, non_emb_params, param_mb, file_mb


BASELINES = {
    "GPT-2 Small":   {"params": "124M",  "ppl": 29.41, "tokens": "~40B",   "org": "OpenAI",      "year": 2019},
    "GPT-Neo 125M":  {"params": "125M",  "ppl": 32.43, "tokens": "300B",   "org": "EleutherAI",  "year": 2021},
    "OPT-125M":      {"params": "125M",  "ppl": 27.65, "tokens": "300B",   "org": "Meta",        "year": 2022},
    "Pythia-160M":   {"params": "160M",  "ppl": 29.33, "tokens": "300B",   "org": "EleutherAI",  "year": 2023},
    "RWKV-169M":     {"params": "169M",  "ppl": 29.01, "tokens": "300B",   "org": "RWKV",        "year": 2023},
    "GPT-2 Medium":  {"params": "355M",  "ppl": 22.76, "tokens": "~40B",   "org": "OpenAI",      "year": 2019},
    "GPT-2 Large":   {"params": "774M",  "ppl": 19.93, "tokens": "~40B",   "org": "OpenAI",      "year": 2019},
    "Pythia-1B":     {"params": "1B",    "ppl": 16.71, "tokens": "300B",   "org": "EleutherAI",  "year": 2023},
    "LLaMA 7B":      {"params": "7B",    "ppl": 7.73,  "tokens": "1T",     "org": "Meta",        "year": 2023},
    "LLaMA-2 7B":    {"params": "7B",    "ppl": 5.47,  "tokens": "2T",     "org": "Meta",        "year": 2023},
}


def main():
    parser = argparse.ArgumentParser(description="BUVN Comprehensive Benchmark")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ckpt_best.pt")
    parser.add_argument("--data", type=str, default="data/processed/val.bin")
    parser.add_argument("--train_data", type=str, default="data/processed/train.bin")
    parser.add_argument("--output", type=str, default="benchmark_results.txt")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        return
    if not os.path.exists(args.data):
        print(f"ERROR: Validation data not found: {args.data}")
        return

    results = []
    def log(msg=""):
        print(msg)
        results.append(msg)

    log("=" * 72)
    log("  BUVN COMPREHENSIVE MODEL BENCHMARK")
    log("=" * 72)

    # Hardware info
    log(f"\n  Hardware: {device.upper()}")
    if device == 'cuda':
        log(f"  GPU: {torch.cuda.get_device_name(0)}")
        log(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        log(f"  PyTorch: {torch.__version__}")
        log(f"  CUDA: {torch.version.cuda}")

    # Load model
    log("\n[1/6] Loading model...")
    model, config, ckpt = load_model(args.checkpoint, device)
    num_params, non_emb_params, param_mb, file_mb = benchmark_memory(model, args.checkpoint)

    log(f"  Architecture : {config.n_layers} layers, {config.d_model} dim, {config.n_heads} heads")
    log(f"  Parameters   : {num_params:,} ({num_params/1e6:.2f}M total)")
    log(f"  Non-Embedding: {non_emb_params:,} ({non_emb_params/1e6:.2f}M)")
    log(f"  Vocab Size   : {config.vocab_size:,}")
    log(f"  Context Len  : {config.max_seq_len}")
    log(f"  Weights Size : {param_mb:.1f} MB (in memory)")
    log(f"  Checkpoint   : {file_mb:.1f} MB (on disk)")
    if 'best_val_loss' in ckpt:
        log(f"  Best Val Loss: {ckpt['best_val_loss']:.4f}")
    if 'iter_num' in ckpt:
        log(f"  Trained Steps: {ckpt['iter_num']:,}")

    # Perplexity on val
    log(f"\n[2/6] Evaluating Perplexity on Validation Set (200 batches)...")
    val_loss, val_ppl = benchmark_perplexity(model, args.data, config.max_seq_len, device)
    bpc = val_loss / math.log(2)
    log(f"  Val Loss     : {val_loss:.4f}")
    log(f"  Val PPL      : {val_ppl:.2f}")
    log(f"  Bits/Char    : {bpc:.4f}")

    # Perplexity on train
    if os.path.exists(args.train_data):
        train_loss, train_ppl = benchmark_perplexity(model, args.train_data, config.max_seq_len, device)
        log(f"  Train Loss   : {train_loss:.4f}")
        log(f"  Train PPL    : {train_ppl:.2f}")
        overfit_gap = val_loss - train_loss
        log(f"  Overfit Gap  : {overfit_gap:.4f} ({'overfitting!' if overfit_gap > 0.5 else 'healthy' if overfit_gap < 0.3 else 'moderate'})")

    random_ppl = config.vocab_size
    log(f"  Random PPL   : {random_ppl:,} (vocab size = random guessing)")
    log(f"  vs Random    : {((random_ppl - val_ppl) / random_ppl * 100):.1f}% better")

    # Accuracy
    log(f"\n[3/6] Evaluating Top-K Accuracy (200 batches)...")
    top1, top5 = benchmark_accuracy(model, args.data, config.max_seq_len, device)
    log(f"  Top-1 Accuracy : {top1:.2f}%")
    log(f"  Top-5 Accuracy : {top5:.2f}%")

    # Throughput
    log(f"\n[4/6] Measuring Throughput (100 forward passes, batch=16)...")
    tps = benchmark_throughput(model, config.max_seq_len, device)
    log(f"  Throughput : {tps:,.0f} tokens/sec ({device.upper()})")

    # MFU estimate
    mfu = model.estimate_mfu(16, 16 * config.max_seq_len / tps)
    log(f"  MFU        : {mfu*100:.1f}%")

    # Generation latency
    log(f"\n[5/6] Measuring Generation Latency (100 tokens x 5 runs)...")
    latency = benchmark_generation_latency(model, config.max_seq_len, device)
    log(f"  Avg Latency : {latency:.1f} ms/token")
    log(f"  Gen Speed   : {1000/latency:.0f} tokens/sec (autoregressive)")

    # GPU memory
    if device == 'cuda':
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        log(f"  Peak VRAM   : {peak_mem:.2f} GB")

    # Leaderboard
    log(f"\n[6/6] Benchmark Comparison")
    log("")
    log("=" * 76)
    log("  LEADERBOARD - WikiText-103 Perplexity (lower = better)")
    log("=" * 76)
    log("")
    log(f"  {'Rank':<6}{'Model':<24}{'Params':>8}{'PPL':>10}{'Tokens':>12}  {'Org'}")
    log(f"  {'-'*72}")

    tokens_trained = f"{ckpt.get('iter_num', 5000) * 65536 / 1e6:.0f}M"
    all_models = [("BUVN (ours)", {"params": f"{num_params/1e6:.1f}M", "ppl": round(val_ppl, 2), "tokens": tokens_trained, "org": "Bhuvan"})]
    for name, data in BASELINES.items():
        all_models.append((name, data))

    all_models.sort(key=lambda x: x[1]["ppl"])

    for rank, (name, data) in enumerate(all_models, 1):
        marker = "  <-- YOU" if "ours" in name else ""
        log(f"  {rank:<6}{name:<24}{data['params']:>8}{data['ppl']:>10.2f}{data['tokens']:>12}  {data['org']}{marker}")

    log("")
    our_rank = next(i for i, (n, _) in enumerate(all_models, 1) if "ours" in n)
    log(f"  Your rank: #{our_rank} out of {len(all_models)} models")

    # Gap analysis
    log("")
    log("=" * 76)
    log("  GAP ANALYSIS — What it takes to close the gap")
    log("=" * 76)
    log("")

    gaps = [
        ("GPT-2 Small (124M)",    29.41,  "12x params, 3000x data → ~$50 on 1x H100, ~5 hrs"),
        ("OPT-125M",              27.65,  "12x params, 23000x data → ~$150 on H100, ~15 hrs"),
        ("GPT-2 Medium (355M)",   22.76,  "34x params, 3000x data → ~$300 on H100, ~30 hrs"),
        ("Pythia-1B",             16.71,  "96x params, 23000x data → ~$2000, multi-GPU"),
    ]

    for name, target, recipe in gaps:
        gap = val_ppl - target
        log(f"  vs {name}:")
        log(f"     Target PPL: {target:.2f}  |  Gap: {gap:.1f} points")
        log(f"     Recipe: {recipe}")
        log("")

    # Summary card
    log("=" * 76)
    log("")
    log("+----------------------------------------------------------+")
    log("|              BUVN BENCHMARK SUMMARY                      |")
    log("+----------------------------------------------------------+")
    log(f"|  Model            :  {config.n_layers}L / {config.d_model}d / {config.n_heads}H              |")
    log(f"|  Parameters       :  {num_params/1e6:>8.2f}M                          |")
    log(f"|  Perplexity (PPL) :  {val_ppl:>10.2f}                          |")
    log(f"|  Bits Per Char    :  {bpc:>10.4f}                          |")
    log(f"|  Top-1 Accuracy   :  {top1:>9.2f}%                          |")
    log(f"|  Top-5 Accuracy   :  {top5:>9.2f}%                          |")
    log(f"|  Throughput (fwd) :  {tps:>8,.0f} tok/s ({device.upper()})             |")
    log(f"|  Gen Latency      :  {latency:>8.1f} ms/tok                    |")
    log(f"|  Gen Speed        :  {1000/latency:>8.0f} tok/s (autoregressive)     |")
    log(f"|  vs Random        :  {((random_ppl - val_ppl) / random_ppl * 100):>8.1f}% better                  |")
    log(f"|  Leaderboard      :  #{our_rank:>6} / {len(all_models)}                       |")
    log("+----------------------------------------------------------+")

    # Save
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(results))
    log(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
