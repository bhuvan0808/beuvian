"""
BUVN-2.0 Parallel Streaming Data Pipeline
==========================================
Streams C4 from HuggingFace using 8 parallel shard workers.
Each worker streams a different shard, tokenizes in memory, writes a chunk file.
Main process merges chunks into train.bin / val.bin.

~8x faster than single-stream. No raw text on disk.

Usage:
    python scripts/prepare_parallel.py
    python scripts/prepare_parallel.py --num_workers 8 --target_tokens 2_000_000_000
"""

import os
import sys
import time
import argparse
import numpy as np
from multiprocessing import Process, Value, Lock
from datasets import load_dataset
from tokenizers import Tokenizer


def worker_stream(
    worker_id, num_workers, dataset_name, tokenizer_path,
    output_dir, tokens_per_worker, shared_counter, lock
):
    """Each worker streams one shard, tokenizes, writes its own chunk file."""
    tokenizer = Tokenizer.from_file(tokenizer_path)

    ds = load_dataset(dataset_name, "en", split="train", streaming=True)
    # Each worker takes every Nth sample (shard by skipping)
    shard = ds.skip(worker_id).take(999_999_999)  # effectively infinite

    chunk_path = os.path.join(output_dir, f"chunk_{worker_id:02d}.bin")
    f_out = open(chunk_path, "wb")

    buffer = []
    my_tokens = 0
    my_docs = 0
    flush_size = 5_000_000  # flush every 5M tokens
    t0 = time.time()

    step = 0
    for sample in shard:
        step += 1
        # Manual sharding: only process every num_workers-th sample
        if step % num_workers != 0:
            continue

        text = sample.get("text", "").strip()
        if len(text) < 50:
            continue

        tokens = tokenizer.encode(text).ids
        buffer.extend(tokens)
        my_tokens += len(tokens)
        my_docs += 1

        if len(buffer) >= flush_size:
            chunk = np.array(buffer, dtype=np.uint16)
            chunk.tofile(f_out)
            buffer = []

            # Update shared counter
            with lock:
                shared_counter.value += flush_size

            elapsed = time.time() - t0
            rate = my_tokens / elapsed / 1e6
            print(f"  [Worker {worker_id}] {my_tokens/1e6:.0f}M tok | "
                  f"{my_docs:,} docs | {rate:.2f}M tok/s", flush=True)

        if my_tokens >= tokens_per_worker:
            break

    # Flush remaining
    if buffer:
        chunk = np.array(buffer, dtype=np.uint16)
        chunk.tofile(f_out)
        with lock:
            shared_counter.value += len(buffer)

    f_out.close()
    elapsed = time.time() - t0
    print(f"  [Worker {worker_id}] DONE — {my_tokens/1e6:.1f}M tokens in {elapsed/60:.1f}min", flush=True)


def merge_chunks(output_dir, num_workers, val_ratio=0.005):
    """Merge worker chunk files into train.bin / val.bin."""
    print(f"\n[STEP 3] Merging {num_workers} chunk files...", flush=True)

    all_data = []
    total = 0
    for i in range(num_workers):
        chunk_path = os.path.join(output_dir, f"chunk_{i:02d}.bin")
        if os.path.exists(chunk_path):
            data = np.fromfile(chunk_path, dtype=np.uint16)
            all_data.append(data)
            total += len(data)
            print(f"  Chunk {i}: {len(data)/1e6:.1f}M tokens", flush=True)

    merged = np.concatenate(all_data)
    print(f"  Total: {len(merged):,} tokens ({merged.nbytes/1e9:.2f} GB)", flush=True)

    # Shuffle at document boundaries would be ideal, but random shuffle of
    # the merged array is good enough for pre-training
    # (each chunk comes from a different part of C4 anyway)

    # Split
    split = int(len(merged) * (1 - val_ratio))

    train_path = os.path.join(output_dir, "train.bin")
    val_path = os.path.join(output_dir, "val.bin")

    merged[:split].tofile(train_path)
    print(f"  Train: {split:,} tokens ({split*2/1e9:.2f} GB) → {train_path}", flush=True)

    merged[split:].tofile(val_path)
    val_count = len(merged) - split
    print(f"  Val:   {val_count:,} tokens ({val_count*2/1e6:.1f} MB) → {val_path}", flush=True)

    # Cleanup chunks
    for i in range(num_workers):
        chunk_path = os.path.join(output_dir, f"chunk_{i:02d}.bin")
        if os.path.exists(chunk_path):
            os.remove(chunk_path)
    print(f"  Cleaned up chunk files", flush=True)

    return len(merged)


def main():
    parser = argparse.ArgumentParser(description="BUVN-2.0 Parallel Streaming Pipeline")
    parser.add_argument("--dataset", type=str, default="allenai/c4")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/tokenizer_32k.json")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--target_tokens", type=int, default=2_000_000_000)
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--val_ratio", type=float, default=0.005)
    args = parser.parse_args()

    tokens_per_worker = args.target_tokens // args.num_workers

    print("=" * 65, flush=True)
    print("  BUVN-2.0 PARALLEL STREAMING PIPELINE", flush=True)
    print("=" * 65, flush=True)
    print(f"  Dataset:       {args.dataset} (streaming, no download)", flush=True)
    print(f"  Workers:       {args.num_workers} parallel streams", flush=True)
    print(f"  Target:        {args.target_tokens/1e9:.1f}B tokens total", flush=True)
    print(f"  Per worker:    {tokens_per_worker/1e6:.0f}M tokens each", flush=True)
    print(f"  Tokenizer:     {args.tokenizer}", flush=True)
    print(f"  Output:        {args.output_dir}/", flush=True)
    print("=" * 65, flush=True)

    if not os.path.exists(args.tokenizer):
        print(f"ERROR: Tokenizer not found at {args.tokenizer}", flush=True)
        print(f"Run prepare_stream.py first to train the tokenizer.", flush=True)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    t0 = time.time()

    # Shared progress counter
    shared_counter = Value('l', 0)
    lock = Lock()

    # Launch workers
    print(f"\n[STEP 1] Launching {args.num_workers} parallel stream workers...", flush=True)
    workers = []
    for i in range(args.num_workers):
        p = Process(target=worker_stream, args=(
            i, args.num_workers, args.dataset, args.tokenizer,
            args.output_dir, tokens_per_worker, shared_counter, lock
        ))
        p.start()
        workers.append(p)
        print(f"  Started worker {i} (PID {p.pid})", flush=True)

    # Monitor progress
    print(f"\n[STEP 2] Streaming & tokenizing...", flush=True)
    while any(w.is_alive() for w in workers):
        time.sleep(15)
        with lock:
            done = shared_counter.value
        elapsed = time.time() - t0
        rate = done / elapsed / 1e6 if elapsed > 0 else 0
        pct = done / args.target_tokens * 100
        eta = (args.target_tokens - done) / (done / elapsed) / 60 if done > 0 else 0
        # Check disk
        disk_used = sum(
            os.path.getsize(os.path.join(args.output_dir, f"chunk_{i:02d}.bin"))
            for i in range(args.num_workers)
            if os.path.exists(os.path.join(args.output_dir, f"chunk_{i:02d}.bin"))
        ) / 1e6
        print(f"  [{pct:5.1f}%] {done/1e6:,.0f}M / {args.target_tokens/1e6:,.0f}M tok | "
              f"{rate:.2f}M tok/s | disk: {disk_used:,.0f}MB | ETA: {eta:.0f}min",
              flush=True)

    # Wait for all to finish
    for w in workers:
        w.join()

    elapsed = time.time() - t0
    print(f"\n  All workers done in {elapsed/60:.1f} min", flush=True)

    # Merge chunks
    total = merge_chunks(args.output_dir, args.num_workers, args.val_ratio)

    total_time = time.time() - t0
    print(f"\n{'='*65}", flush=True)
    print(f"  COMPLETE in {total_time/60:.1f} min", flush=True)
    print(f"  Total: {total/1e9:.2f}B tokens", flush=True)
    print(f"  Effective rate: {total/total_time/1e6:.2f}M tok/s", flush=True)
    print(f"  Files: {args.output_dir}/train.bin + val.bin", flush=True)
    print(f"{'='*65}", flush=True)


if __name__ == "__main__":
    main()
