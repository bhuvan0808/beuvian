"""
BUVN-2.0 Streaming Data Pipeline
==================================
True streaming: HuggingFace → tokenize in memory → write chunks to disk.
No raw text saved. No full dataset in RAM. Writes binary incrementally.

Usage:
    python scripts/prepare_stream.py
    python scripts/prepare_stream.py --target_tokens 2_000_000_000 --vocab_size 32000
"""

import os
import sys
import time
import argparse
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders


# ─── Step 1: Train Tokenizer ─────────────────────────────────────────────────

def train_tokenizer(dataset_name, vocab_size=32000, sample_size=100_000):
    """Train BPE tokenizer on streamed sample. Keeps sample_size texts in RAM only."""
    print(f"\n[STEP 1] Training {vocab_size:,}-vocab BPE tokenizer")
    print(f"         Streaming {sample_size:,} samples from {dataset_name}...")

    ds = load_dataset(dataset_name, "en", split="train", streaming=True)

    # Collect texts for tokenizer training (only these stay in RAM)
    texts = []
    t0 = time.time()
    for sample in ds:
        text = sample.get("text", "").strip()
        if len(text) < 50:
            continue
        texts.append(text)
        if len(texts) % 25_000 == 0:
            print(f"         Collected {len(texts):,} / {sample_size:,} ({time.time()-t0:.0f}s)")
        if len(texts) >= sample_size:
            break

    print(f"         Collected {len(texts):,} texts in {time.time()-t0:.0f}s")

    # Build tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<s>", "</s>"],
        show_progress=True,
        min_frequency=2,
    )

    print(f"         Training BPE...")
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # EOS only (no BOS per line — we concatenate documents with EOS separator)
    eos_id = tokenizer.token_to_id("</s>")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="$A </s>",
        special_tokens=[("</s>", eos_id)],
    )

    # Free the text samples
    del texts

    # Quick test
    test = "The transformer architecture was introduced in 2017."
    enc = tokenizer.encode(test)
    print(f"         Vocab: {tokenizer.get_vocab_size():,} | "
          f"Test: '{test}' → {len(enc.ids)} tokens")

    return tokenizer


# ─── Step 2: Stream → Tokenize → Write Chunks ────────────────────────────────

def stream_tokenize_to_binary(
    dataset_name, tokenizer, output_dir,
    target_tokens=2_000_000_000, val_ratio=0.005,
    chunk_size=10_000_000,  # write every 10M tokens (~20MB)
):
    """
    True streaming pipeline:
    1. Stream text from HuggingFace (no download)
    2. Tokenize each document in memory
    3. Accumulate tokens in a buffer
    4. When buffer hits chunk_size, flush to disk
    5. Final split into train.bin / val.bin
    """
    print(f"\n[STEP 2] Streaming → tokenizing → writing binary")
    print(f"         Target: {target_tokens/1e9:.1f}B tokens")
    print(f"         Chunk size: {chunk_size/1e6:.0f}M tokens per flush")

    os.makedirs(output_dir, exist_ok=True)
    temp_path = os.path.join(output_dir, "tokens_temp.bin")

    # Open temp file for incremental writing
    f_out = open(temp_path, "wb")

    ds = load_dataset(dataset_name, "en", split="train", streaming=True)

    buffer = []
    total_tokens = 0
    total_docs = 0
    skipped = 0
    t0 = time.time()
    last_log = t0

    for sample in ds:
        text = sample.get("text", "").strip()
        if len(text) < 50:
            skipped += 1
            continue

        # Tokenize in memory (no raw text stored)
        tokens = tokenizer.encode(text).ids
        buffer.extend(tokens)
        total_tokens += len(tokens)
        total_docs += 1

        # Flush buffer to disk when it's big enough
        if len(buffer) >= chunk_size:
            chunk = np.array(buffer, dtype=np.uint16)
            chunk.tofile(f_out)
            buffer = []

            # Log progress
            elapsed = time.time() - t0
            tps = total_tokens / elapsed
            eta = (target_tokens - total_tokens) / tps if tps > 0 else 0
            pct = total_tokens / target_tokens * 100
            disk_mb = os.path.getsize(temp_path) / 1e6
            print(f"         {pct:5.1f}% | {total_tokens/1e6:,.0f}M tok | "
                  f"{total_docs:,} docs | {tps/1e6:.2f}M tok/s | "
                  f"disk: {disk_mb:,.0f}MB | ETA: {eta/60:.0f}min")

        # Check target
        if total_tokens >= target_tokens:
            break

    # Flush remaining buffer
    if buffer:
        chunk = np.array(buffer, dtype=np.uint16)
        chunk.tofile(f_out)
        buffer = []

    f_out.close()

    elapsed = time.time() - t0
    print(f"\n         Done! {total_tokens:,} tokens | {total_docs:,} docs | "
          f"{elapsed/60:.1f}min | {total_tokens/elapsed/1e6:.2f}M tok/s")

    # ─── Split into train/val ─────────────────────────────────────────────
    print(f"\n[STEP 3] Splitting train/val ({val_ratio*100:.1f}%)")

    all_data = np.memmap(temp_path, dtype=np.uint16, mode='r')
    n = len(all_data)
    split = int(n * (1 - val_ratio))

    train_path = os.path.join(output_dir, "train.bin")
    val_path = os.path.join(output_dir, "val.bin")

    # Write train
    train_data = np.array(all_data[:split], dtype=np.uint16)
    train_data.tofile(train_path)
    print(f"         Train: {len(train_data):,} tokens ({train_data.nbytes/1e9:.2f} GB) → {train_path}")

    # Write val
    val_data = np.array(all_data[split:], dtype=np.uint16)
    val_data.tofile(val_path)
    print(f"         Val:   {len(val_data):,} tokens ({val_data.nbytes/1e6:.1f} MB) → {val_path}")

    # Cleanup temp file
    del all_data, train_data, val_data
    os.remove(temp_path)
    print(f"         Removed temp file")

    return total_tokens


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BUVN-2.0 Streaming Data Pipeline")
    parser.add_argument("--dataset", type=str, default="allenai/c4",
                        help="HuggingFace dataset (must support streaming)")
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--tokenizer_samples", type=int, default=100_000,
                        help="Samples for tokenizer training")
    parser.add_argument("--target_tokens", type=int, default=2_000_000_000,
                        help="Total tokens to collect")
    parser.add_argument("--tokenizer_out", type=str, default="tokenizer/tokenizer_32k.json")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--val_ratio", type=float, default=0.005)
    args = parser.parse_args()

    print("=" * 65)
    print("  BUVN-2.0 STREAMING DATA PIPELINE")
    print("=" * 65)
    print(f"  Dataset:       {args.dataset} (streaming, no download)")
    print(f"  Vocab:         {args.vocab_size:,} BPE tokens")
    print(f"  Target:        {args.target_tokens/1e9:.1f}B tokens")
    print(f"  Tokenizer out: {args.tokenizer_out}")
    print(f"  Data out:      {args.output_dir}/train.bin + val.bin")
    print("=" * 65)

    t_start = time.time()

    # Step 1: Train tokenizer on streamed sample
    tokenizer = train_tokenizer(args.dataset, args.vocab_size, args.tokenizer_samples)
    os.makedirs(os.path.dirname(args.tokenizer_out), exist_ok=True)
    tokenizer.save(args.tokenizer_out)
    print(f"         Saved tokenizer → {args.tokenizer_out}")

    # Step 2-3: Stream, tokenize, write binary, split
    total = stream_tokenize_to_binary(
        args.dataset, tokenizer, args.output_dir,
        args.target_tokens, args.val_ratio
    )

    total_time = time.time() - t_start
    print(f"\n{'='*65}")
    print(f"  COMPLETE in {total_time/60:.1f} min")
    print(f"  Tokens: {total/1e9:.2f}B | Tokenizer: {args.tokenizer_out}")
    print(f"  Data: {args.output_dir}/train.bin + val.bin")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
