"""
BUVN-2.0 Data Pipeline — FineWeb-Edu Streaming
================================================
Streams FineWeb-Edu from HuggingFace, trains a 32K BPE tokenizer,
and tokenizes directly to binary (train.bin / val.bin).

No raw text stored on disk — only the tokenized binary output.

Usage:
    python scripts/prepare_fineweb.py
    python scripts/prepare_fineweb.py --target_tokens 2_000_000_000 --vocab_size 32000
"""

import os
import sys
import time
import argparse
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders


def stream_texts(dataset_iter, max_texts=500_000, min_length=100):
    """Yields clean text strings from the dataset iterator."""
    count = 0
    for sample in dataset_iter:
        text = sample.get("text", "").strip()
        if len(text) < min_length:
            continue
        yield text
        count += 1
        if count >= max_texts:
            break


def train_tokenizer_from_stream(dataset_name, vocab_size=32000, sample_size=200_000):
    """Train a BPE tokenizer on a sample of streamed data."""
    print(f"[1/3] Training {vocab_size}-vocab BPE tokenizer on {sample_size:,} samples...")
    print(f"      Streaming from {dataset_name}...")

    dataset = load_dataset(dataset_name, split="train", streaming=True)

    # Collect sample texts for tokenizer training
    texts = []
    count = 0
    t0 = time.time()
    for sample in dataset:
        text = sample.get("text", "").strip()
        if len(text) < 100:
            continue
        texts.append(text)
        count += 1
        if count % 50_000 == 0:
            print(f"      Collected {count:,} samples ({time.time()-t0:.0f}s)")
        if count >= sample_size:
            break

    print(f"      Collected {len(texts):,} texts in {time.time()-t0:.0f}s")

    # Train BPE tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<s>", "</s>"],
        show_progress=True,
        min_frequency=2,
    )

    # Train from iterator (memory efficient)
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # Add EOS post-processing (no BOS — we don't want BOS per line)
    eos_id = tokenizer.token_to_id("</s>")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="$A </s>",
        special_tokens=[("</s>", eos_id)],
    )

    # Test
    test_text = "Artificial intelligence is transforming the world."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded.ids)
    print(f"      Vocab size: {tokenizer.get_vocab_size():,}")
    print(f"      Test: '{test_text}' → {len(encoded.ids)} tokens")
    print(f"      Decoded: '{decoded}'")

    return tokenizer


def tokenize_stream_to_binary(
    dataset_name,
    tokenizer,
    output_dir,
    target_tokens=2_000_000_000,
    val_ratio=0.005,
):
    """Stream data, tokenize in memory, write directly to binary files."""
    print(f"\n[2/3] Streaming & tokenizing to {target_tokens/1e9:.1f}B tokens...")
    print(f"      Streaming from {dataset_name}...")

    os.makedirs(output_dir, exist_ok=True)

    # Stream a fresh iterator
    dataset = load_dataset(dataset_name, split="train", streaming=True)

    all_tokens = []
    total_tokens = 0
    lines_processed = 0
    lines_skipped = 0
    t0 = time.time()
    last_log = t0

    for sample in dataset:
        text = sample.get("text", "").strip()
        if len(text) < 100:
            lines_skipped += 1
            continue

        encoded = tokenizer.encode(text)
        tokens = encoded.ids
        all_tokens.extend(tokens)
        total_tokens += len(tokens)
        lines_processed += 1

        # Log progress
        now = time.time()
        if now - last_log > 10:  # log every 10 seconds
            elapsed = now - t0
            tps = total_tokens / elapsed
            eta = (target_tokens - total_tokens) / tps if tps > 0 else 0
            print(f"      {total_tokens/1e6:.1f}M tokens | {lines_processed:,} lines | "
                  f"{tps/1e6:.2f}M tok/s | ETA: {eta/60:.0f}min")
            last_log = now

        # Write in chunks to avoid memory issues (every 50M tokens)
        if len(all_tokens) > 50_000_000:
            # Flush to temp array
            pass  # We'll write all at once at the end for simplicity

        if total_tokens >= target_tokens:
            print(f"      Reached target: {total_tokens/1e6:.1f}M tokens")
            break

    elapsed = time.time() - t0
    print(f"      Done! {total_tokens:,} tokens from {lines_processed:,} lines in {elapsed/60:.1f}min")
    print(f"      Skipped {lines_skipped:,} short lines")

    # Split train/val
    split_idx = int(len(all_tokens) * (1 - val_ratio))
    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:]

    print(f"      Train: {len(train_tokens):,} tokens")
    print(f"      Val:   {len(val_tokens):,} tokens")

    # Validate no overflow for uint16
    max_token = max(all_tokens)
    assert max_token < 65536, f"Token ID {max_token} exceeds uint16 range! Use uint32."

    # Save as uint16 binary
    print(f"\n[3/3] Saving binary files...")

    train_arr = np.array(train_tokens, dtype=np.uint16)
    train_path = os.path.join(output_dir, "train.bin")
    train_arr.tofile(train_path)
    print(f"      Saved {train_path} ({train_arr.nbytes / 1e9:.2f} GB)")

    val_arr = np.array(val_tokens, dtype=np.uint16)
    val_path = os.path.join(output_dir, "val.bin")
    val_arr.tofile(val_path)
    print(f"      Saved {val_path} ({val_arr.nbytes / 1e6:.1f} MB)")

    del all_tokens, train_tokens, val_tokens  # free memory
    return total_tokens


def main():
    parser = argparse.ArgumentParser(description="BUVN-2.0 FineWeb-Edu Data Pipeline")
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu",
                        help="HuggingFace dataset name")
    parser.add_argument("--vocab_size", type=int, default=32000,
                        help="BPE tokenizer vocabulary size")
    parser.add_argument("--tokenizer_samples", type=int, default=200_000,
                        help="Number of samples for tokenizer training")
    parser.add_argument("--target_tokens", type=int, default=2_000_000_000,
                        help="Target number of tokens to tokenize")
    parser.add_argument("--tokenizer_out", type=str, default="tokenizer/tokenizer_32k.json")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--val_ratio", type=float, default=0.005)
    args = parser.parse_args()

    print("=" * 60)
    print("  BUVN-2.0 Data Pipeline — FineWeb-Edu Streaming")
    print("=" * 60)
    print(f"  Dataset:       {args.dataset}")
    print(f"  Vocab size:    {args.vocab_size:,}")
    print(f"  Target tokens: {args.target_tokens/1e9:.1f}B")
    print(f"  Output:        {args.output_dir}/")
    print("=" * 60)

    t_start = time.time()

    # Step 1: Train tokenizer
    tokenizer = train_tokenizer_from_stream(
        args.dataset, args.vocab_size, args.tokenizer_samples
    )

    # Save tokenizer
    os.makedirs(os.path.dirname(args.tokenizer_out), exist_ok=True)
    tokenizer.save(args.tokenizer_out)
    print(f"      Tokenizer saved to {args.tokenizer_out}")

    # Step 2-3: Stream, tokenize, save binary
    total_tokens = tokenize_stream_to_binary(
        args.dataset, tokenizer, args.output_dir,
        args.target_tokens, args.val_ratio
    )

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {total_time/60:.1f} minutes")
    print(f"  Total tokens: {total_tokens/1e9:.2f}B")
    print(f"  Tokenizer: {args.tokenizer_out}")
    print(f"  Data: {args.output_dir}/train.bin + val.bin")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
