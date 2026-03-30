"""
Tokenize corpus.txt → train.bin / val.bin
==========================================
Reads a plain text corpus, tokenizes it with HuggingFace tokenizers,
and saves memory-mapped binary files for the training dataloader.

Usage:
    python scripts/tokenize_corpus.py
    python scripts/tokenize_corpus.py --corpus data/processed/corpus.txt --tokenizer tokenizer/tokenizer.json
"""

import os
import argparse
import numpy as np
from tokenizers import Tokenizer


def tokenize_corpus(
    corpus_path: str = "BUVN-1.1/data/processed/corpus.txt",
    tokenizer_path: str = "BUVN-1.1/tokenizer/tokenizer.json",
    output_dir: str = "BUVN-1.1/data/processed",
    val_ratio: float = 0.01,
):
    """Tokenizes a text corpus and writes train.bin + val.bin."""

    # 1. Load tokenizer
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    eos_id = tokenizer.token_to_id("</s>")
    print(f"Loaded tokenizer (vocab size: {tokenizer.get_vocab_size()}, EOS id: {eos_id})")

    # 2. Read and tokenize the corpus line by line
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    all_tokens = []
    line_count = 0

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            encoded = tokenizer.encode(line)
            tokens = encoded.ids  # already includes BOS/EOS from post-processor
            all_tokens.extend(tokens)
            line_count += 1

            if line_count % 5000 == 0:
                print(f"  Tokenized {line_count:,} lines | {len(all_tokens):,} tokens")

    print(f"\nTotal lines: {line_count:,}")
    print(f"Total tokens: {len(all_tokens):,}")

    # 3. Split into train / val
    split_idx = int(len(all_tokens) * (1 - val_ratio))
    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:]

    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Val tokens:   {len(val_tokens):,}")

    # 4. Save as uint16 binary (supports vocab up to 65535)
    os.makedirs(output_dir, exist_ok=True)

    train_arr = np.array(train_tokens, dtype=np.uint16)
    train_path = os.path.join(output_dir, "train.bin")
    train_arr.tofile(train_path)
    print(f"Saved {train_path} ({train_arr.nbytes / 1e6:.1f} MB)")

    val_arr = np.array(val_tokens, dtype=np.uint16)
    val_path = os.path.join(output_dir, "val.bin")
    val_arr.tofile(val_path)
    print(f"Saved {val_path} ({val_arr.nbytes / 1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize corpus into binary files")
    parser.add_argument("--corpus", type=str, default="data/processed/corpus.txt")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/tokenizer.json")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--val_ratio", type=float, default=0.01)
    args = parser.parse_args()

    tokenize_corpus(args.corpus, args.tokenizer, args.output_dir, args.val_ratio)
