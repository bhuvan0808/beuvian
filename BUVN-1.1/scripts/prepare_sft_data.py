"""
BUVN SFT Data Preparation — Parallel Processing
=================================================
Downloads Alpaca + OpenAssistant, converts to chat format,
tokenizes, and saves as binary. Uses multiprocessing for speed.

Usage:
    python scripts/prepare_sft_data.py
"""

import os
import sys
import time
import json
import numpy as np
from multiprocessing import Pool, cpu_count
from datasets import load_dataset
from tokenizers import Tokenizer

# Chat template tokens
USER_START = "<|user|>"
USER_END = "<|end|>"
ASST_START = "<|assistant|>"
ASST_END = "<|end|>"


def format_alpaca(sample):
    """Convert Alpaca format to chat format."""
    instruction = sample['instruction'].strip()
    inp = sample.get('input', '').strip()
    output = sample['output'].strip()

    if inp:
        prompt = f"{instruction}\n{inp}"
    else:
        prompt = instruction

    return f"{USER_START}\n{prompt}\n{USER_END}\n{ASST_START}\n{output}\n{ASST_END}\n"


def extract_oasst_pairs(dataset):
    """Extract (prompter, assistant) pairs from OASST conversation trees."""
    # Index messages by id
    msg_by_id = {}
    for row in dataset:
        msg_by_id[row['message_id']] = row

    pairs = []
    for row in dataset:
        if row['role'] == 'assistant' and row['lang'] == 'en' and row['parent_id']:
            parent = msg_by_id.get(row['parent_id'])
            if parent and parent['role'] == 'prompter':
                # Check quality: only use if not deleted and has reviews
                if not row.get('deleted', False) and not parent.get('deleted', False):
                    pairs.append({
                        'prompt': parent['text'].strip(),
                        'response': row['text'].strip(),
                        'rank': row.get('rank', 99),
                    })

    # Keep only rank 0 (best) responses when multiple exist
    # Group by prompt
    by_prompt = {}
    for p in pairs:
        key = p['prompt'][:200]  # rough dedup key
        if key not in by_prompt or (p['rank'] or 99) < (by_prompt[key]['rank'] or 99):
            by_prompt[key] = p

    return list(by_prompt.values())


def format_oasst_pair(pair):
    """Convert OASST pair to chat format."""
    return f"{USER_START}\n{pair['prompt']}\n{USER_END}\n{ASST_START}\n{pair['response']}\n{ASST_END}\n"


def tokenize_batch(args):
    """Worker function for parallel tokenization."""
    texts, tokenizer_path = args
    tokenizer = Tokenizer.from_file(tokenizer_path)
    all_tokens = []
    for text in texts:
        tokens = tokenizer.encode(text).ids
        all_tokens.extend(tokens)
    return all_tokens


def main():
    tokenizer_path = "tokenizer/tokenizer_32k.json"
    output_dir = "data/sft"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(tokenizer_path):
        print(f"ERROR: Tokenizer not found at {tokenizer_path}")
        sys.exit(1)

    print("=" * 60)
    print("  BUVN SFT DATA PREPARATION")
    print("=" * 60)
    t0 = time.time()

    # ─── Step 1: Load datasets ────────────────────────────────────────
    print("\n[1/4] Loading datasets...")

    print("  Loading Alpaca...")
    alpaca = load_dataset('tatsu-lab/alpaca', split='train')
    print(f"  Alpaca: {len(alpaca):,} examples")

    print("  Loading OpenAssistant...")
    oasst = load_dataset('OpenAssistant/oasst2', split='train')
    print(f"  OpenAssistant: {len(oasst):,} messages")

    # ─── Step 2: Format into chat template ────────────────────────────
    print("\n[2/4] Formatting to chat template...")

    # Alpaca
    alpaca_texts = []
    for sample in alpaca:
        text = format_alpaca(sample)
        if len(text) > 50:
            alpaca_texts.append(text)
    print(f"  Alpaca: {len(alpaca_texts):,} formatted examples")

    # OpenAssistant — extract English prompter-assistant pairs
    print("  Extracting OASST pairs (English only, best rank)...")
    oasst_pairs = extract_oasst_pairs(oasst)
    oasst_texts = [format_oasst_pair(p) for p in oasst_pairs]
    print(f"  OpenAssistant: {len(oasst_texts):,} pairs extracted")

    # Combine and shuffle
    import random
    random.seed(42)
    all_texts = alpaca_texts + oasst_texts
    random.shuffle(all_texts)
    print(f"  Total: {len(all_texts):,} training examples")

    # Show sample
    print(f"\n  Sample:\n  {all_texts[0][:200]}...")

    # ─── Step 3: Parallel tokenization ────────────────────────────────
    print(f"\n[3/4] Tokenizing with {min(8, cpu_count())} parallel workers...")

    # Split texts into chunks for parallel processing
    n_workers = min(8, cpu_count())
    chunk_size = len(all_texts) // n_workers + 1
    chunks = []
    for i in range(n_workers):
        start = i * chunk_size
        end = min(start + chunk_size, len(all_texts))
        if start < len(all_texts):
            chunks.append((all_texts[start:end], tokenizer_path))

    t1 = time.time()
    with Pool(n_workers) as pool:
        results = pool.map(tokenize_batch, chunks)

    # Merge results
    all_tokens = []
    for r in results:
        all_tokens.extend(r)

    print(f"  Tokenized {len(all_tokens):,} tokens in {time.time()-t1:.1f}s")

    # ─── Step 4: Save as binary ───────────────────────────────────────
    print(f"\n[4/4] Saving binary files...")

    # Split 95/5 train/val
    split = int(len(all_tokens) * 0.95)
    train_tokens = all_tokens[:split]
    val_tokens = all_tokens[split:]

    train_arr = np.array(train_tokens, dtype=np.uint16)
    train_path = os.path.join(output_dir, "train.bin")
    train_arr.tofile(train_path)
    print(f"  Train: {len(train_tokens):,} tokens ({train_arr.nbytes/1e6:.1f} MB) → {train_path}")

    val_arr = np.array(val_tokens, dtype=np.uint16)
    val_path = os.path.join(output_dir, "val.bin")
    val_arr.tofile(val_path)
    print(f"  Val:   {len(val_tokens):,} tokens ({val_arr.nbytes/1e6:.1f} MB) → {val_path}")

    # Save metadata
    meta = {
        "total_examples": len(all_texts),
        "alpaca_examples": len(alpaca_texts),
        "oasst_examples": len(oasst_texts),
        "total_tokens": len(all_tokens),
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "chat_template": {
            "user_start": USER_START,
            "user_end": USER_END,
            "assistant_start": ASST_START,
            "assistant_end": ASST_END,
        }
    }
    with open(os.path.join(output_dir, "sft_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  COMPLETE in {total_time:.0f}s")
    print(f"  Examples: {len(all_texts):,} ({len(alpaca_texts):,} Alpaca + {len(oasst_texts):,} OASST)")
    print(f"  Tokens: {len(all_tokens):,}")
    print(f"  Files: {output_dir}/train.bin + val.bin")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
