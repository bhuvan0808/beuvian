"""
BUVN-1.1 Data Preparation Script
=================================
Builds a clean text corpus from WikiText-103 using Hugging Face streaming.
No full dataset download required — streams data sample-by-sample and writes
incrementally to disk, stopping once the target file size is reached.

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --max_size_mb 120 --min_length 100
"""

import os
import argparse
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_file_size_mb(path: str) -> float:
    """Returns the file size in megabytes."""
    if not os.path.exists(path):
        return 0.0
    return os.path.getsize(path) / (1024 * 1024)


def is_valid_line(text: str, min_length: int) -> bool:
    """
    Returns True if the text passes all quality filters.

    Filters applied:
      - Must not be empty after stripping whitespace
      - Must be at least `min_length` characters long
      - Must not start with '=' (wiki section headers like '= Title =')
    """
    text = text.strip()
    if not text:
        return False
    if len(text) < min_length:
        return False
    if text.startswith("="):
        return False
    return True


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_corpus(
    output_path: str = "BUVN-1.1/data/processed/corpus.txt",
    max_size_mb: float = 150.0,
    min_length: int = 100,
):
    """
    Streams WikiText-103 from Hugging Face, filters it, and writes clean
    text to `output_path` until the file reaches `max_size_mb`.
    """

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Load dataset in STREAMING mode (no full download)
    # -----------------------------------------------------------------------
    print("Loading WikiText-103 (streaming mode)...")
    dataset = load_dataset(
        "wikitext",
        "wikitext-103-raw-v1",
        split="train",
        streaming=True,
    )

    # -----------------------------------------------------------------------
    # 2. Stream, filter, and write incrementally
    # -----------------------------------------------------------------------
    sample_count = 0
    written_count = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in dataset:
            sample_count += 1

            # Extract and clean text
            text = sample["text"].strip()

            # Apply quality filters
            if not is_valid_line(text, min_length):
                continue

            # Write clean line to corpus
            f.write(text + "\n")
            written_count += 1

            # Progress logging every 10,000 samples
            if sample_count % 10_000 == 0:
                size_mb = get_file_size_mb(output_path)
                print(f"Processed {sample_count:,} samples | Written {written_count:,} | Size: {size_mb:.1f} MB")

            # ---------------------------------------------------------------
            # 3. Stop when file reaches target size
            # ---------------------------------------------------------------
            #    Check every 1000 written lines to avoid excessive stat calls
            if written_count % 1000 == 0:
                size_mb = get_file_size_mb(output_path)
                if size_mb >= max_size_mb:
                    print(f"\nReached target dataset size ({size_mb:.1f} MB >= {max_size_mb} MB), stopping.")
                    break

    # Final stats
    final_size = get_file_size_mb(output_path)
    print(f"\nDone! Corpus saved to: {output_path}")
    print(f"  Total samples scanned : {sample_count:,}")
    print(f"  Total lines written   : {written_count:,}")
    print(f"  Final file size       : {final_size:.1f} MB")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a clean text corpus from WikiText-103 (streaming)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="BUVN-1.1/data/processed/corpus.txt",
        help="Path to the output corpus file.",
    )
    parser.add_argument(
        "--max_size_mb",
        type=float,
        default=150.0,
        help="Stop writing when corpus reaches this size in MB (default: 150).",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=100,
        help="Minimum character length for a line to be kept (default: 100).",
    )
    args = parser.parse_args()

    build_corpus(
        output_path=args.output,
        max_size_mb=args.max_size_mb,
        min_length=args.min_length,
    )
