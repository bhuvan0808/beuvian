"""
Train a BPE tokenizer using HuggingFace tokenizers library.
This avoids the SentencePiece C++ crashes on some Windows environments.

Usage:
    python scripts/train_hf_tokenizer.py
    python scripts/train_hf_tokenizer.py --corpus data/processed/corpus.txt --vocab_size 4000
"""

import os
import argparse
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders


def train_tokenizer(corpus_path: str, output_path: str, vocab_size: int = 4000):
    """
    Trains a Byte-Level BPE tokenizer on a text corpus and saves it as a JSON file.
    """
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    # 1. Create a BPE tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # 2. Train on the corpus
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<s>", "</s>"],
        show_progress=True,
    )
    print(f"Training BPE tokenizer on {corpus_path} (vocab_size={vocab_size})...")
    tokenizer.train([corpus_path], trainer)

    # 3. Add BOS/EOS post-processing
    bos_id = tokenizer.token_to_id("<s>")
    eos_id = tokenizer.token_to_id("</s>")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[("<s>", bos_id), ("</s>", eos_id)],
    )

    # 4. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tokenizer.save(output_path)
    print(f"Tokenizer saved to: {output_path}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")

    # 5. Quick test
    test_text = "AI is transforming the world."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded.ids)
    print(f"\nTest encode: '{test_text}' -> {encoded.ids[:20]}...")
    print(f"Test decode: '{decoded}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer")
    parser.add_argument("--corpus", type=str, default="data/processed/corpus.txt")
    parser.add_argument("--output", type=str, default="tokenizer/tokenizer.json")
    parser.add_argument("--vocab_size", type=int, default=4000)
    args = parser.parse_args()

    train_tokenizer(args.corpus, args.output, args.vocab_size)
