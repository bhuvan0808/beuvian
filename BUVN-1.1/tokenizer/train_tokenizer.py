import os
import sentencepiece as spm
from typing import Optional

def train_tokenizer(
    input_file: str, 
    model_prefix: str = "BUVN-1.1/tokenizer/tokenizer",
    vocab_size: int = 50000,
    character_coverage: float = 1.0,
    model_type: str = "bpe",
):
    """
    Trains a SentencePiece tokenizer from a given text file.
    
    Args:
        input_file: Path to the raw text corpus file (e.g., train.txt)
        model_prefix: Where to output the trained tokenizer (.model and .vocab)
        vocab_size: Number of tokens, default 50k
        character_coverage: Proportion of characters to cover
        model_type: Type of model (bpe, unigram, char, word)
    """
    print(f"Training {model_type.upper()} tokenizer on {input_file}...")
    print(f"Target vocab size: {vocab_size}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(model_prefix), exist_ok=True)

    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece='<pad>',
        unk_piece='<unk>',
        bos_piece='<s>',
        eos_piece='</s>',
        user_defined_symbols=[],
        # To make things fast enough on small-scale
        train_extremely_large_corpus=True
    )

    print(f"Tokenizer trained. Saved as {model_prefix}.model and {model_prefix}.vocab")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train SentencePiece Tokenizer")
    parser.add_argument("--input_file", type=str, required=True, help="Path to plain text training corpus.")
    parser.add_argument("--model_prefix", type=str, default="BUVN-1.1/tokenizer/tokenizer", help="Output prefix.")
    parser.add_argument("--vocab_size", type=int, default=50000)
    args = parser.parse_args()

    train_tokenizer(args.input_file, args.model_prefix, args.vocab_size)
