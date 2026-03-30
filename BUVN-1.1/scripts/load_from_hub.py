"""
Download BUVN model weights from HuggingFace Hub.
Restores checkpoint, tokenizer, and config to local disk.

Usage:
    python scripts/load_from_hub.py
    python scripts/load_from_hub.py --repo bhuvan0808/buvn-2.0 --output_dir checkpoints
"""

import os
import argparse
from huggingface_hub import hf_hub_download


def load_from_hub(repo_id, output_dir, tokenizer_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tokenizer_dir, exist_ok=True)

    print(f"Downloading from: https://huggingface.co/{repo_id}")

    # Download checkpoint
    print("Downloading checkpoint (1.3 GB)...")
    ckpt_path = hf_hub_download(
        repo_id=repo_id,
        filename='buvn_2.0_best.pt',
        local_dir=output_dir,
    )
    print(f"  Saved to: {ckpt_path}")

    # Download tokenizer
    print("Downloading tokenizer...")
    tok_path = hf_hub_download(
        repo_id=repo_id,
        filename='tokenizer_32k.json',
        local_dir=tokenizer_dir,
    )
    print(f"  Saved to: {tok_path}")

    # Download config
    print("Downloading config...")
    config_path = hf_hub_download(
        repo_id=repo_id,
        filename='config.json',
        local_dir=output_dir,
    )
    print(f"  Saved to: {config_path}")

    print(f"\nDone! Files downloaded to {output_dir}/ and {tokenizer_dir}/")
    print(f"\nTo use the model:")
    print(f"  python inference/generate.py \\")
    print(f"    --prompt 'Your text here' \\")
    print(f"    --checkpoint {os.path.join(output_dir, 'buvn_2.0_best.pt')} \\")
    print(f"    --tokenizer {os.path.join(tokenizer_dir, 'tokenizer_32k.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download BUVN model from HuggingFace Hub")
    parser.add_argument("--repo", type=str, default="bhuvan0808/buvn-2.0")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--tokenizer_dir", type=str, default="tokenizer")
    args = parser.parse_args()

    load_from_hub(args.repo, args.output_dir, args.tokenizer_dir)
