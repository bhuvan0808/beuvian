"""
Push model weights to HuggingFace Hub.
Auto-uploads checkpoint, tokenizer, and config after training.

Usage:
    python scripts/push_to_hub.py --checkpoint checkpoints/ckpt_best.pt
    python scripts/push_to_hub.py --checkpoint checkpoints/ckpt_best.pt --repo bhuvan0808/buvn-2.0
"""

import os
import sys
import json
import argparse
import torch
from huggingface_hub import HfApi, create_repo


def push_to_hub(checkpoint_path, tokenizer_path, repo_id, token):
    api = HfApi(token=token)

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, repo_type='model', exist_ok=True, token=token)
    except Exception:
        pass  # Repo already exists or token uses fine-grained permissions

    print(f"Pushing to: https://huggingface.co/{repo_id}")

    # Upload checkpoint
    if os.path.exists(checkpoint_path):
        print(f"Uploading checkpoint ({os.path.getsize(checkpoint_path)/1e9:.2f} GB)...")
        api.upload_file(
            path_or_fileobj=checkpoint_path,
            path_in_repo='buvn_2.0_best.pt',
            repo_id=repo_id,
        )
        print("  Checkpoint uploaded.")
    else:
        print(f"  WARNING: Checkpoint not found at {checkpoint_path}")

    # Upload tokenizer
    if os.path.exists(tokenizer_path):
        print("Uploading tokenizer...")
        api.upload_file(
            path_or_fileobj=tokenizer_path,
            path_in_repo=os.path.basename(tokenizer_path),
            repo_id=repo_id,
        )
        print("  Tokenizer uploaded.")

    # Upload config
    if os.path.exists(checkpoint_path):
        print("Uploading config...")
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        config_json = json.dumps(ckpt['model_args'], indent=2)
        api.upload_file(
            path_or_fileobj=config_json.encode(),
            path_in_repo='config.json',
            repo_id=repo_id,
        )
        print("  Config uploaded.")

        # Print training info
        print(f"\n  Model args: {ckpt['model_args']}")
        print(f"  Trained for: {ckpt.get('iter_num', '?')} steps")
        print(f"  Best val loss: {ckpt.get('best_val_loss', '?')}")

    # Upload model card if exists
    card_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'hf_model_card.md')
    if os.path.exists(card_path):
        print("Uploading model card...")
        api.upload_file(path_or_fileobj=card_path, path_in_repo='README.md', repo_id=repo_id)

    print(f"\nDone! Model available at: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push BUVN model to HuggingFace Hub")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/ckpt_best.pt")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/tokenizer_32k.json")
    parser.add_argument("--repo", type=str, default="bhuvan0808/buvn-2.0")
    parser.add_argument("--token", type=str, default=None, help="HF token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: Provide --token or set HF_TOKEN environment variable")
        sys.exit(1)

    push_to_hub(args.checkpoint, args.tokenizer, args.repo, token)
