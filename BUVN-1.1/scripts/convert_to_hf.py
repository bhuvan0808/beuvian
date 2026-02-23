import os
import json
import torch
import argparse
import shutil
from transformers import PretrainedConfig, AutoConfig

def convert_checkpoint(checkpoint_path: str, output_dir: str):
    """
    Dummy conversion script to show how we would map native PyTorch parameters
    to a HuggingFace PreTrainedModel format (e.g. LLaMA/GPTNeoX format).

    In reality, you need a custom defined `configuration_buvn.py` and `modeling_buvn.py` 
    to register on the Auto classes for full native HF support.
    This script writes out the standalone PyTorch state_dict and a basic config.json 
    that could be loaded into a custom HF structure.
    """
    print(f"Loading native checkpoint from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_state = checkpoint["model"]
    model_args = checkpoint["model_args"]
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Clean the state dict
    hf_state_dict = {}
    unwanted_prefix = '_orig_mod.'
    for k, v in list(model_state.items()):
        if k.startswith(unwanted_prefix):
             k = k[len(unwanted_prefix):]
        # Mapping rules could go here. e.g. tok_embeddings -> model.embed_tokens
        hf_state_dict[k] = v
        
    # 2. Save the SafeTensors or PyTorch Bin
    out_bin = os.path.join(output_dir, "pytorch_model.bin")
    torch.save(hf_state_dict, out_bin)
    print(f"Saved model binary to {out_bin}")
    
    # 3. Save Config.json
    hf_config = {
        "architectures": [
            "BUVNForCausalLM" # Custom Architecture Name
        ],
        "bos_token_id": 2,
        "eos_token_id": 3,
        "hidden_size": model_args["d_model"],
        "num_attention_heads": model_args["n_heads"],
        "num_hidden_layers": model_args["n_layers"],
        "vocab_size": model_args["vocab_size"],
        "max_position_embeddings": model_args["max_seq_len"],
        "model_type": "buvn",
        "rope_theta": 10000.0
    }
    
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(hf_config, f, indent=2)
    print("Saved config.json")
    
    # 4. Copy Tokenizer
    tokenizer_src = "BUVN-1.1/tokenizer/tokenizer.model"
    if os.path.exists(tokenizer_src):
        shutil.copy2(tokenizer_src, os.path.join(output_dir, "tokenizer.model"))
        print("Copied tokenizer.model")
        
    print(f"\nHuggingFace format model saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="BUVN-1.1/checkpoints/ckpt.pt")
    parser.add_argument("--out_dir", type=str, default="BUVN-1.1/hf_model")
    args = parser.parse_args()
    
    convert_checkpoint(args.checkpoint, args.out_dir)
