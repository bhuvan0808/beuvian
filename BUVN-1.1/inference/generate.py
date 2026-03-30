import os
import argparse
import torch
import sentencepiece as spm

# Local imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model import BUVNModel
from model.config import BUVNConfig
from inference.sample import generate

def load_generator(checkpoint_path: str, tokenizer_path: str, device: str = 'cpu'):
    # 1. Load Tokenizer (supports .json, .pkl, and .model formats)
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
        
    if tokenizer_path.endswith('.json'):
        # HuggingFace tokenizers format
        from tokenizers import Tokenizer as HFTokenizer
        _hf_tok = HFTokenizer.from_file(tokenizer_path)
        class HFTokenizerWrapper:
            def __init__(self, tok):
                self._tok = tok
                self._eos_id = tok.token_to_id("</s>") or 3
            def encode(self, text, out_type=int):
                return self._tok.encode(text).ids
            def decode(self, tokens):
                return self._tok.decode(tokens)
            def eos_id(self):
                return self._eos_id
        tokenizer = HFTokenizerWrapper(_hf_tok)
    elif tokenizer_path.endswith('.pkl'):
        import pickle
        with open(tokenizer_path, 'rb') as f:
            meta = pickle.load(f)
        class CharTokenizer:
            def __init__(self, meta):
                self.stoi = meta['stoi']
                self.itos = meta['itos']
                self._eos_id = meta.get('eos_id', 0)
            def encode(self, text, out_type=int):
                return [self.stoi[c] for c in text if c in self.stoi]
            def decode(self, tokens):
                return ''.join([self.itos[i] for i in tokens if i in self.itos])
            def eos_id(self):
                return self._eos_id
        tokenizer = CharTokenizer(meta)
    else:
        tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
    
    # 2. Load Model
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config = BUVNConfig.from_dict(checkpoint['model_args'])
    model = BUVNModel(config)
    
    # Handle DDP prefixes if necessary
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument("--checkpoint", type=str, default="BUVN-1.1/checkpoints/ckpt.pt")
    parser.add_argument("--tokenizer", type=str, default="BUVN-1.1/tokenizer/tokenizer.model")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=200)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        model, tokenizer = load_generator(args.checkpoint, args.tokenizer, device)
        
        print(f"\nPrompt: {args.prompt}")
        print("-" * 50)
        
        text, usage = generate(
            model=model,
            tokenizer=tokenizer,
            prompt_str=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device
        )
        
        print(f"Output: {text}")
        print("-" * 50)
        print(f"Tokens Used: {usage}")
        
    except Exception as e:
        print(f"Error during generation: {e}")

if __name__ == "__main__":
    main()
