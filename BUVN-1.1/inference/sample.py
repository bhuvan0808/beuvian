import torch
from torch.nn import functional as F

def sample_top_k(logits, top_k):
    """
    Keep only the top k tokens with highest probability.
    Used for inference generation.
    """
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')
    return logits

@torch.no_grad()
def generate(model, tokenizer, prompt_str, max_new_tokens, temperature=1.0, top_k=None, device='cpu'):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    """
    model.eval()
    
    # 1. Tokenize prompt
    prompt_tokens = tokenizer.encode(prompt_str, out_type=int)
    idx = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    
    # 2. Store context lengths
    input_len = idx.size(1)
    
    token_usage = {
        "prompt_tokens": input_len,
        "completion_tokens": 0,
        "total_tokens": input_len
    }

    # 3. Generation loop
    for _ in range(max_new_tokens):
        # crop ctx if growing too large (for bounded max_seq_len)
        idx_cond = idx if idx.size(1) <= model.config.max_seq_len else idx[:, -model.config.max_seq_len:]
        
        # forward
        logits, _ = model(idx_cond)
        
        # sample at the final step
        logits = logits[:, -1, :] / temperature
        
        # option to only select from top_k
        if top_k is not None:
            logits = sample_top_k(logits, top_k)
            
        # apply softmax
        probs = F.softmax(logits, dim=-1)
        
        # sample next token
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # append sampled index to sequence
        idx = torch.cat((idx, idx_next), dim=1)
        token_usage["completion_tokens"] += 1
        token_usage["total_tokens"] += 1
        
        # Check EOS
        if idx_next.item() == tokenizer.eos_id():
            break

    # 4. Decode
    generated_tokens = idx[0].tolist()[input_len:]
    generated_text = tokenizer.decode(generated_tokens)
    
    return generated_text, token_usage
