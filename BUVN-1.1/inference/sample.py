import torch
from torch.nn import functional as F


def sample_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Keep only the top k tokens with highest probability."""
    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        v, _ = torch.topk(logits, top_k)
        # Clone to avoid in-place modification
        logits = logits.clone()
        logits[logits < v[:, [-1]]] = -float('Inf')
    return logits


def sample_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus sampling: keep smallest set of tokens with cumulative prob >= top_p."""
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least one token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        # Scatter back to original indexing
        logits = logits.clone()
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float('Inf')
    return logits


@torch.no_grad()
def generate(model, tokenizer, prompt_str, max_new_tokens, temperature=1.0, top_k=None, top_p=None, device='cpu'):
    """
    Autoregressive text generation with top-k and top-p sampling.
    """
    model.eval()

    # 1. Tokenize prompt
    prompt_tokens = tokenizer.encode(prompt_str, out_type=int)
    idx = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    # 2. Track token usage
    input_len = idx.size(1)
    token_usage = {
        "prompt_tokens": input_len,
        "completion_tokens": 0,
        "total_tokens": input_len
    }

    # 3. Generation loop
    for _ in range(max_new_tokens):
        # Crop context if needed
        idx_cond = idx if idx.size(1) <= model.config.max_seq_len else idx[:, -model.config.max_seq_len:]

        # Forward pass
        logits, _ = model(idx_cond)

        # Get logits for last position, apply temperature
        logits = logits[:, -1, :]
        if temperature > 0:
            logits = logits / temperature
        else:
            # Greedy: just take argmax
            idx_next = logits.argmax(dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)
            token_usage["completion_tokens"] += 1
            token_usage["total_tokens"] += 1
            if idx_next.item() == tokenizer.eos_id():
                break
            continue

        # Apply top-k filtering
        if top_k is not None:
            logits = sample_top_k(logits, top_k)

        # Apply top-p filtering
        if top_p is not None:
            logits = sample_top_p(logits, top_p)

        # Sample
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        # Append
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
