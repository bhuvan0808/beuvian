# Model Architecture

## High-Level Overview

BUVN is a **decoder-only transformer** — the same architecture family as GPT, LLaMA, and Claude. Text goes in, next-word prediction comes out.

```
"The cat sat on the" → [BUVN Model] → "mat" (42%), "floor" (20%), "roof" (8%)...
```

The full pipeline inside the model:

```
Input: "The cat sat on the"
  ↓
Token Embedding (convert words → 768-dimensional vectors)
  ↓
Dropout
  ↓
Transformer Block 1  ─┐
Transformer Block 2   │  × 12 blocks (each identical structure)
...                    │  Each block: Attention + FeedForward + Norms + Residuals
Transformer Block 12 ─┘
  ↓
Final RMSNorm
  ↓
Output Projection (768 dims → 32,000 vocab scores)
  ↓
Softmax → Probabilities
  ↓
Output: "mat" with 42% probability
```

## Token Embeddings & Weight Tying

The model converts token IDs into dense vectors:

```python
self.tok_embeddings = nn.Embedding(vocab_size, d_model)  # 32000 × 768
```

**Weight tying:** The input embedding and output projection share the same weight matrix:

```python
self.tok_embeddings.weight = self.output.weight  # Same parameters!
```

This saves 32,000 × 768 = **24.6M parameters** and slightly improves quality (the model learns that input and output representations should be consistent).

## Transformer Block

Each of the 12 blocks has this structure:

```
Input x
  ↓
RMSNorm(x)
  ↓
Multi-Head Attention(normalized_x) → attn_output
  ↓
x + attn_output                     ← Residual connection
  ↓
RMSNorm(result)
  ↓
SwiGLU FeedForward(normalized) → ffn_output
  ↓
result + ffn_output                 ← Residual connection
  ↓
Output (same shape as input)
```

**Pre-normalization:** We normalize BEFORE attention/FFN (not after). This is the LLaMA/PaLM style — more stable training than GPT-2's post-norm.

**Residual connections:** The input is added back after each sub-layer. This prevents the "vanishing gradient" problem — gradients can flow directly through the skip connections during backpropagation.

## Multi-Head Attention

Attention lets each word "look at" relevant other words in the context.

### The SDPA Formula

```
Attention(Q, K, V) = softmax(Q × K^T / √d_head) × V
```

### Step by step:

1. **Project** input into Query, Key, Value:
   ```python
   Q = x @ W_q  # "What am I looking for?"
   K = x @ W_k  # "What do I contain?"
   V = x @ W_v  # "What information do I carry?"
   ```

2. **Split** into 12 heads (each head has dim 64):
   ```python
   Q = Q.view(batch, seq_len, 12, 64).transpose(1, 2)  # (batch, 12, seq, 64)
   ```

3. **Apply RoPE** to Q and K (position encoding — see below)

4. **Compute attention scores:**
   ```python
   scores = (Q @ K.transpose(-2, -1)) / sqrt(64)  # (batch, 12, seq, seq)
   ```

5. **Causal mask** — each position can only attend to earlier positions:
   ```python
   # Upper triangle set to -infinity → softmax gives 0 probability
   mask = torch.triu(ones(seq, seq), diagonal=1)
   scores = scores.masked_fill(mask, -inf)
   ```

6. **Softmax → weighted sum:**
   ```python
   weights = softmax(scores)       # (batch, 12, seq, seq)
   output = weights @ V            # (batch, 12, seq, 64)
   ```

7. **Merge heads and project:**
   ```python
   output = output.transpose(1, 2).reshape(batch, seq, 768)
   output = output @ W_o           # Final linear projection
   ```

### Flash Attention (SDPA)

We use PyTorch's `scaled_dot_product_attention` which auto-selects the fastest kernel:

```python
output = F.scaled_dot_product_attention(Q, K, V, is_causal=True,
    dropout_p=self.attn_dropout.p if self.training else 0.0)
```

This is 2–4x more memory efficient than manual attention because it never materializes the full (seq × seq) attention matrix.

## RoPE (Rotary Position Embedding)

The model needs to know word ORDER. RoPE encodes position by rotating Q and K vectors.

### Intuition
Each pair of dimensions in Q/K gets rotated by an angle proportional to the position:

```
Word at position 5: rotate pair (q₁, q₂) by 5 × θ₁
                    rotate pair (q₃, q₄) by 5 × θ₂
                    ...
                    rotate pair (q₆₃, q₆₄) by 5 × θ₃₂
```

### The Math

For position m, dimension pair i:

```
θᵢ = 10000^(-2i/d)

[q'₂ᵢ  ]   [cos(m·θᵢ)  -sin(m·θᵢ)] [q₂ᵢ  ]
[q'₂ᵢ₊₁] = [sin(m·θᵢ)   cos(m·θᵢ)] [q₂ᵢ₊₁]
```

Different frequency per pair: low-index pairs rotate fast (capture local patterns), high-index pairs rotate slowly (capture long-range structure).

### Why RoPE is clever
The dot product Q·K naturally depends on the **relative** distance between positions, not absolute positions. This generalizes better and can sometimes handle sequences longer than training length.

### Implementation

```python
# Precompute frequencies
freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))

# For each position m, compute rotation
freqs_cis = torch.polar(ones, outer(positions, freqs))  # complex exponentials

# Apply to Q and K (as complex multiplication)
q_rotated = view_as_complex(q) * freqs_cis
```

## RMSNorm

Keeps activations in a stable range. Simpler and faster than LayerNorm.

### Formula

```
RMSNorm(x) = x / RMS(x) × γ

where RMS(x) = √(mean(x²) + ε)
```

### Implementation

```python
def _norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

def forward(self, x):
    return self._norm(x.float()).type_as(x) * self.weight  # γ (learnable)
```

- `eps = 1e-5` prevents division by zero (increased from 1e-6 for mixed precision stability)
- `self.weight` (γ) is a learnable scale parameter, initialized to 1.0
- Cast to float32 for the norm computation, then back to original dtype

### vs LayerNorm
- LayerNorm: subtract mean AND divide by std (2 stats)
- RMSNorm: only divide by RMS (1 stat)
- ~10–15% faster, same quality. Used by LLaMA and PaLM.

## SwiGLU FeedForward

The "thinking" layer between attention steps. Uses a gating mechanism.

### Formula

```
SwiGLU(x) = (SiLU(x·W₁) ⊙ x·W₃) · W₂

where SiLU(x) = x × sigmoid(x)
  and ⊙ = element-wise multiplication
```

### Two paths:

```
Input x (768 dims)
  ├── Path 1: x·W₁ → SiLU activation → "proposal"
  ├── Path 2: x·W₃ → "gate" (decides what to keep)
  ↓
  proposal ⊙ gate  → element-wise multiply
  ↓
  result · W₂ → project back to 768 dims
  ↓
  Dropout
```

### Why SwiGLU > ReLU
- **ReLU** kills all negative values (hard cutoff) → information loss
- **SiLU** is smooth — small negative values survive
- **Gating** lets the model learn what to keep vs discard (data-dependent filtering)
- 2–3% better perplexity than ReLU FFN at same parameter count

### Hidden dimension
```python
hidden_dim = 4 * d_model          # 3072
hidden_dim = int(2 * hidden_dim / 3)  # 2048 (SwiGLU uses 2/3 ratio)
hidden_dim = 256 * ((hidden_dim + 255) // 256)  # Round up to 256 for GPU efficiency
```

For d_model=768: hidden_dim = **2048** (after rounding to nearest 256).

## Gradient Checkpointing

Trades compute for memory. Instead of storing all intermediate activations for the backward pass, it recomputes them on-the-fly.

```python
# Without checkpointing: stores all 12 layers' activations in VRAM
for layer in self.layers:
    h = layer(h, freqs_cis)

# With checkpointing: only stores input, recomputes during backward
for layer in self.layers:
    h = grad_checkpoint(layer, h, freqs_cis, use_reentrant=False)
```

- Saves ~30–40% VRAM
- ~20% slower (recomputation cost)
- Enable via `gradient_checkpointing: true` in config
- Useful when batch size is limited by VRAM

## Weight Initialization

```python
# Most layers: normal distribution, std=0.02
nn.init.normal_(module.weight, mean=0.0, std=0.02)

# Residual projections (W_o in attention, W₂ in FFN): depth-scaled
nn.init.normal_(p, mean=0.0, std=0.02 / sqrt(2 * n_layers))
```

Depth-scaled init prevents the residual stream from growing too large in deep models. With 12 layers, residual projections get `std = 0.02 / sqrt(24) ≈ 0.004`.

## Model Configurations

| Config | d_model | n_layers | n_heads | head_dim | FFN hidden | vocab | context | Params |
|--------|---------|----------|---------|----------|-----------|-------|---------|--------|
| Tiny | 128 | 4 | 4 | 32 | 256 | 8K | 128 | 2.8M |
| Small | 384 | 6 | 6 | 64 | 768 | 8K | 512 | 13.7M |
| **Medium** | **768** | **12** | **12** | **64** | **2048** | **32K** | **1024** | **109.5M** |
| Large | 1024 | 24 | 16 | 64 | 2816 | 50K | 2048 | ~350M |
| XL | 1536 | 24 | 16 | 96 | 4096 | 50K | 2048 | ~770M |

## Parameter Count Breakdown (125M model)

| Component | Shape | Parameters | % of Total |
|-----------|-------|-----------|-----------|
| Token Embedding | 32000 × 768 | 24.6M | 22.4% |
| Output Projection | (weight-tied) | 0 | 0% |
| 12× Attention (Wq,Wk,Wv,Wo) | 4 × (768 × 768) | 28.3M | 25.9% |
| 12× FFN (W1,W2,W3) | 3 × (768 × 2048) | 56.6M | 51.7% |
| 12× RMSNorm (attn + ffn) | 24 × 768 | 18K | 0.02% |
| Final RMSNorm | 768 | 768 | 0.001% |
| **Total** | | **109.5M** | **100%** |

The FFN layers contain over half the parameters — this is where most of the model's "knowledge" is stored.
