import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from model.config import BUVNConfig


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Precomputes the frequency tensor for complex exponentials (RoPE)."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshapes the precomputed frequencies to match the input tensor shape."""
    ndim = x.ndim
    assert ndim >= 2, f"Expected tensor with at least 2 dims, got {ndim}"
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding to query and key tensors."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        """Root Mean Square Layer Normalization."""
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, config: BUVNConfig):
        """SwiGLU feedforward network."""
        super().__init__()
        hidden_dim = 4 * config.d_model
        hidden_dim = int(2 * hidden_dim / 3)
        # Round to nearest multiple of 256 for optimal GPU utilization
        hidden_dim = 256 * ((hidden_dim + 255) // 256)

        self.w1 = nn.Linear(config.d_model, hidden_dim, bias=config.bias)
        self.w2 = nn.Linear(hidden_dim, config.d_model, bias=config.bias)
        self.w3 = nn.Linear(config.d_model, hidden_dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Attention(nn.Module):
    def __init__(self, config: BUVNConfig):
        """Multi-head attention module with RoPE."""
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads

        self.wq = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=config.bias)
        self.wk = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=config.bias)
        self.wv = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=config.bias)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=config.bias)

        self.resid_dropout = nn.Dropout(config.dropout)
        self.attn_dropout = nn.Dropout(config.dropout)

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        bsz, seqlen, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # Transpose to (bsz, n_heads, seqlen, head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        if self.flash:
            output = torch.nn.functional.scaled_dot_product_attention(
                xq, xk, xv, attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True
            )
        else:
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            mask = torch.triu(torch.ones(seqlen, seqlen, dtype=torch.bool, device=x.device), diagonal=1)
            scores = scores.masked_fill(mask, float('-inf'))
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.resid_dropout(self.wo(output))


class TransformerBlock(nn.Module):
    def __init__(self, config: BUVNConfig):
        """A single transformer layer."""
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.d_model)
        self.ffn_norm = RMSNorm(config.d_model)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class BUVNModel(nn.Module):
    def __init__(self, config: BUVNConfig):
        """The BUVN Foundation Model."""
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers
        self.gradient_checkpointing = getattr(config, 'gradient_checkpointing', False)

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share embedding and output weights
        self.tok_embeddings.weight = self.output.weight

        # Precompute RoPE frequencies
        freqs_cis = precompute_freqs_cis(
            config.d_model // config.n_heads, config.max_seq_len * 2, config.rope_theta
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # Init weights
        self.apply(self._init_weights)
        # Depth-scaled init for residual projections (GPT-2 / nanoGPT style)
        for pn, p in self.named_parameters():
            if pn.endswith('wo.weight') or pn.endswith('w2.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor = None):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        freqs_cis = self.freqs_cis[:seqlen]

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                h = grad_checkpoint(layer, h, freqs_cis, use_reentrant=False)
            else:
                h = layer(h, freqs_cis)

        h = self.norm(h)
        logits = self.output(h)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            loss = None

        return logits, loss

    def get_num_params(self, non_embedding=True):
        """Return the number of parameters. Exclude embeddings if weight-tied."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.tok_embeddings.weight.numel()
        return n_params

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Estimate model FLOPs utilization (MFU). Auto-detects GPU peak FLOPS."""
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layers, cfg.n_heads, cfg.d_model // cfg.n_heads, cfg.max_seq_len
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter / dt

        # Auto-detect GPU peak FLOPS (bf16)
        flops_promised = 312e12  # default: A100
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0).lower()
            if 'h100' in gpu_name:
                flops_promised = 989.5e12  # H100 SXM bf16
                if 'nvl' in gpu_name:
                    flops_promised = 835e12  # H100 NVL bf16
            elif 'h200' in gpu_name:
                flops_promised = 989.5e12
            elif 'a100' in gpu_name:
                flops_promised = 312e12
            elif '4090' in gpu_name:
                flops_promised = 165.2e12
            elif '3090' in gpu_name:
                flops_promised = 71e12

        mfu = flops_achieved / flops_promised
        return mfu
