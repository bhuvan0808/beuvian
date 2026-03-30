---
language:
- en
license: mit
tags:
- text-generation
- transformer
- pytorch
- from-scratch
- foundation-model
datasets:
- allenai/c4
pipeline_tag: text-generation
model-index:
- name: buvn-2.0
  results:
  - task:
      type: text-generation
      name: Language Modeling
    dataset:
      type: wikitext
      name: WikiText-103
    metrics:
    - type: perplexity
      value: 29.19
      name: Perplexity
---

# BUVN-2.0 — Foundation Language Model

**Built from scratch by Bhuvan** | Part of the [Beuvian AI Ecosystem](https://github.com/bhuvan0808/beuvian)

## Model Summary

| Metric | Value |
|--------|-------|
| **Architecture** | Decoder-only Transformer (GPT-style) |
| **Parameters** | 109.53M (84.95M non-embedding) |
| **Layers** | 12 |
| **Dimensions** | 768 |
| **Attention Heads** | 12 |
| **Context Window** | 1,024 tokens |
| **Vocabulary** | 32,000 BPE |
| **Training Data** | C4 (2B tokens, streamed) |
| **Training Hardware** | NVIDIA H100 NVL (96 GB) |
| **Training Time** | ~2 hours |
| **Val Perplexity** | **29.19** |

## Benchmark Results

| Rank | Model | Params | PPL | Data |
|:---:|-------|:---:|:---:|:---:|
| 6 | OPT-125M (Meta) | 125M | 27.65 | 300B |
| 7 | RWKV-169M | 169M | 29.01 | 300B |
| **8** | **BUVN-2.0 (this model)** | **109.5M** | **29.19** | **2B** |
| 9 | Pythia-160M | 160M | 29.33 | 300B |
| 10 | GPT-2 Small | 124M | 29.41 | ~40B |

**Beats GPT-2 Small** with 9x fewer params and 20,000x less training data.

## Architecture Details

- **Attention:** Multi-head with Rotary Position Embeddings (RoPE)
- **Normalization:** RMSNorm (pre-normalization, eps=1e-5)
- **Feedforward:** SwiGLU (gated activation)
- **Flash Attention:** Via PyTorch SDPA
- **Weight Tying:** Embedding = output projection
- **Bias:** None (LLaMA/PaLM style)
- **Initialization:** Depth-scaled residual (GPT-2/nanoGPT style)

## Usage

```python
import torch
from model.config import BUVNConfig
from model.model import BUVNModel

# Load checkpoint
ckpt = torch.load('buvn_2.0_best.pt', map_location='cpu', weights_only=False)

# Handle torch.compile prefix
state_dict = ckpt['model']
for k in list(state_dict.keys()):
    if k.startswith('_orig_mod.'):
        state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)

# Build model
config = BUVNConfig.from_dict(ckpt['model_args'])
model = BUVNModel(config)
model.load_state_dict(state_dict)
model.eval()
```

## Training Details

- **Optimizer:** AdamW (beta1=0.9, beta2=0.95, weight_decay=0.1)
- **Learning Rate:** 6e-4 peak, cosine decay to 6e-5, 500 step warmup
- **Batch Size:** 64 × 2 gradient accumulation = 128 effective
- **Precision:** bfloat16
- **Compiler:** torch.compile enabled
- **Steps:** 15,000 (processing ~2B tokens)
- **Throughput:** ~320K tokens/sec, 24% MFU on H100

## The Beuvian Ecosystem

BUVN is the foundation model. Two specializations are planned:
- **SRVN** — Coding agent (fine-tuned from BUVN on code data)
- **MNI** — Finance model (trained on market data, SEC filings)

## Links

- **GitHub:** [bhuvan0808/beuvian](https://github.com/bhuvan0808/beuvian)
- **Documentation:** [docs/](https://github.com/bhuvan0808/beuvian/tree/main/BUVN-1.1/docs)

## License

MIT
