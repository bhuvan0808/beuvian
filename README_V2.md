<pre>
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ██████  ███████ ██    ██ ██    ██ ██  █████  ███    ██      ║
║   ██   ██ ██      ██    ██ ██    ██ ██ ██   ██ ████   ██      ║
║   ██████  █████   ██    ██ ██    ██ ██ ███████ ██ ██  ██      ║
║   ██   ██ ██      ██    ██  ██  ██  ██ ██   ██ ██  ██ ██      ║
║   ██████  ███████  ██████    ████   ██ ██   ██ ██   ████      ║
║                                                               ║
║   One Foundation. Three Intelligences.                        ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
</pre>

![ecosystem](https://img.shields.io/badge/Ecosystem-3_Models-F5F0EB?style=flat-square&labelColor=0A0A0A)
![buvn](https://img.shields.io/badge/BUVN-Foundation-F5F0EB?style=flat-square&labelColor=0A0A0A)
![srvn](https://img.shields.io/badge/SRVN-Code_Agent-00FF88?style=flat-square&labelColor=0A0A0A)
![mni](https://img.shields.io/badge/MNI-Finance-FFD600?style=flat-square&labelColor=0A0A0A)
![python](https://img.shields.io/badge/Python-3.10+-F5F0EB?style=flat-square&labelColor=0A0A0A)
![pytorch](https://img.shields.io/badge/PyTorch-2.0+-F5F0EB?style=flat-square&labelColor=0A0A0A)
![license](https://img.shields.io/badge/License-MIT-F5F0EB?style=flat-square&labelColor=0A0A0A)

```
═══════════════════════════════════════════════════════════════
```

## > What is Beuvian

**Beuvian** is an ecosystem of AI models built from the ground up. No pretrained weights, no shortcuts, no black boxes. One foundation model (BUVN) gives rise to two specialists: a coding agent (SRVN) and a financial analyst (MNI). Every layer, every weight, every training decision -- built and owned from scratch.

```
                    ╔═══════════════════════╗
                    ║   BUVN  [Foundation]  ║
                    ║   109.5M params       ║
                    ║   PPL 29.19           ║
                    ╚═══════════╤═══════════╝
                        ┌───────┴───────┐
                        v               v
                ┌───────────────┐ ┌─────────────┐
                │ SRVN  [Code]  │ │ MNI  [Fin]  │
                │ Fine-tuned    │ │ Domain-      │
                │ from BUVN     │ │ trained      │
                │ [PLANNED]     │ │ [PLANNED]    │
                └───────────────┘ └─────────────┘
```

```
═══════════════════════════════════════════════════════════════
```

## | The Model Family

```
BEUVIAN
│
├── BUVN ── Foundation LLM ─────────────────── [v2.0 RELEASED]
│   ├── GPT-style decoder-only transformer
│   ├── RoPE + RMSNorm + SwiGLU + Flash Attention
│   ├── Trained on C4 (2B tokens)
│   └── Serves as weight donor for all downstream models
│
├── SRVN ── Code Agent LLM ─────────────────── [PLANNED]
│   ├── Fine-tuned from BUVN checkpoint
│   ├── Code generation, bug detection, review
│   ├── Fill-in-the-Middle (FIM) training
│   └── Agentic mode: plan -> code -> test -> debug
│
└── MNI ─── Finance LLM ────────────────────── [PLANNED]
    ├── Domain-trained from BUVN checkpoint
    ├── Price prediction, sentiment analysis
    ├── Multi-modal: text + numeric embeddings
    └── Signal generation (not financial advice)
```

```
═══════════════════════════════════════════════════════════════
```

## | BUVN -- The Foundation

```
╔══════════════════════════════════════════════════════════════╗
║  BUVN-2.0 Results                                           ║
╠══════════╦══════════╦══════════╦══════════╦═════════════════╣
║ PPL      ║ Top-1    ║ Top-5    ║ Rank     ║ Overfit Gap     ║
║ 29.19    ║ 37.88%   ║ 60.34%   ║ #8 / 11  ║ 0.03 (healthy)  ║
╚══════════╩══════════╩══════════╩══════════╩═════════════════╝
```

| Spec           | CPU Test    | Production   |
|----------------|-------------|--------------|
| Parameters     | 2.8M        | 109.5M       |
| Layers         | 4           | 12           |
| d_model        | 128         | 768          |
| Heads          | 4           | 12           |
| Vocab          | 8K BPE      | 32K BPE      |
| Context        | 128         | 1,024        |
| Training Data  | WikiText-103| C4 (2B tok)  |
| Training Time  | 5 min       | ~2 hours     |
| Hardware       | Any CPU     | H100         |

### > Architecture

```
  Input Tokens
      |
  [Embedding]
      |
  ┌──────────────┐
  │ Transformer  │ x12
  │   Block      │
  │              │
  │ * RMSNorm    │
  │ * Attention  │
  │   + RoPE     │
  │ * SwiGLU FFN │
  │ * Residuals  │
  └──────┬───────┘
         |
  [RMSNorm]
      |
  [Linear --> Logits]
      |
  Next Token
```

### > The Pipeline

```
  [1] prepare_data.py ──> [2] train_hf_tokenizer.py ──> [3] tokenize_corpus.py
         |                         |                            |
    Download C4             Train BPE vocab             Tokenize to binary
                                                               |
  [6] api/app.py <── [5] generate.py <── [4] train.py <────────┘
         |                  |                   |
    Deploy FastAPI    Generate text        Train transformer
```

```
═══════════════════════════════════════════════════════════════
```

## | Leaderboard

WikiText-103 perplexity (lower = better):

| Rank | Model | Params | PPL | Training Tokens | Org |
|------|-------|--------|-----|-----------------|-----|
| 1 | LLaMA-2 7B | 7B | 5.47 | 2T | Meta |
| 2 | LLaMA 7B | 7B | 7.73 | 1T | Meta |
| 3 | Pythia-1B | 1B | 16.71 | 300B | EleutherAI |
| 4 | GPT-2 Large | 774M | 19.93 | ~40B | OpenAI |
| 5 | GPT-2 Medium | 355M | 22.76 | ~40B | OpenAI |
| 6 | OPT-125M | 125M | 27.65 | 300B | Meta |
| 7 | RWKV-169M | 169M | 29.01 | 300B | RWKV |
| **8** | **BUVN-2.0** | **109.5M** | **29.19** | **2B** | **Bhuvan** |
| 9 | Pythia-160M | 160M | 29.33 | 300B | EleutherAI |
| 10 | GPT-2 Small | 124M | 29.41 | ~40B | OpenAI |
| 11 | GPT-Neo 125M | 125M | 32.43 | 300B | EleutherAI |

```diff
+ BUVN-2.0  PPL 29.19  |  Beat GPT-2 Small (29.41) with 20,000x less data
+ 109.5M params  |  2B tokens  |  ~2 hours on H100
- GPT-2 Small    |  124M params |  ~40B tokens  |  weeks of training
```

```
═══════════════════════════════════════════════════════════════
```

## | SRVN -- The Coder

`STATUS: PLANNED` | Fine-tuned from BUVN checkpoint

| Capability         | Detail                                          |
|--------------------|-------------------------------------------------|
| Code Generation    | Python, JS, Rust, Go, Java, C++, SQL, Bash      |
| Bug Detection      | Static analysis via LLM, root cause, auto-fix    |
| Code Review        | Quality scoring, anti-patterns, refactoring       |
| Agentic Mode       | Multi-step planning, tool use, self-correction    |
| Code Explanation   | Line-by-line breakdown, architecture summaries    |

### > Training Plan

| Phase | Source | Technique | Goal |
|-------|--------|-----------|------|
| 1 | The Stack v2, GitHub | Continued pre-training | Learn syntax across 20+ languages |
| 2 | CodeAlpaca, Code-Instruct | SFT | Follow coding instructions |
| 3 | Tool-use, ReAct traces | RL | Agentic: plan-code-test-debug |
| 4 | Human preference data | DPO / RLHF | Safe, readable, documented code |

### > Target Benchmarks

| Benchmark | Measures | Target |
|-----------|----------|--------|
| HumanEval | Python generation | > 25% pass@1 |
| MBPP | Basic problems | > 30% pass@1 |
| SWE-bench | GitHub issues | > 5% resolved |

```
═══════════════════════════════════════════════════════════════
```

## | MNI -- The Analyst

`STATUS: PLANNED` | Domain-trained from BUVN checkpoint

| Capability           | Detail                                         |
|----------------------|------------------------------------------------|
| Price Prediction     | Short-term (1-5d), medium-term (1-3mo), trend  |
| Sentiment Analysis   | News scoring, earnings parsing, social signals  |
| Risk Assessment      | Portfolio risk, anomaly detection, drawdown      |
| Pattern Recognition  | Technical patterns, volume, support/resistance   |
| Report Generation    | Research summaries, trade rationale, commentary  |

### > Training Plan

| Phase | Source | Technique | Goal |
|-------|--------|-----------|------|
| 1 | SEC filings, transcripts | Continued pre-training | Financial vocabulary |
| 2 | OHLCV, tick data | Specialized tokenization | Price action patterns |
| 3 | News, Reddit, analyst reports | SFT | Sentiment extraction |
| 4 | Labeled outcomes, backtests | RL | Prediction accuracy |

### > Target Markets

| Priority | Asset Class |
|----------|-------------|
| HIGH | US Equities (NYSE, NASDAQ) |
| HIGH | Indian Equities (NSE, BSE) |
| MED | Forex (Major pairs) |
| LOW | Crypto (BTC, ETH, Top 20) |
| LOW | Commodities (Gold, Oil) |

```
═══════════════════════════════════════════════════════════════
```

## * Roadmap

```
PHASE 1 -- FOUNDATION                                      [BUVN]
───────────────────────────────────────────────────────────────
  [x] BUVN-1.1 released (2.8M params, CPU, WikiText-103)
  [x] BUVN-2.0 released (109.5M params, H100, C4 2B tokens)
  [ ] Scale to 120M+ params on multi-GPU
  [ ] INT8/INT4 quantization for efficient inference

PHASE 2 -- CODE INTELLIGENCE                               [SRVN]
───────────────────────────────────────────────────────────────
  [ ] Curate code corpus (The Stack v2, 500GB+)
  [ ] Extend tokenizer (64K with code tokens)
  [ ] Fine-tune BUVN on code data
  [ ] FIM training + instruction tuning
  [ ] Agentic framework (tool use, self-correction)
  [ ] Benchmark: HumanEval, MBPP, SWE-bench

PHASE 3 -- FINANCIAL INTELLIGENCE                          [MNI]
───────────────────────────────────────────────────────────────
  [ ] Financial data pipeline (SEC, Yahoo Finance, news APIs)
  [ ] Numeric-aware tokenization
  [ ] Domain pre-train on financial corpus
  [ ] Sentiment + prediction training
  [ ] Backtest with walk-forward validation
  [ ] Deploy analysis API + dashboard

PHASE 4 -- ECOSYSTEM INTEGRATION                           [ALL]
───────────────────────────────────────────────────────────────
  [ ] Unified API gateway (route to BUVN/SRVN/MNI)
  [ ] Cross-model orchestration
  [ ] HuggingFace Hub publication
  [ ] Web playground
```

```
═══════════════════════════════════════════════════════════════
```

## / Docs

| Document | Path |
|----------|------|
| Architecture | [`BUVN-1.1/docs/architecture.md`](BUVN-1.1/docs/architecture.md) |
| Training | [`BUVN-1.1/docs/training.md`](BUVN-1.1/docs/training.md) |
| Benchmarks | [`BUVN-1.1/docs/benchmarks.md`](BUVN-1.1/docs/benchmarks.md) |
| Usage | [`BUVN-1.1/docs/usage.md`](BUVN-1.1/docs/usage.md) |
| Setup | [`BUVN-1.1/docs/setup.md`](BUVN-1.1/docs/setup.md) |
| Fine-tuning | [`BUVN-1.1/docs/fine-tuning.md`](BUVN-1.1/docs/fine-tuning.md) |
| Scaling | [`BUVN-1.1/docs/scaling.md`](BUVN-1.1/docs/scaling.md) |
| Design System | [`BUVN-1.1/docs/design-system.md`](BUVN-1.1/docs/design-system.md) |

```
═══════════════════════════════════════════════════════════════
```

## # Quick Start

```bash
git clone https://github.com/bhuvan0808/beuvian.git && cd beuvian/BUVN-1.1
pip install -r requirements.txt && export PYTHONPATH=$(pwd)

python scripts/prepare_data.py --max_size_mb 300           # 1. Download data
python scripts/train_hf_tokenizer.py --vocab_size 8000     # 2. Train tokenizer
python scripts/tokenize_corpus.py                           # 3. Tokenize
python training/train.py --config configs/train_config.yaml # 4. Train
python inference/generate.py --prompt "The future of AI is" # 5. Generate
python api/app.py                                           # 6. Deploy API
```

### > Hardware

| Model | Minimum | Recommended |
|-------|---------|-------------|
| BUVN (test) | Any CPU, 8 GB RAM | GPU, 16 GB |
| BUVN (prod) | 1x A100 40 GB | 4x A100 80 GB |
| SRVN | 1x A100 40 GB | 2x A100 80 GB |
| MNI | 1x A100 40 GB | 2x A100 80 GB |

```
═══════════════════════════════════════════════════════════════
```

## > Disclaimers

```
BUVN    Research model. Not instruction-tuned. May generate incoherent text.
SRVN    Coding assistant (when released). Always review generated code.
MNI     NOT financial advice. Research tool only. Consult professionals.
ALL     Do not use for harmful, deceptive, or manipulative content.
```

```
═══════════════════════════════════════════════════════════════
```

<pre>
  Built by Bhuvan  |  github.com/bhuvan0808/beuvian  |  MIT License
</pre>
