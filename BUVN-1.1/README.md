<div align="center">

# 🧠 BUVN-1.1

### A Foundation Language Model — Built From Scratch

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

> A complete, production-quality codebase for training a **GPT-style decoder-only transformer** from scratch — data pipeline, tokenization, training, inference, and deployment-ready API.

---

[Quick Start](#-quick-start) · [Architecture](#-architecture) · [Step-by-Step Guide](#-step-by-step-guide) · [API Docs](#-step-6--start-the-api-server) · [Results](#-training-results)

</div>

---

## 📖 Table of Contents

| Section | Description |
|---------|-------------|
| [What is a Foundation Model?](#-what-is-a-foundation-model) | Core concept explanation |
| [Architecture](#-architecture) | Model design and specs |
| [Project Structure](#-project-structure) | Directory layout |
| [Quick Start](#-quick-start) | All 7 commands at a glance |
| [Step-by-Step Guide](#-step-by-step-guide) | Detailed walkthrough of every step |
| [Training Results](#-training-results) | Loss curves from our test run |
| [API & Swagger](#-step-6--start-the-api-server) | Endpoint docs, Postman, Swagger |
| [Checkpoint Guide](#-using-the-checkpoint-pt-file) | How to load and use .pt files |
| [Hardware Requirements](#-hardware-requirements) | CPU, GPU, Cloud options |
| [Limitations & Roadmap](#%EF%B8%8F-limitations) | Current gaps and future plans |

---

## 💡 What is a Foundation Model?

A **foundation model** is a large neural network trained on massive amounts of raw text. Once trained, it becomes the *foundation* for many downstream applications.

```
Raw Text Data  →  Pre-training (this repo)  →  Foundation Model  →  Fine-tuning  →  AI Assistant
                                                                  →  RLHF        →  ChatGPT-style
                                                                  →  Prompting    →  Zero-shot tasks
```

BUVN-1.1 implements the **pre-training step** — the base upon which everything else is built.

---

## 🏗 Architecture

BUVN-1.1 is a **decoder-only transformer** (GPT-style) with these modern enhancements:

```
Input Tokens → Embedding → [Transformer Block × N] → RMSNorm → Output Logits
                                    │
                          ┌─────────┴─────────┐
                          │  Transformer Block │
                          │                    │
                          │  ┌──────────────┐  │
                          │  │  RMSNorm     │  │
                          │  │  Multi-Head  │  │
                          │  │  Attention   │  │
                          │  │  (+ RoPE)    │  │
                          │  └──────┬───────┘  │
                          │      + residual    │
                          │  ┌──────────────┐  │
                          │  │  RMSNorm     │  │
                          │  │  SwiGLU FFN  │  │
                          │  └──────┬───────┘  │
                          │      + residual    │
                          └────────────────────┘
```

### Model Specs

| Parameter | Production Config | CPU Test Config |
|-----------|:-:|:-:|
| **Parameters** | ~120M | ~2M |
| **Layers** | 12 | 4 |
| **Attention Heads** | 12 | 4 |
| **Embedding Dim** | 768 | 128 |
| **Context Length** | 512 | 128 |
| **Vocab Size** | 50,000 | 8,000 |
| **Positional Encoding** | RoPE | RoPE |
| **Activation** | SwiGLU | SwiGLU |
| **Normalization** | RMSNorm | RMSNorm |

---

## 📂 Project Structure

```
BUVN-1.1/
│
├── 📁 model/                       # Core neural network
│   ├── config.py                   #   Model hyperparameters dataclass
│   ├── model.py                    #   Transformer, Attention, RoPE, SwiGLU
│   └── utils.py                    #   Device detection helpers
│
├── 📁 tokenizer/                   # Trained tokenizer files
│   └── tokenizer.json              #   (generated) HuggingFace BPE tokenizer
│
├── 📁 training/                    # Training pipeline
│   ├── config.py                   #   YAML config parser
│   ├── dataloader.py               #   Memory-mapped binary data loader
│   └── train.py                    #   Training loop (AdamW, cosine LR, mixed precision)
│
├── 📁 inference/                   # Text generation
│   ├── sample.py                   #   Top-k sampling with temperature
│   └── generate.py                 #   CLI & model loader
│
├── 📁 api/                         # FastAPI server
│   ├── app.py                      #   Server bootstrap + CORS
│   └── routes.py                   #   POST /generate endpoint
│
├── 📁 scripts/                     # Pipeline utilities
│   ├── prepare_data.py             #   Download & clean WikiText-103 (streaming)
│   ├── train_hf_tokenizer.py       #   Train BPE tokenizer
│   ├── tokenize_corpus.py          #   Corpus → train.bin / val.bin
│   ├── test_inference.py           #   Quick inference test
│   └── convert_to_hf.py            #   Export to HuggingFace format
│
├── 📁 configs/
│   └── train_config.yaml           #   All hyperparameters in one place
│
├── 📁 data/processed/              #   (generated) corpus.txt, train.bin, val.bin
├── 📁 checkpoints/                 #   (generated) ckpt.pt model files
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

> All 7 commands at a glance. See the detailed guide below for explanations.

```bash
# 0. Setup
cd BUVN-1.1
pip install -r requirements.txt
export PYTHONPATH=$(pwd)            # Linux/Mac
# $env:PYTHONPATH=$(pwd)            # Windows PowerShell

# 1. Download dataset
python scripts/prepare_data.py --max_size_mb 300

# 2. Train tokenizer
python scripts/train_hf_tokenizer.py --vocab_size 8000

# 3. Tokenize into binary
python scripts/tokenize_corpus.py

# 4. Train the model
python training/train.py --config configs/train_config.yaml

# 5. Test inference
python inference/generate.py --prompt "The history of" --checkpoint checkpoints/ckpt.pt --tokenizer tokenizer/tokenizer.json

# 6. Start API server
python api/app.py --checkpoint checkpoints/ckpt.pt --tokenizer tokenizer/tokenizer.json
```

---

## 📘 Step-by-Step Guide

### 📥 Step 0 — Installation

```bash
git clone <your-repo-url>
cd BUVN-1.1
```

Create a virtual environment and install dependencies:

```bash
python -m venv venv

# Activate:
venv\Scripts\activate              # Windows
# source venv/bin/activate         # Linux / Mac

pip install -r requirements.txt
```

**Set PYTHONPATH** (required for all subsequent commands):

```bash
# Windows PowerShell:
$env:PYTHONPATH = "C:\path\to\BUVN-1.1"

# Linux / Mac:
export PYTHONPATH=$(pwd)
```

> ⚠️ You must set `PYTHONPATH` to the `BUVN-1.1/` directory in every new terminal session.

---

### 📊 Step 1 — Download & Prepare the Dataset

Downloads **WikiText-103** from HuggingFace using streaming (no full download needed). Filters out short text, empty lines, and wiki headers. Stops automatically when the file reaches the target size.

```bash
python scripts/prepare_data.py --max_size_mb 300
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--max_size_mb` | 150 | Stop when corpus reaches this size in MB |
| `--min_length` | 100 | Skip lines shorter than this (characters) |
| `--output` | `data/processed/corpus.txt` | Output file path |

**Example output:**
```
Loading WikiText-103 (streaming mode)...
Processed 100,000 samples | Written 57,723 | Size: 38.3 MB
Processed 200,000 samples | Written 109,651 | Size: 75.2 MB
...
Done! Corpus saved to: data/processed/corpus.txt
  Total samples scanned : 837,542
  Total lines written   : 424,310
  Final file size       : 283.2 MB
```

**Our test run result:** `283.2 MB` corpus with `424,310` clean lines.

> 💡 For quick testing, use `--max_size_mb 5` to get a tiny corpus in seconds.

**Scaling up later:** To use C4 instead of WikiText-103, change one line in `prepare_data.py`:
```python
dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
```

---

### 🔤 Step 2 — Train the Tokenizer

Trains a **Byte-Level BPE tokenizer** on your corpus:

```bash
# Option A: HuggingFace tokenizers (recommended)
python scripts/train_hf_tokenizer.py --corpus data/processed/corpus.txt --vocab_size 4000

# Option B: SentencePiece
python tokenizer/train_tokenizer.py --input_file data/processed/corpus.txt --vocab_size 50000
```

Then tokenize the corpus into binary format:

```bash
python scripts/tokenize_corpus.py
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--corpus` | `data/processed/corpus.txt` | Input text file |
| `--tokenizer` | `tokenizer/tokenizer.json` | Tokenizer to use |
| `--output_dir` | `data/processed` | Where to save .bin files |
| `--val_ratio` | 0.01 | Fraction of data for validation |

**Example output:**
```
Loaded tokenizer (vocab size: 8000, EOS id: 3)
  Tokenized 100,000 lines | 17,461,238 tokens
  Tokenized 200,000 lines | 35,082,109 tokens
  Tokenized 300,000 lines | 52,647,382 tokens
  Tokenized 400,000 lines | 70,250,999 tokens

Total lines: 424,310
Total tokens: 74,476,112
Train tokens: 73,731,350
Val tokens:   744,762
Saved data/processed/train.bin (147.5 MB)
Saved data/processed/val.bin (1.5 MB)
```

**Our test run result:** `74.5 million tokens` → `train.bin` (147.5 MB) + `val.bin` (1.5 MB).

---

### 🏋️ Step 4 — Train the Model

All hyperparameters are configured in `configs/train_config.yaml`:

```yaml
model:
  vocab_size: 8000        # Must match your tokenizer
  d_model: 128            # Embedding dimension
  n_layers: 4             # Transformer layers
  n_heads: 4              # Attention heads
  max_seq_len: 128        # Context window (tokens)
  dropout: 0.1
  bias: false

training:
  batch_size: 8
  gradient_accumulation_steps: 1
  max_iters: 500
  lr: 0.001
  min_lr: 0.0001
  warmup_iters: 50
  weight_decay: 0.1
  grad_clip: 1.0
  eval_interval: 100
  log_interval: 10
  checkpoint_dir: "checkpoints"

data:
  data_dir: "data/processed"
```

**Run training:**

```bash
python training/train.py --config configs/train_config.yaml
```

**Example output:**
```
Using device: cpu, dtype: float16
Loaded 73,731,350 tokens from data/processed/train.bin
Loaded 744,762 tokens from data/processed/val.bin
Initializing model...
Model parameters: 2.11 M

step 0:   train loss 9.0148, val loss 8.9822    ← random weights
iter 10:  loss 8.7215, lr 2.000e-04
iter 50:  loss 7.6243, lr 6.000e-04              ← learning!
step 100: train loss 6.8694, val loss 6.8527
step 200: train loss 6.6183, val loss 6.5813
step 300: train loss 6.3728, val loss 6.3337
step 400: train loss 6.2461, val loss 6.2844
step 499: train loss 6.1760, val loss 6.1997     ← converging ✅

saving checkpoint to checkpoints
```

> 📉 **Key signal**: If train loss decreases continuously, the model is learning. Our run went from **9.0 → 6.2**.

**Training features included:**
- ⚡ Mixed Precision (bfloat16/float16 automatic)
- 📈 Cosine LR Decay with linear warmup
- 🔄 Gradient Accumulation for larger effective batches
- ✂️ Gradient Clipping for stability
- 💾 Automatic Checkpointing every N steps

---

### 🧪 Step 5 — Run Inference

Generate text from a prompt using the trained model:

```bash
python inference/generate.py \
    --prompt "The history of" \
    --checkpoint checkpoints/ckpt.pt \
    --tokenizer tokenizer/tokenizer.json \
    --max_new_tokens 80 \
    --temperature 0.8 \
    --top_k 40
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--prompt` | *(required)* | Input text to complete |
| `--checkpoint` | `checkpoints/ckpt.pt` | Path to trained model |
| `--tokenizer` | `tokenizer/tokenizer.json` | Path to tokenizer |
| `--max_new_tokens` | 100 | Maximum tokens to generate |
| `--temperature` | 0.8 | Higher = more creative, lower = more focused |
| `--top_k` | 200 | Only sample from top K most likely tokens |

**Example output:**
```
Prompt: "The history of"
--------------------------------------------------
Output: SG @-@ season , the two of the Ure 's CP. The series
        of the next 54th century , and Cangan .
--------------------------------------------------
Tokens Used: {'prompt_tokens': 5, 'completion_tokens': 50, 'total_tokens': 55}
```

> ⚠️ Text quality depends heavily on model size and training duration. A tiny CPU model produces rough text — this is expected. On GPUs with the full 120M config, results improve dramatically.

---

### 🌐 Step 6 — Start the API Server

Deploy your model as a **FastAPI** server:

```bash
python api/app.py \
    --checkpoint checkpoints/ckpt.pt \
    --tokenizer tokenizer/tokenizer.json \
    --port 8000
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8000` | Port number |
| `--checkpoint` | `checkpoints/ckpt.pt` | Trained model |
| `--tokenizer` | `tokenizer/tokenizer.json` | Tokenizer file |

**Server output:**
```
Successfully loaded model on cpu
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

#### 📚 Swagger Documentation (Interactive)

Once the server is running, open these in your browser:

| URL | Description |
|-----|-------------|
| **http://127.0.0.1:8000/docs** | 🟢 **Swagger UI** — try the API directly in browser |
| **http://127.0.0.1:8000/redoc** | 📘 **ReDoc** — alternative docs view |

#### 📮 Test with Postman

Copy this curl into **Postman → Import → Raw Text → Continue → Import**:

```bash
curl --location 'http://127.0.0.1:8000/generate' \
--header 'Content-Type: application/json' \
--data '{"prompt": "The history of science", "max_tokens": 60, "temperature": 0.8, "top_k": 40}'
```

#### Endpoint: `POST /generate`

**Request Body:**
```json
{
    "prompt": "The history of science",
    "max_tokens": 60,
    "temperature": 0.8,
    "top_k": 40
}
```

**Response:**
```json
{
    "generated_text": "The most of his first the other time , and Nur @-@ episode ...",
    "usage": {
        "prompt_tokens": 6,
        "completion_tokens": 60,
        "total_tokens": 66
    },
    "latency_ms": 293.56
}
```

#### Request & Response Schema

| Field | Type | Default | Range | Description |
|-------|------|---------|-------|-------------|
| `prompt` | `string` | *(required)* | — | Input text to complete |
| `max_tokens` | `int` | 100 | 1–512 | Max tokens to generate |
| `temperature` | `float` | 0.7 | 0.0–2.0 | Sampling randomness |
| `top_k` | `int` | 50 | ≥ 0 | Top-K filtering |

| Response Field | Type | Description |
|----------------|------|-------------|
| `generated_text` | `string` | Model's text continuation |
| `usage.prompt_tokens` | `int` | Tokens in your input |
| `usage.completion_tokens` | `int` | Tokens generated |
| `usage.total_tokens` | `int` | Sum of both |
| `latency_ms` | `float` | Server-side processing time (ms) |

---

## � Training Results

### Run: 283 MB WikiText-103 Corpus

| Metric | Value |
|--------|-------|
| **Corpus** | 283.2 MB (424,310 lines) |
| **Tokenizer** | BPE, 8,000 vocab |
| **Total Tokens** | 74.5 million |
| **Model** | 4 layers, 128 dim, 4 heads (~2.1M params) |
| **Training** | 500 iterations on CPU |
| **Final Loss** | Train: 6.18 / Val: 6.20 |

```
Loss curve:

  9.0 ┤●
      │ \
  8.5 ┤  \
      │   \
  8.0 ┤    \
      │     \
  7.5 ┤      \
      │       \
  7.0 ┤        ╲
      │          ╲
  6.5 ┤           ╲───╲
      │                ╲────╲
  6.0 ┤                      ╲─────── ← 6.18
      └──────────────────────────────
      0   100  200  300  400  500
                iterations
```

---

## 💾 Using the Checkpoint (.pt) File

The `.pt` file contains everything needed to restore the model:

| Key | Contents |
|-----|----------|
| `checkpoint['model']` | Model weights (state_dict) |
| `checkpoint['optimizer']` | Optimizer state (for resuming training) |
| `checkpoint['model_args']` | Config dict (vocab_size, d_model, etc.) |
| `checkpoint['iter_num']` | Last training iteration number |

### Load the Model in Python

```python
import torch
from model.config import BUVNConfig
from model.model import BUVNModel

# Load checkpoint
ckpt = torch.load('checkpoints/ckpt.pt', map_location='cpu')

# Reconstruct model
config = BUVNConfig.from_dict(ckpt['model_args'])
model = BUVNModel(config)
model.load_state_dict(ckpt['model'])
model.eval()

print(f"Loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
print(f"Config: {ckpt['model_args']}")
```

### Resume Training from Checkpoint

```python
optimizer.load_state_dict(ckpt['optimizer'])
start_iter = ckpt['iter_num']
# ... continue training loop from start_iter
```

### Convert to HuggingFace Format

```bash
python scripts/convert_to_hf.py --checkpoint checkpoints/ckpt.pt --out_dir hf_model/
```

---

## 💻 Hardware Requirements

| Setup | Hardware | Batch Size | Use Case |
|-------|----------|:----------:|----------|
| 🟢 **CPU** | Any modern CPU, 8 GB RAM | 4–8 | Testing & validation (this README) |
| 🟡 **Single GPU** | RTX 3080/4080/T4 (16 GB) | 16–32 | Small-scale training runs |
| 🔵 **Cloud GPU** | Azure NC A100 v4 (40/80 GB) | 64–128 | Full 120M model training |
| 🟣 **Multi-GPU** | 4× A100 (DDP) | 256+ | Production pre-training |

### Azure Cost Estimate

| Resource | Hourly Cost | Training Time | Total Cost |
|----------|:-----------:|:------------:|:----------:|
| 1× A100 40 GB | ~$3–4/hr | ~20–30 hrs | **< $150** |

---

## 💰 Token Pricing Concept

Every API response includes a `usage` object — exactly like OpenAI's API:

```json
{
    "usage": {
        "prompt_tokens": 6,       // what you sent
        "completion_tokens": 60,  // what was generated
        "total_tokens": 66        // sum
    }
}
```

This enables per-token billing by integrating with payment systems like Stripe.

---

## ⚠️ Limitations

| Limitation | Why |
|------------|-----|
| **Not instruction-tuned** | Predicts next tokens, doesn't "answer" questions |
| **Hallucinations** | Small models fabricate facts frequently |
| **128-token context** | Limited window for CPU testing (512 in production) |
| **English only** | WikiText-103 is English-only |
| **Rough text quality** | Tiny model + limited iterations = incoherent output |

---

## 🔮 Future Roadmap

- [ ] 🎯 **Scale to 120M params** — Full config on Azure GPUs
- [ ] 📚 **Switch to C4 dataset** — 5B+ tokens for serious training
- [ ] 💬 **Supervised Fine-Tuning** — Train on prompt/response pairs
- [ ] 🧭 **RLHF / DPO** — Align model to human preferences
- [ ] ⚡ **Flash Attention 2** — Faster attention kernels
- [ ] 🖥️ **Multi-GPU DDP** — Distributed training
- [ ] 📏 **Longer context** — Extend to 2048+ tokens
- [ ] 📦 **Quantization** — INT8/INT4 for faster inference

---

<div align="center">

**Built with ❤️ by Bhuvan**

*BUVN-1.1 — Your first step toward building AI from scratch.*

</div>
