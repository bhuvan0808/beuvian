# Usage Guide

## Quick Start

```bash
# Generate text from a trained model
python inference/generate.py \
    --prompt "The future of AI is" \
    --checkpoint checkpoints/ckpt_best.pt \
    --tokenizer tokenizer/tokenizer_32k.json

# Start the API server
python api/app.py \
    --checkpoint checkpoints/ckpt_best.pt \
    --tokenizer tokenizer/tokenizer_32k.json
```

## CLI Inference — generate.py

```bash
python inference/generate.py \
    --prompt "The history of science" \
    --checkpoint checkpoints/ckpt_best.pt \
    --tokenizer tokenizer/tokenizer_32k.json \
    --max_new_tokens 150 \
    --temperature 0.7 \
    --top_k 50
```

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--prompt` | *(required)* | Input text to complete |
| `--checkpoint` | `checkpoints/ckpt.pt` | Path to trained model checkpoint |
| `--tokenizer` | `tokenizer/tokenizer.model` | Path to tokenizer file (.json or .model) |
| `--max_new_tokens` | `100` | Maximum tokens to generate (1-1024) |
| `--temperature` | `0.8` | Sampling temperature (0 = greedy, higher = more creative) |
| `--top_k` | `200` | Only sample from top K tokens |

### Supported Tokenizer Formats

| Extension | Format | Library |
|-----------|--------|---------|
| `.json` | HuggingFace tokenizers | `tokenizers` |
| `.model` | SentencePiece | `sentencepiece` |
| `.pkl` | Character-level (pickle) | Built-in |

## API Server — app.py

```bash
python api/app.py \
    --checkpoint checkpoints/ckpt_best.pt \
    --tokenizer tokenizer/tokenizer_32k.json \
    --port 8000
```

### Server Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8000` | Port number |
| `--checkpoint` | `checkpoints/ckpt.pt` | Model checkpoint |
| `--tokenizer` | `tokenizer/tokenizer.model` | Tokenizer file |
| `--device` | Auto-detect | Force `cpu` or `cuda` |

### API Documentation

Once running, open in your browser:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### POST /generate

**Request:**
```json
{
    "prompt": "The future of artificial intelligence is",
    "max_tokens": 100,
    "temperature": 0.7,
    "top_k": 50
}
```

**Response:**
```json
{
    "generated_text": "The future of artificial intelligence is...",
    "usage": {
        "prompt_tokens": 8,
        "completion_tokens": 100,
        "total_tokens": 108
    },
    "latency_ms": 490.5
}
```

**Request Fields:**

| Field | Type | Default | Range | Description |
|-------|------|---------|-------|-------------|
| `prompt` | string | *(required)* | — | Input text |
| `max_tokens` | int | 100 | 1–512 | Max generation length |
| `temperature` | float | 0.7 | 0.0–2.0 | Randomness control |
| `top_k` | int | 50 | >= 0 | Token pool filter |

**curl Example:**
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The history of science", "max_tokens": 60, "temperature": 0.8, "top_k": 40}'
```

## Loading a Checkpoint in Python

```python
import torch
from model.config import BUVNConfig
from model.model import BUVNModel

# Load checkpoint
ckpt = torch.load('checkpoints/ckpt_best.pt', map_location='cpu', weights_only=False)

# Handle torch.compile prefix if present
state_dict = ckpt['model']
for k in list(state_dict.keys()):
    if k.startswith('_orig_mod.'):
        state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)

# Build model
config = BUVNConfig.from_dict(ckpt['model_args'])
model = BUVNModel(config)
model.load_state_dict(state_dict)
model.eval()

print(f"Loaded model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
print(f"Trained for {ckpt['iter_num']} steps, best val loss: {ckpt['best_val_loss']:.4f}")
```

## Sampling Parameters Explained

### Temperature

Controls randomness. Divides logits by temperature before softmax.

| Value | Effect | Use Case |
|-------|--------|----------|
| 0.0 | Greedy (always pick highest probability) | Factual, deterministic |
| 0.3 | Low creativity, high coherence | Formal writing |
| 0.7 | Balanced (default) | General use |
| 1.0 | Full model distribution | Creative writing |
| 1.5+ | Very random, may produce nonsense | Brainstorming |

### Top-K

Only consider the K most likely next tokens. Eliminates low-probability noise.

| Value | Effect |
|-------|--------|
| 1 | Same as greedy |
| 10 | Very focused, repetitive |
| 50 | Good balance (default) |
| 200 | More diverse |
| 0 | Disabled (consider all tokens) |

### Top-P (Nucleus Sampling)

Keep the smallest set of tokens whose cumulative probability exceeds P.

| Value | Effect |
|-------|--------|
| 0.1 | Very focused |
| 0.5 | Moderate diversity |
| 0.9 | Standard (recommended) |
| 1.0 | Disabled |

Top-P adapts automatically — for confident predictions it considers fewer tokens, for uncertain predictions it considers more. Generally preferred over Top-K.

## Example Outputs

### 125M Model (BUVN-2.0)

**Prompt:** "The history of artificial intelligence began"
> The number of people living with heart disease in the United States is projected to increase by nearly 20 million every year, according to the Centers for Disease Control and Prevention...

**Prompt:** "The president of the United States announced"
> Here at The Ritz and Suites, we are proud to offer a variety of unique and unique packages. Our experienced staff is here to help you find the perfect vacation...

*Note: The model generates fluent, grammatically correct text in a web-text style. It does not follow the prompt topic because it has not been instruction-tuned. See [fine-tuning.md](fine-tuning.md) for how to add instruction following.*
