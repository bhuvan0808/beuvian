# Fine-Tuning Guide

## Pre-training vs Fine-tuning

| Stage | What It Learns | Data | Cost |
|-------|---------------|------|------|
| **Pre-training** | Language structure, grammar, world knowledge | Raw internet text (trillions of tokens) | Expensive (days/weeks) |
| **Fine-tuning** | How to follow instructions, answer questions | Curated prompt-response pairs (thousands) | Cheap (hours) |

BUVN-2.0 has completed **pre-training**. It generates fluent text but doesn't follow instructions. Fine-tuning teaches it to be useful.

## What is Instruction Tuning (SFT)?

**SFT = Supervised Fine-Tuning.** You show the model thousands of examples of "question → answer" and it learns the pattern.

### Before SFT (current BUVN-2.0):

```
Input:  "What is the capital of France?"
Output: "What is the capital of Germany? What is the capital of Spain?
         What is the population of France? These questions and more..."
         ↑ Just continues the text — doesn't answer
```

### After SFT:

```
Input:  "<user>What is the capital of France?</user><assistant>"
Output: "The capital of France is Paris.</assistant>"
         ↑ Actually answers the question
```

## Chat Template Format

For instruction tuning, we wrap conversations in a template:

```
<s><user>What is the capital of France?</user>
<assistant>The capital of France is Paris. It is the largest city in France and serves as the country's political, economic, and cultural center.</assistant></s>

<s><user>Write a haiku about rain</user>
<assistant>Drops on the window
Soft rhythm on the rooftop
Earth drinks and is still</assistant></s>
```

The model learns: when it sees `<user>...<assistant>`, it should generate a helpful response and stop at `</s>`.

## Recommended Datasets

| Dataset | Size | Source | Best For |
|---------|------|--------|----------|
| **OpenAssistant-2** | 65K conversations | Human-written | High quality multi-turn chat |
| **Alpaca** | 52K instructions | GPT-3.5 generated | Single-turn instruction following |
| **ShareGPT** | 90K conversations | Real ChatGPT conversations | Natural conversational style |
| **Dolly-15K** | 15K instructions | Human-written (Databricks) | Clean, diverse instructions |
| **SlimOrca** | 500K instructions | GPT-4 generated | Large-scale, high quality |
| **Code Alpaca** | 20K code instructions | GPT-3.5 generated | Code generation (for SRVN) |

**Recommended start:** OpenAssistant-2 + Alpaca combined (~117K examples).

## How to Prepare Instruction Data

### 1. Download dataset

```python
from datasets import load_dataset

# Example: Alpaca
ds = load_dataset("tatsu-lab/alpaca", split="train")

# Each sample has: instruction, input (optional), output
# Convert to chat format:
for sample in ds:
    prompt = sample['instruction']
    if sample['input']:
        prompt += f"\n{sample['input']}"
    response = sample['output']

    # Format as: <s><user>{prompt}</user><assistant>{response}</assistant></s>
```

### 2. Tokenize with chat template

```python
def format_chat(prompt, response, tokenizer):
    text = f"<s><user>{prompt}</user>\n<assistant>{response}</assistant></s>"
    return tokenizer.encode(text).ids
```

### 3. Save as binary (same format as pre-training)

```python
import numpy as np
all_tokens = []
for sample in dataset:
    tokens = format_chat(sample['prompt'], sample['response'], tokenizer)
    all_tokens.extend(tokens)

arr = np.array(all_tokens, dtype=np.uint16)
arr.tofile("data/sft/train.bin")
```

## How to Fine-tune (Modify train.py)

Key differences from pre-training:

| Setting | Pre-training | Fine-tuning |
|---------|-------------|-------------|
| **Learning rate** | 6e-4 | **1e-5 to 5e-5** (much lower) |
| **Steps** | 15,000 | **500–3,000** (1–3 epochs over data) |
| **Warmup** | 500 steps | **50–100 steps** |
| **Dropout** | 0.0 | **0.05–0.1** (slight regularization) |
| **Weight decay** | 0.1 | **0.01** |
| **Starting weights** | Random | **Load from pre-trained checkpoint** |

### Fine-tuning config example:

```yaml
model:
  vocab_size: 32000
  d_model: 768
  n_layers: 12
  n_heads: 12
  max_seq_len: 1024
  dropout: 0.05
  bias: false

training:
  batch_size: 8
  gradient_accumulation_steps: 8
  max_iters: 2000
  lr: 0.00003           # 30x lower than pre-training
  min_lr: 0.000003
  warmup_iters: 100
  weight_decay: 0.01
  beta1: 0.9
  beta2: 0.95
  grad_clip: 1.0
  eval_interval: 200
  eval_iters: 20
  log_interval: 10
  checkpoint_dir: "checkpoints_sft"

data:
  data_dir: "data/sft"
```

## What is RLHF?

**RLHF = Reinforcement Learning from Human Feedback.** It's the technique that turned GPT-3 into ChatGPT.

```
Step 1: Collect human preferences
        Show humans two responses to the same prompt
        Human picks the better one
        Repeat thousands of times

Step 2: Train a Reward Model
        A separate model that scores responses
        Trained to predict which response humans prefer

Step 3: Optimize with PPO
        The language model generates responses
        The reward model scores them
        PPO (Proximal Policy Optimization) updates the model
        to generate higher-scoring responses
```

RLHF is complex to implement but produces the most helpful, safe models.

## What is DPO?

**DPO = Direct Preference Optimization.** A simpler alternative to RLHF.

Instead of training a separate reward model + PPO, DPO directly optimizes the language model on preference pairs:

```
Prompt: "Explain gravity"
Preferred:  "Gravity is a force that attracts objects toward each other..."
Rejected:   "Gravity is when stuff falls down I guess lol..."

DPO loss pushes the model to increase probability of preferred
response and decrease probability of rejected response.
```

**Advantages over RLHF:**
- No reward model needed
- No PPO (simpler, more stable)
- Often works just as well
- Recommended for small/medium models

## What is LoRA?

**LoRA = Low-Rank Adaptation.** A parameter-efficient fine-tuning method.

Instead of updating ALL 109M parameters, LoRA:
1. Freezes all pre-trained weights
2. Adds tiny trainable matrices (rank 4–16) to attention layers only
3. Trains only ~0.1–1% of total parameters

```
Pre-trained weight W (768 × 768 = 590K params) → FROZEN
LoRA matrices A (768 × 8) + B (8 × 768) = 12K params → TRAINABLE

Output = W·x + A·B·x   (original + small correction)
```

**Benefits:**
- 10–100x fewer trainable parameters
- Fits on much smaller GPUs
- Multiple LoRA adapters for different tasks
- Original model unchanged

## Planned: SRVN (Code Agent)

SRVN takes BUVN's pre-trained weights and fine-tunes for code:

| Phase | Data | Technique |
|-------|------|-----------|
| Code pre-training | The Stack v2 (GitHub, 500GB+) | Continue pre-training on code |
| FIM training | Same data, 50% reformatted | Fill-in-the-Middle objective |
| Instruction tuning | Code Alpaca, CodeInstruct | SFT on coding tasks |
| Agentic training | Tool-use traces, ReAct format | Plan → Code → Test → Debug loop |

## Planned: MNI (Finance)

MNI takes BUVN's pre-trained weights and trains on financial data:

| Phase | Data | Technique |
|-------|------|-----------|
| Domain pre-training | SEC filings, earnings transcripts | Continue pre-training |
| Sentiment training | Financial news with labels | SFT on sentiment classification |
| Prediction training | Historical price data + outcomes | Regression head for price direction |
| Report generation | Analyst reports as examples | SFT on generating research summaries |
