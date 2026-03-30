---
language:
- en
license: mit
tags:
- text-generation
- transformer
- pytorch
- instruction-tuned
- sft
- chat
base_model: bhuvan0808/buvn-2.0
datasets:
- tatsu-lab/alpaca
- OpenAssistant/oasst2
pipeline_tag: text-generation
---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=2,6,11,20,30&height=200&section=header&text=BUVN-2.0-SFT&fontSize=70&fontColor=fff&animation=fadeIn&fontAlignY=30&desc=Instruction-Tuned%20Foundation%20Model%20%7C%20Chat%20Format&descAlignY=55&descSize=18" width="100%"/>

[![Base Model](https://img.shields.io/badge/Base-BUVN--2.0%20(109.5M)-blue?style=for-the-badge&labelColor=0d1117)](https://huggingface.co/bhuvan0808/buvn-2.0)
[![SFT](https://img.shields.io/badge/SFT-Alpaca%20+%20OpenAssistant-green?style=for-the-badge&labelColor=0d1117)](https://huggingface.co/bhuvan0808/buvn-2.0-sft)
[![Val Loss](https://img.shields.io/badge/Val%20PPL-6.3-purple?style=for-the-badge&labelColor=0d1117)](https://huggingface.co/bhuvan0808/buvn-2.0-sft)

</div>

## What is BUVN-2.0-SFT?

This is the **instruction-tuned** version of [BUVN-2.0](https://huggingface.co/bhuvan0808/buvn-2.0), fine-tuned on conversational data from **Alpaca** (52K instructions) and **OpenAssistant** (13.9K English conversation pairs).

The model learns to follow the chat template format and generate responses in a conversational style.

## Training Details

| Setting | Value |
|---------|:-----:|
| **Base Model** | [buvn-2.0](https://huggingface.co/bhuvan0808/buvn-2.0) (109.5M params) |
| **SFT Data** | 52,002 Alpaca + 13,892 OpenAssistant = **65,894 examples** |
| **Total Tokens** | 9,194,803 |
| **Training Steps** | 3,000 |
| **Learning Rate** | 3×10⁻⁵ (20x lower than pre-training) |
| **Batch Size** | 32 × 4 grad accum = 128 effective |
| **Best Val Loss** | **1.8337 (PPL 6.3)** |
| **Training Time** | ~25 min on H100 NVL |
| **Precision** | bfloat16 + torch.compile |

### Training Progress

```
Step     0: val loss 3.19 (ppl 24.4) ← loaded pre-trained weights
Step   200: val loss 1.98 (ppl  7.2) ← rapidly learning chat format
Step   600: val loss 1.87 (ppl  6.5)
Step  1000: val loss 1.85 (ppl  6.4)
Step  2000: val loss 1.84 (ppl  6.3)
Step  3000: val loss 1.83 (ppl  6.3) ← converged
```

## Chat Template

The model was trained with this conversation format:

```
<|user|>
What is the capital of France?
<|end|>
<|assistant|>
The capital of France is Paris.
<|end|>
```

## Honest Assessment

At **109.5M parameters**, the model successfully learns the chat format but has limited instruction-following accuracy. This is a known limitation of small models:

| Model Size | Instruction Quality |
|:---:|---|
| **109.5M (this model)** | Learns format; answers often tangential to the question |
| 350M+ | Basic Q&A works for simple questions |
| 1B+ | Solid instruction following |
| 3B+ (Phi-2, TinyLlama) | Good quality, genuinely useful |
| 7B+ (LLaMA, Mistral) | Strong instruction following |

The model needs to be scaled to 350M+ parameters for meaningful instruction following. The SFT pipeline and infrastructure are validated and ready for scaling.

## Files

| File | Size | Description |
|------|:----:|-------------|
| `buvn_2.0_sft_best.pt` | 1.31 GB | SFT fine-tuned checkpoint |
| `tokenizer_32k.json` | 2.2 MB | 32K BPE tokenizer |
| `config.json` | ~200 B | Model hyperparameters |
| `sft_meta.json` | ~500 B | Training data statistics |

## The Beuvian Ecosystem

| Model | Status | Description |
|:-----:|:------:|-------------|
| [BUVN-2.0](https://huggingface.co/bhuvan0808/buvn-2.0) | ✅ Released | Foundation model (PPL 29.19, beats GPT-2 Small) |
| **BUVN-2.0-SFT (this)** | ✅ Released | Instruction-tuned version |
| SRVN | 🔜 Planned | Code agent (fine-tuned on code data) |
| MNI | 🔜 Planned | Finance model (market data, SEC filings) |

## Links

- **Base model:** [bhuvan0808/buvn-2.0](https://huggingface.co/bhuvan0808/buvn-2.0)
- **GitHub:** [bhuvan0808/beuvian](https://github.com/bhuvan0808/beuvian)
- **Documentation:** [docs/](https://github.com/bhuvan0808/beuvian/tree/main/BUVN-1.1/docs)

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=2,6,11,20,30&height=100&section=footer&animation=twinkling" width="100%"/>

**Built by Bhuvan** | [Beuvian AI Ecosystem](https://github.com/bhuvan0808/beuvian)

</div>
