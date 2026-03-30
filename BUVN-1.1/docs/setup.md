# Setup & Installation

## Prerequisites

- **Python** 3.10 or higher
- **PyTorch** 2.0+ (2.6+ recommended for torch.compile)
- **CUDA** 11.8+ (optional — CPU training works but is slow)
- **Git**

## 1. Clone the Repository

```bash
git clone https://github.com/bhuvan0808/beuvian.git
cd beuvian/BUVN-1.1
```

## 2. Create Virtual Environment

```bash
python -m venv venv

# Linux / Mac:
source venv/bin/activate

# Windows PowerShell:
venv\Scripts\activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Key packages installed:

| Package | Purpose |
|---------|---------|
| `torch>=2.0.0` | Neural network framework |
| `transformers` | HuggingFace utilities |
| `datasets` | Streaming datasets from HuggingFace |
| `tokenizers` | BPE tokenizer training |
| `numpy` | Numerical operations |
| `fastapi` | API server |
| `uvicorn` | ASGI server for FastAPI |
| `pyyaml` | Config file parsing |

## 4. Set PYTHONPATH (Critical)

This must be done in **every new terminal session**:

```bash
# Linux / Mac:
export PYTHONPATH=$(pwd)

# Windows PowerShell:
$env:PYTHONPATH = "C:\full\path\to\BUVN-1.1"
```

Without this, you'll get `ModuleNotFoundError: No module named 'model'`.

## 5. Verify Installation

```bash
# Check PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__}')"

# Check CUDA (if GPU available)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Check model imports
python -c "from model.config import BUVNConfig; from model.model import BUVNModel; print('Imports OK')"
```

## 6. GPU Verification

```bash
# Check GPU details
nvidia-smi

# Check PyTorch sees the GPU
python -c "
import torch
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print(f'BF16 supported: {torch.cuda.is_bf16_supported()}')
print(f'torch.compile available: {hasattr(torch, \"compile\")}')
"
```

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: No module named 'model'` | PYTHONPATH not set | Run `export PYTHONPATH=$(pwd)` from BUVN-1.1 directory |
| `ImportError: attempted relative import` | Running script from wrong directory | Always run from BUVN-1.1 root |
| `torch.cuda.is_available()` returns False | CUDA not installed or wrong PyTorch | Install PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu128` |
| `RuntimeError: CUDA out of memory` | Batch size too large | Reduce `batch_size` in config YAML |
| `WeightsUnpickler error` | PyTorch 2.6+ safe loading | Use `weights_only=False` in `torch.load()` |
| Tokenizer not found | Haven't run tokenizer training | Run Step 2 of the pipeline first |
