import os
import math
import time
import argparse
import torch
import random
import numpy as np
from contextlib import nullcontext

# Local imports
from training.config import AppConfig
from training.dataloader import get_train_val_dataloaders
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.config import BUVNConfig
from model.model import BUVNModel


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True  # safe with fixed seed + fixed input sizes


def get_lr(it: int, config: AppConfig):
    """Cosine learning rate schedule with warmup."""
    t_cfg = config.training
    if t_cfg.warmup_iters > 0 and it < t_cfg.warmup_iters:
        return t_cfg.lr * (it + 1) / t_cfg.warmup_iters
    if it > t_cfg.max_iters:
        return t_cfg.min_lr
    decay_ratio = (it - t_cfg.warmup_iters) / (t_cfg.max_iters - t_cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return t_cfg.min_lr + coeff * (t_cfg.lr - t_cfg.min_lr)


@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, eval_iters, ctx):
    """Evaluates the model and estimates loss + perplexity on train/val splits."""
    out = {}
    model.eval()
    for split, dataloader in [('train', train_loader), ('val', val_loader)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = dataloader.get_batch()
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        avg_loss = losses.mean().item()
        out[split] = avg_loss
        out[f'{split}_ppl'] = math.exp(min(avg_loss, 20.0))  # cap to avoid overflow
    model.train()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true", help="Use torch.compile (PyTorch 2.0+)")
    parser.add_argument("--from_checkpoint", type=str, default=None,
                        help="Load pre-trained weights from checkpoint (for fine-tuning)")
    args = parser.parse_args()

    # Reproducibility
    set_seed(args.seed)

    app_cfg = AppConfig.load(args.config)
    t_cfg = app_cfg.training
    m_cfg_dict = app_cfg.model

    # 1. Device and Context Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)

    print(f"Using device: {device}, dtype: {dtype}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 2. Data Loaders
    train_loader, val_loader = get_train_val_dataloaders(
        data_dir=app_cfg.data.data_dir,
        batch_size=t_cfg.batch_size,
        seq_len=m_cfg_dict['max_seq_len'],
        device=device
    )

    # 3. Model Initialization
    print("Initializing model...")
    m_config = BUVNConfig.from_dict(m_cfg_dict)
    model = BUVNModel(m_config)

    # Load pre-trained weights for fine-tuning
    if args.from_checkpoint:
        print(f"Loading pre-trained weights from {args.from_checkpoint}...")
        ckpt = torch.load(args.from_checkpoint, map_location=device, weights_only=False)
        state_dict = ckpt['model']
        # Handle torch.compile _orig_mod. prefix
        for k in list(state_dict.keys()):
            if k.startswith('_orig_mod.'):
                state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
        model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded weights from step {ckpt.get('iter_num', '?')}, "
              f"val loss {ckpt.get('best_val_loss', '?'):.4f}")

    model.to(device)

    n_params = model.get_num_params()
    print(f"Model parameters: {n_params/1e6:.2f}M (non-embedding)")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Optional: torch.compile for speedups on PyTorch 2.0+
    if args.compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # 4. Optimizer Setup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=t_cfg.lr,
        betas=(t_cfg.beta1, t_cfg.beta2),
        weight_decay=t_cfg.weight_decay
    )

    # GradScaler only needed for float16, bfloat16 doesn't need it
    scaler = torch.amp.GradScaler(device, enabled=(dtype == 'float16'))

    # 5. Training Loop
    os.makedirs(t_cfg.checkpoint_dir, exist_ok=True)

    best_val_loss = float('inf')
    tokens_per_iter = t_cfg.batch_size * m_cfg_dict['max_seq_len'] * t_cfg.gradient_accumulation_steps
    print(f"Tokens per iteration: {tokens_per_iter:,}")
    print(f"Total tokens to process: {tokens_per_iter * t_cfg.max_iters:,}")
    print(f"Starting training for {t_cfg.max_iters} iterations...")
    print("=" * 60)

    t0 = time.time()

    for iter_num in range(t_cfg.max_iters):

        # Determine and set learning rate
        lr = get_lr(iter_num, app_cfg)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Evaluation and checkpointing
        if iter_num % t_cfg.eval_interval == 0 or iter_num == t_cfg.max_iters - 1:
            losses = estimate_loss(model, train_loader, val_loader, t_cfg.eval_iters, ctx)
            print(f"step {iter_num:>5d}: train loss {losses['train']:.4f} (ppl {losses['train_ppl']:.1f}), "
                  f"val loss {losses['val']:.4f} (ppl {losses['val_ppl']:.1f})")

            # Save best checkpoint
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'model_args': m_cfg_dict,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': app_cfg,
                }
                ckpt_path = os.path.join(t_cfg.checkpoint_dir, 'ckpt_best.pt')
                torch.save(checkpoint, ckpt_path)
                print(f"  -> new best val loss! saved to {ckpt_path}")

            # Also save latest checkpoint (for resume)
            if iter_num > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'model_args': m_cfg_dict,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': app_cfg,
                }
                torch.save(checkpoint, os.path.join(t_cfg.checkpoint_dir, 'ckpt.pt'))

        # Forward-backward with gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        for micro_step in range(t_cfg.gradient_accumulation_steps):
            X, Y = train_loader.get_batch()
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / t_cfg.gradient_accumulation_steps
            scaler.scale(loss).backward()

        # Gradient clipping
        if t_cfg.grad_clip > 0.0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t_cfg.grad_clip)
        else:
            grad_norm = None

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        # NaN detection
        lossf = loss.item() * t_cfg.gradient_accumulation_steps
        if math.isnan(lossf) or math.isinf(lossf):
            print(f"WARNING: NaN/Inf loss at iter {iter_num}, skipping...")
            continue

        # Timing and logging
        if iter_num % t_cfg.log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num > 0:
                tokens_per_sec = tokens_per_iter / dt * t_cfg.log_interval
                mfu = model.estimate_mfu(t_cfg.batch_size * t_cfg.gradient_accumulation_steps, dt / t_cfg.log_interval)
                grad_str = f", grad_norm {grad_norm:.2f}" if grad_norm is not None else ""
                print(f"iter {iter_num:>5d}: loss {lossf:.4f}, lr {lr:.2e}, "
                      f"{tokens_per_sec:,.0f} tok/s, MFU {mfu*100:.1f}%{grad_str}")

    # Final summary
    total_time = time.time()
    print("=" * 60)
    print(f"Training complete! Best val loss: {best_val_loss:.4f} (ppl {math.exp(min(best_val_loss, 20.0)):.1f})")
    print(f"Checkpoints saved to: {t_cfg.checkpoint_dir}/")


if __name__ == "__main__":
    main()
