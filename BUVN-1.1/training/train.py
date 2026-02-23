import os
import math
import argparse
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext

# Local imports
from training.config import AppConfig
from training.dataloader import get_train_val_dataloaders
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.config import BUVNConfig
from model.model import BUVNModel

def get_lr(it: int, config: AppConfig):
    """Cosine learning rate schedule with warmup."""
    t_cfg = config.training
    if it < t_cfg.warmup_iters:
        return t_cfg.lr * it / t_cfg.warmup_iters
    if it > t_cfg.max_iters:
        return t_cfg.min_lr
    decay_ratio = (it - t_cfg.warmup_iters) / (t_cfg.max_iters - t_cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return t_cfg.min_lr + coeff * (t_cfg.lr - t_cfg.min_lr)

@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, eval_iters, ctx):
    """Evaluates the model and estimates loss on train/val splits."""
    out = {}
    model.eval()
    for split, dataloader in [('train', train_loader), ('val', val_loader)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
             X, Y = dataloader.get_batch()
             with ctx:
                 logits, loss = model(X, Y)
             losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="BUVN-1.1/configs/train_config.yaml")
    args = parser.parse_args()

    app_cfg = AppConfig.load(args.config)
    t_cfg = app_cfg.training
    m_cfg_dict = app_cfg.model
    
    # 1. Device and Context Setup
    # Assumes single GPU. For DDP, you would modify this block to use torch.distributed
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)

    print(f"Using device: {device}, dtype: {dtype}")

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
    model.to(device)

    # Optional: torch.compile for massive speedups on PyTorch 2.0+ (Linux/Ampere+ usually)
    # print("Compiling model...")
    # model = torch.compile(model)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    # 4. Optimizer Setup
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=t_cfg.lr, 
        betas=(t_cfg.beta1, t_cfg.beta2), 
        weight_decay=t_cfg.weight_decay
    )
    
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    # 5. Training Loop
    os.makedirs(t_cfg.checkpoint_dir, exist_ok=True)
    
    X, Y = train_loader.get_batch()
    
    for iter_num in range(t_cfg.max_iters):
        
        # Determine and set learning rate
        lr = get_lr(iter_num, app_cfg)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        # Evaluation and checkpointing
        if iter_num % t_cfg.eval_interval == 0 or iter_num == t_cfg.max_iters - 1:
             losses = estimate_loss(model, train_loader, val_loader, t_cfg.eval_iters, ctx)
             print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
             
             # Save checkpoint
             if iter_num > 0:
                 checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': m_cfg_dict,
                    'iter_num': iter_num,
                    'config': app_cfg,
                 }
                 print(f"saving checkpoint to {t_cfg.checkpoint_dir}")
                 torch.save(checkpoint, os.path.join(t_cfg.checkpoint_dir, 'ckpt.pt'))
                 
        # Forward backward
        for micro_step in range(t_cfg.gradient_accumulation_steps):
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / t_cfg.gradient_accumulation_steps
            
            # Prefetch next batch while backward accumulates computation on GPU
            X, Y = train_loader.get_batch()
            
            # backward pass
            scaler.scale(loss).backward()
            
        # gradient clipping
        if t_cfg.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), t_cfg.grad_clip)
            
        # optimizer step
        scaler.step(optimizer)
        scaler.update()
        
        # flush gradients
        optimizer.zero_grad(set_to_none=True)
        
        # logging
        if iter_num % t_cfg.log_interval == 0:
            lossf = loss.item() * t_cfg.gradient_accumulation_steps
            print(f"iter {iter_num}: loss {lossf:.4f}, lr {lr:e}")

if __name__ == "__main__":
    main()
