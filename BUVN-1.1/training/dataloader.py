import os
import numpy as np
import torch

class MemmapDataLoader:
    """
    A simple dataloader that reads memory-mapped token arrays and yields
    batches of contiguous token sequences (X) and their shifted targets (Y).
    """
    def __init__(self, data_path: str, batch_size: int, seq_len: int, device: str):
        self.data_path = data_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        
        if not os.path.exists(data_path):
             raise FileNotFoundError(f"Binary token file not found at {data_path}. Run prepare_data.py first.")
             
        # memory map the binary file
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.n_tokens = len(self.data)
        print(f"Loaded {self.n_tokens:,} tokens from {data_path}")

    def get_batch(self):
        """Samples random chunks from the dataset."""
        # Random starting indices
        ix = torch.randint(len(self.data) - self.seq_len - 1, (self.batch_size,))
        
        # x is the sequence, y is the sequence shifted by 1
        x = torch.stack([torch.from_numpy((self.data[i:i+self.seq_len]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((self.data[i+1:i+1+self.seq_len]).astype(np.int64)) for i in ix])
        
        if self.device == 'cuda':
            # pin arrays x,y to cuda
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
            
        return x, y

def get_train_val_dataloaders(data_dir: str, batch_size: int, seq_len: int, device: str):
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")
    
    train_loader = MemmapDataLoader(train_path, batch_size, seq_len, device)
    val_loader = MemmapDataLoader(val_path, batch_size, seq_len, device)
    
    return train_loader, val_loader
