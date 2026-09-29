import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional


class TextDataset(Dataset):
    def __init__(self, data_path: str, seq_len: int):
        self.seq_len = seq_len
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        
    def __len__(self):
        return len(self.data) // self.seq_len
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = torch.from_numpy(self.data[start:end].astype(np.int64))
        
        x = chunk[:-1]
        y = chunk[1:]
        
        return x, y


def get_dataloader(data_path: str, seq_len: int, batch_size: int, shuffle: bool = True, num_workers: int = 0):
    dataset = TextDataset(data_path, seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader
