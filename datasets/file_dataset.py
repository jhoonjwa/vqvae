import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable

class FileDataset(Dataset):
    """Generic dataset for loading images stored in a NumPy ``.npy`` file.

    The file is expected to contain an array of shape ``(N, C, H, W)`` or
    ``(N, H, W, C)``.  This loader keeps the data in memory and returns a
    tuple ``(x, 0)`` to remain compatible with existing training code that
    expects a target value.
    """

    def __init__(self, file_path: str, transform: Optional[Callable] = None):
        data = np.load(file_path)
        if data.ndim != 4:
            raise ValueError(
                f"Expected data of shape (N,C,H,W) or (N,H,W,C), got {data.shape}")
        # ensure channel-first layout
        if data.shape[1] != 9 and data.shape[-1] == 9:
            data = np.transpose(data, (0, 3, 1, 2))
        self.data = data.astype(np.float32)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.data[idx])
        if self.transform:
            x = self.transform(x)
        return x, 0
