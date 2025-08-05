import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Callable
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mesh_to_grid import process_glb_to_uv_grid

class GLBDataset(Dataset):
    """Dataset for loading GLB files with on-demand processing to 512x512x9 arrays.
    
    This dataset loads .glb files from a directory and processes them using
    the process_glb_to_uv_grid function to create 512x512x9 arrays.
    Returns a tuple (x, 0) to remain compatible with existing training code.
    """

    def __init__(self, glb_dir: str, transform: Optional[Callable] = None):
        self.glb_dir = Path(glb_dir)
        if not self.glb_dir.exists():
            raise ValueError(f"GLB directory does not exist: {glb_dir}")
        
        # Find all .glb files in the directory
        self.glb_files = list(self.glb_dir.glob("*.glb"))
        if len(self.glb_files) == 0:
            raise ValueError(f"No .glb files found in directory: {glb_dir}")
        
        # Sort for consistent ordering
        self.glb_files.sort()
        self.transform = transform
        
        print(f"Found {len(self.glb_files)} GLB files in {glb_dir}")

    def __len__(self) -> int:
        return len(self.glb_files)

    def __getitem__(self, idx: int):
        glb_path = self.glb_files[idx]
        
        try:
            # Process GLB file to 512x512x9 array
            final_array, _, _, _ = process_glb_to_uv_grid(str(glb_path))
            
            # Convert to tensor and ensure channel-first format (9, 512, 512)
            x = torch.from_numpy(final_array.astype(np.float32))
            x = x.permute(2, 0, 1)  # Change from (512, 512, 9) to (9, 512, 512)
            
            if self.transform:
                x = self.transform(x)
                
            return x, 0
            
        except Exception as e:
            print(f"Error processing {glb_path}: {e}")
            # Return zeros in case of error to avoid breaking training
            x = torch.zeros(9, 512, 512, dtype=torch.float32)
            return x, 0