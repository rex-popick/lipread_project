import os
import numpy as np
import torch
from torch.utils.data import Dataset

class CachedLipreadingDataset(Dataset):
    def __init__(self, cache_dir, labels, transform=None):
        super().__init__()
        self.cache_dir = cache_dir
        self.transform = transform
        self.labels    = labels
        self.files     = sorted(f for f in os.listdir(cache_dir)
                                if f.endswith(".npy"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        # print("Loading", fname)
        arr   = np.load(os.path.join(self.cache_dir, fname))    # [T,112,112,3]
        vid   = torch.from_numpy(arr).permute(3,0,1,2).float() / 255.0
        transcript = self.labels.get(fname.replace(".npy", ".mpg"), "")
        if self.transform:
            vid = self.transform(vid)
        return vid, transcript