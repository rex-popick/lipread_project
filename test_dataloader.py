# test_dataloader.py

import os
import json
import torch
import matplotlib.pyplot as plt

from datasets import LipreadingDataset
from dataloaders import get_loader

# 0. Check folder contents
video_files = sorted(os.listdir("data/videos"))
print("⚙️  Files in data/videos:", video_files)

# 1. Load labels.json
with open("labels.json", "r") as f:
    labels = json.load(f)
print(f"⚙️  Loaded {len(labels)} labels")

# 2. Instantiate dataset
ds = LipreadingDataset(video_dir="data/videos", labels=labels)
print(f"⚙️  Dataset length: {len(ds)}")

if len(ds) == 0:
    raise RuntimeError("❌  No videos found or labels.json keys don't match filenames!")

# 3. Instantiate DataLoader
loader = get_loader(ds, batch_size=4, shuffle=False, num_workers=0)
print("⚙️  DataLoader ready — fetching one batch...")

# 4. Grab and inspect a single batch
for vids, lengths, transcripts in loader:
    print("→ vids.shape   :", vids.shape)      # [B, 3, F, 112, 112]
    print("→ lengths     :", lengths)          # list of frame counts
    print("→ transcripts :", transcripts)       # list of strings
    # visualize the middle frame of the first sample
    mid = lengths[0] // 2
    frame = vids[0, :, mid, :, :].permute(1,2,0).numpy()
    plt.imshow(frame)
    plt.axis("off")
    plt.title(f"Sample 0 • frame {mid}")
    break