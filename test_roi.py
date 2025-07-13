# test_roi.py

import os
import json
import matplotlib.pyplot as plt
from datasets import LipreadingDataset

print("⚙️  Running test_roi.py")
print("  • Files in data/videos:", os.listdir("data/videos"))

# Load labels
with open("labels.json", "r") as f:
    labels = json.load(f)

# Create the dataset
ds = LipreadingDataset(
    video_dir="data/videos",
    labels=labels
)
print(f"  • Found {len(ds)} videos")

# Guard against empty dataset
if len(ds) == 0:
    raise RuntimeError("No videos found in data/videos/")

# Grab the first sample
video_tensor, transcript = ds[0]
print("  • Transcript:", transcript)

# Warn if no face detected (fallback tensor is zeros)
if video_tensor.sum() == 0:
    print("⚠️  No face detected in sample 0—check your video and MediaPipe confidence settings.")

# Show the first frame
frame0 = video_tensor[:, 0, :, :].permute(1, 2, 0).numpy()
plt.imshow(frame0)
plt.axis("off")
plt.title("First mouth crop")
plt.show()