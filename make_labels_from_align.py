# make_labels_from_align.py

import os
import json

ALIGN_DIR = "data/grid_align"       # folder with all .align files
VIDEO_DIR = "data/videos_small"     # folder with your 10 .mpg test clips

# 1. Build a set of the video basenames you care about
video_files = {
    os.path.splitext(f)[0]  # strip extension
    for f in os.listdir(VIDEO_DIR)
    if f.lower().endswith(".mpg")
}

labels = {}
for fname in os.listdir(ALIGN_DIR):
    if not fname.endswith(".align"):
        continue

    speaker_sent, _ = fname.rsplit(".", 1)
    # Only proceed if we have that video in VIDEO_DIR
    if speaker_sent not in video_files:
        continue

    video_name = speaker_sent + ".mpg"

    words = []
    with open(os.path.join(ALIGN_DIR, fname), "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            _, _, token = parts
            token = token.lower()
            if token not in ("sil", "sp"):
                words.append(token)

    labels[video_name] = " ".join(words)

# 2. Write out only the filtered labels
with open("labels.json", "w") as out:
    json.dump(labels, out, indent=2)

print(f"✔️  Generated labels.json with {len(labels)} entries")