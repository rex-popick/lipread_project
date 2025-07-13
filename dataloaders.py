# dataloaders.py

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

def collate_fn(batch, max_frames=75):
    videos, transcripts = zip(*batch)
    # videos: list of [3, T_i, 112, 112]
    # Pad/truncate along dim=1 (time)
    vids_padded = []
    lengths = []
    for vid in videos:
        T = vid.size(1)
        if T > max_frames:
            vid = vid[:, :max_frames]
        else:
            pad = torch.zeros((3, max_frames - T, 112, 112))
            vid = torch.cat([vid, pad], dim=1)
        vids_padded.append(vid)
        lengths.append(min(T, max_frames))
    vids_tensor = torch.stack(vids_padded)  # [B, 3, F, 112, 112]
    return vids_tensor, lengths, transcripts

def get_loader(dataset, batch_size=8, shuffle=True, num_workers=4):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )