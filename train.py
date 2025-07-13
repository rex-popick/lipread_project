# train.py

import os, json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split

from datasets import CachedLipreadingDataset
from dataloaders import get_loader
from model import LipNet
from tqdm import tqdm  # progress bar

# ─── Config ───────────────────────────────────────────────────────────────────
CACHE_DIR   = "data/cache_small"
LABELS_JSON = "labels.json"
CHECKPOINT_DIR = "checkpoints"
BATCH_SIZE  = 32       # you can bump this on GPU
NUM_WORKERS = 2        # Colab suggests ≤2
LR          = 1e-4
EPOCHS      = 30
VAL_RATIO   = 0.1      # 10% for validation
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ────────────────────────────────────────────────────────────────────────────────

def text_to_tensor(txt, char_list):
    idxs = [char_list.index(c) for c in txt]
    return torch.tensor(idxs, dtype=torch.long), len(idxs)

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print("Using device:", DEVICE)

    # 1) Load the full cached dataset
    labels = json.load(open(LABELS_JSON))
    full_ds = CachedLipreadingDataset(CACHE_DIR, labels)

    # 2) Split into train/validation
    val_size   = int(len(full_ds) * VAL_RATIO)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    # 3) DataLoaders
    train_loader = get_loader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = get_loader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS)

    # 4) Model, loss, optimizer
    char_list = ["<blank>"] + list("abcdefghijklmnopqrstuvwxyz ")
    model   = LipNet(num_classes=len(char_list)).to(DEVICE)
    ctc     = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 5) Training + validation loop
    for epoch in range(1, EPOCHS+1):
        # ——— Train ——————————————————————————————————————————————
        model.train()
        total_train_loss = 0.0
        for vids, lengths, texts in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            vids = vids.to(DEVICE)           # [B,3,T,112,112]
            # prepare targets
            targets, tgt_lens = zip(*(text_to_tensor(t, char_list) for t in texts))
            targets    = torch.cat(targets).to(DEVICE)
            tgt_lens   = torch.tensor(tgt_lens, dtype=torch.long).to(DEVICE)
            input_lens = torch.tensor(lengths, dtype=torch.long).to(DEVICE)

            log_probs = model(vids, lengths)           # [T, B, C]
            loss = ctc(log_probs, targets, input_lens, tgt_lens)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train = total_train_loss / len(train_loader)

        # ——— Validate —————————————————————————————————————————————
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for vids, lengths, texts in tqdm(val_loader, desc=f"Epoch {epoch} [ val ]"):
                vids = vids.to(DEVICE)
                targets, tgt_lens = zip(*(text_to_tensor(t, char_list) for t in texts))
                targets    = torch.cat(targets).to(DEVICE)
                tgt_lens   = torch.tensor(tgt_lens, dtype=torch.long).to(DEVICE)
                input_lens = torch.tensor(lengths, dtype=torch.long).to(DEVICE)

                log_probs = model(vids, lengths)
                loss = ctc(log_probs, targets, input_lens, tgt_lens)
                total_val_loss += loss.item()

        avg_val = total_val_loss / len(val_loader)

        # ——— Checkpoint ————————————————————————————————————————————
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"lipnet_epoch{epoch}.pt")
        torch.save(model.state_dict(), ckpt_path)

        print(f"Epoch {epoch}/{EPOCHS} — "
              f"train loss: {avg_train:.4f} — val loss: {avg_val:.4f}")

if __name__ == "__main__":
    main()