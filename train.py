# train.py

import os
import json
import editdistance
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from tqdm import tqdm

from datasets import CachedLipreadingDataset
from dataloaders import get_loader
from model import LipNet

# ─── Config ───────────────────────────────────────────────────────────────────
CACHE_DIR      = "data/cache_small"
LABELS_JSON    = "labels.json"
CHECKPOINT_DIR = "checkpoints"
BATCH_SIZE     = 64      # bump higher if your GPU can handle it
NUM_WORKERS    = 2       # lower if you hit Colab worker‐freeze warnings
LR             = 1e-4
EPOCHS         = 100     # we’ll early‐stop
VAL_RATIO      = 0.1     # 10% as validation
PATIENCE       = 5       # stop after 5 epochs w/o val‐loss improvement

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)
# ────────────────────────────────────────────────────────────────────────────────

def text_to_tensor(txt: str, char_list: list[str]):
    idxs = [char_list.index(c) for c in txt]
    return torch.tensor(idxs, dtype=torch.long), len(idxs)

def ctc_decode(log_probs: torch.Tensor, char_list: list[str]) -> list[str]:
    """
    Greedy CTC decode: collapse repeats & drop blank (0).
    log_probs: [T, B, C]
    """
    seq = log_probs.argmax(dim=2).cpu().numpy()  # [T, B]
    decoded = []
    for b in range(seq.shape[1]):
        prev = 0
        chars = []
        for t in seq[:, b]:
            if t != prev and t != 0:
                chars.append(char_list[t])
            prev = t
        decoded.append("".join(chars))
    return decoded

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print("Using device:", DEVICE)

    # 1) Load & split dataset
    labels = json.load(open(LABELS_JSON, "r"))
    full_ds = CachedLipreadingDataset(CACHE_DIR, labels)
    val_size   = int(len(full_ds) * VAL_RATIO)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    train_loader = get_loader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = get_loader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS)

    # 2) Model, loss, optimizer
    char_list = ["<blank>"] + list("abcdefghijklmnopqrstuvwxyz ")
    model     = LipNet(num_classes=len(char_list)).to(DEVICE)
    ctc       = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS+1):
        # ——— Training ——————————————————————————————————————————————
        model.train()
        train_loss = 0.0
        for vids, lengths, texts in tqdm(train_loader, desc=f"[Epoch {epoch}] train"):
            vids = vids.to(DEVICE)
            # prepare targets
            targets, tgt_lens = zip(*(text_to_tensor(t, char_list) for t in texts))
            targets    = torch.cat(targets).to(DEVICE)
            tgt_lens   = torch.tensor(tgt_lens, dtype=torch.long).to(DEVICE)
            input_lens = torch.tensor(lengths, dtype=torch.long).to(DEVICE)

            optimizer.zero_grad()
            log_probs = model(vids, lengths)           # [T, B, C]
            loss = ctc(log_probs, targets, input_lens, tgt_lens)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)

        # ——— Validation —————————————————————————————————————————————
        model.eval()
        val_loss = 0.0
        total_cer = 0.0
        n_samples = 0

        with torch.no_grad():
            # loss
            for vids, lengths, texts in tqdm(val_loader, desc=f"[Epoch {epoch}] val"):
                vids = vids.to(DEVICE)
                targets, tgt_lens = zip(*(text_to_tensor(t, char_list) for t in texts))
                targets    = torch.cat(targets).to(DEVICE)
                tgt_lens   = torch.tensor(tgt_lens, dtype=torch.long).to(DEVICE)
                input_lens = torch.tensor(lengths, dtype=torch.long).to(DEVICE)

                log_probs = model(vids, lengths)
                loss = ctc(log_probs, targets, input_lens, tgt_lens)
                val_loss += loss.item()

                # CER
                preds = ctc_decode(log_probs, char_list)
                for pred, truth in zip(preds, texts):
                    dist = editdistance.eval(list(pred), list(truth))
                    total_cer += dist / max(len(truth), 1)
                    n_samples += 1

        avg_val_loss = val_loss / len(val_loader)
        avg_cer      = total_cer / n_samples

        # ——— Check for improvement / early stopping ————————————————————
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # save best model
            best_path = os.path.join(CHECKPOINT_DIR, "lipnet_best.pt")
            torch.save(model.state_dict(), best_path)
        else:
            epochs_no_improve += 1

        # summary
        print(
            f"Epoch {epoch:2d}/{EPOCHS} — train loss: {avg_train_loss:.4f} — "
            f"val loss: {avg_val_loss:.4f} — val CER: {avg_cer:.2%}"
        )

        if epochs_no_improve >= PATIENCE:
            print(f"No improvement for {PATIENCE} epochs—early stopping.")
            break

if __name__ == "__main__":
    main()