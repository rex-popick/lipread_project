# train.py

import json
import torch
import torch.nn as nn
import torch.optim as optim
import multiprocessing

from datasets import CachedLipreadingDataset
from dataloaders import get_loader
from model import LipNet

def main():
    # ─── Hyperparams ─────────────────────────────────────────────
    CACHE_DIR   = "data/cache_small"
    LABELS_JSON = "labels.json"
    BATCH_SIZE  = 100
    NUM_WORKERS = 2
    LR          = 1e-4
    EPOCHS      = 20
    # ─────────────────────────────────────────────────────────────

    # 1) Pick device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )
    print("Using device:", device)

    # 2) Data
    labels  = json.load(open(LABELS_JSON, "r"))
    dataset = CachedLipreadingDataset(CACHE_DIR, labels)
    loader  = get_loader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    # 3) Model, loss, optimizer
    char_list = ["<blank>"] + list("abcdefghijklmnopqrstuvwxyz ")
    model     = LipNet(num_classes=len(char_list)).to(device)
    ctc       = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 4) Text → tensor helper
    def text_to_tensor(txt):
        idxs = [char_list.index(c) for c in txt]
        return torch.tensor(idxs, dtype=torch.long), len(idxs)

    # 5) Training loop
    model.train()
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        for vids, lengths, texts in loader:
            vids = vids.to(device)

            # Prepare targets
            targets, tgt_lens = zip(*(text_to_tensor(t) for t in texts))
            targets    = torch.cat(targets).to(device)
            tgt_lens   = torch.tensor(tgt_lens, dtype=torch.long).to(device)
            input_lens = torch.tensor(lengths, dtype=torch.long).to(device)

            # Forward + CTC loss
            log_probs = model(vids, lengths)
            loss = ctc(log_probs, targets, input_lens, tgt_lens)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch}/{EPOCHS} — avg loss: {avg_loss:.4f}")

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("fork")
    except RuntimeError:
        pass
    main()