# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class LipNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 3D conv block
        self.conv3d = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3,5,5), padding=(1,2,2)),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2)),
            nn.Conv3d(32, 64, kernel_size=(3,5,5), padding=(1,2,2)),
            nn.ReLU(),
            nn.MaxPool3d((1,2,2)),
        )
        # after two (1×2×2) pools spatial dims 112→28
        rnn_input = 64 * 28 * 28
        self.gru = nn.GRU(
            input_size=rnn_input,
            hidden_size=256,
            num_layers=2,
            batch_first=False,
            bidirectional=True
        )
        self.fc = nn.Linear(256 * 2, num_classes)

    def forward(self, x, lengths):
        # x: [B, 3, T, 112, 112]
        x = self.conv3d(x)               # → [B, 64, T, 28, 28]
        B, C, T, H, W = x.size()

        # reshape for RNN: [T, B, C*H*W]
        x = x.permute(2, 0, 1, 3, 4).contiguous().view(T, B, -1)

        # pack & run GRU
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out)
        # out: [T, B, 512]

        logits = self.fc(out)            # → [T, B, num_classes]
        return F.log_softmax(logits, dim=2)