import torch
import torch.nn as nn
import torch.nn.functional as F


class HFreqCNN(nn.Module):
    """
    Input: [B,1,256,256] frequency map
    Output: logits [B]
    """
    def __init__(self):
        super().__init__()

        def block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.net = nn.Sequential(
            block(1, 16),   # 256 -> 128
            block(16, 32),  # 128 -> 64
            block(32, 64),  # 64 -> 32
            block(64, 128), # 32 -> 16
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B,128,1,1]
            nn.Flatten(),             # [B,128]
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.net(x)
        x = self.head(x).squeeze(1)  # [B]
        return x
