import torch
import torch.nn as nn


def _up(ci, co):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        nn.Conv2d(ci, co, 3, padding=1),
        nn.BatchNorm2d(co),
        nn.GELU(),
    )


class SegDecoder(nn.Module):
    def __init__(self, in_dim=1024, n_classes=5, grid=14):
        super().__init__()
        self.grid = grid
        self.proj = nn.Sequential(nn.Conv2d(in_dim, 256, 1), nn.GELU())
        self.up1 = _up(256, 128)
        self.up2 = _up(128, 64)
        self.up3 = _up(64, 32)
        self.head = nn.Conv2d(32, n_classes, 1)

    def forward(self, tokens):
        B, N, D = tokens.shape
        x = tokens.transpose(1, 2).reshape(B, D, self.grid, self.grid)
        x = self.proj(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        return self.head(x)