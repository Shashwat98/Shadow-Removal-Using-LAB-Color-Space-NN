"""
LAB-Net (lightweight) for Shadow Removal in LAB space.

- Shared encoder (U-Net style) over input x = [L_adj, a, b, mask] -> 4 channels
- Dual decoders:
    * Decoder_L  -> 1 channel (L_hat) with sigmoid in [0,1]
    * Decoder_AB -> 2 channels (a_hat, b_hat) with tanh in [-1,1]
- Skip connections from encoder stages to both decoders
- Lightweight (≈ <2M params by default), depth=4

Author: LAB-Net Shadow Removal (ISTD+)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Basic building blocks
# -----------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_ch, out_ch, 3, 1, 1),
            ConvBNReLU(out_ch, out_ch, 3, 1, 1),
        )
    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, mode: str = "bilinear"):
        super().__init__()
        self.mode = mode
        self.conv1x1 = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        # ⬇️ rename to avoid clashing with nn.Module.double()
        self.double_conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        # Upsample encoder feature to exactly match the skip's HxW
        if self.mode == "nearest":
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        else:
            x = F.interpolate(x, size=skip.shape[-2:], mode=self.mode, align_corners=False)

        x = self.conv1x1(x)

        # (Usually unnecessary after size=..., but keep a safe-guard for odd rounding cases)
        if x.shape[-2:] != skip.shape[-2:]:
            dy = skip.size(-2) - x.size(-2)
            dx = skip.size(-1) - x.size(-1)
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])

        x = torch.cat([x, skip], dim=1)
        return self.double_conv(x)


# -----------------------------
# Encoder
# -----------------------------
class Encoder(nn.Module):
    def __init__(self, in_ch: int = 4, widths = (64, 128, 256, 512)):
        super().__init__()
        w1, w2, w3, w4 = widths
        self.e1 = DoubleConv(in_ch, w1)
        self.d1 = nn.Conv2d(w1, w1, kernel_size=3, stride=2, padding=1)  # downsample
        self.e2 = DoubleConv(w1, w2)
        self.d2 = nn.Conv2d(w2, w2, kernel_size=3, stride=2, padding=1)
        self.e3 = DoubleConv(w2, w3)
        self.d3 = nn.Conv2d(w3, w3, kernel_size=3, stride=2, padding=1)
        self.e4 = DoubleConv(w3, w4)
        # bottleneck keeps w4

    def forward(self, x):
        # x: (B,4,H,W)
        s1 = self.e1(x)              # (B,w1,H,W)
        x = self.d1(s1)              # (B,w1,H/2,W/2)
        s2 = self.e2(x)              # (B,w2,H/2,W/2)
        x = self.d2(s2)              # (B,w2,H/4,W/4)
        s3 = self.e3(x)              # (B,w3,H/4,W/4)
        x = self.d3(s3)              # (B,w3,H/8,W/8)
        s4 = self.e4(x)              # (B,w4,H/8,W/8)
        return s1, s2, s3, s4


# -----------------------------
# Decoders (two heads)
# -----------------------------
class DecoderL(nn.Module):
    def __init__(self, widths = (64, 128, 256, 512)):
        super().__init__()
        w1, w2, w3, w4 = widths
        self.up3 = UpBlock(w4, w3, w3)
        self.up2 = UpBlock(w3, w2, w2)
        self.up1 = UpBlock(w2, w1, w1)
        self.out_conv = nn.Conv2d(w1, 1, kernel_size=1)
    def forward(self, s1, s2, s3, s4):
        x = self.up3(s4, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        L_hat = torch.sigmoid(self.out_conv(x))  # [0,1]
        return L_hat


class DecoderAB(nn.Module):
    def __init__(self, widths = (64, 128, 256, 512)):
        super().__init__()
        w1, w2, w3, w4 = widths
        self.up3 = UpBlock(w4, w3, w3)
        self.up2 = UpBlock(w3, w2, w2)
        self.up1 = UpBlock(w2, w1, w1)
        self.out_conv = nn.Conv2d(w1, 2, kernel_size=1)
    def forward(self, s1, s2, s3, s4):
        x = self.up3(s4, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        ab_hat = torch.tanh(self.out_conv(x))   # [-1,1]
        return ab_hat


# -----------------------------
# LAB-Net (shared encoder + dual decoders)
# -----------------------------
@dataclass
class LabNetConfig:
    in_ch: int = 4
    widths: Tuple[int, int, int, int] = (64, 128, 256, 512)
    up_mode: str = "bilinear"


class LabNet(nn.Module):
    def __init__(self, cfg: LabNetConfig = LabNetConfig()):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg.in_ch, cfg.widths)
        self.decL = DecoderL(cfg.widths)
        self.decAB = DecoderAB(cfg.widths)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B,4,H,W) with [L_adj, a, b, mask]
        s1, s2, s3, s4 = self.encoder(x)
        L_hat = self.decL(s1, s2, s3, s4)
        ab_hat = self.decAB(s1, s2, s3, s4)
        return L_hat, ab_hat


# -----------------------------
# Utilities
# -----------------------------

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    cfg = LabNetConfig()
    net = LabNet(cfg)
    print("Trainable params:", count_params(net))
    x = torch.randn(2, 4, 256, 256)
    Lh, abh = net(x)
    print(Lh.shape, abh.shape)
