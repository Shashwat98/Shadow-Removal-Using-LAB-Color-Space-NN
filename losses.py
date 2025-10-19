"""
Loss functions for LAB-Net shadow removal.

Includes:
- Masked L1 on L and ab inside shadow region (mask==1) and valid (non-saturated) pixels
- SSIM on L inside shadow region (PyTorch implementation)
- Outside-mask consistency: keep non-shadow area similar to input (L and ab)

Return a dict with total loss and components.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# SSIM (PyTorch, differentiable)
# -----------------------------
# Simple, single-scale SSIM for grayscale images in [0,1]
# Based on standard formulation with Gaussian window

def _gaussian_window(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = g / g.sum()
    window_1d = g.view(1, 1, -1)
    window_2d = (window_1d.transpose(2, 1) @ window_1d).squeeze(0)
    window_2d = window_2d / window_2d.sum()
    return window_2d


def ssim_L_torch(pred_L: torch.Tensor, gt_L: torch.Tensor, mask01: torch.Tensor | None = None,
                  window_size: int = 11, sigma: float = 1.5, eps: float = 1e-6) -> torch.Tensor:
    """Compute SSIM over L channel in [0,1]. If mask provided, compute over full image
    but weight mean by mask. Returns (B,) tensor with SSIM per sample.
    """
    B, C, H, W = pred_L.shape
    assert C == 1
    device, dtype = pred_L.device, pred_L.dtype

    w = _gaussian_window(window_size, sigma, device, dtype)
    w = w.view(1, 1, window_size, window_size)

    mu1 = F.conv2d(pred_L, w, padding=window_size // 2, groups=1)
    mu2 = F.conv2d(gt_L,   w, padding=window_size // 2, groups=1)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred_L * pred_L, w, padding=window_size // 2, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(gt_L * gt_L,   w, padding=window_size // 2, groups=1) - mu2_sq
    sigma12   = F.conv2d(pred_L * gt_L, w, padding=window_size // 2, groups=1) - mu1_mu2

    # constants for L in [0,1]
    C1 = (0.01 ** 2)
    C2 = (0.03 ** 2)

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + eps)

    if mask01 is not None:
        # mask01: (B,1,H,W) with 1=shadow. Weight mean over mask>0.5
        m = (mask01 > 0.5).float()
        # Avoid division by zero
        denom = m.sum(dim=(2, 3)).clamp_min(1.0)
        ssim_val = (ssim_map * m).sum(dim=(2, 3)) / denom
    else:
        ssim_val = ssim_map.mean(dim=(2, 3))

    return ssim_val.squeeze(1) if ssim_val.ndim == 3 else ssim_val


# -----------------------------
# Config and loss wrapper
# -----------------------------
@dataclass
class LossConfig:
    lambda_ab: float = 1.0
    lambda_ssim: float = 0.2
    lambda_out: float = 0.05


class LabLoss(nn.Module):
    def __init__(self, cfg: LossConfig = LossConfig()):
        super().__init__()
        self.cfg = cfg
        self.l1 = nn.L1Loss(reduction='none')  # we'll mask manually

    def forward(self, x: torch.Tensor, y: torch.Tensor, pred_L: torch.Tensor, pred_ab: torch.Tensor,
                valid_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B,4,H,W)  input [L_adj, a, b, mask]
            y: (B,3,H,W)  target [L_gt, a_gt, b_gt]
            pred_L: (B,1,H,W)
            pred_ab: (B,2,H,W)
            valid_mask: (B,1,H,W)  1 = valid (not saturated in either img)
        """
        B, _, H, W = x.shape
        L_in  = x[:, 0:1]
        a_in  = x[:, 1:2]
        b_in  = x[:, 2:3]
        mask  = x[:, 3:4]  # 1=shadow

        L_gt  = y[:, 0:1]
        ab_gt = y[:, 1:3]

        inside = (mask * valid_mask).detach()
        outside = ((1.0 - mask) * valid_mask).detach()

        # L1 inside shadow region
        l1_L_in = (self.l1(pred_L, L_gt) * inside).sum() / (inside.sum() + 1e-8)
        l1_ab_in = (self.l1(pred_ab, ab_gt) * inside).sum() / (inside.sum() + 1e-8)

        # SSIM on L inside mask (maximize SSIM -> minimize 1-SSIM)
        ssim_vals = ssim_L_torch(pred_L, L_gt, mask01=inside)
        ssim_loss = (1.0 - ssim_vals).mean()

        # Outside-mask consistency (keep non-shadow unchanged)
        l1_L_out = (self.l1(pred_L, L_in) * outside).sum() / (outside.sum() + 1e-8)
        l1_ab_out = (self.l1(pred_ab, torch.cat([a_in, b_in], dim=1)) * outside).sum() / (outside.sum() + 1e-8)
        l1_out = l1_L_out + l1_ab_out

        total = l1_L_in + self.cfg.lambda_ab * l1_ab_in + self.cfg.lambda_ssim * ssim_loss + self.cfg.lambda_out * l1_out

        return {
            'total': total,
            'l1_L_in': l1_L_in.detach(),
            'l1_ab_in': l1_ab_in.detach(),
            'ssim_loss': ssim_loss.detach(),
            'l1_out': l1_out.detach(),
        }
