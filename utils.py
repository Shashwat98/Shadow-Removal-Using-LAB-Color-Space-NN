"""
Utilities for LAB-Net Shadow Removal (ISTD+)

Includes:
- RGB <-> LAB normalized conversions (OpenCV semantics)
- Mask utilities and mean-matching on L outside shadow
- Metrics (PSNR, SSIM, RMSE) with optional masking
- Visualization helpers for side-by-side grids
"""
from __future__ import annotations

import os
import math
import random
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch

# -----------------------------
# Randomness / reproducibility
# -----------------------------

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Color space helpers (OpenCV CIE Lab)
# -----------------------------

def rgb_to_lab_norm(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """uint8 RGB [0..255] -> normalized (L in [0,1], a,b in [-1,1])."""
    assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[-1] == 3
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0] / 255.0
    a = (lab[..., 1] - 128.0) / 127.0
    b = (lab[..., 2] - 128.0) / 127.0
    return L.astype(np.float32), a.astype(np.float32), b.astype(np.float32)


def lab_norm_to_rgb(L: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Inverse: normalized (L,a,b) -> RGB."""
    L255 = (np.clip(L, 0.0, 1.0) * 255.0).astype(np.float32)
    a255 = (np.clip(a, -1.0, 1.0) * 127.0 + 128.0).astype(np.float32)
    b255 = (np.clip(b, -1.0, 1.0) * 127.0 + 128.0).astype(np.float32)
    lab = np.stack([L255, a255, b255], axis=-1)
    rgb = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_Lab2RGB)
    return rgb


# -----------------------------
# Mean match on L outside shadow
# -----------------------------

def mean_match_L_nonshadow(L_shadow: np.ndarray, L_gt: np.ndarray, mask01: np.ndarray) -> np.ndarray:
    """Additive mean match using non-shadow (mask==0) region."""
    non_shadow = (mask01 < 0.5)
    if not np.any(non_shadow):
        return L_shadow
    offset = float(L_gt[non_shadow].mean() - L_shadow[non_shadow].mean())
    return np.clip(L_shadow + offset, 0.0, 1.0).astype(np.float32)


# -----------------------------
# Metrics (with optional mask)
# -----------------------------

def _apply_mask(arr: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None:
        return arr
    if mask.dtype != bool:
        mask = mask >= 0.5
    return arr[mask]


def rmse(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if mask is not None:
        if mask.dtype != bool:
            mask = mask >= 0.5
        a = a[mask]
        b = b[mask]
    if a.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def psnr(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None, data_range: float = 1.0) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if mask is not None:
        if mask.dtype != bool:
            mask = mask >= 0.5
        a = a[mask]
        b = b[mask]
    mse = np.mean((a - b) ** 2) if a.size else np.nan
    if not np.isfinite(mse) or mse <= 0:
        return float("inf")
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def ssim_L(L_pred: np.ndarray, L_gt: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """SSIM on L channel in [0,1]. If mask provided, compute on masked crop.
    Skimage's SSIM doesn't support sparse masks natively; we approximate by
    cropping to the tightest bounding box that encloses the mask==1 region.
    """
    assert L_pred.ndim == 2 and L_gt.ndim == 2
    if mask is not None:
        ys, xs = np.where(mask >= 0.5)
        if ys.size == 0:
            return float("nan")
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        Lp = L_pred[y0:y1, x0:x1]
        Lg = L_gt[y0:y1, x0:x1]
    else:
        Lp, Lg = L_pred, L_gt
    return float(ssim(Lg, Lp, data_range=1.0))


def metrics_lab(L_pred: np.ndarray, a_pred: np.ndarray, b_pred: np.ndarray,
                L_gt: np.ndarray, a_gt: np.ndarray, b_gt: np.ndarray,
                mask01: Optional[np.ndarray] = None,
                valid_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute PSNR/SSIM/RMSE on L channel inside shadow mask & valid region."""
    mask = None
    if mask01 is not None and valid_mask is not None:
        mask = (mask01 >= 0.5) & (valid_mask >= 0.5)
    elif mask01 is not None:
        mask = (mask01 >= 0.5)
    elif valid_mask is not None:
        mask = (valid_mask >= 0.5)

    m = {}
    m["rmse_L"] = rmse(L_pred, L_gt, mask)
    m["psnr_L"] = psnr(L_pred, L_gt, mask, data_range=1.0)
    try:
        m["ssim_L"] = ssim_L(L_pred, L_gt, mask)
    except Exception:
        m["ssim_L"] = float("nan")

    # Optional: color error inside mask (Euclidean on a,b)
    if mask is not None:
        da = _apply_mask(a_pred - a_gt, mask)
        db = _apply_mask(b_pred - b_gt, mask)
        if da.size:
            m["rmse_ab"] = float(np.sqrt(np.mean(da**2 + db**2)))
        else:
            m["rmse_ab"] = float("nan")
    else:
        m["rmse_ab"] = float(np.sqrt(np.mean((a_pred - a_gt)**2 + (b_pred - b_gt)**2)))
    return m


# -----------------------------
# Visualization helpers
# -----------------------------

def make_grid(images: Tuple[np.ndarray, ...], pad: int = 4) -> np.ndarray:
    """Horizontally concatenate uint8 RGB images with padding."""
    assert len(images) > 0
    h = min(img.shape[0] for img in images)
    rs = [cv2.resize(img, (int(img.shape[1] * (h / img.shape[0])), h), interpolation=cv2.INTER_AREA) for img in images]
    pad_arr = 255 * np.ones((h, pad, 3), dtype=np.uint8)
    out = rs[0]
    for im in rs[1:]:
        out = np.concatenate([out, pad_arr, im], axis=1)
    return out


def overlay_error_map(rgb_ref: np.ndarray, err: np.ndarray) -> np.ndarray:
    """Create a simple heat overlay of absolute error on top of rgb_ref.
    err should be in [0,1]; we'll map to a colormap and alpha blend.
    """
    err = np.clip(err, 0.0, 1.0).astype(np.float32)
    heat = (cv2.applyColorMap((err * 255).astype(np.uint8), cv2.COLORMAP_JET))
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    out = (0.7 * rgb_ref.astype(np.float32) + 0.3 * heat.astype(np.float32))
    return np.clip(out, 0, 255).astype(np.uint8)


# -----------------------------
# Tensor helpers (PyTorch ↔ NumPy)
# -----------------------------

def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def lab_tensor_to_rgb_uint8(L: torch.Tensor, ab: torch.Tensor) -> np.ndarray:
    """Convert batched LAB tensors to a stacked grid RGB (uint8) for preview.
    L: (B,1,H,W) in [0,1]; ab: (B,2,H,W) in [-1,1]
    Returns: image created by tiling first 4 samples.
    """
    Lb = L[:4, 0].detach().cpu().numpy()
    ab_b = ab[:4].detach().cpu().numpy()
    tiles = []
    for i in range(Lb.shape[0]):
        Li = Lb[i]
        ai = ab_b[i, 0]
        bi = ab_b[i, 1]
        rgb = lab_norm_to_rgb(Li, ai, bi)
        tiles.append(rgb)
    if not tiles:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    h = tiles[0].shape[0]
    grid = tiles[0]
    for im in tiles[1:]:
        grid = np.concatenate([grid, 255*np.ones((h, 4, 3), dtype=np.uint8), im], axis=1)
    return grid
