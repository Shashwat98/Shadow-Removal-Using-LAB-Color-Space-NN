"""
ISTD+ (Adjusted ISTD) Dataset Loader for LAB-Net Shadow Removal

- Groups triplets by filename prefix where:
    * "<id>-1.png" : shadow image (RGB)
    * "<id>-2.png" : binary shadow mask (white=shadow, black=non-shadow)
    * "<id>-3.png" : shadow-free ground truth (RGB)
- Converts RGB -> LAB (OpenCV CIE Lab) and normalizes as:
    * L in [0, 1]   (OpenCV L is [0..255])
    * a, b in [-1, 1] where a' = (a-128)/127, b' = (b-128)/127
- Mean-matches the non-shadow (mask==0) region between shadow and GT on L channel
    using an additive offset on L to align global illumination.
- Computes a saturation validity mask (exclude pixels with any RGB >= 250 in either
    shadow or GT); this is for losses.
- Returns tensors:
    * x  : (4,H,W)  -> [L, a, b, mask]
    * y  : (3,H,W)  -> [L_gt, a_gt, b_gt]
    * aux: dict with paths and validity mask ("valid_mask")

"""
from __future__ import annotations

import os
import re
import glob
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ----------------------
# Utility: image I/O
# ----------------------

def _imread_rgb(path: str) -> np.ndarray:
    """Read an image as RGB [H,W,3]. Raises if missing."""
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def _imread_mask01(path: str) -> np.ndarray:
    """Read a binary mask as float32 in {0.0, 1.0}. Accepts grayscale/PNG.
    """
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Failed to read mask: {path}")
    m = (m >= 128).astype(np.float32)  # binarize
    return m


# ----------------------
# Utility: color space
# ----------------------

def _rgb_to_lab_norm(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert uint8 RGB [0..255] -> normalized LAB channels.

    Using OpenCV's CIE Lab:
      - L: 0..255 (not 0..100). We map to [0,1] via /255.
      - a,b: 0..255 with 128 at neutral; map to [-1,1] via (v-128)/127.
    Returns L, a, b each as float32 arrays in ranges [0,1], [-1,1], [-1,1].
    """
    if rgb.dtype != np.uint8:
        raise ValueError("_rgb_to_lab_norm expects RGB input")
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0] / 255.0
    a = (lab[..., 1] - 128.0) / 127.0
    b = (lab[..., 2] - 128.0) / 127.0
    return L.astype(np.float32), a.astype(np.float32), b.astype(np.float32)


def _lab_norm_to_rgb(L: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Inverse of _rgb_to_lab_norm.
    Inputs: L in [0,1], a,b in [-1,1]. Returns RGB image.
    """
    L255 = (np.clip(L, 0.0, 1.0) * 255.0).astype(np.float32)
    a255 = (np.clip(a, -1.0, 1.0) * 127.0 + 128.0).astype(np.float32)
    b255 = (np.clip(b, -1.0, 1.0) * 127.0 + 128.0).astype(np.float32)
    lab = np.stack([L255, a255, b255], axis=-1)
    rgb = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_Lab2RGB)
    return rgb


# ----------------------
# Utility: mean-match on L (non-shadow region)
# ----------------------

def mean_match_L_nonshadow(L_shadow: np.ndarray, L_gt: np.ndarray, mask01: np.ndarray) -> np.ndarray:
    """Additive mean matching on L channel using non-shadow (mask==0) region.

    Args:
        L_shadow: float32 [H,W] in [0,1]
        L_gt    : float32 [H,W] in [0,1]
        mask01  : float32 [H,W] in {0,1} where 1 = SHADOW region

    Returns:
        L_shadow_adj: float32 [H,W] in [0,1]
    """
    non_shadow = (mask01 < 0.5)
    if not np.any(non_shadow):
        return L_shadow  # degenerate mask; skip
    mu_s = float(L_shadow[non_shadow].mean())
    mu_g = float(L_gt[non_shadow].mean())
    offset = mu_g - mu_s
    L_adj = np.clip(L_shadow + offset, 0.0, 1.0)
    return L_adj.astype(np.float32)


# ----------------------
# File grouping
# ----------------------
_triplet_re = re.compile(r"^(?P<prefix>.+?)-(?:[1-9][0-9]*)\\.png$", re.IGNORECASE)
_suffix_re = re.compile(r"^(?P<prefix>.+?)-(?P<suffix>[0-9]+)\\.png$", re.IGNORECASE)


def _gather_triplets(root_dir, split_tag):
    """
    root_dir: dataset root, e.g. data/ISTD_plus
    split_tag: "train" or "test"
    """
    import glob, os

    A_dir = os.path.join(root_dir, f"{split_tag}_A")
    B_dir = os.path.join(root_dir, f"{split_tag}_B")
    C_dir = os.path.join(root_dir,
                         "train_C_fixed_ours" if split_tag == "train" else "test_C_fixed_official")

    def base(p): return os.path.splitext(os.path.basename(p))[0]

    As = {base(p): p for p in glob.glob(os.path.join(A_dir, "*.*"))}
    Bs = {base(p): p for p in glob.glob(os.path.join(B_dir, "*.*"))}
    Cs = {base(p): p for p in glob.glob(os.path.join(C_dir, "*.*"))}

    keys = sorted(As.keys() & Bs.keys() & Cs.keys())
    triplets = [(As[k], Bs[k], Cs[k]) for k in keys]

    if not triplets:
        raise RuntimeError(f"No valid triplets found under root: {root_dir} (split={split_tag})")

    return triplets


# ----------------------
# Dataset
# ----------------------
@dataclass
class ISTDPlusConfig:
    root_dir: str = os.path.join("data", "ISTD_plus")
    split: str = "train"  # "train" or "test"
    train_dirname: str = "train_C_fixed_ours"
    test_dirname: str = "test_C_fixed_official"
    resize: Optional[Tuple[int, int]] = None  # (H,W) - optional
    augment: bool = False  # simple flips/rotations (not implemented yet)


class ISTDPlusDataset(Dataset):
    def __init__(self, cfg: ISTDPlusConfig):
        super().__init__()
        self.cfg = cfg

        # NEW: set data_root (used below)
        self.data_root = cfg.root_dir

        # build split directory path just for sanity checks/other uses
        split_dirname = (
            cfg.train_dirname if cfg.split.lower().startswith("train") else cfg.test_dirname
        )
        self.split_dir = os.path.join(cfg.root_dir, split_dirname)
        if not os.path.isdir(self.split_dir):
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        # figure out split tag
        split_tag = "train" if "train" in os.path.basename(self.split_dir).lower() else "test"

        # gather triplets from the ROOT + split tag (NOT from split_dir alone)
        self.triplets = _gather_triplets(self.data_root, split_tag)

    def __len__(self) -> int:
        return len(self.triplets)

    def _maybe_resize(self, img: np.ndarray, size_hw: Optional[Tuple[int, int]]) -> np.ndarray:
        if size_hw is None:
            return img
        H, W = size_hw
        return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

    def __getitem__(self, idx: int):
        shadow_path, mask_path, gt_path = self.triplets[idx]
        rgb_s = _imread_rgb(shadow_path)
        rgb_g = _imread_rgb(gt_path)
        mask01 = _imread_mask01(mask_path)  # 1=shadow, 0=non

        # Optional resize (applied equally)
        if self.cfg.resize is not None:
            H, W = self.cfg.resize
            rgb_s = self._maybe_resize(rgb_s, (H, W))
            rgb_g = self._maybe_resize(rgb_g, (H, W))
            mask01 = self._maybe_resize((mask01 * 255).astype(np.uint8), (H, W)).astype(np.float32) / 255.0
            mask01 = (mask01 >= 0.5).astype(np.float32)

        # Saturation validity mask (exclude very bright/near-clipped pixels in either image)
        sat_s = (rgb_s >= 250).any(axis=-1)
        sat_g = (rgb_g >= 250).any(axis=-1)
        valid_mask = (~(sat_s | sat_g)).astype(np.float32)

        # Convert to normalized LAB
        Ls, as_, bs_ = _rgb_to_lab_norm(rgb_s)
        Lg, ag, bg = _rgb_to_lab_norm(rgb_g)

        # Mean-match shadow L to GT L outside the shadow (mask==0)
        Ls_adj = mean_match_L_nonshadow(Ls, Lg, mask01)

        # Stack inputs/targets
        x = np.stack([Ls_adj, as_, bs_, mask01], axis=0).astype(np.float32)  # (4,H,W)
        y = np.stack([Lg, ag, bg], axis=0).astype(np.float32)                # (3,H,W)
        aux = {
            "shadow_path": shadow_path,
            "mask_path": mask_path,
            "gt_path": gt_path,
            "valid_mask": valid_mask.astype(np.float32),
        }

        return torch.from_numpy(x), torch.from_numpy(y), aux


# ----------------------
# Convenience: DataLoaders
# ----------------------

def create_loaders(
    data_root: str = os.path.join("data", "ISTD_plus"),
    batch_size: int = 8,
    num_workers: int = 0,  # Windows-safe default; user can raise if needed
    resize: Optional[Tuple[int, int]] = None,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Create train/test DataLoaders for ISTD+.

    On Windows, keep num_workers small at first (0 or 2) to avoid spawn issues.
    """
    train_cfg = ISTDPlusConfig(root_dir=data_root, split="train", resize=resize)
    test_cfg  = ISTDPlusConfig(root_dir=data_root, split="test", resize=resize)

    ds_train = ISTDPlusDataset(train_cfg)
    ds_test  = ISTDPlusDataset(test_cfg)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=shuffle_train,
                          num_workers=num_workers, pin_memory=True, drop_last=False)
    dl_test  = DataLoader(ds_test, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True, drop_last=False)
    return dl_train, dl_test


# ----------------------
# Quick smoke test (optional): run as script
# ----------------------
if __name__ == "__main__":
    # Adjust paths if running this file directly
    root = os.path.join("data", "ISTD_plus")
    try:
        dl_tr, dl_te = create_loaders(root, batch_size=2, num_workers=0, resize=None)
        x, y, aux = next(iter(dl_tr))
        print("Batch shapes:", x.shape, y.shape)
        print("Example paths:", aux["shadow_path"][0])
        print("Valid mask mean:", aux["valid_mask"][0].float().mean().item())
    except Exception as e:
        print("[SmokeTest]", e)
