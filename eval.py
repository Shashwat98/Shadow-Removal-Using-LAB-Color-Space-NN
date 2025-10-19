"""
Evaluation script for LAB-Net on ISTD+ (Adjusted ISTD).

Computes PSNR/SSIM/RMSE inside shadow mask (and valid pixels),
saves a CSV with per-image and summary metrics, and dumps qualitative grids.

Usage:
  python -m src.eval --ckpt runs/ckpts/labnet_stepXXXX.pt --data_root data/ISTD_plus --bs 4 --resize 256 256
"""
from __future__ import annotations

import os
import csv
from pathlib import Path
import argparse
from typing import Dict

import cv2
import numpy as np
import torch
from tqdm import tqdm

from dataset import create_loaders
from model import LabNet, LabNetConfig
from utils import lab_tensor_to_rgb_uint8, to_numpy, metrics_lab, lab_norm_to_rgb


def load_checkpoint(model: torch.nn.Module, ckpt_path: str | Path):
    state = torch.load(str(ckpt_path), map_location='cpu')
    model.load_state_dict(state['model'], strict=True)
    return state


def save_grid_triplet(x, L_hat, ab_hat, y, aux: Dict, out_dir: Path, idx: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        pred_grid = lab_tensor_to_rgb_uint8(L_hat, ab_hat)
        in_grid   = lab_tensor_to_rgb_uint8(x[:, 0:1], x[:, 1:3])
        gt_grid   = lab_tensor_to_rgb_uint8(y[:, 0:1], y[:, 1:3])
    h = min(pred_grid.shape[0], in_grid.shape[0], gt_grid.shape[0])
    pad = 255 * np.ones((h, 6, 3), dtype=np.uint8)
    strip = np.concatenate([in_grid, pad, pred_grid, pad, gt_grid], axis=1)
    out_path = out_dir / f"sample_{idx:05d}.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    return out_path


def evaluate(ckpt: str, data_root: str, bs: int = 4, workers: int = 0, resize=None, save_every: int = 20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data (test only)
    _, dl_test = create_loaders(data_root=data_root, batch_size=bs, num_workers=workers, resize=resize, shuffle_train=False)

    # Model
    net = LabNet(LabNetConfig()).to(device)

    # Load weights
    load_checkpoint(net, ckpt)
    net.eval()

    # Output dirs
    runs = Path('runs')
    eval_dir = runs / 'eval'
    imgs_dir = eval_dir / 'images'
    eval_dir.mkdir(parents=True, exist_ok=True)

    # CSV setup
    csv_path = eval_dir / 'metrics.csv'
    fout = open(csv_path, 'w', newline='')
    writer = csv.writer(fout)
    writer.writerow(['shadow_path', 'rmse_L', 'psnr_L', 'ssim_L', 'rmse_ab'])

    # Accumulators
    rmse_L_all, psnr_L_all, ssim_L_all, rmse_ab_all = [], [], [], []

    with torch.no_grad():
        idx = 0
        for x, y, aux in tqdm(dl_test, desc='Evaluating'):
            x = x.to(device)
            y = y.to(device)

            # valid mask collate (dict-or-list handling)
            if isinstance(aux, dict) and isinstance(aux.get('valid_mask'), torch.Tensor):
                valid = aux['valid_mask'].unsqueeze(1).to(device)
                spaths = aux['shadow_path'] if isinstance(aux['shadow_path'], list) else [aux['shadow_path']]
            else:
                vmaps = [torch.from_numpy(d['valid_mask']).float() if isinstance(d['valid_mask'], np.ndarray) else d['valid_mask'].float() for d in aux]
                valid = torch.stack(vmaps, dim=0).unsqueeze(1).to(device)
                spaths = [d['shadow_path'] for d in aux]

            L_hat, ab_hat = net(x)

            # Compute metrics per item in batch
            for bi in range(x.size(0)):
                Lp = to_numpy(L_hat[bi, 0])
                ap = to_numpy(ab_hat[bi, 0])
                bp = to_numpy(ab_hat[bi, 1])
                Lg = to_numpy(y[bi, 0])
                ag = to_numpy(y[bi, 1])
                bg = to_numpy(y[bi, 2])
                mask01 = to_numpy(x[bi, 3])
                vmask = to_numpy(valid[bi, 0])

                m = metrics_lab(Lp, ap, bp, Lg, ag, bg, mask01=mask01, valid_mask=vmask)
                writer.writerow([spaths[bi], f"{m['rmse_L']:.6f}", f"{m['psnr_L']:.6f}", f"{m['ssim_L']:.6f}", f"{m['rmse_ab']:.6f}"])
                rmse_L_all.append(m['rmse_L'])
                psnr_L_all.append(m['psnr_L'])
                ssim_L_all.append(m['ssim_L'])
                rmse_ab_all.append(m['rmse_ab'])

                # Save qualitative strip every N images
                if (idx % save_every) == 0:
                    save_grid_triplet(x[bi:bi+1], L_hat[bi:bi+1], ab_hat[bi:bi+1], y[bi:bi+1], aux, imgs_dir, idx)
                idx += 1

    fout.close()

    # Summary
    def _avg(v):
        v = [vv for vv in v if np.isfinite(vv)]
        return float(np.mean(v)) if v else float('nan')

    summary = {
        'rmse_L_mean': _avg(rmse_L_all),
        'psnr_L_mean': _avg(psnr_L_all),
        'ssim_L_mean': _avg(ssim_L_all),
        'rmse_ab_mean': _avg(rmse_ab_all),
        'count': len(rmse_L_all),
        'csv': str(csv_path),
        'images_dir': str(imgs_dir),
    }

    # Also write a small summary.txt
    with open(Path('runs') / 'eval' / 'summary.txt', 'w') as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--data_root', type=str, default=os.path.join('data', 'ISTD_plus'))
    ap.add_argument('--bs', type=int, default=4)
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--resize', type=int, nargs=2, metavar=('H', 'W'))
    ap.add_argument('--save_every', type=int, default=20)
    args = ap.parse_args()
    if args.resize is not None:
        resize = (args.resize[0], args.resize[1])
    else:
        resize = None
    return args, resize


if __name__ == '__main__':
    args, resize = parse_args()
    evaluate(ckpt=args.ckpt, data_root=args.data_root, bs=args.bs, workers=args.workers, resize=resize, save_every=args.save_every)
