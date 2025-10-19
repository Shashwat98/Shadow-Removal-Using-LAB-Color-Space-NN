"""
Training script for LAB-Net on ISTD+ (Adjusted ISTD).

Usage (from project root):
  python -m train --data_root data/ISTD_plus --epochs 6 --bs 8 --lr 2e-4 --resize 256 256
"""
from __future__ import annotations

import os
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import create_loaders
from model import LabNet, LabNetConfig
from losses import LabLoss, LossConfig
from utils import seed_everything, lab_tensor_to_rgb_uint8


# -----------------------------
# Helpers
# -----------------------------

def save_ckpt(model, opt, step, out_dir: Path):
    """Save model + optimizer + global step."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"labnet_step{step}.pt"
    torch.save({
        'model': model.state_dict(),
        'opt': opt.state_dict(),
        'step': step,
    }, path)
    return path


def save_sample_grid(x, L_hat, ab_hat, y, out_dir: Path, tag: str):
    """Save an input / prediction / GT triptych image for visual sanity check."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        pred_grid = lab_tensor_to_rgb_uint8(L_hat, ab_hat)
        in_grid = lab_tensor_to_rgb_uint8(x[:, 0:1], x[:, 1:3])
        gt_grid = lab_tensor_to_rgb_uint8(y[:, 0:1], y[:, 1:3])

    h = min(pred_grid.shape[0], in_grid.shape[0], gt_grid.shape[0])

    def _resize(img):
        return cv2.resize(img, (img.shape[1], h), interpolation=cv2.INTER_AREA)

    pred_grid = _resize(pred_grid)
    in_grid = _resize(in_grid)
    gt_grid = _resize(gt_grid)
    pad = 255 * np.ones((h, 6, 3), dtype=np.uint8)
    strip = np.concatenate([in_grid, pad, pred_grid, pad, gt_grid], axis=1)

    out_path = out_dir / f"{tag}.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    return out_path


# -----------------------------
# Training
# -----------------------------

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    seed_everything(args.seed)

    # -------------------------
    # Data
    # -------------------------
    resize = (args.resize_h, args.resize_w) if args.resize_h and args.resize_w else None
    dl_train, dl_test = create_loaders(
        data_root=args.data_root,
        batch_size=args.bs,
        num_workers=args.workers,
        resize=resize,
        shuffle_train=True
    )

    steps_per_epoch = len(dl_train)

    # -------------------------
    # Model, loss, optimizer
    # -------------------------
    net = LabNet(LabNetConfig()).to(device)
    crit = LabLoss(LossConfig(
        lambda_ab=args.lambda_ab,
        lambda_ssim=args.lambda_ssim,
        lambda_out=args.lambda_out
    ))
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # -------------------------
    # Resume checkpoint (if provided)
    # -------------------------
    start_epoch = 1
    step = 0
    seen_in_epoch = 0

    if args.ckpt is not None:
        print(f"[resume] Loading checkpoint: {args.ckpt}")
        state = torch.load(args.ckpt, map_location='cpu')

        # model weights
        if 'model' in state:
            net.load_state_dict(state['model'])
        else:
            net.load_state_dict(state)  # fallback if plain state_dict

        # optimizer state (optional)
        if 'opt' in state:
            try:
                opt.load_state_dict(state['opt'])
            except Exception as e:
                print(f"[resume] Skipping optimizer state load: {e}")

        # global step
        if 'step' in state:
            step = int(state['step'])

        # infer epoch position
        start_epoch = step // steps_per_epoch + 1
        seen_in_epoch = step % steps_per_epoch
        print(f"[resume] Loaded step={step}, start_epoch={start_epoch}, "
              f"seen_in_epoch={seen_in_epoch}/{steps_per_epoch}")

    # -------------------------
    # Output dirs
    # -------------------------
    runs = Path('runs')
    ckpt_dir = runs / 'ckpts'
    samp_dir = runs / 'samples'

    # -------------------------
    # Training loop
    # -------------------------
    net.train()
    for epoch in range(start_epoch, args.epochs + 1):
        pbar = tqdm(dl_train, desc=f"Epoch {epoch}/{args.epochs}")
        for i, batch in enumerate(pbar):
            # skip batches already processed in resumed epoch
            if epoch == start_epoch and i < seen_in_epoch:
                continue

            x, y, aux = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            valid = torch.ones_like(x[:, 0:1], dtype=torch.float32, device=device)
            opt.zero_grad(set_to_none=True)
            L_hat, ab_hat = net(x)
            loss_dict = crit(x, y, L_hat, ab_hat, valid)
            loss = loss_dict['total']
            loss.backward()
            opt.step()

            step += 1
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'L1_in': f"{loss_dict['l1_L_in'].item():.3f}",
                'ab_in': f"{loss_dict['l1_ab_in'].item():.3f}",
                'ssim': f"{loss_dict['ssim_loss'].item():.3f}",
                'out': f"{loss_dict['l1_out'].item():.3f}",
            })

            if step % args.sample_every == 0:
                save_sample_grid(x, L_hat, ab_hat, y, samp_dir, tag=f"ep{epoch}_step{step}")

        # Save checkpoint each epoch
        save_ckpt(net, opt, step, ckpt_dir)

    # -------------------------
    # Simple test preview
    # -------------------------
    net.eval()
    with torch.no_grad():
        batch = next(iter(dl_test))
        x, y, aux = batch
        x = x.to(device)
        y = y.to(device)
        L_hat, ab_hat = net(x)
        save_sample_grid(x, L_hat, ab_hat, y, samp_dir, tag="test_preview_final")


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, default=os.path.join('data', 'ISTD_plus'))
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--workers', type=int, default=0)  # Windows-friendly
    ap.add_argument('--resize', type=int, nargs=2, metavar=('H', 'W'))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--lambda_ab', type=float, default=1.0)
    ap.add_argument('--lambda_ssim', type=float, default=0.2)
    ap.add_argument('--lambda_out', type=float, default=0.05)
    ap.add_argument('--sample_every', type=int, default=200)
    ap.add_argument('--ckpt', type=str, default=None, help='Path to checkpoint to resume from')
    args = ap.parse_args()
    if args.resize is not None:
        args.resize_h, args.resize_w = args.resize
    else:
        args.resize_h = args.resize_w = None
    return args


if __name__ == '__main__':
    args = parse_args()
    train(args)
