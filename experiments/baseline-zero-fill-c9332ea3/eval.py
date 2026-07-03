"""Re-evaluate a saved zero-fill baseline checkpoint on the held-out test split.

Usage:
    python eval.py --data_root <BraTS-GLI/train dir> --cache_dir <scratch cache> \
        --code_dir <this code/ dir> --checkpoint weights/best.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.unet import MultiEncoderUNet3D  # noqa: E402
from utils.dataset import BraTSZeroFillDataset  # noqa: E402
from utils.losses import DiceBCELoss, hard_dice  # noqa: E402
from utils.preprocess import run as preprocess_run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--code_dir", required=True)
    ap.add_argument("--checkpoint", default="weights/best.pt")
    ap.add_argument("--base_ch", type=int, default=24)
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(os.path.join(args.code_dir, "split.json")) as f:
        split = json.load(f)
    test_ids = split["test"]

    subjects = sorted(os.listdir(args.data_root))
    preprocess_run(args.data_root, args.cache_dir, subjects, device=device)

    test_ds = BraTSZeroFillDataset(args.cache_dir, test_ids, train=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = MultiEncoderUNet3D(n_modalities=4, base_ch=args.base_ch, n_classes=3).to(device)
    ckpt_path = args.checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(args.code_dir, ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}, best_metric={ckpt['best_metric']:.4f}")

    loss_fn = DiceBCELoss()
    total_loss, n_batches = 0.0, 0
    dice_sum = torch.zeros(3, device=device)
    n_samples = 0
    with torch.no_grad():
        for images, regions, sids in test_loader:
            images, regions = images.to(device), regions.to(device)
            mods = [images[:, i:i + 1] for i in range(4)]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(mods)
                loss = loss_fn(logits, regions)
            total_loss += loss.item()
            n_batches += 1
            d = hard_dice(logits.float(), regions)
            dice_sum += d * images.shape[0]
            n_samples += images.shape[0]

    dice_per_region = (dice_sum / n_samples).cpu().numpy().tolist()
    result = {
        "test_loss": total_loss / n_batches,
        "test_dice_WT": dice_per_region[0],
        "test_dice_TC": dice_per_region[1],
        "test_dice_ET": dice_per_region[2],
        "test_mean_dice": float(np.mean(dice_per_region)),
        "n_test_subjects": len(test_ids),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
