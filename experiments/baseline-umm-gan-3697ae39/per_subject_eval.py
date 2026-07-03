"""Compute per-subject test-set Dice scores from a trained checkpoint, for the
`gan_imputed_dice_distribution` figure (distribution of segmentation Dice
across held-out subjects). Run after train.py has produced weights/best.pt
and split.json.

Usage:
    python per_subject_eval.py --data_root <BraTS-GLI/train dir> \
        --cache_dir <seg cache> --gan_cache_dir <gan-imputed cache> \
        --code_dir . --out per_subject_dice.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.unet import MultiEncoderUNet3D  # noqa: E402
from utils.dataset import BraTSGANImputedDataset  # noqa: E402
from utils.losses import hard_dice  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--gan_cache_dir", required=True)
    ap.add_argument("--code_dir", required=True)
    ap.add_argument("--checkpoint", default="weights/best.pt")
    ap.add_argument("--base_ch_unet", type=int, default=24)
    ap.add_argument("--out", default="per_subject_dice.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(os.path.join(args.code_dir, "split.json")) as f:
        split = json.load(f)
    test_ids = split["test"]

    ds = BraTSGANImputedDataset(args.cache_dir, args.gan_cache_dir, test_ids, train=False)

    model = MultiEncoderUNet3D(n_modalities=4, base_ch=args.base_ch_unet, n_classes=3).to(device)
    ckpt_path = args.checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(args.code_dir, ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    per_subject = []
    with torch.no_grad():
        for i in range(len(ds)):
            images, regions, sid = ds[i]
            images = images.unsqueeze(0).to(device)
            regions = regions.unsqueeze(0).to(device)
            mods = [images[:, c:c + 1] for c in range(4)]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(mods)
            d = hard_dice(logits.float(), regions).cpu().numpy().tolist()
            mean_d = float(np.mean(d))
            per_subject.append({"subject_id": sid, "dice_WT": d[0], "dice_TC": d[1],
                                 "dice_ET": d[2], "mean_dice": mean_d})
            print(f"{sid}: mean_dice={mean_d:.4f} (WT={d[0]:.4f} TC={d[1]:.4f} ET={d[2]:.4f})")

    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(args.code_dir, out_path)
    with open(out_path, "w") as f:
        json.dump(per_subject, f, indent=2)
    mean_of_means = float(np.mean([p["mean_dice"] for p in per_subject]))
    print(f"=== wrote {len(per_subject)} per-subject dice scores to {out_path}, "
          f"mean_of_means={mean_of_means:.4f} ===")


if __name__ == "__main__":
    main()
