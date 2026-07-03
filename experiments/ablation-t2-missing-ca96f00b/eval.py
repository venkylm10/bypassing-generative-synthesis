"""Evaluate a trained latent-imputed 3D U-Net checkpoint on the held-out test split.

Reports mean Dice (foreground classes), per-class Dice, per-subject Dice
(for the distribution figure), inference latency, and peak inference memory.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from models.unet3d import FusionUNet3D, MODALITIES
from utils.dataset import BraTSSubset
from utils.losses import dice_per_class

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.abspath(os.path.join(CODE_DIR, ".."))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default="seed0")
    parser.add_argument("--n_latency_reps", type=int, default=30)
    args = parser.parse_args()

    with open(os.path.join(CODE_DIR, "configs", "config.json")) as f:
        cfg = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = os.path.join(CODE_DIR, f"weights_{args.tag}", "best.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = FusionUNet3D(widths=tuple(cfg["widths"]), bottleneck_extra=cfg["bottleneck_extra"],
                          n_classes=cfg["n_classes"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint {ckpt_path} (epoch {ckpt['epoch']}, best_metric={ckpt['best_metric']:.4f})")

    with open(os.path.join(OUTPUT_DIR, f"splits_{args.tag}.json")) as f:
        splits = json.load(f)
    test_ids = splits["test"]
    test_ds = BraTSSubset(test_ids)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

    missing = cfg["missing_modality"]
    per_subject_dice = []
    per_class_dices = []

    with torch.no_grad():
        for volumes, label, sids in test_loader:
            volumes = {m: v.to(device) for m, v in volumes.items()}
            label = label.to(device)
            logits = model(volumes, missing=missing)
            dpc = dice_per_class(logits, label).cpu().numpy()
            per_class_dices.append(dpc)
            per_subject_dice.append({"subject": sids[0], "mean_dice": float(dpc[1:].mean()),
                                      "dice_per_class": dpc.tolist()})

    mean_dice = float(np.mean([r["mean_dice"] for r in per_subject_dice]))
    std_dice = float(np.std([r["mean_dice"] for r in per_subject_dice]))
    mean_dice_per_class = np.mean(np.stack(per_class_dices), axis=0).tolist()

    # Inference latency + peak memory: single-subject forward pass, batch=1, real volumes
    sample_volumes, sample_label, _ = test_ds[0]
    sample_volumes = {m: v.unsqueeze(0).to(device) for m, v in sample_volumes.items()}
    torch.cuda.reset_peak_memory_stats(device) if device == "cuda" else None
    with torch.no_grad():
        for _ in range(5):  # warmup
            _ = model(sample_volumes, missing=missing)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(args.n_latency_reps):
            _ = model(sample_volumes, missing=missing)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - t0
    latency_ms = 1000.0 * elapsed / args.n_latency_reps
    peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1e6 if device == "cuda" else 0.0

    result = {
        "tag": args.tag,
        "mean_dice_score": mean_dice,
        "std_dice_score": std_dice,
        "mean_dice_per_class": mean_dice_per_class,
        "per_subject_dice": per_subject_dice,
        "inference_latency_ms": latency_ms,
        "peak_inference_memory_mb": peak_mem_mb,
        "n_test_subjects": len(test_ids),
        "best_epoch": ckpt["epoch"],
    }
    out_path = os.path.join(OUTPUT_DIR, f"eval_{args.tag}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"mean_dice_score={mean_dice:.4f} +/- {std_dice:.4f}")
    print(f"inference_latency_ms={latency_ms:.3f}")
    print(f"peak_inference_memory_mb={peak_mem_mb:.2f}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
