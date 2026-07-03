"""Re-evaluate saved baseline_umm_gan checkpoints (GAN generator + U-Net) on
the held-out test split, independently reproducing mean_dice_score and the
inference latency / peak memory benchmark from results.json.

Usage:
    python eval.py --data_root <BraTS-GLI/train dir> --cache_dir <seg cache> \
        --gan_cache_dir <gan-imputed cache> --code_dir <this code/ dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.gan import UMMGenerator3D  # noqa: E402
from models.unet import MultiEncoderUNet3D  # noqa: E402
from utils.dataset import BraTSGANImputedDataset  # noqa: E402
from utils.losses import DiceBCELoss, hard_dice  # noqa: E402
from utils.preprocess import run as preprocess_run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--gan_cache_dir", required=True)
    ap.add_argument("--code_dir", required=True)
    ap.add_argument("--unet_checkpoint", default="weights/best.pt")
    ap.add_argument("--gan_checkpoint", default="weights_gan/best.pt")
    ap.add_argument("--base_ch_unet", type=int, default=24)
    ap.add_argument("--base_ch_g", type=int, default=24)
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(os.path.join(args.code_dir, "split.json")) as f:
        split = json.load(f)
    test_ids = split["test"]

    subjects = sorted(os.listdir(args.data_root))
    preprocess_run(args.data_root, args.cache_dir, subjects, device=device)

    test_ds = BraTSGANImputedDataset(args.cache_dir, args.gan_cache_dir, test_ids, train=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = MultiEncoderUNet3D(n_modalities=4, base_ch=args.base_ch_unet, n_classes=3).to(device)
    ckpt_path = args.unet_checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(args.code_dir, ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded U-Net checkpoint from epoch {ckpt['epoch']}, best_metric={ckpt['best_metric']:.4f}")

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

    # inference benchmark (GAN + U-Net, single forward pass)
    gan_ckpt_path = args.gan_checkpoint
    if not os.path.isabs(gan_ckpt_path):
        gan_ckpt_path = os.path.join(args.code_dir, gan_ckpt_path)
    G = UMMGenerator3D(in_ch=3, base_ch=args.base_ch_g, out_ch=1).to(device)
    gckpt = torch.load(gan_ckpt_path, map_location=device, weights_only=False)
    G.load_state_dict(gckpt["model_g"])
    G.eval()

    sid = test_ids[0]
    npz = np.load(os.path.join(args.cache_dir, f"{sid}.npz"))
    images = npz["images"]
    condition = torch.from_numpy(np.stack([images[0], images[2], images[3]], axis=0)[None]).to(device)
    t1n = torch.from_numpy(images[0:1][None]).to(device)
    t2w = torch.from_numpy(images[2:3][None]).to(device)
    t2f = torch.from_numpy(images[3:4][None]).to(device)

    def full_forward():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            t1c_synth = G(condition)
            logits = model([t1n, t1c_synth, t2w, t2f])
        return logits

    for _ in range(5):
        full_forward()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    full_forward()
    torch.cuda.synchronize()
    peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6

    n_runs = 20
    t0 = time.time()
    for _ in range(n_runs):
        full_forward()
    torch.cuda.synchronize()
    latency_ms = (time.time() - t0) / n_runs * 1000.0
    print(json.dumps({"inference_latency_ms": latency_ms, "peak_inference_memory_mb": peak_mem_mb}, indent=2))


if __name__ == "__main__":
    main()
