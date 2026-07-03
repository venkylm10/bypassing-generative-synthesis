"""Train the zero-fill baseline: multi-encoder late-fusion 3D U-Net, BraTS T1ce
zero-filled at input. Establishes the no-imputation lower bound.

Usage:
    python train.py --data_root <BraTS-GLI/train dir> --cache_dir <scratch cache> \
        --code_dir <this code/ dir, for weights/> --epochs 120 --batch_size 4
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.unet import MultiEncoderUNet3D  # noqa: E402
from utils.dataset import BraTSZeroFillDataset, split_subjects  # noqa: E402
from utils.losses import DiceBCELoss, hard_dice  # noqa: E402
from utils.preprocess import run as preprocess_run  # noqa: E402

REGION_NAMES = ["WT", "TC", "ET"]
PROGRESS_PATH = "/workspace/output/PROGRESS.json"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_progress(phase: str, current: int, total: int) -> None:
    try:
        with open(PROGRESS_PATH, "w") as f:
            json.dump({"phase": phase, "current": current, "total": total}, f)
    except Exception:
        pass


def save_checkpoint(path: str, state: dict) -> None:
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def rng_state_dict() -> dict:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def evaluate(model, loader, device, loss_fn) -> dict:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    dice_sum = torch.zeros(3, device=device)
    n_samples = 0
    with torch.no_grad():
        for images, regions, _sids in loader:
            images = images.to(device, non_blocking=True)
            regions = regions.to(device, non_blocking=True)
            mods = [images[:, i:i + 1] for i in range(4)]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(mods)
                loss = loss_fn(logits, regions)
            total_loss += loss.item()
            n_batches += 1
            d = hard_dice(logits.float(), regions)
            dice_sum += d * images.shape[0]
            n_samples += images.shape[0]
    mean_loss = total_loss / max(n_batches, 1)
    mean_dice_per_region = (dice_sum / max(n_samples, 1)).cpu().numpy().tolist()
    mean_dice = float(np.mean(mean_dice_per_region))
    return {"loss": mean_loss, "dice_per_region": mean_dice_per_region, "mean_dice": mean_dice}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--code_dir", required=True)
    ap.add_argument("--train_log", required=True)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--base_ch", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_val", type=int, default=8)
    ap.add_argument("--n_test", type=int, default=12)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--max_train_minutes", type=float, default=150.0)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "GPU required for this experiment (constraints.gpu_required=true)"

    weights_dir = os.path.join(args.code_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    last_path = os.path.join(weights_dir, "last.pt")
    best_path = os.path.join(weights_dir, "best.pt")

    subjects = sorted(os.listdir(args.data_root))
    train_ids, val_ids, test_ids = split_subjects(subjects, args.seed, args.n_val, args.n_test)
    with open(os.path.join(args.code_dir, "split.json"), "w") as f:
        json.dump({"train": train_ids, "val": val_ids, "test": test_ids}, f, indent=2)

    write_progress("loading_model", 0, 1)
    print(f"=== Loaded: MultiEncoderUNet3D (zero-fill baseline) | subjects total={len(subjects)} "
          f"train={len(train_ids)} val={len(val_ids)} test={len(test_ids)} ===", flush=True)

    write_progress("downloading_model", 0, 1)  # preprocessing / caching stage
    print("=== preprocessing (resize+cache) all subjects ===", flush=True)
    preprocess_run(args.data_root, args.cache_dir, subjects, device=device)

    train_ds = BraTSZeroFillDataset(args.cache_dir, train_ids, train=True)
    val_ds = BraTSZeroFillDataset(args.cache_dir, val_ids, train=False)
    test_ds = BraTSZeroFillDataset(args.cache_dir, test_ids, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    model = MultiEncoderUNet3D(n_modalities=4, base_ch=args.base_ch, n_classes=3).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"=== model params: {n_params/1e6:.2f}M ===", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = DiceBCELoss()

    wandb_run = None
    wandb_run_id = None
    if os.environ.get("WANDB_API_KEY"):
        try:
            import wandb
            wandb_run = wandb.init(
                project="prof-f9e1ad4b-plan-fa67285a",
                name="baseline_zero_fill",
                config={
                    "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                    "base_ch": args.base_ch, "seed": args.seed, "n_train": len(train_ids),
                    "n_val": len(val_ids), "n_test": len(test_ids), "resolution": 96,
                    "missing_modality": "t1c", "n_params": n_params,
                },
            )
            wandb_run_id = wandb_run.id
            print(f"WANDB_RUN_URL: {wandb_run.url}", flush=True)
        except Exception as e:
            print(f"[warn] wandb init failed: {e}", flush=True)

    start_epoch = 0
    global_step = 0
    best_metric = -1.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_mean_dice": []}

    if os.path.exists(last_path):
        print(f"=== resuming from {last_path} ===", flush=True)
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        torch.set_rng_state(ckpt["rng_torch"].cpu())
        if ckpt.get("rng_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["rng_cuda"]])
        np.random.set_state(ckpt["rng_numpy"])
        random.setstate(ckpt["rng_python"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        best_metric = ckpt["best_metric"]
        patience_counter = ckpt.get("patience_counter", 0)
        history = ckpt.get("history", history)
        wandb_run_id = ckpt.get("wandb_run_id", wandb_run_id)

    write_progress("training", start_epoch, args.epochs)
    train_start_time = time.time()
    stopped_reason = "completed_all_epochs"

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        epoch_start = time.time()
        for images, regions, _sids in train_loader:
            images = images.to(device, non_blocking=True)
            regions = regions.to(device, non_blocking=True)
            mods = [images[:, i:i + 1] for i in range(4)]

            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(mods)
                loss = loss_fn(logits, regions)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1
            lr_now = opt.param_groups[0]["lr"]
            if global_step % 5 == 0:
                print(f"[step {global_step}, epoch {epoch}] loss={loss.item():.4f} lr={lr_now:.6f}", flush=True)
            if wandb_run is not None:
                wandb_run.log({"train/loss_step": loss.item(), "lr": lr_now}, step=global_step)

        scheduler.step()
        train_loss = epoch_loss / max(n_batches, 1)
        val_metrics = evaluate(model, val_loader, device, loss_fn)
        epoch_elapsed = time.time() - epoch_start

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_mean_dice"].append(val_metrics["mean_dice"])

        print(f"=== epoch {epoch} done | avg_train_loss={train_loss:.4f} "
              f"val_loss={val_metrics['loss']:.4f} val_mean_dice={val_metrics['mean_dice']:.4f} "
              f"val_dice_WT={val_metrics['dice_per_region'][0]:.4f} "
              f"val_dice_TC={val_metrics['dice_per_region'][1]:.4f} "
              f"val_dice_ET={val_metrics['dice_per_region'][2]:.4f} "
              f"elapsed={epoch_elapsed:.1f}s ===", flush=True)

        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch, "train/loss_epoch": train_loss,
                "val/loss": val_metrics["loss"], "val/mean_dice": val_metrics["mean_dice"],
                "val/dice_WT": val_metrics["dice_per_region"][0],
                "val/dice_TC": val_metrics["dice_per_region"][1],
                "val/dice_ET": val_metrics["dice_per_region"][2],
            }, step=global_step)

        write_progress("training", epoch + 1, args.epochs)

        improved = val_metrics["mean_dice"] > best_metric
        if improved:
            best_metric = val_metrics["mean_dice"]
            patience_counter = 0
        else:
            patience_counter += 1

        state = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "rng_numpy": np.random.get_state(),
            "rng_python": random.getstate(),
            "epoch": epoch,
            "global_step": global_step,
            "best_metric": best_metric,
            "patience_counter": patience_counter,
            "history": history,
            "wandb_run_id": wandb_run_id,
        }
        save_checkpoint(last_path, state)
        if improved:
            save_checkpoint(best_path, state)
            print(f"=== new best val_mean_dice={best_metric:.4f} @ epoch {epoch}, saved best.pt ===", flush=True)

        elapsed_min = (time.time() - train_start_time) / 60.0
        if patience_counter >= args.patience:
            stopped_reason = f"early_stopping (patience={args.patience})"
            print(f"=== early stopping at epoch {epoch}: no val improvement for {args.patience} epochs ===", flush=True)
            break
        if elapsed_min >= args.max_train_minutes:
            stopped_reason = f"max_train_minutes budget reached ({args.max_train_minutes} min)"
            print(f"=== stopping at epoch {epoch}: time budget {args.max_train_minutes} min reached ===", flush=True)
            break

    total_train_time_min = (time.time() - train_start_time) / 60.0
    print(f"=== training loop finished: reason={stopped_reason} total_time={total_train_time_min:.2f} min ===", flush=True)

    write_progress("evaluating", 0, 1)
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"=== loaded best.pt (epoch {ckpt['epoch']}, val_mean_dice={ckpt['best_metric']:.4f}) for test eval ===", flush=True)
    test_metrics = evaluate(model, test_loader, device, loss_fn)
    print(f"=== FINAL TEST | test_loss={test_metrics['loss']:.4f} test_mean_dice={test_metrics['mean_dice']:.4f} "
          f"test_dice_WT={test_metrics['dice_per_region'][0]:.4f} "
          f"test_dice_TC={test_metrics['dice_per_region'][1]:.4f} "
          f"test_dice_ET={test_metrics['dice_per_region'][2]:.4f} ===", flush=True)

    if wandb_run is not None:
        wandb_run.log({
            "test/loss": test_metrics["loss"], "test/mean_dice": test_metrics["mean_dice"],
            "test/dice_WT": test_metrics["dice_per_region"][0],
            "test/dice_TC": test_metrics["dice_per_region"][1],
            "test/dice_ET": test_metrics["dice_per_region"][2],
            "mean_dice_score": test_metrics["mean_dice"],
        })

    summary = {
        "n_params": n_params,
        "train_ids": train_ids, "val_ids": val_ids, "test_ids": test_ids,
        "history": history,
        "best_val_mean_dice": best_metric,
        "test_metrics": test_metrics,
        "total_train_time_min": total_train_time_min,
        "stopped_reason": stopped_reason,
        "wandb_run_url": wandb_run.url if wandb_run is not None else None,
        "epochs_ran": len(history["train_loss"]),
    }
    with open(os.path.join(args.code_dir, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    write_progress("writing_results", 0, 1)
    if wandb_run is not None:
        wandb_run.finish()
    print("=== train.py finished successfully ===", flush=True)


if __name__ == "__main__":
    main()
