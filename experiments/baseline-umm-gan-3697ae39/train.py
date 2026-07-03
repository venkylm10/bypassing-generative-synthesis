"""End-to-end baseline_umm_gan pipeline:

  1. Preprocess/cache BraTS volumes (reuses the shared 96^3 cache; a no-op if
     already cached, as it is on this pod).
  2. Train a UMM-GAN-style generator+discriminator (models/gan.py) to
     synthesize the missing T1ce modality from (T1n, T2w, T2f).
  3. Run the trained generator over every subject to produce GAN-imputed T1c
     volumes (impute.py logic, inlined below as `run_imputation`).
  4. Train the SAME multi-encoder late-fusion 3D U-Net architecture used by
     the baseline_zero_fill sibling (models/unet.py) — differing only in
     that the T1c branch now receives GAN-synthesized pixels instead of
     zeros — to establish the "heavy generative synthesis" target pipeline's
     segmentation accuracy.
  5. Evaluate test-set Dice, and benchmark end-to-end (GAN + U-Net) inference
     latency and peak memory for a single forward pass.

Usage:
    python train.py --data_root <BraTS-GLI/train dir> --cache_dir <seg cache> \
        --gan_cache_dir <gan-imputed cache> --code_dir <this code/ dir> \
        --train_log ../train.log
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
from models.gan import PatchDiscriminator3D, UMMGenerator3D, gradient_l1_loss  # noqa: E402
from models.unet import MultiEncoderUNet3D  # noqa: E402
from utils.dataset import (  # noqa: E402
    BraTSGANDataset,
    BraTSGANImputedDataset,
    TARGET_CLAMP,
    _from_tanh_range,
    split_subjects,
)
from utils.losses import DiceBCELoss, hard_dice  # noqa: E402
from utils.preprocess import MODALITIES, run as preprocess_run  # noqa: E402

REGION_NAMES = ["WT", "TC", "ET"]
PROGRESS_PATH = "/workspace/output/PROGRESS.json"
WANDB_PROJECT = "prof-f9e1ad4b-plan-fa67285a"
WANDB_RUN_NAME = "baseline_umm_gan"


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


# --------------------------------------------------------------------------
# Phase 1: GAN training
# --------------------------------------------------------------------------

def train_gan(args, device, train_ids, val_ids, weights_dir, wandb_run, log):
    train_ds = BraTSGANDataset(args.cache_dir, train_ids, train=True)
    val_ds = BraTSGANDataset(args.cache_dir, val_ids, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    G = UMMGenerator3D(in_ch=3, base_ch=args.base_ch_g, out_ch=1).to(device)
    D = PatchDiscriminator3D(in_ch=4, base_ch=args.base_ch_d).to(device)
    n_params_g = sum(p.numel() for p in G.parameters())
    n_params_d = sum(p.numel() for p in D.parameters())
    log(f"=== GAN: G params={n_params_g/1e6:.2f}M D params={n_params_d/1e6:.2f}M ===")

    opt_g = torch.optim.AdamW(G.parameters(), lr=args.lr_g, betas=(0.5, 0.999), weight_decay=1e-5)
    opt_d = torch.optim.AdamW(D.parameters(), lr=args.lr_d, betas=(0.5, 0.999), weight_decay=1e-5)
    mse = nn.MSELoss()
    l1 = nn.L1Loss()

    last_path = os.path.join(weights_dir, "last.pt")
    best_path = os.path.join(weights_dir, "best.pt")

    start_epoch = 0
    global_step = 0
    best_metric = float("inf")  # lower val_l1 is better
    patience_counter = 0
    history = {"g_loss": [], "d_loss": [], "val_l1": []}

    if os.path.exists(last_path):
        log(f"=== resuming GAN from {last_path} ===")
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        G.load_state_dict(ckpt["model_g"])
        D.load_state_dict(ckpt["model_d"])
        opt_g.load_state_dict(ckpt["opt_g"])
        opt_d.load_state_dict(ckpt["opt_d"])
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

    write_progress("training", start_epoch, args.gan_epochs)
    t0 = time.time()
    stopped_reason = "completed_all_epochs"

    for epoch in range(start_epoch, args.gan_epochs):
        G.train()
        D.train()
        epoch_g, epoch_d = 0.0, 0.0
        n_batches = 0
        ep_t0 = time.time()
        for condition, target, _sids in train_loader:
            condition = condition.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # --- D step ---
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    fake = G(condition)
                real_score = D(torch.cat([condition, target], dim=1))
                fake_score = D(torch.cat([condition, fake], dim=1))
                d_loss = 0.5 * (mse(real_score, torch.ones_like(real_score)) +
                                 mse(fake_score, torch.zeros_like(fake_score)))
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), 5.0)
            opt_d.step()

            # --- G step ---
            with torch.autocast("cuda", dtype=torch.bfloat16):
                fake = G(condition)
                fake_score = D(torch.cat([condition, fake], dim=1))
                g_adv = mse(fake_score, torch.ones_like(fake_score))
                g_l1 = l1(fake, target)
                g_gdl = gradient_l1_loss(fake.float(), target.float())
                g_loss = args.lambda_adv * g_adv + args.lambda_l1 * g_l1 + args.lambda_gdl * g_gdl
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), 5.0)
            opt_g.step()

            epoch_g += g_loss.item()
            epoch_d += d_loss.item()
            n_batches += 1
            global_step += 1
            if global_step % 5 == 0:
                log(f"[gan step {global_step}, epoch {epoch}] g_loss={g_loss.item():.4f} "
                    f"d_loss={d_loss.item():.4f} g_l1={g_l1.item():.4f} g_adv={g_adv.item():.4f}")
            if wandb_run is not None:
                wandb_run.log({"gan/g_loss_step": g_loss.item(), "gan/d_loss_step": d_loss.item()})

        train_g = epoch_g / max(n_batches, 1)
        train_d = epoch_d / max(n_batches, 1)

        # validation: L1 only (no discriminator involved)
        G.eval()
        val_l1_sum, n_val = 0.0, 0
        with torch.no_grad():
            for condition, target, _sids in val_loader:
                condition = condition.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    fake = G(condition)
                val_l1_sum += l1(fake.float(), target.float()).item() * condition.shape[0]
                n_val += condition.shape[0]
        val_l1 = val_l1_sum / max(n_val, 1)
        ep_elapsed = time.time() - ep_t0

        history["g_loss"].append(train_g)
        history["d_loss"].append(train_d)
        history["val_l1"].append(val_l1)

        log(f"=== gan epoch {epoch} done | train_g_loss={train_g:.4f} train_d_loss={train_d:.4f} "
            f"val_l1={val_l1:.4f} elapsed={ep_elapsed:.1f}s ===")
        if wandb_run is not None:
            wandb_run.log({"epoch_gan": epoch, "gan/train_g_loss": train_g,
                            "gan/train_d_loss": train_d, "gan/val_l1": val_l1})

        write_progress("training", epoch + 1, args.gan_epochs)

        improved = val_l1 < best_metric
        if improved:
            best_metric = val_l1
            patience_counter = 0
        else:
            patience_counter += 1

        state = {
            "model_g": G.state_dict(), "model_d": D.state_dict(),
            "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "rng_numpy": np.random.get_state(), "rng_python": random.getstate(),
            "epoch": epoch, "global_step": global_step, "best_metric": best_metric,
            "patience_counter": patience_counter, "history": history,
        }
        save_checkpoint(last_path, state)
        if improved:
            save_checkpoint(best_path, state)
            log(f"=== new best GAN val_l1={best_metric:.4f} @ epoch {epoch}, saved best.pt ===")

        elapsed_min = (time.time() - t0) / 60.0
        if patience_counter >= args.patience_gan:
            stopped_reason = f"early_stopping (patience={args.patience_gan})"
            log(f"=== GAN early stopping at epoch {epoch} ===")
            break
        if elapsed_min >= args.max_train_minutes_gan:
            stopped_reason = f"max_train_minutes_gan budget reached ({args.max_train_minutes_gan} min)"
            log(f"=== GAN stopping at epoch {epoch}: time budget reached ===")
            break

    total_time_min = (time.time() - t0) / 60.0
    log(f"=== GAN training finished: reason={stopped_reason} total_time={total_time_min:.2f} min ===")

    if not os.path.exists(best_path):
        save_checkpoint(best_path, state)  # degenerate case: never improved past inf

    return {
        "n_params_g": n_params_g, "n_params_d": n_params_d, "history": history,
        "best_val_l1": best_metric, "total_time_min": total_time_min,
        "stopped_reason": stopped_reason, "epochs_ran": len(history["g_loss"]),
    }


# --------------------------------------------------------------------------
# Phase 2: run trained generator over every subject -> GAN-imputed T1c cache
# --------------------------------------------------------------------------

def run_imputation(args, device, all_ids, weights_dir, log):
    os.makedirs(args.gan_cache_dir, exist_ok=True)
    G = UMMGenerator3D(in_ch=3, base_ch=args.base_ch_g, out_ch=1).to(device)
    ckpt = torch.load(os.path.join(weights_dir, "best.pt"), map_location=device, weights_only=False)
    G.load_state_dict(ckpt["model_g"])
    G.eval()
    log(f"=== imputation: loaded GAN generator from epoch {ckpt['epoch']}, "
        f"best_val_l1={ckpt['best_metric']:.4f} ===")

    t0 = time.time()
    with torch.no_grad():
        for i, sid in enumerate(all_ids):
            out_path = os.path.join(args.gan_cache_dir, f"{sid}.npz")
            if os.path.exists(out_path):
                continue
            npz = np.load(os.path.join(args.cache_dir, f"{sid}.npz"))
            images = npz["images"]  # (4,S,S,S)
            condition = np.stack([images[0], images[2], images[3]], axis=0)[None]  # (1,3,S,S,S)
            cond_t = torch.from_numpy(condition).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                fake = G(cond_t)
            t1c_synth = _from_tanh_range(fake[0, 0].float().cpu().numpy())
            np.savez_compressed(out_path, t1c_synth=t1c_synth.astype(np.float32))
            if (i + 1) % 20 == 0 or i == len(all_ids) - 1:
                log(f"[impute] {i+1}/{len(all_ids)} subjects synthesized, elapsed={time.time()-t0:.1f}s")
    log(f"=== imputation done: {len(all_ids)} subjects, elapsed={time.time()-t0:.1f}s ===")


# --------------------------------------------------------------------------
# Phase 3: segmentation U-Net training (identical arch/hparams to zero_fill)
# --------------------------------------------------------------------------

def evaluate_unet(model, loader, device, loss_fn):
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


def train_unet(args, device, train_ids, val_ids, test_ids, weights_dir, wandb_run, log):
    train_ds = BraTSGANImputedDataset(args.cache_dir, args.gan_cache_dir, train_ids, train=True)
    val_ds = BraTSGANImputedDataset(args.cache_dir, args.gan_cache_dir, val_ids, train=False)
    test_ds = BraTSGANImputedDataset(args.cache_dir, args.gan_cache_dir, test_ids, train=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    model = MultiEncoderUNet3D(n_modalities=4, base_ch=args.base_ch_unet, n_classes=3).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"=== U-Net params: {n_params/1e6:.2f}M ===")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr_unet, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.unet_epochs)
    loss_fn = DiceBCELoss()

    last_path = os.path.join(weights_dir, "last.pt")
    best_path = os.path.join(weights_dir, "best.pt")

    start_epoch = 0
    global_step = 0
    best_metric = -1.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_mean_dice": []}

    if os.path.exists(last_path):
        log(f"=== resuming U-Net from {last_path} ===")
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

    write_progress("training", start_epoch, args.unet_epochs)
    t0 = time.time()
    stopped_reason = "completed_all_epochs"

    for epoch in range(start_epoch, args.unet_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        ep_t0 = time.time()
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
                log(f"[unet step {global_step}, epoch {epoch}] loss={loss.item():.4f} lr={lr_now:.6f}")
            if wandb_run is not None:
                wandb_run.log({"train/loss_step": loss.item(), "lr": lr_now})

        scheduler.step()
        train_loss = epoch_loss / max(n_batches, 1)
        val_metrics = evaluate_unet(model, val_loader, device, loss_fn)
        ep_elapsed = time.time() - ep_t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_mean_dice"].append(val_metrics["mean_dice"])

        log(f"=== unet epoch {epoch} done | avg_train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_mean_dice={val_metrics['mean_dice']:.4f} "
            f"val_dice_WT={val_metrics['dice_per_region'][0]:.4f} "
            f"val_dice_TC={val_metrics['dice_per_region'][1]:.4f} "
            f"val_dice_ET={val_metrics['dice_per_region'][2]:.4f} elapsed={ep_elapsed:.1f}s ===")

        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch, "train/loss_epoch": train_loss,
                "val/loss": val_metrics["loss"], "val/mean_dice": val_metrics["mean_dice"],
                "val/dice_WT": val_metrics["dice_per_region"][0],
                "val/dice_TC": val_metrics["dice_per_region"][1],
                "val/dice_ET": val_metrics["dice_per_region"][2],
            })

        write_progress("training", epoch + 1, args.unet_epochs)

        improved = val_metrics["mean_dice"] > best_metric
        if improved:
            best_metric = val_metrics["mean_dice"]
            patience_counter = 0
        else:
            patience_counter += 1

        state = {
            "model": model.state_dict(), "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "rng_numpy": np.random.get_state(), "rng_python": random.getstate(),
            "epoch": epoch, "global_step": global_step, "best_metric": best_metric,
            "patience_counter": patience_counter, "history": history,
        }
        save_checkpoint(last_path, state)
        if improved:
            save_checkpoint(best_path, state)
            log(f"=== new best val_mean_dice={best_metric:.4f} @ epoch {epoch}, saved best.pt ===")

        elapsed_min = (time.time() - t0) / 60.0
        if patience_counter >= args.patience_unet:
            stopped_reason = f"early_stopping (patience={args.patience_unet})"
            log(f"=== U-Net early stopping at epoch {epoch} ===")
            break
        if elapsed_min >= args.max_train_minutes_unet:
            stopped_reason = f"max_train_minutes_unet budget reached ({args.max_train_minutes_unet} min)"
            log(f"=== U-Net stopping at epoch {epoch}: time budget reached ===")
            break

    total_time_min = (time.time() - t0) / 60.0
    log(f"=== U-Net training finished: reason={stopped_reason} total_time={total_time_min:.2f} min ===")

    write_progress("evaluating", 0, 1)
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        log(f"=== loaded best.pt (epoch {ckpt['epoch']}, val_mean_dice={ckpt['best_metric']:.4f}) for test eval ===")
    test_metrics = evaluate_unet(model, test_loader, device, loss_fn)
    log(f"=== FINAL TEST | test_loss={test_metrics['loss']:.4f} test_mean_dice={test_metrics['mean_dice']:.4f} "
        f"test_dice_WT={test_metrics['dice_per_region'][0]:.4f} "
        f"test_dice_TC={test_metrics['dice_per_region'][1]:.4f} "
        f"test_dice_ET={test_metrics['dice_per_region'][2]:.4f} ===")

    return {
        "n_params": n_params, "history": history, "best_val_mean_dice": best_metric,
        "test_metrics": test_metrics, "total_time_min": total_time_min,
        "stopped_reason": stopped_reason, "epochs_ran": len(history["train_loss"]),
    }


# --------------------------------------------------------------------------
# Phase 4: latency + peak-memory benchmark (GAN synthesis + U-Net, end to end)
# --------------------------------------------------------------------------

def benchmark_inference(args, device, subject_id, weights_gan_dir, weights_unet_dir, log):
    G = UMMGenerator3D(in_ch=3, base_ch=args.base_ch_g, out_ch=1).to(device)
    ckpt_g = torch.load(os.path.join(weights_gan_dir, "best.pt"), map_location=device, weights_only=False)
    G.load_state_dict(ckpt_g["model_g"])
    G.eval()

    unet = MultiEncoderUNet3D(n_modalities=4, base_ch=args.base_ch_unet, n_classes=3).to(device)
    ckpt_u = torch.load(os.path.join(weights_unet_dir, "best.pt"), map_location=device, weights_only=False)
    unet.load_state_dict(ckpt_u["model"])
    unet.eval()

    npz = np.load(os.path.join(args.cache_dir, f"{subject_id}.npz"))
    images = npz["images"]
    condition = torch.from_numpy(np.stack([images[0], images[2], images[3]], axis=0)[None]).to(device)
    t1n = torch.from_numpy(images[0:1][None]).to(device)
    t2w = torch.from_numpy(images[2:3][None]).to(device)
    t2f = torch.from_numpy(images[3:4][None]).to(device)

    def full_forward():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            t1c_synth = G(condition)
            mods = [t1n, t1c_synth, t2w, t2f]
            logits = unet(mods)
        return logits

    for _ in range(5):  # warm-up
        full_forward()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    full_forward()
    torch.cuda.synchronize()
    peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6

    n_runs = 20
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_runs):
        full_forward()
    torch.cuda.synchronize()
    latency_ms = (time.time() - t0) / n_runs * 1000.0

    log(f"=== inference benchmark (subject={subject_id}): latency={latency_ms:.2f} ms, "
        f"peak_mem={peak_mem_mb:.2f} MB (n_runs={n_runs}) ===")
    return {"inference_latency_ms": latency_ms, "peak_inference_memory_mb": peak_mem_mb}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--gan_cache_dir", required=True)
    ap.add_argument("--code_dir", required=True)
    ap.add_argument("--train_log", required=True)

    ap.add_argument("--gan_epochs", type=int, default=150)
    ap.add_argument("--lr_g", type=float, default=2e-4)
    ap.add_argument("--lr_d", type=float, default=2e-4)
    ap.add_argument("--base_ch_g", type=int, default=24)
    ap.add_argument("--base_ch_d", type=int, default=32)
    ap.add_argument("--patience_gan", type=int, default=20)
    ap.add_argument("--max_train_minutes_gan", type=float, default=90.0)
    ap.add_argument("--lambda_l1", type=float, default=100.0)
    ap.add_argument("--lambda_gdl", type=float, default=10.0)
    ap.add_argument("--lambda_adv", type=float, default=1.0)

    ap.add_argument("--unet_epochs", type=int, default=120)
    ap.add_argument("--lr_unet", type=float, default=1e-4)
    ap.add_argument("--base_ch_unet", type=int, default=24)
    ap.add_argument("--patience_unet", type=int, default=25)
    ap.add_argument("--max_train_minutes_unet", type=float, default=150.0)

    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_val", type=int, default=8)
    ap.add_argument("--n_test", type=int, default=12)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "GPU required for this experiment (constraints.gpu_required=true)"

    log_f = open(args.train_log, "a")

    def log(msg: str):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    weights_gan_dir = os.path.join(args.code_dir, "weights_gan")
    weights_unet_dir = os.path.join(args.code_dir, "weights")
    os.makedirs(weights_gan_dir, exist_ok=True)
    os.makedirs(weights_unet_dir, exist_ok=True)

    subjects = sorted(os.listdir(args.data_root))
    train_ids, val_ids, test_ids = split_subjects(subjects, args.seed, args.n_val, args.n_test)
    with open(os.path.join(args.code_dir, "split.json"), "w") as f:
        json.dump({"train": train_ids, "val": val_ids, "test": test_ids}, f, indent=2)

    log(f"=== baseline_umm_gan | subjects total={len(subjects)} train={len(train_ids)} "
        f"val={len(val_ids)} test={len(test_ids)} ===")
    log(f"=== modality order: {MODALITIES}, missing/synthesized modality: t1c (index 1) ===")

    write_progress("downloading_model", 0, 1)
    log("=== preprocessing (resize+cache) all subjects ===")
    preprocess_run(args.data_root, args.cache_dir, subjects, device=device)

    wandb_run = None
    if os.environ.get("WANDB_API_KEY"):
        try:
            import wandb
            wandb_run = wandb.init(
                project=WANDB_PROJECT, name=WANDB_RUN_NAME,
                config={
                    "gan_epochs": args.gan_epochs, "unet_epochs": args.unet_epochs,
                    "batch_size": args.batch_size, "lr_g": args.lr_g, "lr_d": args.lr_d,
                    "lr_unet": args.lr_unet, "base_ch_g": args.base_ch_g,
                    "base_ch_d": args.base_ch_d, "base_ch_unet": args.base_ch_unet,
                    "seed": args.seed, "n_train": len(train_ids), "n_val": len(val_ids),
                    "n_test": len(test_ids), "resolution": 96, "missing_modality": "t1c",
                    "lambda_l1": args.lambda_l1, "lambda_gdl": args.lambda_gdl,
                    "lambda_adv": args.lambda_adv,
                },
            )
            log(f"WANDB_RUN_URL: {wandb_run.url}")
        except Exception as e:
            log(f"[warn] wandb init failed: {e}")

    write_progress("loading_model", 0, 1)
    log("=== Loaded: UMMGenerator3D + PatchDiscriminator3D + MultiEncoderUNet3D (baseline_umm_gan) ===")

    gan_summary = train_gan(args, device, train_ids, val_ids, weights_gan_dir, wandb_run, log)

    write_progress("computing_metrics", 0, 1)
    log("=== running GAN imputation over all subjects ===")
    run_imputation(args, device, subjects, weights_gan_dir, log)

    unet_summary = train_unet(args, device, train_ids, val_ids, test_ids, weights_unet_dir, wandb_run, log)

    bench = benchmark_inference(args, device, test_ids[0], weights_gan_dir, weights_unet_dir, log)

    if wandb_run is not None:
        wandb_run.log({
            "test/loss": unet_summary["test_metrics"]["loss"],
            "test/mean_dice": unet_summary["test_metrics"]["mean_dice"],
            "test/dice_WT": unet_summary["test_metrics"]["dice_per_region"][0],
            "test/dice_TC": unet_summary["test_metrics"]["dice_per_region"][1],
            "test/dice_ET": unet_summary["test_metrics"]["dice_per_region"][2],
            "mean_dice_score": unet_summary["test_metrics"]["mean_dice"],
            "inference_latency_ms": bench["inference_latency_ms"],
            "peak_inference_memory_mb": bench["peak_inference_memory_mb"],
        })

    summary = {
        "train_ids": train_ids, "val_ids": val_ids, "test_ids": test_ids,
        "gan": gan_summary, "unet": unet_summary, "benchmark": bench,
        "wandb_run_url": wandb_run.url if wandb_run is not None else None,
    }
    with open(os.path.join(args.code_dir, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    write_progress("writing_results", 0, 1)
    if wandb_run is not None:
        wandb_run.finish()
    log("=== train.py finished successfully ===")
    log_f.close()


if __name__ == "__main__":
    main()
