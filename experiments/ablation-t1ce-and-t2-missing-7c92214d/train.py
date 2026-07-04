"""Train the multi-encoder late-fusion 3D U-Net with deterministic latent
imputation (Normalized Mean Algorithm) replacing the missing T1ce branch.

Checkpoints: weights/last.pt (every epoch) and weights/best.pt (on val Dice
improvement), each a complete state (model/optimizer/scheduler/rng/epoch/
best_metric/patience_counter/history/wandb_run_id) per the checkpoint contract.
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from models.unet3d import FusionUNet3D, MODALITIES
from utils.dataset import BraTSSubset, make_splits
from utils.losses import ce_dice_loss, dice_per_class

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.abspath(os.path.join(CODE_DIR, ".."))
PROGRESS_PATH = os.path.join(OUTPUT_DIR, "PROGRESS.json")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_progress(phase, current, total):
    try:
        with open(PROGRESS_PATH, "w") as f:
            json.dump({"phase": phase, "current": current, "total": total}, f)
    except Exception:
        pass


def save_checkpoint(state, path):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def evaluate(model, loader, device, missing):
    model.eval()
    all_dice = []
    losses = []
    with torch.no_grad():
        for volumes, label, sids in loader:
            volumes = {m: v.to(device) for m, v in volumes.items()}
            label = label.to(device)
            logits = model(volumes, missing=missing)
            loss = ce_dice_loss(logits, label)
            losses.append(loss.item())
            dpc = dice_per_class(logits, label)
            all_dice.append(dpc.detach().cpu().numpy())
    mean_loss = float(np.mean(losses))
    mean_dice_per_class = np.mean(np.stack(all_dice), axis=0)
    mean_dice = float(mean_dice_per_class[1:].mean())  # exclude background class 0
    return mean_loss, mean_dice, mean_dice_per_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None, help="override config seed")
    parser.add_argument("--tag", type=str, default="seed0", help="run tag; weights/logs land under weights_<tag>/")
    args = parser.parse_args()

    with open(os.path.join(CODE_DIR, "configs", "config.json")) as f:
        cfg = json.load(f)
    if args.seed is not None:
        cfg["seed"] = args.seed

    weights_dir = os.path.join(CODE_DIR, f"weights_{args.tag}")
    os.makedirs(weights_dir, exist_ok=True)

    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    train_ids, val_ids, test_ids = make_splits(cfg["seed"], cfg["n_train"], cfg["n_val"], cfg["n_test"])
    with open(os.path.join(OUTPUT_DIR, f"splits_{args.tag}.json"), "w") as f:
        json.dump({"train": train_ids, "val": val_ids, "test": test_ids}, f, indent=2)
    print(f"Splits: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}", flush=True)

    train_ds = BraTSSubset(train_ids)
    val_ds = BraTSSubset(val_ids)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=2)

    model = FusionUNet3D(widths=tuple(cfg["widths"]), bottleneck_extra=cfg["bottleneck_extra"],
                          n_classes=cfg["n_classes"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    wandb_run = None
    wandb_run_id = None
    use_wandb = bool(os.environ.get("WANDB_API_KEY"))
    if use_wandb:
        import wandb
        run_name = "ablation_t1ce_and_t2_missing" if args.tag == "seed0" else f"ablation_t1ce_and_t2_missing_{args.tag}"
        wandb_run = wandb.init(project="prof-f9e1ad4b-plan-fa67285a", name=run_name, config=cfg)
        wandb_run_id = wandb_run.id
        print(f"WANDB_RUN_URL: {wandb_run.url}", flush=True)

    start_epoch = 0
    best_metric = -1.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_dice": []}

    last_ckpt_path = os.path.join(weights_dir, "last.pt")
    if os.path.exists(last_ckpt_path):
        ckpt = torch.load(last_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        torch.set_rng_state(ckpt["rng_torch"])
        if torch.cuda.is_available() and ckpt.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
        np.random.set_state(ckpt["rng_numpy"])
        random.setstate(ckpt["rng_python"])
        start_epoch = ckpt["epoch"] + 1
        best_metric = ckpt["best_metric"]
        patience_counter = ckpt["patience_counter"]
        history = ckpt["history"]
        wandb_run_id = ckpt.get("wandb_run_id", wandb_run_id)
        print(f"Resumed from epoch {start_epoch}", flush=True)

    missing = cfg["missing_modality"]
    n_epochs = cfg["epochs"]
    global_step = start_epoch * (len(train_ds) // cfg["batch_size"])

    for epoch in range(start_epoch, n_epochs):
        write_progress("training", epoch, n_epochs)
        model.train()
        # re-seed sampler shuffle deterministically per (epoch, seed) so resume matches
        torch.manual_seed(cfg["seed"] * 100000 + epoch)
        epoch_t0 = time.time()
        train_losses = []
        for volumes, label, sids in train_loader:
            volumes = {m: v.to(device) for m, v in volumes.items()}
            label = label.to(device)
            optimizer.zero_grad()
            logits = model(volumes, missing=missing)
            loss = ce_dice_loss(logits, label)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            global_step += 1
            if global_step % 5 == 0:
                line = f"[step {global_step}, epoch {epoch}] loss={loss.item():.4f} lr={optimizer.param_groups[0]['lr']:.6f}"
                print(line, flush=True)
                if use_wandb:
                    wandb_run.log({"train/loss_step": loss.item(), "epoch": epoch}, step=global_step)

        scheduler.step()
        val_loss, val_dice, val_dice_pc = evaluate(model, val_loader, device, missing)
        avg_train_loss = float(np.mean(train_losses))
        elapsed = time.time() - epoch_t0

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)

        summary = (f"=== epoch {epoch} done | avg_train_loss={avg_train_loss:.4f} "
                   f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} elapsed={elapsed:.1f}s ===")
        print(summary, flush=True)
        if use_wandb:
            wandb_run.log({"train/loss_epoch": avg_train_loss, "val/loss": val_loss,
                            "val/mean_dice": val_dice, "epoch": epoch}, step=global_step)

        improved = val_dice > best_metric
        if improved:
            best_metric = val_dice
            patience_counter = 0
        else:
            patience_counter += 1

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
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
        save_checkpoint(state, last_ckpt_path)
        if improved:
            save_checkpoint(state, os.path.join(weights_dir, "best.pt"))
            print(f"  -> new best val_dice={best_metric:.4f}, saved best.pt", flush=True)

        if patience_counter >= cfg["early_stop_patience"]:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience_counter} epochs)", flush=True)
            break

    print("=== TRAINING COMPLETE ===", flush=True)
    print(f"Best val Dice: {best_metric:.4f}", flush=True)
    with open(os.path.join(OUTPUT_DIR, f"history_{args.tag}.json"), "w") as f:
        json.dump(history, f, indent=2)
    if use_wandb:
        wandb_run.summary["best_val_dice"] = best_metric
    write_progress("training", n_epochs, n_epochs)


if __name__ == "__main__":
    main()
