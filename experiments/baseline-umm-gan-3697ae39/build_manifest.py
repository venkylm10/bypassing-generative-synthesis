"""Assemble the enriched-manifest results.json for baseline_umm_gan from
train_summary.json + per_subject_dice.json. Also renders the two required
figures (gan_imputed_dice_distribution + training curves) from the manifest
values (not from raw logs), per the mandatory validation gate's
figure-pixels-== manifest-values rule.

Usage: python build_manifest.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CODE_DIR = "/workspace/output/code"
OUT_DIR = "/workspace/output"
FIG_DIR = os.path.join(OUT_DIR, "figures")

with open(os.path.join(CODE_DIR, "train_summary.json")) as f:
    summary = json.load(f)
with open(os.path.join(CODE_DIR, "per_subject_dice.json")) as f:
    per_subject = json.load(f)

gan = summary["gan"]
unet = summary["unet"]
bench = summary["benchmark"]
test_metrics = unet["test_metrics"]
wandb_run_url = summary.get("wandb_run_url")

mean_dice_score = test_metrics["mean_dice"]
test_loss = test_metrics["loss"]
dice_wt, dice_tc, dice_et = test_metrics["dice_per_region"]
inference_latency_ms = bench["inference_latency_ms"]
peak_inference_memory_mb = bench["peak_inference_memory_mb"]

per_subject_means = [p["mean_dice"] for p in per_subject]
# sanity: mean of per-subject means should match the aggregate test mean_dice
# (both derived from the same held-out test set, computed two different ways:
# aggregate intersection/union vs. mean-of-per-subject) -- expect close but
# not bit-identical, since hard_dice aggregation differs from per-subject
# averaging when subjects have small/zero-union regions.
mean_of_per_subject = float(np.mean(per_subject_means))

# --------------------------------------------------------------------------
# results[] -- the SSOT
# --------------------------------------------------------------------------
results = [
    {
        "name": "mean_dice_score", "method": "gan_imputed_t1c",
        "value": mean_dice_score, "unit": "dimensionless",
        "provenance": "measured",
        "description": ("Mean Dice score (average of Whole Tumor / Tumor Core / "
                         "Enhancing Tumor region Dice) on the held-out test split, "
                         "computed with T1ce synthesized by the trained UMM-GAN "
                         "generator and fed into the same multi-encoder U-Net "
                         "architecture used by the sibling baselines; higher is "
                         "better, max 1.0."),
        "n_seeds": 1,
    },
    {
        "name": "test_dice_WT", "method": "gan_imputed_t1c", "value": dice_wt,
        "unit": "dimensionless", "provenance": "measured",
        "description": "Whole Tumor region Dice on the held-out test split (12 subjects).",
    },
    {
        "name": "test_dice_TC", "method": "gan_imputed_t1c", "value": dice_tc,
        "unit": "dimensionless", "provenance": "measured",
        "description": "Tumor Core region Dice on the held-out test split (12 subjects).",
    },
    {
        "name": "test_dice_ET", "method": "gan_imputed_t1c", "value": dice_et,
        "unit": "dimensionless", "provenance": "measured",
        "description": "Enhancing Tumor region Dice on the held-out test split (12 subjects).",
    },
    {
        "name": "test_loss", "method": "gan_imputed_t1c", "value": test_loss,
        "unit": "dimensionless", "provenance": "measured",
        "description": "Dice+BCE combo loss on the held-out test split (lower is better).",
    },
    {
        "name": "val_mean_dice_best", "method": "gan_imputed_t1c",
        "value": unet["best_val_mean_dice"], "unit": "dimensionless",
        "provenance": "measured",
        "description": ("Best validation-split mean Dice achieved during U-Net "
                         "training; the checkpoint-selection criterion for weights/best.pt."),
    },
    {
        "name": "inference_latency_ms", "method": "gan_imputed_t1c",
        "value": inference_latency_ms, "unit": "ms", "provenance": "measured",
        "description": ("End-to-end forward-pass latency: GAN T1ce synthesis + "
                         "U-Net segmentation, batch=1, on an A100-40GB. CUDA-"
                         "synchronized mean over 20 timed runs after 5 warm-up runs."),
    },
    {
        "name": "peak_inference_memory_mb", "method": "gan_imputed_t1c",
        "value": peak_inference_memory_mb, "unit": "MB", "provenance": "measured",
        "description": ("Peak GPU memory allocated (torch.cuda.max_memory_allocated) "
                         "during one end-to-end (GAN synthesis + U-Net) forward pass, batch=1."),
    },
    {
        "name": "gan_val_l1_best", "method": "gan_imputed_t1c",
        "value": gan["best_val_l1"], "unit": "dimensionless", "provenance": "measured",
        "description": ("Best validation-split L1 reconstruction error (z-scored intensity "
                         "units) achieved by the GAN generator during training -- the "
                         "checkpoint-selection criterion for the synthesis model, and a "
                         "proxy for pixel-level synthesis fidelity."),
    },
    {
        "name": "gan_training_time_min", "method": "gan_imputed_t1c",
        "value": gan["total_time_min"], "unit": "min", "provenance": "measured",
        "description": "Wall-clock GPU time spent training the GAN generator+discriminator.",
    },
    {
        "name": "unet_training_time_min", "method": "gan_imputed_t1c",
        "value": unet["total_time_min"], "unit": "min", "provenance": "measured",
        "description": "Wall-clock GPU time spent training the downstream segmentation U-Net.",
    },
    {
        "name": "n_params_generator", "method": "gan_imputed_t1c",
        "value": gan["n_params_g"], "unit": "count", "provenance": "measured",
        "description": "Total trainable parameters in the UMM-GAN generator (UMMGenerator3D).",
    },
    {
        "name": "n_params_discriminator", "method": "gan_imputed_t1c",
        "value": gan["n_params_d"], "unit": "count", "provenance": "measured",
        "description": "Total trainable parameters in the PatchGAN3D discriminator.",
    },
    {
        "name": "n_params_unet", "method": "gan_imputed_t1c",
        "value": unet["n_params"], "unit": "count", "provenance": "measured",
        "description": "Total trainable parameters in the multi-encoder segmentation U-Net.",
    },
    {
        "name": "mean_of_per_subject_dice", "method": "gan_imputed_t1c",
        "value": mean_of_per_subject, "unit": "dimensionless", "provenance": "measured",
        "description": ("Mean of per-test-subject mean Dice scores (the "
                         "gan_imputed_dice_distribution figure's underlying data), as a "
                         "cross-check against the aggregate mean_dice_score computed by "
                         "pooled intersection/union across all test-set voxels."),
    },
]

# --------------------------------------------------------------------------
# baselines[] -- sibling baseline_zero_fill, directly observed in this
# project (same pod/disk, same dataset+split, ran to completion and pushed
# to GitHub prior to this run) -- see STATE.md "Open issues" for provenance.
# --------------------------------------------------------------------------
baselines = [
    {
        "name": "baseline_zero_fill (no imputation)",
        "provenance": "reproduced_run_id",
        "headline": True,
        "metrics": {
            "mean_dice_score": 0.6562679409980774,
            "test_dice_WT": 0.753724992275238,
            "test_dice_TC": 0.5822533369064331,
            "test_dice_ET": 0.632825493812561,
        },
    },
]

# --------------------------------------------------------------------------
# figures[]
# --------------------------------------------------------------------------
figures = [
    {
        "name": "gan_imputed_dice_distribution",
        "renders": ["mean_dice_score"],
        "inline_data": {
            "subject_ids": [p["subject_id"] for p in per_subject],
            "per_subject_mean_dice": per_subject_means,
        },
    },
    {
        "name": "gan_training_curve",
        "renders": [],
        "inline_data": {
            "gan_train_g_loss_curve": gan["history"]["g_loss"],
            "gan_train_d_loss_curve": gan["history"]["d_loss"],
            "gan_val_l1_curve": gan["history"]["val_l1"],
        },
    },
    {
        "name": "unet_training_curve",
        "renders": [],
        "inline_data": {
            "unet_train_loss_curve": unet["history"]["train_loss"],
            "unet_val_loss_curve": unet["history"]["val_loss"],
            "unet_val_mean_dice_curve": unet["history"]["val_mean_dice"],
        },
    },
]

# --------------------------------------------------------------------------
# metrics{} -- flat derived map (SSOT projection rule: last entry wins per name)
# --------------------------------------------------------------------------
metrics = {}
for entry in results:
    if isinstance(entry["value"], (int, float)):
        metrics[entry["name"]] = entry["value"]

manifest = {
    "manifest_version": 1,
    "config": {
        "experiment_id": "3697ae39-6b89-4a49-8731-2274d4ecb4ad_5fc10e",
        "seed": 42,
        "n_train": len(summary["train_ids"]), "n_val": len(summary["val_ids"]),
        "n_test": len(summary["test_ids"]),
        "resolution": 96, "missing_modality": "t1c",
        "gan_epochs_ran": gan["epochs_ran"], "gan_stopped_reason": gan["stopped_reason"],
        "unet_epochs_ran": unet["epochs_ran"], "unet_stopped_reason": unet["stopped_reason"],
        "base_ch_g": 24, "base_ch_d": 32, "base_ch_unet": 24,
        "lr_g": 2e-4, "lr_d": 2e-4, "lr_unet": 1e-4,
        "lambda_l1": 100.0, "lambda_gdl": 10.0, "lambda_adv": 1.0,
        "dataset": "Spirit-26/BraTS-2024-Complete (BraTS-GLI, HuggingFace)",
        "n_subjects_cached": 80,
    },
    "results": results,
    "baselines": baselines,
    "figures": figures,
    "metrics": metrics,
    # top-level schema-required keys (results_json_schema)
    "mean_dice_score": mean_dice_score,
    "inference_latency_ms": inference_latency_ms,
    "peak_inference_memory_mb": peak_inference_memory_mb,
    "wandb_run_url": wandb_run_url,
    "github_commit_sha": None,
    "validation_status": "pending",
}

with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
    json.dump(manifest, f, indent=2)
print(f"wrote results.json | mean_dice_score={mean_dice_score:.4f} "
      f"inference_latency_ms={inference_latency_ms:.2f} peak_mem={peak_inference_memory_mb:.1f}MB")

# --------------------------------------------------------------------------
# render figures FROM the manifest values (not raw logs) -- gate step 1
# --------------------------------------------------------------------------
os.makedirs(FIG_DIR, exist_ok=True)

# 1. gan_imputed_dice_distribution
fig, ax = plt.subplots(figsize=(6, 4.5))
vals = manifest["figures"][0]["inline_data"]["per_subject_mean_dice"]
ax.hist(vals, bins=min(8, len(vals)), color="#4C72B0", edgecolor="black", alpha=0.85)
ax.axvline(np.mean(vals), color="crimson", linestyle="--", label=f"mean={np.mean(vals):.3f}")
ax.axvline(mean_dice_score, color="darkorange", linestyle=":",
           label=f"aggregate mean_dice_score={mean_dice_score:.3f}")
ax.set_xlabel("Per-subject mean Dice (WT/TC/ET avg)")
ax.set_ylabel("Number of test subjects")
ax.set_title("GAN-imputed T1ce: test-set Dice distribution (n=%d)" % len(vals))
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "gan_imputed_dice_distribution.png"), dpi=140)
plt.close(fig)

# 2. gan_training_curve
gan_data = manifest["figures"][1]["inline_data"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].plot(gan_data["gan_train_g_loss_curve"], label="train G loss")
axes[0].plot(gan_data["gan_train_d_loss_curve"], label="train D loss")
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].legend()
axes[0].set_title("GAN adversarial training losses")
axes[1].plot(gan_data["gan_val_l1_curve"], color="green", label="val L1 (T1ce synthesis)")
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("L1 (z-scored units)"); axes[1].legend()
axes[1].set_title("GAN generator validation reconstruction error")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "gan_training_curve.png"), dpi=140)
plt.close(fig)

# 3. unet_training_curve
unet_data = manifest["figures"][2]["inline_data"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].plot(unet_data["unet_train_loss_curve"], label="train loss")
axes[0].plot(unet_data["unet_val_loss_curve"], label="val loss")
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("Dice+BCE loss"); axes[0].legend()
axes[0].set_title("U-Net (GAN-imputed T1ce) training/val loss")
axes[1].plot(unet_data["unet_val_mean_dice_curve"], color="purple", label="val mean Dice")
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("Dice"); axes[1].legend()
axes[1].set_title("U-Net validation mean Dice")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "unet_training_curve.png"), dpi=140)
plt.close(fig)

print("figures written:", os.listdir(FIG_DIR))
