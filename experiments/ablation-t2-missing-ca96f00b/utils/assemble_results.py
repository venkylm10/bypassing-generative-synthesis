"""Aggregate per-seed eval results + training history into the enriched
results.json manifest, and render figures FROM the manifest's inline_data
(never from raw eval/history files directly) so figure pixels == table values
by construction.

This is the T2-missing ablation of `main_latent_imputation` (which used T1ce
as the missing modality). Same architecture, same training recipe, same
80-subject preprocessed cache — only `missing_modality` changed (t1c -> t2w)
via configs/config.json.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.abspath(os.path.join(CODE_DIR, ".."))
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
SEEDS_TAGS = ["seed0", "seed1", "seed2"]

os.makedirs(FIGURES_DIR, exist_ok=True)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    with open(os.path.join(CODE_DIR, "configs", "config.json")) as f:
        cfg = json.load(f)

    evals = []
    for tag in SEEDS_TAGS:
        p = os.path.join(OUTPUT_DIR, f"eval_{tag}.json")
        if os.path.exists(p):
            evals.append(load_json(p))
        else:
            print(f"WARNING: missing {p}, skipping seed {tag}", flush=True)

    if not evals:
        print("FATAL: no eval results found", flush=True)
        sys.exit(1)

    per_seed_mean_dice = [e["mean_dice_score"] for e in evals]
    mean_dice_score = float(np.mean(per_seed_mean_dice))
    dice_variance = float(np.var(per_seed_mean_dice))
    dice_std = float(np.std(per_seed_mean_dice))
    n_seeds = len(evals)
    if n_seeds > 1:
        ci_halfwidth = 1.96 * dice_std / np.sqrt(n_seeds)
        ci = [mean_dice_score - ci_halfwidth, mean_dice_score + ci_halfwidth]
    else:
        ci = [mean_dice_score, mean_dice_score]

    inference_latency_ms = float(np.mean([e["inference_latency_ms"] for e in evals]))
    peak_inference_memory_mb = float(np.mean([e["peak_inference_memory_mb"] for e in evals]))

    # per-subject dice pooled across seeds' test splits (each seed has its own held-out
    # 10 test subjects drawn from the same 80-subject pool) -> distribution figure
    all_subject_dice = []
    for e in evals:
        for r in e["per_subject_dice"]:
            all_subject_dice.append(r["mean_dice"])

    # primary (seed0) training history for the loss/dice curve figure
    history_path = os.path.join(OUTPUT_DIR, "history_seed0.json")
    history = load_json(history_path) if os.path.exists(history_path) else {"train_loss": [], "val_loss": [], "val_dice": []}

    # per-class dice, averaged across seeds
    per_class_matrix = np.stack([e["mean_dice_per_class"] for e in evals])  # (n_seeds, 4)
    per_class_mean = per_class_matrix.mean(axis=0).tolist()

    wandb_run_url = None
    train_log_path = os.path.join(OUTPUT_DIR, "train.log")
    if os.path.exists(train_log_path):
        with open(train_log_path) as f:
            for line in f:
                if line.startswith("WANDB_RUN_URL:"):
                    url = line.split("WANDB_RUN_URL:", 1)[1].strip()
                    if url and url != "None":
                        wandb_run_url = url
                        break

    # ---- enriched manifest ----
    results = [
        {
            "name": "mean_dice_score",
            "method": "latent_imputed_unet_t2_missing",
            "description": ("Absolute mean Dice score (foreground classes: necrotic core, edema, "
                             "enhancing tumor; background excluded) of the deterministic "
                             "latent-imputed 3D U-Net on held-out test subjects when T2 (t2w) is "
                             "the missing modality, averaged across "
                             f"{n_seeds} random seeds. Higher is better; 1.0 = perfect overlap. "
                             "Reported as an absolute figure per this ablation's objective "
                             "(no non-inferiority claim against GAN/zero-fill baselines is made "
                             "here, since those T2-missing baselines were not run in this experiment)."),
            "value": mean_dice_score,
            "unit": "dimensionless",
            "provenance": "measured",
            "variance": dice_variance,
            "formula": "ci = mean +/- 1.96 * population_std(per_seed_means) / sqrt(n_seeds)",
            "ci": ci,
            "n_seeds": n_seeds,
        },
        {
            "name": "mean_dice_score_necrotic_core",
            "method": "latent_imputed_unet_t2_missing",
            "description": "Mean Dice score for the necrotic/non-enhancing core class (raw BraTS labels 1 and 3 merged), T2-missing scenario, averaged across seeds.",
            "value": per_class_mean[1],
            "unit": "dimensionless",
            "provenance": "measured",
            "n_seeds": n_seeds,
        },
        {
            "name": "mean_dice_score_edema",
            "method": "latent_imputed_unet_t2_missing",
            "description": "Mean Dice score for the peritumoral edema class (raw BraTS label 2), T2-missing scenario, averaged across seeds.",
            "value": per_class_mean[2],
            "unit": "dimensionless",
            "provenance": "measured",
            "n_seeds": n_seeds,
        },
        {
            "name": "mean_dice_score_enhancing_tumor",
            "method": "latent_imputed_unet_t2_missing",
            "description": "Mean Dice score for the enhancing tumor class (raw BraTS label 4), T2-missing scenario, averaged across seeds.",
            "value": per_class_mean[3],
            "unit": "dimensionless",
            "provenance": "measured",
            "n_seeds": n_seeds,
        },
        {
            "name": "inference_latency_ms",
            "method": "latent_imputed_unet_t2_missing",
            "description": ("End-to-end forward-pass latency of the latent-imputed architecture "
                             "(3 real modality encoders + NMA-imputed T2 branch + shared decoder) "
                             "for a single 96^3 volume on the eval GPU, averaged across seeds "
                             "(30 timed reps after 5 warmup iterations each)."),
            "value": inference_latency_ms,
            "unit": "ms",
            "provenance": "measured",
        },
        {
            "name": "peak_inference_memory_mb",
            "method": "latent_imputed_unet_t2_missing",
            "description": "Peak CUDA memory allocated during a single-subject inference forward pass, T2-missing scenario, averaged across seeds.",
            "value": peak_inference_memory_mb,
            "unit": "MB",
            "provenance": "measured",
        },
        {
            "name": "val_loss",
            "method": "latent_imputed_unet_t2_missing",
            "description": "Final-epoch validation loss (cross-entropy + Dice loss) for the primary seed (seed 0) training run (best val loss was "
                            f"{min(history['val_loss']):.4f} at an earlier epoch; training continued under early-stopping patience).",
            "value": float(history["val_loss"][-1]) if history["val_loss"] else None,
            "unit": "dimensionless",
            "provenance": "measured",
        },
        {
            "name": "train_loss",
            "method": "latent_imputed_unet_t2_missing",
            "description": "Final-epoch training loss (cross-entropy + Dice loss) for the primary seed (seed 0) training run.",
            "value": float(history["train_loss"][-1]) if history["train_loss"] else None,
            "unit": "dimensionless",
            "provenance": "measured",
        },
    ]

    # No zero-fill / UMM-GAN baselines re-run for the T2-missing condition in this
    # experiment (out of scope per TASK.yaml.objective: report absolute performance,
    # analyzing robustness WITHOUT a non-inferiority claim) -- see report.md self-critique.
    baselines = []

    figures = []

    # --- Figure 1 (expected_figures: t2_missing_dice_distribution) ---
    fig1_inline = {"per_subject_dice": all_subject_dice, "mean_dice_score": mean_dice_score}
    fig1_path = os.path.join(FIGURES_DIR, "t2_missing_dice_distribution.png")
    plt.figure(figsize=(6, 4))
    plt.hist(all_subject_dice, bins=12, color="#4C72B0", edgecolor="black", alpha=0.85)
    plt.axvline(mean_dice_score, color="red", linestyle="--", label=f"mean={mean_dice_score:.3f}")
    plt.xlabel("Per-subject mean Dice (foreground classes)")
    plt.ylabel("Count")
    plt.title(f"Dice distribution, T2 missing, across {len(all_subject_dice)} test-subject evals ({n_seeds} seeds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig1_path, dpi=150)
    plt.close()
    figures.append({
        "name": "t2_missing_dice_distribution",
        "renders": ["mean_dice_score"],
        "inline_data": fig1_inline,
    })

    # --- Figure 2: training/validation loss + dice curve (seed 0) ---
    fig2_inline = {
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
        "val_dice": history["val_dice"],
    }
    fig2_path = os.path.join(FIGURES_DIR, "loss_curve.png")
    fig, ax1 = plt.subplots(figsize=(6, 4))
    epochs_x = list(range(len(history["train_loss"])))
    ax1.plot(epochs_x, history["train_loss"], label="train_loss", color="#4C72B0")
    ax1.plot(epochs_x, history["val_loss"], label="val_loss", color="#DD8452")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss (CE + Dice)")
    ax2 = ax1.twinx()
    ax2.plot(epochs_x, history["val_dice"], label="val_dice", color="#55A868", linestyle="--")
    ax2.set_ylabel("val Dice")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    plt.title("Training curve, T2 missing (seed 0)")
    plt.tight_layout()
    plt.savefig(fig2_path, dpi=150)
    plt.close()
    figures.append({
        "name": "loss_curve",
        "renders": ["val_loss", "train_loss"],
        "inline_data": fig2_inline,
    })

    metrics = {}
    for entry in results:
        if isinstance(entry["value"], (int, float)):
            metrics[entry["name"]] = entry["value"]

    manifest = {
        "manifest_version": 1,
        "config": {
            **cfg,
            "seeds": [0, 1, 2][:n_seeds],
            "n_seeds": n_seeds,
            "dataset": "Spirit-26/BraTS-2024-Complete (BraTS-GLI/train, 80-subject subset)",
            "resolution": "96x96x96 (resampled from native 182x218x182)",
            "architecture": "FusionUNet3D: 4 independent per-modality encoders, "
                             "channel-concat fusion at every scale, shared decoder",
            "missing_modality": cfg["missing_modality"],
            "imputation_method": "Normalized Mean Algorithm (deterministic, no training) "
                                  "applied to per-scale encoder feature maps",
        },
        "results": results,
        "baselines": baselines,
        "figures": figures,
        "metrics": metrics,
        # schema-declared flat keys (TASK.yaml deliverables.results_json_schema)
        "mean_dice_score": mean_dice_score,
        "wandb_run_url": wandb_run_url,
        "github_commit_sha": None,
    }

    out_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {out_path}")
    print(f"mean_dice_score={mean_dice_score:.4f} (n_seeds={n_seeds}, ci={ci})")
    print(f"inference_latency_ms={inference_latency_ms:.3f}")
    print(f"peak_inference_memory_mb={peak_inference_memory_mb:.2f}")
    print(f"wandb_run_url={wandb_run_url}")


if __name__ == "__main__":
    main()
