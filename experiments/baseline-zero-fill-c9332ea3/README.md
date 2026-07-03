# baseline_zero_fill — no-imputation lower bound

Trains a multi-encoder, late-fusion 3D U-Net for BraTS brain-tumor
segmentation where the missing modality (T1ce) is **zero-filled** at input
(no imputation of any kind). This establishes the lower-bound performance
that pixel-level GAN synthesis (UMM-GAN) or latent-space imputation must beat
to justify their added compute.

Architecture: one encoder branch per modality (T1n, T1c, T2w, T2f), features
fused by channel-concat + 1x1x1 conv at every scale, one shared decoder. This
shape is shared across the sibling experiments in this study
(`baseline_umm_gan`, `main_latent_imputation`) so only the imputation method
varies — see `models/unet.py` docstring for the full rationale. For this
baseline, the T1ce input volume is replaced with zeros before it reaches its
encoder branch (`utils/dataset.py:MISSING_MODALITY_INDEX`).

## Hardware
- 1x NVIDIA A100-40GB (or any >=16GB CUDA GPU; batch size may need reducing
  on smaller cards)
- CUDA 12.8, PyTorch 2.11

## Install
```bash
pip install -r requirements.txt
```

## Data
BraTS-2024-GLI (glioma, labeled `train` split), pinned dataset per
`TASK.yaml.datasets`: `Spirit-26/BraTS-2024-Complete` on HuggingFace. This run
used an 80-subject cached subset at
`/workspace/datasets/BraTS-2024-subset/hf_snapshot/BraTS-GLI/train/` (see that
directory's `README.md` for exact subject selection and provenance). Volumes
are resized to 96^3 isotropic and z-score normalized (within brain mask) as a
one-time preprocessing/caching step (`utils/preprocess.py`).

Split (seed 42, deterministic shuffle): 60 train / 8 val / 12 test subjects.
Exact subject IDs are written to `split.json` by `train.py`.

## Train
```bash
python train.py \
  --data_root /workspace/datasets/BraTS-2024-subset/hf_snapshot/BraTS-GLI/train \
  --cache_dir /workspace/cache/brats96 \
  --code_dir . \
  --train_log ../train.log \
  --epochs 120 --batch_size 4 --lr 1e-4 --base_ch 24 --seed 42 \
  --n_val 8 --n_test 12 --patience 25 --max_train_minutes 150
```
Checkpoints: `weights/last.pt` (every epoch) and `weights/best.pt` (on val
mean-Dice improvement). Both are complete checkpoints (model, optimizer,
scheduler, RNG states, epoch, best_metric, patience_counter, history) —
resumable by re-running the same command (`train.py` auto-resumes from
`weights/last.pt` if present).

## Eval (re-evaluate a saved checkpoint on the held-out test split)
```bash
python eval.py \
  --data_root /workspace/datasets/BraTS-2024-subset/hf_snapshot/BraTS-GLI/train \
  --cache_dir /workspace/cache/brats96 \
  --code_dir . --checkpoint weights/best.pt
```

## Seed
`--seed 42` controls the train/val/test split shuffle and all RNG state
(python/numpy/torch/cuda).

## Metric
Mean Dice score = mean of Dice(WT), Dice(TC), Dice(ET) — the 3 canonical
overlapping BraTS regions (Whole Tumor / Tumor Core / Enhancing Tumor),
derived from the raw segmentation labels defensively (robust to the label-1
vs label-3 "necrotic core" convention drift observed across subjects in this
dataset dump — see `utils/preprocess.py:_region_masks`).
