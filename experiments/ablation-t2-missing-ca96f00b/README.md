# ablation_t2_missing

Ablation of `main_latent_imputation`: same deterministic latent-space
imputation (Normalized Mean Algorithm, NMA) and the same multi-encoder
late-fusion 3D U-Net, but the **missing modality is changed to T2 (t2w)**
instead of T1ce (t1c), to evaluate the latent-imputation approach's
robustness under a distinct missing-modality condition. This experiment
reports absolute segmentation performance only — it does not make a
non-inferiority claim against GAN/zero-fill baselines (those were not
re-run for the T2-missing condition; see report.md self-critique).

## Hardware
- 1x NVIDIA A100-40GB (or any CUDA GPU with >=8GB free); CUDA 12.8, PyTorch 2.11.0+cu128.
- Each of the 3 seed runs (150 epochs, 60 train subjects, 96^3 volumes,
  batch size 4) takes ~15-25 minutes on an A100.

## Install
```bash
pip install -r requirements.txt
```

## Data
Pinned dataset: `Spirit-26/BraTS-2024-Complete` (HuggingFace), BraTS-GLI/train
cohort. This experiment reuses the same 80-subject subset and preprocessed
cache produced by the parent experiment `main_latent_imputation`, already
present at `/workspace/datasets/BraTS-2024-subset/derived/preprocessed_96/`
(resampled to 96^3, per-volume z-score normalization, label remap
{0,1,2,3,4}->{0,1,2,3} merging raw label 3 into 1). Regenerate via:
```bash
python utils/preprocess.py
```

## Train
Trains 3 random seeds (0, 1, 2) sequentially, each with a fixed 60/10/10
train/val/test subject split derived from `(seed)`:
```bash
bash run_all_seeds.sh
# or a single seed:
python train.py --seed 0 --tag seed0
```
Checkpoints: `weights_<tag>/last.pt` (every epoch, full resumable state) and
`weights_<tag>/best.pt` (best val Dice). Config: `configs/config.json`
(`missing_modality: "t2w"`).

## Evaluate
```bash
python eval.py --tag seed0
```
Writes `../eval_<tag>.json` with mean Dice (foreground classes, excludes
background), per-class Dice, per-subject Dice, inference latency (ms,
single-subject forward pass, 30 reps after 5 warmup), and peak inference
memory (MB).

## Model
`models/unet3d.py`: `FusionUNet3D` — 4 independent per-modality 3D encoders
(t1n, t1c, t2w, t2f), fused via channel-concat + 1x1x1 conv at every scale
into skip connections for a shared decoder. When a modality is "missing"
(here: **t2w**), its contribution at every scale is NOT a real encoder
output — it is `normalized_mean_impute()`: each available modality's feature
map is instance-normalized (per-sample, per-channel zero-mean/unit-std over
spatial dims) and the normalized maps are averaged. This is deterministic
and requires no training or ground-truth access to the missing modality.

## Checkpoints
`weights_seed0/`, `weights_seed1/`, `weights_seed2/` (~200MB each) are
produced by training but are excluded from the GitHub mirror of this
directory (binary checkpoints don't belong in git history). The primary
(seed 0) checkpoint is delivered separately at
`/workspace/output/latent_imputed_t2_missing_weights/{best,last}.pt`.

## Seed / reproducibility
- Seeds: 0, 1, 2 (matches the reduced-from-5 sweep used by
  `main_latent_imputation`; see report.md self-critique).
- Missing modality: **T2 (`t2w`)**, per this experiment's role
  (`TASK.yaml.context.plan.this_experiment`).
- Splits are re-derived deterministically per seed inside `train.py`
  (`utils/dataset.make_splits`), not hardcoded — same seed reproduces the
  same 60/10/10 split.
