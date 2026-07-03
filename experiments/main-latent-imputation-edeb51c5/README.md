# main_latent_imputation

Deterministic latent-space imputation for a missing MRI modality (T1ce) in
brain tumor segmentation — testing whether a Normalized Mean Algorithm (NMA)
applied to a 3D U-Net's per-modality latent features can replace pixel-level
GAN synthesis (UMM-GAN) without losing segmentation accuracy.

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
cohort. This experiment uses an 80-subject subset already downloaded to
`/workspace/datasets/BraTS-2024-subset/hf_snapshot/`. Preprocessing (resample
to 96^3, per-volume z-score normalization, label remap {0,1,2,3,4}->
{0,1,2,3} merging raw label 3 into 1) is cached at
`/workspace/datasets/BraTS-2024-subset/derived/preprocessed_96/` by:
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
`weights_<tag>/best.pt` (best val Dice). Config: `configs/config.json`.

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
(here: t1c), its contribution at every scale is NOT a real encoder output —
it is `normalized_mean_impute()`: each available modality's feature map is
instance-normalized (per-sample, per-channel zero-mean/unit-std over spatial
dims) and the normalized maps are averaged. This is deterministic and
requires no training or ground-truth access to the missing modality.

## Checkpoints
`weights_seed0/`, `weights_seed1/`, `weights_seed2/` (~200MB each) are
produced by training but are excluded from the GitHub mirror of this
directory (binary checkpoints don't belong in git history). The primary
(seed 0) checkpoint is delivered separately at
`/workspace/output/latent_imputed_unet_weights/{best,last}.pt`.

## Seed / reproducibility
- Seeds: 0, 1, 2 (a reduced-from-5 sweep; see report.md self-critique).
- Missing modality: T1ce (`t1c`), per `TASK.yaml.context.method`.
- Splits are re-derived deterministically per seed inside `train.py`
  (`utils/dataset.make_splits`), not hardcoded — same seed reproduces the
  same 60/10/10 split.
