# ablation_t1ce_and_t2_missing

Ablation of `main_latent_imputation`: deterministic latent-space imputation
(Normalized Mean Algorithm, NMA) for **two simultaneously missing** MRI
modalities — T1ce (`t1c`) AND T2 (`t2w`) — in brain tumor segmentation.
Tests robustness of the latent-imputation approach to severe multi-modality
dropout, where only T1 (`t1n`) and FLAIR (`t2f`) remain available.

## Hardware
- 1x NVIDIA A100-40GB (or any CUDA GPU with >=8GB free); CUDA 12.8, PyTorch 2.11.0+cu128.
- Each of the 3 seed runs (150 epochs, 60 train subjects, 96^3 volumes,
  batch size 4) takes ~15-25 minutes on an A100 (slightly faster than the
  single-missing sibling runs since only 2 of 4 encoder branches are real).

## Install
```bash
pip install -r requirements.txt
```

## Data
Pinned dataset: `Spirit-26/BraTS-2024-Complete` (HuggingFace), BraTS-GLI/train
cohort. Reuses the 80-subject subset and its 96^3 preprocessed cache already
produced by the parent experiment at
`/workspace/datasets/BraTS-2024-subset/derived/preprocessed_96/` (resample to
96^3, per-volume z-score normalization, label remap {0,1,2,3,4}->{0,1,2,3}
merging raw label 3 into 1). No re-preprocessing was needed. If the cache is
absent, regenerate with:
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
(`"missing_modality": ["t1c", "t2w"]` — a list, since both are dropped).

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
into skip connections for a shared decoder. `forward(volumes, missing=...)`
now accepts `missing` as `None`, a single modality name, or a **list/set of
names**. When one or more modalities are "missing" (here: `t1c` and `t2w`),
their contribution at every scale is NOT a real encoder output — it is
`normalized_mean_impute()` computed over whatever modalities remain
available (here: `t1n`, `t2f`): each available feature map is
instance-normalized (per-sample, per-channel zero-mean/unit-std over spatial
dims) and the normalized maps are averaged. The SAME imputed value is
reinserted into every missing branch. This is deterministic and requires no
training or ground-truth access to the missing modalities.

## Checkpoints
`weights_seed0/`, `weights_seed1/`, `weights_seed2/` (~200MB each) are
produced by training but are excluded from the GitHub mirror of this
directory (binary checkpoints don't belong in git history). The primary
(seed 0) checkpoint is delivered separately at
`/workspace/output/latent_imputed_t1ce_t2_missing_weights/{best,last}.pt`.

## Seed / reproducibility
- Seeds: 0, 1, 2 (a reduced-from-5 sweep, matching sibling ablations; see
  report.md self-critique).
- Missing modalities: T1ce (`t1c`) AND T2 (`t2w`) simultaneously, per
  `TASK.yaml.objective` for this experiment.
- Splits are re-derived deterministically per seed inside `train.py`
  (`utils/dataset.make_splits`), not hardcoded — same seed reproduces the
  same 60/10/10 split.
