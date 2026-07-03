# baseline_umm_gan — pixel-level generative synthesis target pipeline

Reproduces the "heavy generative synthesis" pipeline this study's hypothesis
is testing against: synthesize the missing T1ce modality's *pixels* with a
GAN (UMM-GAN reimplementation, Zhang et al. 2023, arXiv:2304.05340), then
train a standard segmentation network on the completed 4-modality volume.
This establishes the target accuracy and the inference compute cost that
`main_latent_imputation`'s deterministic latent-space imputation must match
within a 2% non-inferiority margin while avoiding.

## Architecture

Two networks, matching the methodology review's shared-architecture
requirement across siblings (`baseline_zero_fill`, `baseline_umm_gan`,
`main_latent_imputation`):

1. **`models/gan.py` — `UMMGenerator3D` + `PatchDiscriminator3D`.** A 3D
   pix2pix-style encoder-decoder generator maps the 3 available modalities
   (T1n, T2w, T2f) to the missing T1ce volume; a PatchGAN discriminator
   provides the adversarial signal. Trained with L1 + gradient-difference +
   LSGAN adversarial loss. *Scoping note:* the original paper's generator is
   "unified" across arbitrary missing-modality combinations via a mask input;
   we fix the direction to T1n+T2w+T2f→T1ce since that is this experiment's
   specific role — see `models/gan.py` docstring.
2. **`models/unet.py` — `MultiEncoderUNet3D`.** Identical file/class to the
   `baseline_zero_fill` sibling: one encoder branch per modality, fused by
   channel-concat + 1x1x1 conv at every scale, one shared decoder. Only the
   T1c branch's *input* differs across siblings — here it receives
   GAN-synthesized pixels instead of zeros.

## Pipeline (`train.py`, single entry point)

1. Preprocess/cache BraTS volumes to 96³ isotropic, z-scored (`utils/preprocess.py`,
   reused verbatim from `baseline_zero_fill` — cache is shared and was already
   populated on this pod).
2. Train the GAN generator+discriminator on the train split only.
3. Run the trained generator over **all** subjects (train/val/test) to
   produce GAN-synthesized T1c, cached to `--gan_cache_dir`.
4. Train `MultiEncoderUNet3D` on the GAN-imputed 4-modality volumes — same
   hyperparameters (`base_ch=24`, `lr=1e-4`, `epochs=120`, `batch=4`,
   `patience=25`, `max_train_minutes=150`, AdamW+cosine schedule, Dice+BCE
   loss) as `baseline_zero_fill`, for a fair, controlled comparison.
5. Evaluate test-set Dice (WT/TC/ET + mean).
6. Benchmark end-to-end (GAN synthesis + U-Net) inference latency and peak
   GPU memory for a single forward pass.

## Hardware

1x NVIDIA A100-40GB (or any ≥16GB CUDA GPU; reduce `--batch_size` on smaller
cards). CUDA 12.8, PyTorch 2.11.

## Install

```bash
pip install -r requirements.txt
```

## Data

BraTS-2024-GLI (glioma, labeled `train` split), pinned dataset per
`TASK.yaml.datasets`: `Spirit-26/BraTS-2024-Complete` on HuggingFace. Uses
the same 80-subject cached subset as `baseline_zero_fill` at
`/workspace/datasets/BraTS-2024-subset/hf_snapshot/BraTS-GLI/train/` (see
that directory's `README.md` for subject-selection provenance) and the
**same train/val/test split** (seed 42, deterministic shuffle of the sorted
subject list: 60 train / 8 val / 12 test) — reproduced by re-running
`split_subjects(sorted(os.listdir(data_root)), seed=42, n_val=8, n_test=12)`,
identical to the sibling's `split.json`, for an apples-to-apples comparison.

## Train

```bash
python train.py \
  --data_root /workspace/datasets/BraTS-2024-subset/hf_snapshot/BraTS-GLI/train \
  --cache_dir /workspace/cache/brats96 \
  --gan_cache_dir /workspace/cache/brats96_gan_t1c \
  --code_dir . --train_log ../train.log \
  --gan_epochs 150 --patience_gan 20 --max_train_minutes_gan 90 \
  --unet_epochs 120 --patience_unet 25 --max_train_minutes_unet 150 \
  --batch_size 4 --seed 42 --n_val 8 --n_test 12
```

Checkpoints: `weights_gan/{last,best}.pt` (GAN generator+discriminator) and
`weights/{last,best}.pt` (U-Net). All are complete checkpoints (model(s),
optimizer(s), RNG states, epoch, best_metric, patience_counter, history) —
resumable by re-running the same command (`train.py` auto-resumes each
stage from its `last.pt` if present).

## Eval (re-evaluate saved checkpoints + reproduce the latency/memory benchmark)

```bash
python eval.py \
  --data_root /workspace/datasets/BraTS-2024-subset/hf_snapshot/BraTS-GLI/train \
  --cache_dir /workspace/cache/brats96 --gan_cache_dir /workspace/cache/brats96_gan_t1c \
  --code_dir . --unet_checkpoint weights/best.pt --gan_checkpoint weights_gan/best.pt
```

## Seed

`--seed 42` controls the train/val/test split shuffle and all RNG state
(python/numpy/torch/cuda), matching `baseline_zero_fill`.

## Metrics

- `mean_dice_score`: mean of Dice(WT), Dice(TC), Dice(ET) on the held-out
  test split (3 canonical overlapping BraTS regions, derived defensively
  from raw labels — see `utils/preprocess.py:_region_masks`).
- `inference_latency_ms`: wall-clock time (CUDA-synchronized, 5 warm-up +
  20 timed runs, batch=1) for the full GAN-synthesis-then-U-Net forward pass.
- `peak_inference_memory_mb`: `torch.cuda.max_memory_allocated()` for one
  such forward pass.
