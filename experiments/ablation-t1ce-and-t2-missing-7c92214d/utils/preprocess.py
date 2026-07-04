"""Resample BraTS-2024 subset to a fixed resolution and cache as .npy.

Run once; caches land in /workspace/datasets/BraTS-2024-subset/derived/preprocessed_96/
so re-runs (and sibling experiments) can reuse them without re-touching raw data.
"""
import os
import sys
import json
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

RAW_ROOT = "/workspace/datasets/BraTS-2024-subset/hf_snapshot/BraTS-GLI/train"
OUT_ROOT = "/workspace/datasets/BraTS-2024-subset/derived/preprocessed_96"
TARGET_SHAPE = (96, 96, 96)
MODALITIES = ["t1n", "t1c", "t2w", "t2f"]

# Merge label 3 (rare alt convention for necrotic core) into label 1.
# Result: 0=background, 1=necrotic/non-enhancing core, 2=edema, 3=enhancing tumor (orig 4)
LABEL_REMAP = {0: 0, 1: 1, 2: 2, 3: 1, 4: 3}


def load_volume(path):
    img = nib.load(path)
    return np.asarray(img.get_fdata(), dtype=np.float32)


def resample(vol, target_shape, order):
    factors = [t / s for t, s in zip(target_shape, vol.shape)]
    return zoom(vol, factors, order=order)


def normalize_intensity(vol):
    mask = vol > 0
    if mask.sum() == 0:
        return vol
    mean = vol[mask].mean()
    std = vol[mask].std() + 1e-8
    out = (vol - mean) / std
    out[~mask] = 0.0
    return out.astype(np.float32)


def remap_labels(seg):
    out = np.zeros_like(seg, dtype=np.uint8)
    for src, dst in LABEL_REMAP.items():
        out[seg == src] = dst
    return out


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    subjects = sorted(os.listdir(RAW_ROOT))
    manifest = []
    for i, subj in enumerate(subjects):
        subj_dir = os.path.join(RAW_ROOT, subj)
        out_path = os.path.join(OUT_ROOT, f"{subj}.npz")
        if os.path.exists(out_path):
            manifest.append(subj)
            print(f"[{i+1}/{len(subjects)}] {subj} cached, skip", flush=True)
            continue
        modal_vols = []
        for m in MODALITIES:
            fp = os.path.join(subj_dir, f"{subj}-{m}.nii.gz")
            vol = load_volume(fp)
            vol = resample(vol, TARGET_SHAPE, order=1)
            vol = normalize_intensity(vol)
            modal_vols.append(vol)
        image = np.stack(modal_vols, axis=0).astype(np.float32)  # (4, D, H, W)

        seg_fp = os.path.join(subj_dir, f"{subj}-seg.nii.gz")
        seg = load_volume(seg_fp)
        seg = resample(seg, TARGET_SHAPE, order=0)
        seg = remap_labels(np.round(seg).astype(np.int32))

        np.savez_compressed(out_path, image=image, label=seg)
        manifest.append(subj)
        print(f"[{i+1}/{len(subjects)}] {subj} done, image={image.shape} label={seg.shape} classes={np.unique(seg)}", flush=True)

    with open(os.path.join(OUT_ROOT, "manifest.json"), "w") as f:
        json.dump({"subjects": manifest, "modalities": MODALITIES,
                    "target_shape": TARGET_SHAPE, "label_remap": LABEL_REMAP}, f, indent=2)
    print(f"Preprocessing complete: {len(manifest)} subjects -> {OUT_ROOT}")


if __name__ == "__main__":
    main()
