"""One-time BraTS preprocessing: resize to a fixed isotropic resolution, z-score
normalize each modality within the brain mask, and derive the 3 canonical BraTS
region masks (WT/TC/ET) from the raw segmentation labels.

Caches one .npz per subject so training doesn't repeat expensive I/O/resize
every epoch. Cache lives outside /workspace/output (ephemeral scratch) since it
is a derived intermediate, not a deliverable.
"""
from __future__ import annotations

import os
import sys
import time

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

MODALITIES = ["t1n", "t1c", "t2w", "t2f"]  # fixed encoder-branch order
RESIZE = 96  # isotropic target resolution


def _load_vol(path: str) -> np.ndarray:
    return np.asarray(nib.load(path).get_fdata(dtype=np.float32))


def _resize(vol: np.ndarray, size: int, mode: str, device: str) -> np.ndarray:
    t = torch.from_numpy(vol).to(device)[None, None]  # (1,1,D,H,W)
    if mode == "nearest":
        t = F.interpolate(t, size=(size, size, size), mode="nearest")
    else:
        t = F.interpolate(t, size=(size, size, size), mode="trilinear", align_corners=False)
    return t[0, 0].cpu().numpy()


def _zscore(vol: np.ndarray) -> np.ndarray:
    mask = vol > 0
    if mask.sum() < 10:
        return vol
    mean = vol[mask].mean()
    std = vol[mask].std() + 1e-6
    out = (vol - mean) / std
    out[~mask] = 0.0
    return out.astype(np.float32)


def _region_masks(seg: np.ndarray) -> np.ndarray:
    """Map raw multi-class labels to the 3 canonical overlapping BraTS regions.

    Robust to label-convention drift observed in this dataset dump (some
    subjects use {0,2,4}, others {0,2,3,4}, and label 1 appears in the full
    80-subject union) — WT/TC/ET are built from label *sets*, not a single
    fixed integer per region.
      WT (whole tumor)      = any tumor label > 0
      TC (tumor core)       = necrotic/non-enhancing (1 or 3) + enhancing (4)
      ET (enhancing tumor)  = label 4
    """
    seg_int = np.rint(seg).astype(np.int16)
    wt = (seg_int > 0).astype(np.float32)
    tc = np.isin(seg_int, [1, 3, 4]).astype(np.float32)
    et = (seg_int == 4).astype(np.float32)
    return np.stack([wt, tc, et], axis=0)


def preprocess_subject(subject_dir: str, subject_id: str, device: str) -> dict:
    vols = {}
    for m in MODALITIES:
        p = os.path.join(subject_dir, f"{subject_id}-{m}.nii.gz")
        raw = _load_vol(p)
        resized = _resize(raw, RESIZE, "trilinear", device)
        vols[m] = _zscore(resized)

    seg_p = os.path.join(subject_dir, f"{subject_id}-seg.nii.gz")
    seg_raw = _load_vol(seg_p)
    seg_resized = _resize(seg_raw, RESIZE, "nearest", device)
    regions = _region_masks(seg_resized)

    images = np.stack([vols[m] for m in MODALITIES], axis=0).astype(np.float32)  # (4,S,S,S)
    return {"images": images, "regions": regions.astype(np.uint8)}


def run(data_root: str, cache_dir: str, subjects: list[str], device: str = "cuda") -> None:
    os.makedirs(cache_dir, exist_ok=True)
    t0 = time.time()
    for i, sid in enumerate(subjects):
        out_path = os.path.join(cache_dir, f"{sid}.npz")
        if os.path.exists(out_path):
            continue
        subject_dir = os.path.join(data_root, sid)
        data = preprocess_subject(subject_dir, sid, device)
        np.savez_compressed(out_path, images=data["images"], regions=data["regions"])
        if (i + 1) % 10 == 0 or i == len(subjects) - 1:
            elapsed = time.time() - t0
            print(f"[preprocess] {i+1}/{len(subjects)} subjects cached, elapsed={elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    data_root = sys.argv[1]
    cache_dir = sys.argv[2]
    subjects = sorted(os.listdir(data_root))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run(data_root, cache_dir, subjects, device)
