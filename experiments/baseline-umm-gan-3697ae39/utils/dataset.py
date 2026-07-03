"""BraTS datasets for the baseline_umm_gan experiment.

Two datasets share the same preprocessed cache (utils.preprocess, MODALITIES
order ["t1n", "t1c", "t2w", "t2f"], MISSING_MODALITY_INDEX=1 for t1c):

- BraTSGANDataset: for training the UMM-GAN generator/discriminator.
  condition = (t1n, t2w, t2f), target = real t1c (both z-scored, target
  additionally clamped+rescaled to [-1, 1] to match the generator's tanh head).
- BraTSGANImputedDataset: for training the downstream segmentation U-Net.
  Identical to the sibling baseline_zero_fill's dataset except index 1 (t1c)
  is replaced with the GAN-synthesized volume (loaded from a second cache
  populated by impute.py) instead of zeros.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

MISSING_MODALITY_INDEX = 1  # t1c, per preprocess.MODALITIES order
TARGET_CLAMP = 3.0  # z-score std clamp before rescaling to [-1, 1] for tanh


def split_subjects(subjects: list[str], seed: int, n_val: int, n_test: int):
    rng = random.Random(seed)
    shuffled = subjects[:]
    rng.shuffle(shuffled)
    test = shuffled[:n_test]
    val = shuffled[n_test:n_test + n_val]
    train = shuffled[n_test + n_val:]
    return train, val, test


def _to_tanh_range(x: np.ndarray) -> np.ndarray:
    return np.clip(x, -TARGET_CLAMP, TARGET_CLAMP) / TARGET_CLAMP


def _from_tanh_range(x: np.ndarray) -> np.ndarray:
    return x * TARGET_CLAMP


class BraTSGANDataset(Dataset):
    """condition=(t1n,t2w,t2f) -> target=t1c, for GAN training."""

    def __init__(self, cache_dir: str, subject_ids: list[str], train: bool = False):
        self.cache_dir = cache_dir
        self.subject_ids = subject_ids
        self.train = train

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, idx: int):
        sid = self.subject_ids[idx]
        npz = np.load(os.path.join(self.cache_dir, f"{sid}.npz"))
        images = npz["images"].copy()  # (4,S,S,S): t1n,t1c,t2w,t2f

        condition = np.stack([images[0], images[2], images[3]], axis=0)  # t1n,t2w,t2f
        target = _to_tanh_range(images[MISSING_MODALITY_INDEX:MISSING_MODALITY_INDEX + 1])

        if self.train:
            for axis in (1, 2, 3):
                if random.random() < 0.5:
                    condition = np.flip(condition, axis=axis).copy()
                    target = np.flip(target, axis=axis).copy()

        return torch.from_numpy(condition), torch.from_numpy(target), sid


class BraTSGANImputedDataset(Dataset):
    """Same 4-modality tensor as the zero-fill sibling, but t1c (index 1) is
    read from a second cache of GAN-synthesized volumes instead of zero-filled.
    """

    def __init__(self, cache_dir: str, gan_cache_dir: str, subject_ids: list[str], train: bool = False):
        self.cache_dir = cache_dir
        self.gan_cache_dir = gan_cache_dir
        self.subject_ids = subject_ids
        self.train = train

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, idx: int):
        sid = self.subject_ids[idx]
        npz = np.load(os.path.join(self.cache_dir, f"{sid}.npz"))
        images = npz["images"].copy()  # (4,S,S,S) float32
        regions = npz["regions"].astype(np.float32)  # (3,S,S,S)

        gan_npz = np.load(os.path.join(self.gan_cache_dir, f"{sid}.npz"))
        images[MISSING_MODALITY_INDEX] = gan_npz["t1c_synth"]

        if self.train:
            for axis in (1, 2, 3):
                if random.random() < 0.5:
                    images = np.flip(images, axis=axis).copy()
                    regions = np.flip(regions, axis=axis).copy()

        images_t = torch.from_numpy(images.copy())
        regions_t = torch.from_numpy(regions.copy())
        return images_t, regions_t, sid
