"""BraTS dataset for the zero-fill missing-modality baseline.

The T1ce (index 1, see MODALITIES order in preprocess.py) modality is always
replaced with zeros before it reaches its encoder branch — this is the
"no-imputation lower bound" this experiment establishes. All other modalities
(T1n, T2w, T2f) are passed through normally.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

MISSING_MODALITY_INDEX = 1  # t1c, per preprocess.MODALITIES order


class BraTSZeroFillDataset(Dataset):
    def __init__(self, cache_dir: str, subject_ids: list[str], train: bool = False):
        self.cache_dir = cache_dir
        self.subject_ids = subject_ids
        self.train = train

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, idx: int):
        sid = self.subject_ids[idx]
        npz = np.load(os.path.join(self.cache_dir, f"{sid}.npz"))
        images = npz["images"].copy()  # (4, S, S, S) float32
        regions = npz["regions"].astype(np.float32)  # (3, S, S, S)

        images[MISSING_MODALITY_INDEX] = 0.0  # zero-fill the missing modality

        if self.train:
            for axis in (1, 2, 3):
                if random.random() < 0.5:
                    images = np.flip(images, axis=axis).copy()
                    regions = np.flip(regions, axis=axis).copy()

        images_t = torch.from_numpy(images)  # (4,S,S,S)
        regions_t = torch.from_numpy(regions)  # (3,S,S,S)
        return images_t, regions_t, sid


def split_subjects(subjects: list[str], seed: int, n_val: int, n_test: int):
    rng = random.Random(seed)
    shuffled = subjects[:]
    rng.shuffle(shuffled)
    test = shuffled[:n_test]
    val = shuffled[n_test:n_test + n_val]
    train = shuffled[n_test + n_val:]
    return train, val, test
