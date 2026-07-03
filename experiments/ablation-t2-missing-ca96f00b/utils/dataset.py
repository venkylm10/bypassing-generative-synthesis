import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

PREPROC_ROOT = "/workspace/datasets/BraTS-2024-subset/derived/preprocessed_96"
MODALITIES = ["t1n", "t1c", "t2w", "t2f"]


def load_manifest():
    with open(os.path.join(PREPROC_ROOT, "manifest.json")) as f:
        return json.load(f)


def make_splits(seed=0, n_train=60, n_val=10, n_test=10):
    manifest = load_manifest()
    subjects = sorted(manifest["subjects"])
    assert len(subjects) == n_train + n_val + n_test, len(subjects)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(subjects))
    idx_train = perm[:n_train]
    idx_val = perm[n_train:n_train + n_val]
    idx_test = perm[n_train + n_val:]
    return (
        [subjects[i] for i in idx_train],
        [subjects[i] for i in idx_val],
        [subjects[i] for i in idx_test],
    )


class BraTSSubset(Dataset):
    def __init__(self, subject_ids):
        self.subject_ids = subject_ids

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx):
        sid = self.subject_ids[idx]
        npz = np.load(os.path.join(PREPROC_ROOT, f"{sid}.npz"))
        image = npz["image"]  # (4, D, H, W)
        label = npz["label"].astype(np.int64)  # (D, H, W)
        volumes = {mod: torch.from_numpy(image[i:i + 1]) for i, mod in enumerate(MODALITIES)}
        return volumes, torch.from_numpy(label), sid
