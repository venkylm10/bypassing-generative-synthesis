"""Dice + BCE combo loss and Dice metric for multi-label (overlapping WT/TC/ET) segmentation."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-channel soft Dice, averaged over channels and batch. logits: raw (pre-sigmoid)."""
    probs = torch.sigmoid(logits)
    dims = (0, 2, 3, 4)
    intersection = (probs * targets).sum(dim=dims)
    union = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2 * intersection + eps) / (union + eps)
    return dice  # (C,)


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        dice = soft_dice(logits, targets)
        dice_loss = 1.0 - dice.mean()
        return self.bce_weight * bce + self.dice_weight * dice_loss


@torch.no_grad()
def hard_dice(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    """Per-channel hard Dice (thresholded predictions) — used for reported metrics."""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    dims = (0, 2, 3, 4)
    intersection = (preds * targets).sum(dim=dims)
    union = preds.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2 * intersection + eps) / (union + eps)
    return dice  # (C,) order matches regions: [WT, TC, ET]
