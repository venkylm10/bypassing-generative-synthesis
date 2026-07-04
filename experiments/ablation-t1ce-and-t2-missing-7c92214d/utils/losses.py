import torch
import torch.nn.functional as F

N_CLASSES = 4


def dice_per_class(logits, target, eps=1e-6):
    """Returns per-class Dice (tensor of shape (n_classes,)), computed over the batch."""
    probs = F.softmax(logits, dim=1)
    target_onehot = F.one_hot(target, N_CLASSES).permute(0, 4, 1, 2, 3).float()
    dims = (0, 2, 3, 4)
    intersection = (probs * target_onehot).sum(dims)
    union = probs.sum(dims) + target_onehot.sum(dims)
    return (2 * intersection + eps) / (union + eps)


def dice_loss(logits, target):
    dice = dice_per_class(logits, target)
    return 1.0 - dice.mean()


def ce_dice_loss(logits, target):
    ce = F.cross_entropy(logits, target)
    return ce + dice_loss(logits, target)
