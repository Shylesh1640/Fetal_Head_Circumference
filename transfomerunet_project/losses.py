"""
losses.py
Hybrid Dice + BCE loss for binary segmentation under class imbalance,
as described in the paper. Operates on raw logits (numerically stable).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class DiceLoss(nn.Module):
    """Soft Dice loss computed from logits via sigmoid."""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.contiguous().view(probs.size(0), -1)
        targets = targets.contiguous().view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class HybridDiceBCELoss(nn.Module):
    """
    L_hybrid = bce_weight * BCEWithLogits + dice_weight * DiceLoss

    Using BCEWithLogitsLoss (not plain BCELoss on sigmoid outputs) for
    numerical stability -- this is the standard, bug-safe combination and
    avoids log(0) issues that plain BCE-on-probabilities can hit.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def build_loss() -> HybridDiceBCELoss:
    return HybridDiceBCELoss(bce_weight=config.BCE_WEIGHT, dice_weight=config.DICE_WEIGHT)
