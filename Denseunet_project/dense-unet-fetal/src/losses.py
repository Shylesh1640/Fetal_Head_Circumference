"""
Loss functions
===============
Hybrid Dice + BCE loss, matching the loss used in the reference paper
(L_hybrid = L_BCE + (1 - Dice)), implemented for raw logits (numerically
stable: uses BCEWithLogitsLoss instead of separately applying sigmoid then
binary_cross_entropy).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """Differentiable soft Dice loss computed on probabilities (post-sigmoid)."""

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
    """L = bce_weight * BCE(logits, targets) + dice_weight * (1 - Dice)."""

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, smooth: float = 1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss(smooth=smooth)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
