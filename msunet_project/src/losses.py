"""
losses.py
Hybrid Dice + Binary Cross-Entropy loss, exactly the loss family described
in the paper for handling class imbalance between the (small) fetal-head
foreground and the (large) background in ultrasound images.

The model outputs raw logits (no activation); BCEWithLogitsLoss is used for
numerical stability and Dice is computed on sigmoid(logits).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = probs.contiguous().view(probs.size(0), -1)
        targets = targets.contiguous().view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice_score.mean()


class HybridDiceBCELoss(nn.Module):
    """L = bce_weight * BCEWithLogits + dice_weight * DiceLoss"""

    def __init__(self, bce_weight: float = cfg.BCE_WEIGHT, dice_weight: float = cfg.DICE_WEIGHT):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
