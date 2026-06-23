"""
Loss functions.

Hybrid Dice-BCE loss, as used in the reference paper, for training stability
under the class imbalance typical of fetal-head ultrasound masks (head region
is a small fraction of total pixels).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    """
    L_hybrid = BCE(logits, target) + (1 - Dice(probs, target))

    Operates directly on raw logits (numerically stable BCE-with-logits),
    converts to probabilities internally for the Dice term.
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()

        bce = F.binary_cross_entropy_with_logits(logits, targets)

        probs = torch.sigmoid(logits)
        probs_flat = probs.reshape(probs.size(0), -1)
        targets_flat = targets.reshape(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        dice_coeff = (2.0 * intersection + self.smooth) / (
            probs_flat.sum(dim=1) + targets_flat.sum(dim=1) + self.smooth
        )
        dice_loss = 1.0 - dice_coeff.mean()

        return bce + dice_loss
