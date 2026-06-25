import torch
import torch.nn as nn


class DiceBCELoss(nn.Module):
    """L_hybrid = L_BCE + (1 - Dice), same formulation as the paper (Eq. 1)."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.bce = nn.BCELoss()
        self.eps = eps

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        intersection = (pred_flat * target_flat).sum(dim=1)
        dice = (2 * intersection + self.eps) / (
            pred_flat.sum(dim=1) + target_flat.sum(dim=1) + self.eps
        )
        dice_loss = 1 - dice.mean()
        return bce_loss + dice_loss
