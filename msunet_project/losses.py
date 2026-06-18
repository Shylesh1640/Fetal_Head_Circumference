"""
losses.py
---------
Implements the hybrid loss function described in Section III-C of the
paper (Equation 1):

    L_Hybrid = L_BCE + (1 - Dice)

where Dice is the soft Dice coefficient computed on sigmoid-activated
predictions vs. the ground-truth binary mask, and L_BCE is the standard
binary cross-entropy on logits (using BCEWithLogitsLoss for numerical
stability instead of plain BCE on probabilities).
"""

import torch
import torch.nn as nn


def dice_coefficient(probs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Soft Dice coefficient, averaged over the batch.

    probs, targets : [B, 1, H, W], probs are post-sigmoid in [0, 1],
                       targets are binary {0, 1}.
    """
    probs = probs.contiguous().view(probs.size(0), -1)
    targets = targets.contiguous().view(targets.size(0), -1)

    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)

    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean()


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dice = dice_coefficient(probs, targets, self.eps)
        return 1.0 - dice


class HybridDiceBCELoss(nn.Module):
    """
    L_Hybrid = L_BCE + (1 - Dice)     -- exactly Equation (1) of the paper.

    bce_weight / dice_weight default to 1.0 each, matching the paper's
    unweighted sum, but are exposed in case you want to tune the balance.
    """

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(eps=eps)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


if __name__ == "__main__":
    torch.manual_seed(0)
    logits = torch.randn(4, 1, 64, 64)
    targets = (torch.rand(4, 1, 64, 64) > 0.5).float()

    criterion = HybridDiceBCELoss()
    loss = criterion(logits, targets)
    print("Hybrid loss:", loss.item())

    # Sanity: near-perfect prediction -> loss should be much smaller
    perfect_logits = (targets * 20 - 10)
    loss_perfect = criterion(perfect_logits, targets)
    print("Near-perfect prediction loss:", loss_perfect.item())
    assert loss_perfect.item() < loss.item()
    print("OK: loss decreases for better predictions")
