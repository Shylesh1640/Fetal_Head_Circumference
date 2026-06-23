"""
Segmentation metrics: Dice Score, IoU, Precision, Recall, Accuracy, F1 Score.

All metrics are computed per-batch from the pixel-level confusion matrix
(TP, FP, FN, TN) on binarized predictions (threshold = 0.5 on sigmoid probs),
then accumulated across an epoch using torchmetrics-style running sums so the
final epoch metric is mathematically exact (not a naive average-of-batch-means).
"""

import torch


class SegmentationMetrics:
    """Accumulates TP/FP/FN/TN over an epoch and reports all 6 metrics + loss."""

    def __init__(self, threshold: float = 0.5, smooth: float = 1e-6, device="cpu"):
        self.threshold = threshold
        self.smooth = smooth
        self.device = device
        self.reset()

    def reset(self):
        self.tp = torch.tensor(0.0, device=self.device)
        self.fp = torch.tensor(0.0, device=self.device)
        self.fn = torch.tensor(0.0, device=self.device)
        self.tn = torch.tensor(0.0, device=self.device)
        self.loss_sum = 0.0
        self.n_batches = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss_value: float = None):
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()
        targets = targets.float()

        self.tp += (preds * targets).sum()
        self.fp += (preds * (1 - targets)).sum()
        self.fn += ((1 - preds) * targets).sum()
        self.tn += ((1 - preds) * (1 - targets)).sum()

        if loss_value is not None:
            self.loss_sum += loss_value
            self.n_batches += 1

    @torch.no_grad()
    def compute(self) -> dict:
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        s = self.smooth

        precision = (tp + s) / (tp + fp + s)
        recall = (tp + s) / (tp + fn + s)
        iou = (tp + s) / (tp + fp + fn + s)
        dice = (2 * tp + s) / (2 * tp + fp + fn + s)
        accuracy = (tp + tn + s) / (tp + tn + fp + fn + s)
        f1 = (2 * precision * recall) / (precision + recall + s)

        avg_loss = self.loss_sum / self.n_batches if self.n_batches > 0 else float("nan")

        return {
            "Dice": dice.item(),
            "IoU": iou.item(),
            "Precision": precision.item(),
            "Recall": recall.item(),
            "Accuracy": accuracy.item(),
            "F1": f1.item(),
            "Loss": avg_loss,
        }
