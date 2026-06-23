"""
metrics.py
Pixel-wise binary segmentation metrics, accumulated over an entire
epoch/split using running TP/FP/FN/TN counts (numerically stable, matches
how the metrics are reported in segmentation papers — computed once over
the whole dataset rather than averaged per-batch).

Metrics:
    Dice Score  = 2*TP / (2*TP + FP + FN)
    IoU         = TP / (TP + FP + FN)
    Precision   = TP / (TP + FP)
    Recall      = TP / (TP + FN)
    Accuracy    = (TP + TN) / (TP + TN + FP + FN)
    F1 Score    = 2 * Precision * Recall / (Precision + Recall)   (== Dice for
                  binary masks, reported separately because the paper lists
                  it as its own metric)
"""

import torch

import config as cfg


class SegmentationMetricAccumulator:
    def __init__(self, threshold: float = cfg.THRESHOLD, eps: float = 1e-7):
        self.threshold = threshold
        self.eps = eps
        self.reset()

    def reset(self):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tn = 0.0
        self.loss_sum = 0.0
        self.n_batches = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss_value: float = None):
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()
        targets = targets.float()

        tp = (preds * targets).sum().item()
        fp = (preds * (1 - targets)).sum().item()
        fn = ((1 - preds) * targets).sum().item()
        tn = ((1 - preds) * (1 - targets)).sum().item()

        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn

        if loss_value is not None:
            self.loss_sum += loss_value
            self.n_batches += 1

    def compute(self) -> dict:
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        eps = self.eps

        dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        iou = (tp + eps) / (tp + fp + fn + eps)
        precision = (tp + eps) / (tp + fp + eps)
        recall = (tp + eps) / (tp + fn + eps)
        accuracy = (tp + tn + eps) / (tp + tn + fp + fn + eps)
        f1 = (2 * precision * recall + eps) / (precision + recall + eps)

        avg_loss = self.loss_sum / self.n_batches if self.n_batches > 0 else float("nan")

        return {
            "Dice": dice,
            "IoU": iou,
            "Precision": precision,
            "Recall": recall,
            "Accuracy": accuracy,
            "F1": f1,
            "Loss": avg_loss,
        }
