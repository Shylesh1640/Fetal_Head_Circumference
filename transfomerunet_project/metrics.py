"""
metrics.py
Computes Dice Score, IoU, Precision, Recall, Accuracy, F1 Score using
`segmentation_models_pytorch.metrics`, the library's official, tested
metrics module (confusion-matrix based: tp/fp/fn/tn -> derived metrics).

Reference: https://smp.readthedocs.io/en/latest/metrics.html

Note: For binary segmentation, F1 Score and Dice Score are mathematically
identical (Dice = F1 = 2TP / (2TP + FP + FN)), so both are reported for
clarity/transparency with the paper's terminology, but they will always
match exactly.
"""

from dataclasses import dataclass, asdict

import torch
import segmentation_models_pytorch as smp

import config


@dataclass
class EpochMetrics:
    loss: float = 0.0
    dice: float = 0.0
    iou: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    accuracy: float = 0.0
    f1: float = 0.0

    def as_dict(self):
        return asdict(self)


class MetricAccumulator:
    """
    Accumulates confusion-matrix statistics (tp, fp, fn, tn) across all
    batches in an epoch, then computes corpus-level ("micro") metrics once
    at the end. This is more correct than averaging per-batch metrics,
    since per-batch averaging is biased by variable foreground pixel counts
    across batches.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._tp = []
        self._fp = []
        self._fn = []
        self._tn = []
        self._loss_sum = 0.0
        self._n_batches = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss_value: float):
        """
        logits:  raw model output, shape [B, 1, H, W]
        targets: binary ground-truth mask, shape [B, 1, H, W], values in {0., 1.}
        """
        probs = torch.sigmoid(logits)
        targets_long = targets.long()

        tp, fp, fn, tn = smp.metrics.get_stats(
            probs, targets_long, mode="binary", threshold=config.PIXEL_THRESHOLD
        )

        self._tp.append(tp)
        self._fp.append(fp)
        self._fn.append(fn)
        self._tn.append(tn)

        self._loss_sum += loss_value
        self._n_batches += 1

    def compute(self) -> EpochMetrics:
        if self._n_batches == 0:
            return EpochMetrics()

        tp = torch.cat(self._tp)
        fp = torch.cat(self._fp)
        fn = torch.cat(self._fn)
        tn = torch.cat(self._tn)

        # "micro" reduction = global TP/FP/FN/TN pooled over every pixel and
        # every image before computing the ratio -> standard corpus-level
        # segmentation metric reporting, matches paper-style validation tables.
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item()
        f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro").item()
        precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro").item()
        recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro").item()
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro").item()

        # For binary segmentation Dice == F1; kept as separate named field
        # since the paper reports it under "Dice Score" explicitly.
        dice = f1

        return EpochMetrics(
            loss=self._loss_sum / self._n_batches,
            dice=dice,
            iou=iou,
            precision=precision,
            recall=recall,
            accuracy=accuracy,
            f1=f1,
        )
