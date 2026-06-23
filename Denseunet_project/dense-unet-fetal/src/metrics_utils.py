"""
Metrics
=======
Builds one torchmetrics.MetricCollection per stage (train / val / test) so
metric states never leak between stages, as required by Lightning when
metrics are logged via `self.log(..., on_step=False, on_epoch=True)`.

Reported metrics: Dice Score, IoU, Precision, Recall, Accuracy, F1 Score.
("Loss" is logged separately in the LightningModule since it is not a
torchmetrics object.)

Note: Dice Score here is the *soft* Dice computed on sigmoid probabilities
(continuous), matching how it is typically reported in segmentation papers.
Precision / Recall / F1 / Accuracy / IoU are computed on the thresholded
(binarized at 0.5) prediction, via torchmetrics' binary classification
metrics operating pixel-wise.
"""

from __future__ import annotations

import torch
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinaryJaccardIndex,
)


class DiceScore(Metric):
    """Soft Dice score accumulated over an epoch: 2*sum(p*t) / (sum(p)+sum(t))."""

    full_state_update = False

    def __init__(self, smooth: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.smooth = smooth
        self.add_state("intersection", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("union", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        preds = preds.float().reshape(-1)
        target = target.float().reshape(-1)
        self.intersection += (preds * target).sum()
        self.union += preds.sum() + target.sum()

    def compute(self) -> torch.Tensor:
        return (2.0 * self.intersection + self.smooth) / (self.union + self.smooth)


def build_metric_collection(prefix: str, threshold: float = 0.5) -> MetricCollection:
    """Returns a MetricCollection with Dice, IoU, Precision, Recall, Accuracy, F1.

    `prefix` should be one of "train_", "val_", "test_" — torchmetrics will
    prepend it to every metric name automatically, e.g. "val_Dice".
    """
    return MetricCollection(
        {
            "Dice": DiceScore(),
            "IoU": BinaryJaccardIndex(threshold=threshold),
            "Precision": BinaryPrecision(threshold=threshold),
            "Recall": BinaryRecall(threshold=threshold),
            "Accuracy": BinaryAccuracy(threshold=threshold),
            "F1": BinaryF1Score(threshold=threshold),
        },
        prefix=prefix,
    )
