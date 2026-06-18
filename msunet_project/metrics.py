"""
metrics.py
----------
Segmentation evaluation metrics. The paper states "Eleven metrics were
used for the segmentation performance, which included Dice and IoU among
others." We implement a comprehensive set of standard binary-segmentation
metrics computed per-batch from confusion-matrix counts (TP/FP/TN/FN):

    1. Pixel Accuracy
    2. Precision
    3. Recall (Sensitivity)
    4. Specificity
    5. F1-score
    6. IoU (Jaccard index)
    7. Dice coefficient
    8. Balanced Accuracy
    9. Matthews Correlation Coefficient (MCC)
   10. Cohen's Kappa
   11. Hausdorff-style boundary metric (mean absolute HC error in mm,
       computed separately downstream via the ellipse-fitting pipeline)

All functions operate on already-thresholded binary masks
(torch.uint8 / bool tensors), shape [B, 1, H, W] or [B, H, W].
"""

import torch


def _flatten(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(t.size(0), -1).float()


def confusion_counts(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7):
    """
    preds, targets: binary {0, 1} tensors of the same shape.
    Returns batch-summed TP, FP, TN, FN (scalars).
    """
    preds = _flatten(preds)
    targets = _flatten(targets)

    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    tn = ((1 - preds) * (1 - targets)).sum()

    return tp, fp, tn, fn, eps


@torch.no_grad()
def compute_all_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict:
    """
    Compute the full metrics suite from raw logits + ground-truth mask.

    Parameters
    ----------
    logits  : [B, 1, H, W] raw network output (pre-sigmoid)
    targets : [B, 1, H, W] binary ground-truth mask {0, 1}

    Returns
    -------
    dict of python floats
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = targets.float()

    tp, fp, tn, fn, eps = confusion_counts(preds, targets)

    accuracy = (tp + tn) / (tp + fp + tn + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)            # sensitivity
    specificity = tn / (tn + fp + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    balanced_acc = (recall + specificity) / 2

    # Matthews Correlation Coefficient
    mcc_num = (tp * tn) - (fp * fn)
    mcc_den = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + eps)
    mcc = mcc_num / (mcc_den + eps)

    # Cohen's Kappa
    total = tp + fp + tn + fn
    p_o = (tp + tn) / (total + eps)
    p_e = (((tp + fp) * (tp + fn)) + ((fn + tn) * (fp + tn))) / (total * total + eps)
    kappa = (p_o - p_e) / (1 - p_e + eps)

    return {
        "accuracy": accuracy.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "specificity": specificity.item(),
        "f1": f1.item(),
        "iou": iou.item(),
        "dice": dice.item(),
        "balanced_accuracy": balanced_acc.item(),
        "mcc": mcc.item(),
        "kappa": kappa.item(),
    }


class MetricTracker:
    """Accumulates per-batch metrics (weighted by batch size) over an epoch."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sums = {}
        self.count = 0

    def update(self, metrics_dict: dict, batch_size: int = 1):
        for k, v in metrics_dict.items():
            self.sums[k] = self.sums.get(k, 0.0) + v * batch_size
        self.count += batch_size

    def average(self) -> dict:
        if self.count == 0:
            return {k: 0.0 for k in self.sums}
        return {k: v / self.count for k, v in self.sums.items()}


if __name__ == "__main__":
    torch.manual_seed(0)
    logits = torch.randn(4, 1, 32, 32) * 3
    targets = (torch.rand(4, 1, 32, 32) > 0.5).float()
    m = compute_all_metrics(logits, targets)
    for k, v in m.items():
        print(f"{k:>18}: {v:.4f}")

    # Sanity: identical pred/target -> dice/iou/acc should all be 1.0
    perfect_logits = targets * 20 - 10
    m_perfect = compute_all_metrics(perfect_logits, targets)
    print("\nPerfect prediction sanity check:")
    for k, v in m_perfect.items():
        print(f"{k:>18}: {v:.4f}")
    assert abs(m_perfect["dice"] - 1.0) < 1e-3
    assert abs(m_perfect["iou"] - 1.0) < 1e-3
    assert abs(m_perfect["accuracy"] - 1.0) < 1e-3
    print("OK")
