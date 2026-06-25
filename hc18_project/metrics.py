import numpy as np
import cv2


def segmentation_metrics(pred, target, threshold=0.5, eps=1e-6):
    """pred, target: numpy arrays, same shape, values in [0,1]/{0,1}."""
    pred_bin = (pred > threshold).astype(np.uint8)
    target_bin = (target > threshold).astype(np.uint8)

    tp = np.logical_and(pred_bin == 1, target_bin == 1).sum()
    tn = np.logical_and(pred_bin == 0, target_bin == 0).sum()
    fp = np.logical_and(pred_bin == 1, target_bin == 0).sum()
    fn = np.logical_and(pred_bin == 0, target_bin == 1).sum()

    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)

    return {
        "acc": acc, "precision": precision, "recall": recall,
        "f1": f1, "iou": iou, "dice": dice,
    }


def mask_to_hc_mm(mask, pixel_size_mm=1.0, threshold=0.5):
    """
    Morphological cleanup -> largest contour -> ellipse fit ->
    Ramanujan's perimeter approximation (paper Eq. 2). Returns HC in mm.
    pixel_size_mm: the dataset's per-image pixel spacing
    (see *_pixel_size_and_HC.csv in the HC18 download).
    """
    mask_bin = (mask > threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_clean = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:
        return None

    ellipse = cv2.fitEllipse(largest)
    (_, _), (minor_ax, major_ax), _ = ellipse
    a = (major_ax / 2.0) * pixel_size_mm
    b = (minor_ax / 2.0) * pixel_size_mm

    hc = np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))
    return hc
