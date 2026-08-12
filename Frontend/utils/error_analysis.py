"""Segmentation Error Analysis & Pixel Confusion Matrix Utility Module."""
from __future__ import annotations

import io
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# Updated Color Mapping per User Specification
COLOR_TP = (255, 255, 255)  # ⚪ White (#FFFFFF) - True Positive
COLOR_FP = (255, 0, 0)      # 🔴 Red (#FF0000) - False Positive
COLOR_FN = (0, 255, 0)      # 🟢 Green (#00FF00) - False Negative
COLOR_TN = (0, 0, 0)        # ⚫ Black (#000000) - True Negative


def compute_confusion_masks(predicted: np.ndarray, ground_truth: np.ndarray) -> dict[str, np.ndarray]:
    """
    Compute pixel-wise boolean masks for TP, FP, FN, TN.
    """
    pred_bool = (predicted > 0).astype(bool)
    gt_bool = (ground_truth > 0).astype(bool)

    tp_mask = pred_bool & gt_bool
    fp_mask = pred_bool & (~gt_bool)
    fn_mask = (~pred_bool) & gt_bool
    tn_mask = (~pred_bool) & (~gt_bool)

    return {
        "tp_mask": tp_mask,
        "fp_mask": fp_mask,
        "fn_mask": fn_mask,
        "tn_mask": tn_mask,
        "pred_bool": pred_bool,
        "gt_bool": gt_bool,
    }


def compute_error_metrics(predicted: np.ndarray, ground_truth: np.ndarray) -> dict[str, float | int]:
    """
    Compute all 12 quantitative metrics for pixel-wise segmentation error analysis:
    - TP Pixels
    - FP Pixels
    - FN Pixels
    - TN Pixels
    - Pixel Accuracy
    - Sensitivity (Recall)
    - Specificity
    - False Positive Rate
    - False Negative Rate
    - Precision
    - Dice Score
    - IoU
    """
    masks = compute_confusion_masks(predicted, ground_truth)
    tp = int(masks["tp_mask"].sum())
    fp = int(masks["fp_mask"].sum())
    fn = int(masks["fn_mask"].sum())
    tn = int(masks["tn_mask"].sum())
    total_pixels = tp + fp + fn + tn

    eps = 1e-8

    accuracy = (tp + tn) / (total_pixels + eps)
    sensitivity = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    fpr = fp / (fp + tn + eps)
    fnr = fn / (tp + fn + eps)
    precision = tp / (tp + fp + eps)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)

    return {
        "TP Pixels": tp,
        "FP Pixels": fp,
        "FN Pixels": fn,
        "TN Pixels": tn,
        "Total Pixels": total_pixels,
        "Pixel Accuracy": float(accuracy),
        "Sensitivity (Recall)": float(sensitivity),
        "Specificity": float(specificity),
        "False Positive Rate": float(fpr),
        "False Negative Rate": float(fnr),
        "Precision": float(precision),
        "Dice Score": float(dice),
        "IoU": float(iou),
        "TP Percent": (tp / total_pixels) * 100.0 if total_pixels > 0 else 0.0,
        "FP Percent": (fp / total_pixels) * 100.0 if total_pixels > 0 else 0.0,
        "FN Percent": (fn / total_pixels) * 100.0 if total_pixels > 0 else 0.0,
        "TN Percent": (tn / total_pixels) * 100.0 if total_pixels > 0 else 0.0,
    }


def render_error_map_rgb(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    vis_tp: bool = True,
    vis_fp: bool = True,
    vis_fn: bool = True,
    vis_tn: bool = True,
) -> np.ndarray:
    """
    Generate an RGB uint8 confusion map image matching the user's requested color scheme:
    ⚪ True Positive: White (#FFFFFF)
    🔴 False Positive: Red (#FF0000)
    🟢 False Negative: Green (#00FF00)
    ⚫ True Negative: Black (#000000)
    """
    masks = compute_confusion_masks(predicted, ground_truth)
    h, w = predicted.shape[:2]
    error_map = np.zeros((h, w, 3), dtype=np.uint8)

    if vis_tp:
        error_map[masks["tp_mask"]] = COLOR_TP
    if vis_fp:
        error_map[masks["fp_mask"]] = COLOR_FP
    if vis_fn:
        error_map[masks["fn_mask"]] = COLOR_FN
    if vis_tn:
        error_map[masks["tn_mask"]] = COLOR_TN

    return error_map


def render_error_overlay(
    base_image: np.ndarray,
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    alpha: float = 0.50,
    vis_tp: bool = True,
    vis_fp: bool = True,
    vis_fn: bool = True,
    vis_tn: bool = False,  # Default False so background image shows cleanly through TN pixels
) -> np.ndarray:
    """
    Overlay the pixel-wise error map onto a base image (Original Ultrasound or Predicted Mask).
    Blends visible classes with opacity alpha; hidden classes retain original base image pixels.
    """
    masks = compute_confusion_masks(predicted, ground_truth)

    # Ensure base_image is RGB uint8
    if base_image.ndim == 2:
        base_rgb = cv2.cvtColor(base_image, cv2.COLOR_GRAY2RGB)
    elif base_image.shape[2] == 1:
        base_rgb = cv2.cvtColor(base_image[:, :, 0], cv2.COLOR_GRAY2RGB)
    else:
        base_rgb = base_image.copy()

    if base_rgb.dtype != np.uint8:
        base_rgb = (np.clip(base_rgb, 0, 1) * 255).astype(np.uint8)

    # Resize base_rgb if shapes mismatch
    h, w = predicted.shape[:2]
    if base_rgb.shape[:2] != (h, w):
        base_rgb = cv2.resize(base_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

    overlay = base_rgb.astype(np.float32).copy()

    active_mask = np.zeros((h, w), dtype=bool)
    color_layer = np.zeros((h, w, 3), dtype=np.float32)

    if vis_tp:
        m = masks["tp_mask"]
        active_mask |= m
        color_layer[m] = COLOR_TP
    if vis_fp:
        m = masks["fp_mask"]
        active_mask |= m
        color_layer[m] = COLOR_FP
    if vis_fn:
        m = masks["fn_mask"]
        active_mask |= m
        color_layer[m] = COLOR_FN
    if vis_tn:
        m = masks["tn_mask"]
        active_mask |= m
        color_layer[m] = COLOR_TN

    # Blend active error pixels with base image
    overlay[active_mask] = (1.0 - alpha) * base_rgb[active_mask] + alpha * color_layer[active_mask]
    return np.clip(overlay, 0, 255).astype(np.uint8)


def export_pixel_stats_csv(metrics_dict: dict[str, float | int], model_name: str = "nnUNet_ResAttn") -> bytes:
    """
    Generate CSV data containing pixel counts, percentages, and performance metrics.
    """
    formatted_data = {
        "Metric / Statistic": [
            "True Positive (TP) Pixels",
            "False Positive (FP) Pixels",
            "False Negative (FN) Pixels",
            "True Negative (TN) Pixels",
            "Total Image Pixels",
            "Pixel Accuracy",
            "Sensitivity (Recall)",
            "Specificity",
            "False Positive Rate (FPR)",
            "False Negative Rate (FNR)",
            "Precision",
            "Dice Score (F1)",
            "IoU (Jaccard Index)",
            "TP Pixel Ratio (%)",
            "FP Pixel Ratio (%)",
            "FN Pixel Ratio (%)",
            "TN Pixel Ratio (%)",
            "Model Checkpoint",
        ],
        "Value": [
            f"{metrics_dict['TP Pixels']:,}",
            f"{metrics_dict['FP Pixels']:,}",
            f"{metrics_dict['FN Pixels']:,}",
            f"{metrics_dict['TN Pixels']:,}",
            f"{metrics_dict['Total Pixels']:,}",
            f"{metrics_dict['Pixel Accuracy']:.4f}",
            f"{metrics_dict['Sensitivity (Recall)']:.4f}",
            f"{metrics_dict['Specificity']:.4f}",
            f"{metrics_dict['False Positive Rate']:.4f}",
            f"{metrics_dict['False Negative Rate']:.4f}",
            f"{metrics_dict['Precision']:.4f}",
            f"{metrics_dict['Dice Score']:.4f}",
            f"{metrics_dict['IoU']:.4f}",
            f"{metrics_dict['TP Percent']:.2f}%",
            f"{metrics_dict['FP Percent']:.2f}%",
            f"{metrics_dict['FN Percent']:.2f}%",
            f"{metrics_dict['TN Percent']:.2f}%",
            model_name,
        ],
    }

    df = pd.DataFrame(formatted_data)
    return df.to_csv(index=False).encode("utf-8")


def export_png_bytes(img_array: np.ndarray) -> bytes:
    """
    Convert a NumPy uint8 RGB image into PNG bytes for st.download_button.
    """
    if img_array.ndim == 2:
        pil_img = Image.fromarray(img_array, mode="L")
    else:
        pil_img = Image.fromarray(img_array, mode="RGB")

    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return buffer.getvalue()
