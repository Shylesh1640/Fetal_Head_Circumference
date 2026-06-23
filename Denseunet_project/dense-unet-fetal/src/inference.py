"""
inference.py
=============
Loads a trained Dense U-Net checkpoint and runs full inference on a single
ultrasound image, including the post-processing pipeline used in the
reference paper: morphological cleanup -> contour extraction -> ellipse
fitting -> head circumference (HC) via Ramanujan's approximation.

Usage:
    python src/inference.py \
        --checkpoint outputs/checkpoints/best.ckpt \
        --image path/to/000_HC.png \
        --pixel_size_mm 0.1  # from the HC18 dataset's pixel size CSV
"""

from __future__ import annotations

import os
import sys
import math
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import torch

from lightning_module import DenseUNetLightningModule
from dataset import get_eval_transforms


def predict_mask(model: DenseUNetLightningModule, image_gray: np.ndarray, image_size: int = 256) -> np.ndarray:
    transform = get_eval_transforms(image_size)
    augmented = transform(image=image_gray, mask=np.zeros_like(image_gray))
    tensor = augmented["image"].unsqueeze(0).float()

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

    mask = (prob > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (image_gray.shape[1], image_gray.shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask


def clean_mask(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    return cleaned


def fit_ellipse_and_hc(mask: np.ndarray, pixel_size_mm: float = 1.0):
    """Returns (ellipse, hc_mm) where ellipse = ((cx, cy), (MA, ma), angle)
    as returned by cv2.fitEllipse, and hc_mm is the circumference in mm
    computed via Ramanujan's ellipse-perimeter approximation."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contour found in predicted mask; segmentation may have failed.")

    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:
        raise ValueError("Largest contour has too few points to fit an ellipse.")

    ellipse = cv2.fitEllipse(largest)
    (_, _), (major_axis_px, minor_axis_px), _ = ellipse

    a = (major_axis_px / 2.0) * pixel_size_mm  # semi-major axis, mm
    b = (minor_axis_px / 2.0) * pixel_size_mm  # semi-minor axis, mm

    # Ramanujan's second approximation for ellipse perimeter.
    h = ((a - b) ** 2) / ((a + b) ** 2 + 1e-12)
    hc_mm = math.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)))
    return ellipse, hc_mm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--pixel_size_mm", type=float, default=1.0,
                    help="mm per pixel for this scan (see HC18's pixel_size CSV).")
    p.add_argument("--save_mask", type=str, default=None)
    args = p.parse_args()

    model = DenseUNetLightningModule.load_from_checkpoint(args.checkpoint, map_location="cpu")

    image_gray = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if image_gray is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    raw_mask = predict_mask(model, image_gray, args.image_size)
    mask = clean_mask(raw_mask)
    ellipse, hc_mm = fit_ellipse_and_hc(mask, pixel_size_mm=args.pixel_size_mm)

    print(f"Fitted ellipse: center={ellipse[0]}, axes={ellipse[1]}, angle={ellipse[2]:.2f} deg")
    print(f"Estimated Head Circumference: {hc_mm:.2f} mm")

    if args.save_mask:
        cv2.imwrite(args.save_mask, mask)
        print(f"Saved cleaned mask to: {args.save_mask}")


if __name__ == "__main__":
    main()
