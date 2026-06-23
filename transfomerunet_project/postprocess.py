"""
postprocess.py
Converts a predicted binary mask into a head-circumference (HC) measurement,
following the paper's pipeline:
    predicted mask -> morphological cleanup -> largest contour ->
    cv2.fitEllipse -> Ramanujan's ellipse perimeter approximation -> HC (mm)

Note on pixel-to-mm conversion: HC18 images have a per-image pixel spacing
stored in the dataset's `training_set_pixel_size_and_HC.csv` file (column
'pixel size(mm)'). If you want HC in millimetres rather than pixels, load
that CSV and multiply the pixel-unit measurements below by the per-image
pixel spacing. This module returns HC in PIXELS unless a pixel_size_mm is
supplied.
"""

import math
import os

import cv2
import numpy as np
import pandas as pd


def clean_mask(binary_mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Morphological closing to fill small holes / remove speckle noise."""
    mask_u8 = (binary_mask > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return opened


def extract_largest_contour(mask_u8: np.ndarray):
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def fit_ellipse(contour) -> dict:
    """
    Fits an ellipse via OpenCV (least-squares, cv2.fitEllipse).
    Requires at least 5 contour points.
    Returns semi-major axis a, semi-minor axis b (in pixels), center, angle.
    """
    if contour is None or len(contour) < 5:
        return None

    (cx, cy), (major_axis, minor_axis), angle = cv2.fitEllipse(contour)
    # cv2 returns FULL axis lengths; semi-axes are half of these
    a = major_axis / 2.0
    b = minor_axis / 2.0
    return {"center": (cx, cy), "semi_major_a": a, "semi_minor_b": b, "angle_deg": angle}


def ramanujan_perimeter(a: float, b: float) -> float:
    """
    Ramanujan's second approximation for the perimeter of an ellipse with
    semi-major axis a and semi-minor axis b. Highly accurate (error < 0.04%
    for all aspect ratios), which is why it's the standard formula used in
    fetal HC estimation literature instead of the cruder pi*(a+b) estimate.

        h = ((a-b)^2) / ((a+b)^2)
        P ~= pi * (a + b) * (1 + (3h) / (10 + sqrt(4 - 3h)))
    """
    if a <= 0 or b <= 0:
        return 0.0
    h = ((a - b) ** 2) / ((a + b) ** 2)
    perimeter = math.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)))
    return perimeter


def mask_to_hc(binary_mask: np.ndarray, pixel_size_mm: float = None) -> dict:
    """
    Full pipeline: mask -> cleaned mask -> contour -> ellipse -> HC.

    Args:
        binary_mask: 2D array, predicted segmentation mask (any non-zero = head)
        pixel_size_mm: optional, mm-per-pixel spacing for this specific image.
                        If provided, HC and axes are also returned in mm.

    Returns dict with keys: hc_pixels, hc_mm (or None), semi_major_a,
    semi_minor_b, center, angle_deg. Returns None values if fitting failed
    (e.g. empty mask).
    """
    cleaned = clean_mask(binary_mask)
    contour = extract_largest_contour(cleaned)
    ellipse = fit_ellipse(contour)

    if ellipse is None:
        return {
            "hc_pixels": None, "hc_mm": None,
            "semi_major_a": None, "semi_minor_b": None,
            "center": None, "angle_deg": None,
        }

    hc_pixels = ramanujan_perimeter(ellipse["semi_major_a"], ellipse["semi_minor_b"])
    hc_mm = hc_pixels * pixel_size_mm if pixel_size_mm is not None else None

    return {
        "hc_pixels": hc_pixels,
        "hc_mm": hc_mm,
        "semi_major_a": ellipse["semi_major_a"],
        "semi_minor_b": ellipse["semi_minor_b"],
        "center": ellipse["center"],
        "angle_deg": ellipse["angle_deg"],
    }


def load_pixel_size_lookup(csv_path: str) -> dict:
    """
    Loads HC18's 'training_set_pixel_size_and_HC.csv' (shipped alongside the
    dataset) and returns {filename: pixel_size_mm}.
    Expected columns: 'filename', 'pixel size(mm)'.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Pixel-size CSV not found at {csv_path}. This file ships with "
            f"the HC18 dataset download and is required for mm-accurate HC."
        )
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    return dict(zip(df["filename"], df["pixel size(mm)"]))
