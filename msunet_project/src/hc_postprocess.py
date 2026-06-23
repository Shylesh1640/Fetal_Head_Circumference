"""
hc_postprocess.py
Post-processing pipeline that converts a predicted binary mask into a
head-circumference (HC) measurement in millimetres, exactly mirroring the
paper's pipeline:

    predicted mask -> morphological cleanup -> largest contour
    -> ellipse fit (cv2.fitEllipse) -> Ramanujan's ellipse-perimeter
    approximation -> HC in mm (using the per-image pixel size)
"""

import cv2
import numpy as np


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """mask: uint8 binary {0,255}. Removes small holes/specks."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    return opened


def fit_head_ellipse(mask: np.ndarray):
    """Returns ((cx, cy), (major_axis, minor_axis), angle) or None if no
    contour is found. Axis lengths are the FULL major/minor axis lengths
    in pixels (cv2.fitEllipse convention)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:  # fitEllipse requires >= 5 points
        return None
    ellipse = cv2.fitEllipse(largest)
    return ellipse


def ramanujan_ellipse_perimeter(semi_major_mm: float, semi_minor_mm: float) -> float:
    """Ramanujan's second approximation for the perimeter of an ellipse.

    P ≈ pi * [3(a+b) - sqrt((3a+b)(a+3b))]

    where a, b are the semi-major and semi-minor axes.
    """
    a, b = semi_major_mm, semi_minor_mm
    h = ((a - b) ** 2) / ((a + b) ** 2 + 1e-12)
    perimeter = np.pi * (a + b) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h)))
    return perimeter


def compute_head_circumference_mm(binary_mask_255: np.ndarray, pixel_size_mm: float):
    """Full pipeline: cleaned mask -> ellipse -> HC in mm.

    Returns dict with hc_mm, ellipse params, or None if segmentation failed.
    """
    cleaned = clean_mask(binary_mask_255)
    ellipse = fit_head_ellipse(cleaned)
    if ellipse is None:
        return None

    (cx, cy), (axis1_px, axis2_px), angle = ellipse
    semi_major_px = max(axis1_px, axis2_px) / 2.0
    semi_minor_px = min(axis1_px, axis2_px) / 2.0

    semi_major_mm = semi_major_px * pixel_size_mm
    semi_minor_mm = semi_minor_px * pixel_size_mm

    hc_mm = ramanujan_ellipse_perimeter(semi_major_mm, semi_minor_mm)

    return {
        "hc_mm": hc_mm,
        "center_px": (cx, cy),
        "semi_major_mm": semi_major_mm,
        "semi_minor_mm": semi_minor_mm,
        "angle_deg": angle,
    }


if __name__ == "__main__":
    # quick synthetic sanity check: a perfect circle of radius 50px, 0.1mm/px
    canvas = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(canvas, (128, 128), 50, 255, thickness=-1)
    result = compute_head_circumference_mm(canvas, pixel_size_mm=0.1)
    expected = 2 * np.pi * 50 * 0.1
    print(f"Computed HC: {result['hc_mm']:.3f} mm | Expected (circle): {expected:.3f} mm")
