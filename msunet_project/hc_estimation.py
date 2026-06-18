"""
hc_estimation.py
-----------------
Post-processing pipeline: predicted binary mask -> morphological cleanup
-> contour extraction -> least-squares ellipse fit -> head circumference
(HC) via Ramanujan's second approximation for ellipse perimeter
(Equation 2 in the paper):

    HC ~ pi * [ 3(a+b) - sqrt( (3a+b)(a+3b) ) ]

where a, b are the semi-major / semi-minor axes of the fitted ellipse.

cv2.fitEllipse() already performs a least-squares ellipse fit (it wraps
the same direct least-squares method commonly cited for this task), so
we use it directly rather than re-implementing the fit from scratch.
"""

import cv2
import numpy as np


def clean_mask(binary_mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Morphological cleanup: closing fills small gaps, opening removes
    small spurious blobs, matching the paper's description of
    "morphological operations is to fill in gaps and eliminate minor
    artifacts."
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    return opened


def extract_largest_contour(binary_mask: np.ndarray):
    """Returns the largest external contour (by area), or None if no contour found."""
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return None
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:
        # cv2.fitEllipse requires at least 5 points
        return None
    return largest


def fit_ellipse(contour: np.ndarray):
    """
    Least-squares ellipse fit via cv2.fitEllipse.

    Returns
    -------
    (center, (minor_axis_px, major_axis_px), angle_deg)
        Note: cv2 returns FULL axis lengths (diameters), not semi-axes.
    """
    ellipse = cv2.fitEllipse(contour)
    return ellipse  # ((cx, cy), (d_minor, d_major), angle)


def hc_from_axes(semi_major_a: float, semi_minor_b: float) -> float:
    """
    Ramanujan's second approximation for the perimeter of an ellipse
    (Equation 2 of the paper):

        HC ~ pi * [ 3(a+b) - sqrt( (3a+b)(a+3b) ) ]
    """
    a, b = semi_major_a, semi_minor_b
    term = (3 * a + b) * (a + 3 * b)
    term = max(term, 0.0)  # numerical safety
    hc = np.pi * (3 * (a + b) - np.sqrt(term))
    return float(hc)


def estimate_hc_from_mask(
    binary_mask: np.ndarray,
    pixel_size_mm: float = 1.0,
    do_morphology: bool = True,
):
    """
    Full pipeline: binary mask (uint8, values {0,255} or {0,1}) -> cleaned
    mask -> contour -> fitted ellipse -> HC in millimetres.

    Parameters
    ----------
    binary_mask   : 2D array, predicted segmentation mask at the
                     resolution it was produced (e.g. 256x256).
    pixel_size_mm : physical size of one pixel in mm (from the HC18 CSV).
                     If your mask resolution differs from the original
                     ultrasound image resolution, rescale pixel_size_mm
                     accordingly (see `rescale_pixel_size` below) before
                     calling this function.
    do_morphology : whether to run the morphological cleanup step.

    Returns
    -------
    dict with keys: hc_mm, hc_px, center, semi_major_px, semi_minor_px,
                     angle_deg, contour (np.ndarray or None), success (bool)
    """
    mask = binary_mask.copy()
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)

    if do_morphology:
        mask = clean_mask(mask)

    contour = extract_largest_contour(mask)
    if contour is None:
        return {
            "hc_mm": float("nan"),
            "hc_px": float("nan"),
            "center": None,
            "semi_major_px": float("nan"),
            "semi_minor_px": float("nan"),
            "angle_deg": float("nan"),
            "contour": None,
            "success": False,
        }

    (cx, cy), (d_minor, d_major), angle = fit_ellipse(contour)
    semi_major_px = d_major / 2.0
    semi_minor_px = d_minor / 2.0

    hc_px = hc_from_axes(semi_major_px, semi_minor_px)
    hc_mm = hc_px * pixel_size_mm

    return {
        "hc_mm": hc_mm,
        "hc_px": hc_px,
        "center": (cx, cy),
        "semi_major_px": semi_major_px,
        "semi_minor_px": semi_minor_px,
        "angle_deg": angle,
        "contour": contour,
        "success": True,
    }


def rescale_pixel_size(original_pixel_size_mm: float, orig_dim: int, resized_dim: int) -> float:
    """
    If the network operates on a resized image (e.g. 256x256) but the
    pixel_size_mm value in the CSV refers to the ORIGINAL image resolution
    (e.g. 800x540), the effective pixel size at the resized resolution is:

        new_pixel_size = original_pixel_size * (orig_dim / resized_dim)

    Apply this per-axis if width/height scale differently, then average,
    or (simplest, used here) assume an isotropic resize ratio.
    """
    return original_pixel_size_mm * (orig_dim / resized_dim)


def draw_contour_overlay(image_rgb: np.ndarray, contour: np.ndarray, color=(0, 255, 0), thickness: int = 2) -> np.ndarray:
    """Draws the fitted contour onto a copy of the RGB image for visualization."""
    overlay = image_rgb.copy()
    if contour is not None:
        cv2.drawContours(overlay, [contour], -1, color, thickness)
    return overlay


if __name__ == "__main__":
    # Sanity check: draw a known ellipse, segment it back, verify HC recovery
    H, W = 256, 256
    true_a, true_b = 90.0, 60.0  # semi-axes in px
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.ellipse(mask, (W // 2, H // 2), (int(true_a), int(true_b)), 0, 0, 360, 255, -1)

    pixel_size_mm = 0.15
    result = estimate_hc_from_mask(mask, pixel_size_mm=pixel_size_mm)

    expected_hc_mm = hc_from_axes(true_a, true_b) * pixel_size_mm
    print("Expected HC (mm):", expected_hc_mm)
    print("Estimated HC (mm):", result["hc_mm"])
    print("Recovered semi-axes (px):", result["semi_major_px"], result["semi_minor_px"])
    rel_err = abs(result["hc_mm"] - expected_hc_mm) / expected_hc_mm
    print(f"Relative error: {rel_err*100:.3f}%")
    assert rel_err < 0.02, "Ellipse fitting sanity check failed"
    print("OK")
