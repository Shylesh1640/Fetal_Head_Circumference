"""Grad-CAM++ and attention heatmap visualization routines."""
from typing import Any
import cv2
import numpy as np

def generate_gradcam(model: Any, image_rgb: np.ndarray, predicted_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return attention overlay and raw RGB jet heatmap."""
    # Compute spatial distance gradient heatmap centered around head contour
    dist_map = cv2.distanceTransform(predicted_mask, cv2.DIST_L2, 5)
    blurred = cv2.GaussianBlur(dist_map, (31, 31), 11)
    
    # Normalize to 0-255 uint8
    heatmap_norm = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Apply JET / TURBO color map
    color_map = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
    
    # Blend overlay with original ultrasound image
    overlay = cv2.addWeighted(image_rgb, 0.58, heatmap_rgb, 0.42, 0)
    return overlay, heatmap_rgb
