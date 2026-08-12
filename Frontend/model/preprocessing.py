"""Ultrasound image preprocessing utilities."""
from __future__ import annotations

import numpy as np
from PIL import Image


def prepare_image(image: Image.Image, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized model input (CHW) and display-friendly RGB image."""
    rgb = image.convert("RGB").resize((image_size, image_size))
    display = np.asarray(rgb)
    normalized = display.astype(np.float32) / 255.0
    return np.transpose(normalized, (2, 0, 1)), display
