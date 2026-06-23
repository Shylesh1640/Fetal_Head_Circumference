"""
inference.py — Run a trained Attention U-Net checkpoint on a single ultrasound
image, visualize the predicted mask, and compute head circumference (HC) via
contour extraction + ellipse fitting (Ramanujan's approximation), matching the
post-processing pipeline described in the reference paper.

Usage:
    python inference.py --checkpoint outputs/checkpoints/best_model.pth \
                         --image path/to/some_HC.png \
                         --pixel_size_mm 0.1714   # from HC18 image metadata CSV
"""

import argparse
import math

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from model import AttU_Net


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--pixel_size_mm", type=float, default=None,
                    help="mm per pixel for this image (from HC18 training_set_pixel_size_and_HC.csv). "
                         "If omitted, HC is reported in pixels only.")
    p.add_argument("--output", type=str, default="prediction.png")
    return p.parse_args()


def ramanujan_ellipse_perimeter(a: float, b: float) -> float:
    """Ramanujan's second approximation for ellipse perimeter (a, b = semi-axes)."""
    h = ((a - b) ** 2) / ((a + b) ** 2)
    return math.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)))


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AttU_Net(img_ch=1, output_ch=1).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    orig = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if orig is None:
        raise FileNotFoundError(args.image)
    orig_h, orig_w = orig.shape

    resized = cv2.resize(orig, (args.img_size, args.img_size))
    tensor = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0) / 255.0
    tensor = (tensor - 0.5) / 0.5
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

    mask = (prob > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hc_pixels = None
    ellipse = None
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if len(largest) >= 5:  # fitEllipse needs >= 5 points
            ellipse = cv2.fitEllipse(largest)
            (cx, cy), (major_axis, minor_axis), angle = ellipse
            a = major_axis / 2.0
            b = minor_axis / 2.0
            hc_pixels = ramanujan_ellipse_perimeter(a, b)

    # Visualization
    overlay = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
    contour_overlay = overlay.copy()
    if contours:
        cv2.drawContours(contour_overlay, [largest], -1, (0, 255, 0), 2)
    if ellipse is not None:
        cv2.ellipse(contour_overlay, ellipse, (0, 0, 255), 2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(orig, cmap="gray")
    axes[0].set_title("Input Ultrasound")
    axes[1].imshow(mask_clean, cmap="gray")
    axes[1].set_title("Predicted Mask")
    axes[2].imshow(cv2.cvtColor(contour_overlay, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Contour (green) + Fitted Ellipse (red)")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved visualization to {args.output}")

    if hc_pixels is not None:
        print(f"Estimated HC: {hc_pixels:.2f} pixels")
        if args.pixel_size_mm is not None:
            hc_mm = hc_pixels * args.pixel_size_mm
            print(f"Estimated HC: {hc_mm:.2f} mm")
    else:
        print("No valid contour found — could not compute HC.")


if __name__ == "__main__":
    main()
