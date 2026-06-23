"""
inference.py
End-to-end inference on a single ultrasound image:
    image -> model -> binary mask -> HC (mm) -> Grad-CAM heatmap

Requires the `grad-cam` package (pip install grad-cam), specifically
pytorch-grad-cam by jacobgil, the standard reference implementation:
https://github.com/jacobgil/pytorch-grad-cam

Usage:
    python inference.py --image path/to/some_HC.png --pixel_size_mm 0.123
"""

import argparse
import os

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

import config
from model import build_model
from dataset import get_eval_transforms
from postprocess import mask_to_hc


class SegmentationGradCAMTarget:
    """
    Grad-CAM target for segmentation: sums the activations within the
    predicted foreground region, following the recommended pattern in the
    pytorch-grad-cam repo's semantic segmentation example.
    """

    def __init__(self, mask: torch.Tensor):
        self.mask = mask

    def __call__(self, model_output):
        return (model_output[0, :, :] * self.mask).sum()


def run_inference(image_path: str, checkpoint_path: str, pixel_size_mm: float = None,
                   use_gradcam: bool = True):
    device = config.DEVICE

    model = build_model().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint (epoch {checkpoint['epoch']}, val_dice={checkpoint['val_dice']:.4f})")

    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    original_h, original_w = gray.shape
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    transforms = get_eval_transforms(config.IMAGE_SIZE)
    transformed = transforms(image=rgb)
    input_tensor = transformed["image"].unsqueeze(0).to(device)  # [1,3,H,W]

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits)
        pred_mask = (probs > config.PIXEL_THRESHOLD).float()

    pred_mask_np = pred_mask.squeeze().cpu().numpy().astype(np.uint8)  # at IMAGE_SIZE resolution
    # Resize predicted mask back to original image resolution for HC calculation
    pred_mask_resized = cv2.resize(pred_mask_np, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    hc_result = mask_to_hc(pred_mask_resized, pixel_size_mm=pixel_size_mm)

    print("\n========== HC RESULT ==========")
    print(f"Image: {image_path}")
    print(f"HC (pixels): {hc_result['hc_pixels']}")
    if hc_result["hc_mm"] is not None:
        print(f"HC (mm): {hc_result['hc_mm']:.2f}")
    else:
        print("HC (mm): not computed (no --pixel_size_mm provided)")
    print("================================\n")

    fig, axes = plt.subplots(1, 3 if use_gradcam else 2, figsize=(15, 5))
    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Input ultrasound")
    axes[0].axis("off")

    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    overlay_mask = np.zeros_like(overlay)
    overlay_mask[pred_mask_resized > 0] = [255, 0, 0]
    blended = cv2.addWeighted(overlay, 0.7, overlay_mask, 0.3, 0)
    axes[1].imshow(blended)
    axes[1].set_title(f"Predicted mask + ellipse  (HC={hc_result['hc_pixels']:.1f}px)"
                       if hc_result['hc_pixels'] else "Predicted mask (fit failed)")
    axes[1].axis("off")

    if use_gradcam:
        try:
            from pytorch_grad_cam import GradCAM
            target_layers = [model.encoder.encoder.block4[-1].norm1]  # last MiT-B2 stage block
            cam = GradCAM(model=model, target_layers=target_layers)
            grayscale_cam = cam(input_tensor=input_tensor,
                                 targets=[SegmentationGradCAMTarget(pred_mask.squeeze(0))])[0]
            heatmap_resized = cv2.resize(grayscale_cam, (original_w, original_h))
            axes[2].imshow(gray, cmap="gray")
            axes[2].imshow(heatmap_resized, cmap="jet", alpha=0.5)
            axes[2].set_title("Grad-CAM")
            axes[2].axis("off")
        except Exception as e:
            print(f"[Grad-CAM] Skipped due to error: {e}")
            print("If this is a layer-naming error, inspect `model.encoder` "
                  "with print(model) and update target_layers accordingly.")

    plt.tight_layout()
    out_path = os.path.join(config.PRED_DIR, os.path.basename(image_path).replace(".png", "_result.png"))
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved visualization to {out_path}")

    return hc_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to an ultrasound PNG image")
    parser.add_argument("--checkpoint", type=str,
                         default=os.path.join(config.CHECKPOINT_DIR, "best_model.pth"))
    parser.add_argument("--pixel_size_mm", type=float, default=None,
                         help="mm-per-pixel spacing for this image (from HC18 CSV), optional")
    parser.add_argument("--no_gradcam", action="store_true")
    args = parser.parse_args()

    run_inference(args.image, args.checkpoint, args.pixel_size_mm, use_gradcam=not args.no_gradcam)
