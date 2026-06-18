"""
inference.py
------------
Runs the trained MS-UNet model on a set of images and produces the final
clinical-use outputs described in the paper's workflow (Fig. 1):

    input ultrasound -> predicted mask -> contour -> HC estimation
                                              |
                                              v
                                 Grad-CAM heatmap + HC value + overlay image

Outputs (written under config.OUTPUT_DIR):
    overlays/<filename>_overlay.png   - input image with green contour + HC text
    gradcam/<filename>_gradcam.png    - Grad-CAM heatmap overlay
    hc_predictions.csv                - filename, predicted_hc_mm, true_hc_mm,
                                          abs_error_mm, dice, iou (per image)

Run with:
    python inference.py --checkpoint checkpoints/best_msunet.pth
"""

import os
import argparse

import cv2
import numpy as np
import pandas as pd
import torch

import config
from dataset import HC18Dataset, patient_level_split
from transforms import get_val_transform
from model import MSUNet
from gradcam import SegmentationGradCAM, overlay_heatmap_on_image
from hc_estimation import estimate_hc_from_mask, draw_contour_overlay, rescale_pixel_size
from metrics import compute_all_metrics


def load_model(checkpoint_path: str, device=config.DEVICE) -> torch.nn.Module:
    model = MSUNet(
        encoder_name=config.ENCODER_NAME,
        encoder_weights=None,  # weights are restored from checkpoint, not ImageNet
        in_channels=config.IN_CHANNELS,
        classes=config.NUM_CLASSES,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def tensor_image_to_uint8_rgb(image_tensor: torch.Tensor) -> np.ndarray:
    """[3, H, W] float tensor in [0,1] -> [H, W, 3] uint8 RGB numpy array."""
    arr = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def run_inference(model, dataset, device=config.DEVICE, save_outputs=True):
    os.makedirs(config.OVERLAY_DIR, exist_ok=True)
    os.makedirs(config.GRADCAM_DIR, exist_ok=True)

    cam_extractor = SegmentationGradCAM(model, target_layer=model.encoder.patch_embed4)

    results = []

    for idx in range(len(dataset)):
        image_tensor, mask_tensor, meta = dataset[idx]
        fname = meta["filename"]

        input_batch = image_tensor.unsqueeze(0).to(device)
        target_batch = mask_tensor.unsqueeze(0).to(device)

        # --- segmentation forward pass ---
        with torch.no_grad():
            logits = model(input_batch)
        probs = torch.sigmoid(logits)
        pred_mask = (probs > 0.5).float()

        batch_metrics = compute_all_metrics(logits, target_batch)

        # --- Grad-CAM needs a gradient-enabled forward+backward pass ---
        cam = cam_extractor(input_batch)

        # --- HC estimation from the predicted mask ---
        pred_mask_np = pred_mask.squeeze().cpu().numpy()
        # CSV pixel_size_mm refers to the ORIGINAL image resolution; rescale
        # to the network's working resolution (config.IMG_SIZE) since that
        # is the resolution our predicted mask lives at.
        effective_pixel_size = rescale_pixel_size(
            meta["pixel_size_mm"], orig_dim=meta["orig_w"], resized_dim=config.IMG_SIZE
        )
        hc_result = estimate_hc_from_mask(pred_mask_np, pixel_size_mm=effective_pixel_size)

        true_hc = meta["true_hc_mm"]
        pred_hc = hc_result["hc_mm"]
        abs_error = abs(pred_hc - true_hc) if not np.isnan(pred_hc) and not np.isnan(true_hc) else float("nan")

        results.append({
            "filename": fname,
            "predicted_hc_mm": pred_hc,
            "true_hc_mm": true_hc,
            "abs_error_mm": abs_error,
            "dice": batch_metrics["dice"],
            "iou": batch_metrics["iou"],
            "precision": batch_metrics["precision"],
            "recall": batch_metrics["recall"],
        })

        if save_outputs:
            rgb_img = tensor_image_to_uint8_rgb(image_tensor)

            overlay = draw_contour_overlay(rgb_img, hc_result["contour"])
            text = f"HC: {pred_hc:.2f} mm" if not np.isnan(pred_hc) else "HC: N/A"
            cv2.putText(overlay, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(config.OVERLAY_DIR, f"{os.path.splitext(fname)[0]}_overlay.png"), overlay_bgr)

            cam_overlay = overlay_heatmap_on_image(rgb_img, cam)
            cam_overlay_bgr = cv2.cvtColor(cam_overlay, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(config.GRADCAM_DIR, f"{os.path.splitext(fname)[0]}_gradcam.png"), cam_overlay_bgr)

    cam_extractor.remove_hooks()

    df = pd.DataFrame(results)
    if save_outputs:
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        df.to_csv(config.RESULTS_CSV, index=False)
        print(f"Saved HC predictions to {config.RESULTS_CSV}")
        valid = df.dropna(subset=["abs_error_mm"])
        if len(valid) > 0:
            print(f"Mean Absolute Error (HC): {valid['abs_error_mm'].mean():.3f} mm")
            print(f"Mean Dice: {df['dice'].mean():.4f}  Mean IoU: {df['iou'].mean():.4f}")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=config.BEST_MODEL_PATH)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"])
    args = parser.parse_args()

    all_files = sorted(
        f for f in os.listdir(config.DATA_DIR)
        if f.endswith(".png") and "Annotation" not in f
    )
    train_files, val_files, test_files = patient_level_split(
        all_files, val_split=config.VAL_SPLIT, test_split=config.TEST_SPLIT, seed=config.RANDOM_SEED
    )
    split_map = {"train": train_files, "val": val_files, "test": test_files, "all": all_files}
    chosen_files = split_map[args.split]

    dataset = HC18Dataset(
        config.DATA_DIR, config.CSV_PATH, chosen_files,
        img_size=config.IMG_SIZE, transform=get_val_transform(config.IMG_SIZE),
    )

    model = load_model(args.checkpoint, device=config.DEVICE)
    run_inference(model, dataset, device=config.DEVICE, save_outputs=True)


if __name__ == "__main__":
    main()
