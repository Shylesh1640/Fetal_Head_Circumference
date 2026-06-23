"""
evaluate.py
Standalone evaluation script: loads a trained checkpoint, runs it over a
chosen split (default: test), reports segmentation metrics (Dice, IoU,
Precision, Recall, Accuracy, F1, Loss), AND the downstream head-circumference
error (HC MAE / HC MSE in mm), matching the paper's full evaluation suite.

Usage:
    python src/evaluate.py --split test --ckpt outputs/checkpoints/best_model.pth
"""

import os
import sys
import argparse

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from dataset import HC18Dataset, get_eval_transforms
from model import build_model
from losses import HybridDiceBCELoss
from metrics import SegmentationMetricAccumulator
from hc_postprocess import compute_head_circumference_mm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--ckpt", default=os.path.join(cfg.CKPT_DIR, "best_model.pth"))
    args = parser.parse_args()

    device = cfg.DEVICE
    ds = HC18Dataset(args.split, transforms=get_eval_transforms())
    loader = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)

    model = build_model().to(device)
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = HybridDiceBCELoss()
    acc = SegmentationMetricAccumulator()

    splits_df = pd.read_csv(cfg.SPLITS_CSV)
    splits_df = splits_df.set_index("case_id")

    hc_abs_errors = []

    with torch.no_grad():
        for images, masks, case_ids in tqdm(loader, desc=f"Evaluating [{args.split}]"):
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks)
            acc.update(logits, masks, loss_value=loss.item())

            probs = torch.sigmoid(logits).cpu().numpy()
            for i, case_id in enumerate(case_ids):
                pred_mask = (probs[i, 0] > cfg.THRESHOLD).astype(np.uint8) * 255
                row = splits_df.loc[case_id]
                pixel_size_mm = float(row["pixel_size_mm"])
                gt_hc_mm = float(row["hc_mm_groundtruth"])

                result = compute_head_circumference_mm(pred_mask, pixel_size_mm)
                if result is not None:
                    hc_abs_errors.append(abs(result["hc_mm"] - gt_hc_mm))

    metrics = acc.compute()
    print(f"\n{'='*60}\n{args.split.upper()} SET — SEGMENTATION METRICS\n{'='*60}")
    for k in ["Dice", "IoU", "Precision", "Recall", "Accuracy", "F1", "Loss"]:
        print(f"  {k:10s}: {metrics[k]:.4f}")

    if hc_abs_errors:
        hc_mae = float(np.mean(hc_abs_errors))
        hc_mse = float(np.mean(np.square(hc_abs_errors)))
        print(f"\n{'='*60}\n{args.split.upper()} SET — HEAD CIRCUMFERENCE ERROR\n{'='*60}")
        print(f"  HC MAE (mm): {hc_mae:.3f}")
        print(f"  HC MSE (mm^2): {hc_mse:.3f}")
    else:
        print("\nNo valid ellipse fits obtained — check mask quality.")


if __name__ == "__main__":
    main()
