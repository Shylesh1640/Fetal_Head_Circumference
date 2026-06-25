"""
Evaluate one or more trained checkpoints on the held-out test set and
write a leaderboard CSV (matches the 11-metric style of the paper's
Table I/II, plus HC MAE/MSE if you supply pixel-size CSV).

Usage:
    python evaluate.py --data_dir ./data/training_set \
        --models unet_baseline attention_unet dilated_unet dense_unet ms_unet transformer_unet segformer \
        --ckpt_dir ./checkpoints --out ./results/leaderboard.csv
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import HC18Dataset, build_splits
from models import build_model
from metrics import segmentation_metrics


def evaluate_model(model, loader, device):
    model.eval()
    agg = {"acc": [], "precision": [], "recall": [], "f1": [], "iou": [], "dice": []}
    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(device)
            preds = model(images).cpu().numpy()
            masks_np = masks.numpy()
            for i in range(preds.shape[0]):
                m = segmentation_metrics(preds[i], masks_np[i])
                for k in agg:
                    agg[k].append(m[k])
    return {k: float(np.mean(v)) for k, v in agg.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/training_set")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--ckpt_dir", default="./checkpoints")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--out", default="./results/leaderboard.csv")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, test_pairs = build_splits(args.data_dir)
    test_ds = HC18Dataset(test_pairs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    rows = []
    for name in args.models:
        ckpt_path = os.path.join(args.ckpt_dir, f"{name}_best.pth")
        if not os.path.exists(ckpt_path):
            print(f"Skipping {name}: no checkpoint at {ckpt_path}")
            continue
        model = build_model(name).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        metrics = evaluate_model(model, test_loader, device)
        metrics["model"] = name
        rows.append(metrics)
        print(name, metrics)

    df = pd.DataFrame(rows).sort_values("dice", ascending=False)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nLeaderboard saved to {args.out}\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
