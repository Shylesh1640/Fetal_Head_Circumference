"""
Step 2 of the plan: combine your top-2 models from leaderboard.csv.

Two strategies, run both and compare:

  A) WEIGHTED OUTPUT AVERAGING (zero retraining, ~5 min to test)
     final_mask = w * sigmoid_A + (1-w) * sigmoid_B
     Quick way to see if fusion helps at all before investing in (B).

  B) FEATURE-LEVEL FUSION (models.DualEncoderFusion, needs ~20-30 epochs
     of fine-tuning on top of the two frozen/unfrozen encoders)
     Generally stronger because the two encoders' representations are
     merged before decoding, not just their final probability maps.

Usage (strategy A):
    python ensemble.py --data_dir ./data/training_set \
        --model_a transformer_unet --model_b attention_unet \
        --ckpt_dir ./checkpoints --weight_a 0.6
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import HC18Dataset, build_splits
from models import build_model
from metrics import segmentation_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/training_set")
    parser.add_argument("--model_a", required=True)
    parser.add_argument("--model_b", required=True)
    parser.add_argument("--ckpt_dir", default="./checkpoints")
    parser.add_argument("--weight_a", type=float, default=0.5,
                         help="0.5 = simple average; tune on val set first")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, test_pairs = build_splits(args.data_dir)
    test_ds = HC18Dataset(test_pairs)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model_a = build_model(args.model_a).to(device)
    model_a.load_state_dict(torch.load(f"{args.ckpt_dir}/{args.model_a}_best.pth", map_location=device))
    model_a.eval()

    model_b = build_model(args.model_b).to(device)
    model_b.load_state_dict(torch.load(f"{args.ckpt_dir}/{args.model_b}_best.pth", map_location=device))
    model_b.eval()

    agg = {"acc": [], "precision": [], "recall": [], "f1": [], "iou": [], "dice": []}
    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(device)
            pred_a = model_a(images).cpu().numpy()
            pred_b = model_b(images).cpu().numpy()
            fused = args.weight_a * pred_a + (1 - args.weight_a) * pred_b
            masks_np = masks.numpy()
            for i in range(fused.shape[0]):
                m = segmentation_metrics(fused[i], masks_np[i])
                for k in agg:
                    agg[k].append(m[k])

    print(f"Ensemble ({args.model_a} w={args.weight_a:.2f} + "
          f"{args.model_b} w={1-args.weight_a:.2f}) on test set:")
    for k, v in agg.items():
        print(f"  {k}: {np.mean(v):.4f}")


if __name__ == "__main__":
    main()
