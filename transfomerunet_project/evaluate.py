"""
evaluate.py
Loads the best checkpoint and reports final Dice/IoU/Precision/Recall/
Accuracy/F1/Loss on the held-out TEST split. Also plots training curves
from the CSV log produced by train.py.

Usage:
    python evaluate.py
"""

import os
import json

import torch
import pandas as pd
import matplotlib.pyplot as plt

import config
from dataset import build_dataloaders
from model import build_model
from losses import build_loss
from metrics import MetricAccumulator
from train import run_one_epoch  # reuse the exact same eval-mode loop


def evaluate_test_set():
    device = config.DEVICE
    _train_loader, _val_loader, test_loader = build_dataloaders()

    model = build_model().to(device)
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}. Run train.py first.")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[Eval] Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(val_dice={checkpoint['val_dice']:.4f})")

    criterion = build_loss()

    # run_one_epoch with train=False does a clean no-grad evaluation pass;
    # optimizer/scaler args are unused when train=False but required by signature.
    dummy_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=config.AMP_ENABLED)

    test_metrics = run_one_epoch(model, test_loader, criterion, dummy_optimizer, device, scaler, train=False)

    print("\n========== TEST SET RESULTS ==========")
    print(f"Loss      : {test_metrics.loss:.4f}")
    print(f"Dice Score: {test_metrics.dice:.4f}")
    print(f"IoU       : {test_metrics.iou:.4f}")
    print(f"Precision : {test_metrics.precision:.4f}")
    print(f"Recall    : {test_metrics.recall:.4f}")
    print(f"Accuracy  : {test_metrics.accuracy:.4f}")
    print(f"F1 Score  : {test_metrics.f1:.4f}")
    print("=======================================\n")

    results_path = os.path.join(config.LOG_DIR, "test_results.json")
    with open(results_path, "w") as f:
        json.dump(test_metrics.as_dict(), f, indent=2)
    print(f"Saved test metrics to {results_path}")

    return test_metrics


def plot_training_curves():
    csv_path = os.path.join(config.LOG_DIR, "training_log.csv")
    if not os.path.exists(csv_path):
        print(f"[Plot] No training log found at {csv_path}, skipping plots.")
        return

    df = pd.read_csv(csv_path)

    metric_pairs = [
        ("train_loss", "val_loss", "Loss"),
        ("train_dice", "val_dice", "Dice Score"),
        ("train_iou", "val_iou", "IoU"),
        ("train_precision", "val_precision", "Precision"),
        ("train_recall", "val_recall", "Recall"),
        ("train_accuracy", "val_accuracy", "Accuracy"),
        ("train_f1", "val_f1", "F1 Score"),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(14, 18))
    axes = axes.flatten()

    for i, (train_col, val_col, title) in enumerate(metric_pairs):
        ax = axes[i]
        ax.plot(df["epoch"], df[train_col], label="Train", linewidth=1.8)
        ax.plot(df["epoch"], df[val_col], label="Validation", linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(alpha=0.3)

    # Hide the unused 8th subplot
    axes[-1].axis("off")

    plt.tight_layout()
    out_path = os.path.join(config.PLOT_DIR, "training_curves.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Plot] Saved training curves to {out_path}")


if __name__ == "__main__":
    plot_training_curves()
    evaluate_test_set()
