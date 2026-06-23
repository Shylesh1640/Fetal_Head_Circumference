"""
plot_curves.py
Plots train/val curves for every metric in outputs/logs/training_log.csv.

Usage:
    python src/plot_curves.py
"""

import os
import sys

import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config as cfg


def main():
    log_path = os.path.join(cfg.LOG_DIR, "training_log.csv")
    df = pd.read_csv(log_path)

    metrics = ["Loss", "Dice", "IoU", "Precision", "Recall", "Accuracy", "F1"]
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for split, style in [("train", "-"), ("val", "--")]:
            sub = df[df["split"] == split]
            ax.plot(sub["epoch"], sub[metric], style, label=split)
        ax.set_title(metric)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    axes[-1].axis("off")
    plt.tight_layout()
    out_path = os.path.join(cfg.LOG_DIR, "training_curves.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved training curves to: {out_path}")


if __name__ == "__main__":
    main()
