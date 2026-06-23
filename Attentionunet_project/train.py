"""
train.py — End-to-end training script for Attention U-Net on HC18.

Tracks Dice, IoU, Precision, Recall, Accuracy, F1, and Loss for
train / validation / test splits every epoch, saves the best checkpoint
(by validation Dice), and writes a CSV log + final test report.

Usage (on Lightning AI Studio terminal):
    python train.py --data_root /teamspace/studios/this_studio/hc18/training_set \
                     --epochs 100 --batch_size 16 --img_size 256 --lr 1e-4

See README.md for full setup instructions.
"""

import os
import csv
import argparse
import time

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import AttU_Net
from losses import DiceBCELoss
from metrics import SegmentationMetrics
from dataset import HC18Dataset, patient_level_split, get_train_transforms, get_eval_transforms


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True,
                    help="Path to HC18 training_set folder (contains *_HC.png and *_HC_Annotation.png)")
    p.add_argument("--output_dir", type=str, default="./outputs")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=15,
                    help="Early stopping patience (epochs without val Dice improvement)")
    p.add_argument("--base_ch", type=int, default=64, help="Base channel width of AttU_Net")
    return p.parse_args()


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, metrics_tracker, device, optimizer=None):
    """One pass over `loader`. If optimizer is provided, runs in training mode."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    metrics_tracker.reset()

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, masks, _ in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, masks)

            if is_train:
                loss.backward()
                optimizer.step()

            metrics_tracker.update(logits.detach(), masks.detach(), loss_value=loss.item())

    return metrics_tracker.compute()


def log_row(csv_path, row, header):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def fmt(metrics: dict) -> str:
    return (f"Loss {metrics['Loss']:.4f} | Dice {metrics['Dice']:.4f} | "
            f"IoU {metrics['IoU']:.4f} | Prec {metrics['Precision']:.4f} | "
            f"Rec {metrics['Recall']:.4f} | Acc {metrics['Accuracy']:.4f} | "
            f"F1 {metrics['F1']:.4f}")


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Data ----
    train_files, val_files, test_files = patient_level_split(args.data_root, seed=args.seed)
    print(f"Patient-level split -> train: {len(train_files)} | val: {len(val_files)} | test: {len(test_files)} images")

    train_ds = HC18Dataset(args.data_root, train_files, transform=get_train_transforms(args.img_size))
    val_ds = HC18Dataset(args.data_root, val_files, transform=get_eval_transforms(args.img_size))
    test_ds = HC18Dataset(args.data_root, test_files, transform=get_eval_transforms(args.img_size))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # ---- Model / Loss / Optimizer ----
    model = AttU_Net(img_ch=1, output_ch=1, base_ch=args.base_ch).to(device)
    criterion = DiceBCELoss()
    optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    train_tracker = SegmentationMetrics(device=device)
    val_tracker = SegmentationMetrics(device=device)
    test_tracker = SegmentationMetrics(device=device)

    csv_path = os.path.join(args.output_dir, "training_log.csv")
    header = ["epoch", "split", "Loss", "Dice", "IoU", "Precision", "Recall", "Accuracy", "F1", "lr", "time_sec"]

    best_val_dice = -1.0
    epochs_no_improve = 0
    best_ckpt_path = os.path.join(ckpt_dir, "best_model.pth")

    print("\n" + "=" * 90)
    print("Starting training: Attention U-Net on HC18")
    print("=" * 90)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = run_epoch(model, train_loader, criterion, train_tracker, device, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, val_tracker, device, optimizer=None)

        scheduler.step(val_metrics["Dice"])
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(f"\nEpoch {epoch:3d}/{args.epochs} ({elapsed:.1f}s, lr={current_lr:.2e})")
        print(f"  Train | {fmt(train_metrics)}")
        print(f"  Val   | {fmt(val_metrics)}")

        log_row(csv_path, {"epoch": epoch, "split": "train", "lr": current_lr,
                            "time_sec": round(elapsed, 2), **train_metrics}, header)
        log_row(csv_path, {"epoch": epoch, "split": "val", "lr": current_lr,
                            "time_sec": round(elapsed, 2), **val_metrics}, header)

        if val_metrics["Dice"] > best_val_dice:
            best_val_dice = val_metrics["Dice"]
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": best_val_dice,
                "args": vars(args),
            }, best_ckpt_path)
            print(f"  -> New best model saved (val Dice={best_val_dice:.4f}) -> {best_ckpt_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"\nEarly stopping triggered after {epoch} epochs "
                      f"(no val Dice improvement for {args.patience} epochs).")
                break

    # ---- Final test evaluation using BEST checkpoint ----
    print("\n" + "=" * 90)
    print("Loading best checkpoint for final test-set evaluation")
    print("=" * 90)
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded best model from epoch {checkpoint['epoch']} (val Dice={checkpoint['val_dice']:.4f})")

    test_metrics = run_epoch(model, test_loader, criterion, test_tracker, device, optimizer=None)
    print(f"\n  Test  | {fmt(test_metrics)}")

    log_row(csv_path, {"epoch": "final", "split": "test", "lr": "-",
                        "time_sec": "-", **test_metrics}, header)

    report_path = os.path.join(args.output_dir, "final_report.txt")
    with open(report_path, "w") as f:
        f.write("Attention U-Net on HC18 — Final Results (best checkpoint)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Best epoch: {checkpoint['epoch']}\n\n")
        f.write(f"VALIDATION METRICS (at best epoch):\n  {fmt(val_metrics)}\n\n")
        f.write(f"TEST METRICS:\n  {fmt(test_metrics)}\n")

    print(f"\nFull metric log: {csv_path}")
    print(f"Final report:    {report_path}")
    print(f"Best checkpoint: {best_ckpt_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
