"""
train.py
Trains U-Net + MiT-B2 on HC18 with hybrid Dice-BCE loss.
Logs Dice, IoU, Precision, Recall, Accuracy, F1, Loss for train + val every
epoch to CSV, saves best checkpoint (by validation Dice), and plots curves.

Usage (on Lightning AI Studio terminal):
    python train.py
"""

import os
import time
import csv

import torch
import torch.optim as optim
from tqdm import tqdm

import config
from dataset import build_dataloaders
from model import build_model
from losses import build_loss
from metrics import MetricAccumulator, EpochMetrics


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_one_epoch(model, loader, criterion, optimizer, device, scaler, train: bool):
    model.train() if train else model.eval()
    accumulator = MetricAccumulator()

    context = torch.enable_grad() if train else torch.no_grad()
    pbar = tqdm(loader, desc="train" if train else "val", leave=False)

    with context:
        for images, masks, _meta in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            if config.AMP_ENABLED:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(images)
                    loss = criterion(logits, masks)
            else:
                logits = model(images)
                loss = criterion(logits, masks)

            if train:
                if config.AMP_ENABLED:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
                    optimizer.step()

            # Metrics always computed in fp32 for numerical stability
            accumulator.update(logits.detach().float(), masks.detach().float(), loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    return accumulator.compute()


def write_csv_header(path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss", "train_dice", "train_iou", "train_precision", "train_recall", "train_accuracy", "train_f1",
            "val_loss", "val_dice", "val_iou", "val_precision", "val_recall", "val_accuracy", "val_f1",
            "lr", "epoch_time_sec",
        ])


def append_csv_row(path, epoch, train_m: EpochMetrics, val_m: EpochMetrics, lr, epoch_time):
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            train_m.loss, train_m.dice, train_m.iou, train_m.precision, train_m.recall, train_m.accuracy, train_m.f1,
            val_m.loss, val_m.dice, val_m.iou, val_m.precision, val_m.recall, val_m.accuracy, val_m.f1,
            lr, epoch_time,
        ])


def main():
    set_seed(config.RANDOM_SEED)
    device = config.DEVICE
    print(f"[Setup] device={device}  amp={config.AMP_ENABLED}")

    train_loader, val_loader, _test_loader = build_dataloaders()

    model = build_model().to(device)
    criterion = build_loss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=config.SCHEDULER_FACTOR,
        patience=config.SCHEDULER_PATIENCE
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config.AMP_ENABLED)

    csv_path = os.path.join(config.LOG_DIR, "training_log.csv")
    write_csv_header(csv_path)

    best_val_dice = -1.0
    epochs_no_improve = 0
    best_ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    last_ckpt_path = os.path.join(config.CHECKPOINT_DIR, "last_model.pth")

    for epoch in range(1, config.EPOCHS + 1):
        t0 = time.time()

        train_metrics = run_one_epoch(model, train_loader, criterion, optimizer, device, scaler, train=True)
        val_metrics = run_one_epoch(model, val_loader, criterion, optimizer, device, scaler, train=False)

        scheduler.step(val_metrics.dice)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - t0

        print(
            f"[Epoch {epoch:03d}/{config.EPOCHS}] "
            f"train_loss={train_metrics.loss:.4f} train_dice={train_metrics.dice:.4f} train_iou={train_metrics.iou:.4f} | "
            f"val_loss={val_metrics.loss:.4f} val_dice={val_metrics.dice:.4f} val_iou={val_metrics.iou:.4f} "
            f"val_precision={val_metrics.precision:.4f} val_recall={val_metrics.recall:.4f} "
            f"val_acc={val_metrics.accuracy:.4f} val_f1={val_metrics.f1:.4f} | "
            f"lr={current_lr:.2e} time={epoch_time:.1f}s"
        )

        append_csv_row(csv_path, epoch, train_metrics, val_metrics, current_lr, epoch_time)

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_dice": val_metrics.dice,
        }, last_ckpt_path)

        if val_metrics.dice > best_val_dice:
            best_val_dice = val_metrics.dice
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_dice": val_metrics.dice,
                "val_iou": val_metrics.iou,
                "val_precision": val_metrics.precision,
                "val_recall": val_metrics.recall,
                "val_accuracy": val_metrics.accuracy,
                "val_f1": val_metrics.f1,
            }, best_ckpt_path)
            print(f"  -> New best model saved (val_dice={best_val_dice:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
                print(f"[EarlyStopping] No improvement for {config.EARLY_STOPPING_PATIENCE} epochs. Stopping.")
                break

    print(f"[Done] Best validation Dice: {best_val_dice:.4f}")
    print(f"Best checkpoint: {best_ckpt_path}")
    print(f"Training log CSV: {csv_path}")


if __name__ == "__main__":
    main()
