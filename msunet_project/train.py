"""
train.py
--------
Main training loop for MS-UNet (U-Net + MiT-B2) on the HC18 dataset.

Run with:
    python train.py

Make sure you have edited config.py to point DATA_DIR / CSV_PATH at your
local copy of the HC18 dataset first (see README.md).
"""

import os
import time
import csv

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import config
from dataset import HC18Dataset, patient_level_split
from transforms import get_train_transform, get_val_transform
from model import MSUNet
from losses import HybridDiceBCELoss
from metrics import compute_all_metrics, MetricTracker


def get_dataloaders():
    all_files = sorted(
        f for f in os.listdir(config.DATA_DIR)
        if f.endswith(".png") and "Annotation" not in f
    )
    if len(all_files) == 0:
        raise RuntimeError(
            f"No training images found in {config.DATA_DIR}. "
            f"Expected files like '000_HC.png'. Check config.DATA_DIR."
        )

    train_files, val_files, test_files = patient_level_split(
        all_files, val_split=config.VAL_SPLIT, test_split=config.TEST_SPLIT, seed=config.RANDOM_SEED
    )
    print(f"Patient-level split -> train: {len(train_files)}, val: {len(val_files)}, test: {len(test_files)} images")

    train_ds = HC18Dataset(
        config.DATA_DIR, config.CSV_PATH, train_files,
        img_size=config.IMG_SIZE, transform=get_train_transform(config.IMG_SIZE),
    )
    val_ds = HC18Dataset(
        config.DATA_DIR, config.CSV_PATH, val_files,
        img_size=config.IMG_SIZE, transform=get_val_transform(config.IMG_SIZE),
    )
    test_ds = HC18Dataset(
        config.DATA_DIR, config.CSV_PATH, test_files,
        img_size=config.IMG_SIZE, transform=get_val_transform(config.IMG_SIZE),
    )

    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def run_epoch(model, loader, criterion, optimizer=None, device=config.DEVICE, train_mode=True):
    model.train() if train_mode else model.eval()
    tracker = MetricTracker()
    running_loss = 0.0
    n_batches = 0

    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for images, masks, _meta in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            if train_mode:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, masks)

            if train_mode:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
                optimizer.step()

            batch_metrics = compute_all_metrics(logits.detach(), masks)
            tracker.update(batch_metrics, batch_size=images.size(0))

            running_loss += loss.item()
            n_batches += 1

    avg_loss = running_loss / max(n_batches, 1)
    avg_metrics = tracker.average()
    avg_metrics["loss"] = avg_loss
    return avg_metrics


def main():
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print(f"Device: {config.DEVICE}")
    train_loader, val_loader, test_loader = get_dataloaders()

    model = MSUNet(
        encoder_name=config.ENCODER_NAME,
        encoder_weights=config.ENCODER_WEIGHTS,
        in_channels=config.IN_CHANNELS,
        classes=config.NUM_CLASSES,
    ).to(config.DEVICE)

    criterion = HybridDiceBCELoss()
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=config.LR_SCHEDULER_FACTOR,
        patience=config.LR_SCHEDULER_PATIENCE,
    )

    best_val_loss = float("inf")
    patience_counter = 0

    log_rows = []
    log_fields = [
        "epoch", "train_loss", "train_dice", "train_iou",
        "val_loss", "val_dice", "val_iou", "val_acc", "val_precision",
        "val_recall", "val_f1", "lr", "epoch_time_s",
    ]

    for epoch in range(1, config.NUM_EPOCHS + 1):
        t0 = time.time()

        train_metrics = run_epoch(model, train_loader, criterion, optimizer, config.DEVICE, train_mode=True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer=None, device=config.DEVICE, train_mode=False)

        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{config.NUM_EPOCHS} | "
            f"train_loss={train_metrics['loss']:.4f} dice={train_metrics['dice']:.4f} iou={train_metrics['iou']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} dice={val_metrics['dice']:.4f} iou={val_metrics['iou']:.4f} "
            f"acc={val_metrics['accuracy']:.4f} prec={val_metrics['precision']:.4f} "
            f"rec={val_metrics['recall']:.4f} f1={val_metrics['f1']:.4f} | "
            f"lr={current_lr:.2e} | {epoch_time:.1f}s"
        )

        log_rows.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"], "train_dice": train_metrics["dice"], "train_iou": train_metrics["iou"],
            "val_loss": val_metrics["loss"], "val_dice": val_metrics["dice"], "val_iou": val_metrics["iou"],
            "val_acc": val_metrics["accuracy"], "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"], "val_f1": val_metrics["f1"],
            "lr": current_lr, "epoch_time_s": epoch_time,
        })

        torch.save(
            {"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()},
            config.LAST_MODEL_PATH,
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(), "val_metrics": val_metrics},
                config.BEST_MODEL_PATH,
            )
            print(f"  -> New best model saved (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch} (no improvement for {config.EARLY_STOPPING_PATIENCE} epochs).")
                break

    with open(config.METRICS_LOG_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"Training log saved to {config.METRICS_LOG_CSV}")

    print("\nEvaluating best model on held-out test set...")
    checkpoint = torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = run_epoch(model, test_loader, criterion, optimizer=None, device=config.DEVICE, train_mode=False)
    print("Test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k:>18}: {v:.4f}")


if __name__ == "__main__":
    main()
