"""
train.py
End-to-end training script for U-Net + MiT-B2 fetal-head segmentation.

Logs Dice, IoU, Precision, Recall, Accuracy, F1, and Loss for every epoch,
for BOTH the train and validation splits, to a CSV file. After training
completes (or early-stops), it loads the best checkpoint (highest val Dice)
and reports final metrics on the held-out TEST split.

Usage:
    python src/train.py
"""

import os
import sys
import time
import copy
import csv

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from dataset import HC18Dataset, get_train_transforms, get_eval_transforms
from model import build_model, count_parameters
from losses import HybridDiceBCELoss
from metrics import SegmentationMetricAccumulator


def set_seed(seed: int = cfg.SEED):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, device, train: bool, scaler=None):
    model.train() if train else model.eval()
    acc = SegmentationMetricAccumulator()

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        pbar = tqdm(loader, desc="train" if train else "eval", leave=False)
        for images, masks, _ in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            if cfg.USE_AMP and device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(images)
                    loss = criterion(logits, masks)
            else:
                logits = model(images)
                loss = criterion(logits, masks)

            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_NORM)
                    optimizer.step()

            acc.update(logits.detach().float(), masks.detach().float(), loss_value=loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return acc.compute()


def init_csv_logger(path, fieldnames):
    f = open(path, "w", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    return f, writer


def main():
    set_seed(cfg.SEED)
    device = cfg.DEVICE
    print(f"Using device: {device}")

    train_ds = HC18Dataset("train", transforms=get_train_transforms())
    val_ds = HC18Dataset("val", transforms=get_eval_transforms())
    test_ds = HC18Dataset("test", transforms=get_eval_transforms())

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                               num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                             num_workers=cfg.NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                              num_workers=cfg.NUM_WORKERS, pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    model = build_model().to(device)
    print(f"Model: U-Net + {cfg.ENCODER_NAME} | Trainable params: {count_parameters(model):,}")

    criterion = HybridDiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg.LR_SCHEDULER_FACTOR, patience=cfg.LR_SCHEDULER_PATIENCE
    )
    scaler = torch.cuda.amp.GradScaler() if (cfg.USE_AMP and device == "cuda") else None

    fieldnames = ["epoch", "split", "Dice", "IoU", "Precision", "Recall", "Accuracy", "F1", "Loss", "lr"]
    log_path = os.path.join(cfg.LOG_DIR, "training_log.csv")
    log_file, log_writer = init_csv_logger(log_path, fieldnames)

    best_val_dice = -1.0
    best_state_dict = None
    epochs_without_improvement = 0
    best_ckpt_path = os.path.join(cfg.CKPT_DIR, "best_model.pth")

    print(f"\n{'='*90}\nTraining for up to {cfg.EPOCHS} epochs "
          f"(early stopping patience={cfg.EARLY_STOPPING_PATIENCE})\n{'='*90}\n")

    for epoch in range(1, cfg.EPOCHS + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, train=True, scaler=scaler)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        scheduler.step(val_metrics["Dice"])

        for split_name, m in [("train", train_metrics), ("val", val_metrics)]:
            row = {"epoch": epoch, "split": split_name, "lr": current_lr, **m}
            log_writer.writerow(row)
        log_file.flush()

        dt = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{cfg.EPOCHS} ({dt:.1f}s) | lr={current_lr:.2e}\n"
            f"  Train | Loss={train_metrics['Loss']:.4f} Dice={train_metrics['Dice']:.4f} "
            f"IoU={train_metrics['IoU']:.4f} Prec={train_metrics['Precision']:.4f} "
            f"Rec={train_metrics['Recall']:.4f} Acc={train_metrics['Accuracy']:.4f} F1={train_metrics['F1']:.4f}\n"
            f"  Val   | Loss={val_metrics['Loss']:.4f} Dice={val_metrics['Dice']:.4f} "
            f"IoU={val_metrics['IoU']:.4f} Prec={val_metrics['Precision']:.4f} "
            f"Rec={val_metrics['Recall']:.4f} Acc={val_metrics['Accuracy']:.4f} F1={val_metrics['F1']:.4f}"
        )

        if val_metrics["Dice"] > best_val_dice:
            best_val_dice = val_metrics["Dice"]
            best_state_dict = copy.deepcopy(model.state_dict())
            torch.save({
                "epoch": epoch,
                "model_state_dict": best_state_dict,
                "val_metrics": val_metrics,
                "config": {
                    "encoder_name": cfg.ENCODER_NAME,
                    "image_size": cfg.IMAGE_SIZE,
                },
            }, best_ckpt_path)
            epochs_without_improvement = 0
            print(f"  -> New best model saved (val Dice={best_val_dice:.4f}) at {best_ckpt_path}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= cfg.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch} epochs "
                  f"(no val Dice improvement for {cfg.EARLY_STOPPING_PATIENCE} epochs).")
            break

    log_file.close()

    # ------------------------------------------------------------------
    # Final evaluation on the held-out TEST split using the BEST checkpoint
    # ------------------------------------------------------------------
    print(f"\n{'='*90}\nFinal evaluation on TEST split using best checkpoint "
          f"(val Dice={best_val_dice:.4f})\n{'='*90}")

    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = run_epoch(model, test_loader, criterion, optimizer, device, train=False)

    print("\nTEST SET RESULTS:")
    for k in ["Dice", "IoU", "Precision", "Recall", "Accuracy", "F1", "Loss"]:
        print(f"  {k:10s}: {test_metrics[k]:.4f}")

    test_log_path = os.path.join(cfg.LOG_DIR, "test_results.csv")
    with open(test_log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Dice", "IoU", "Precision", "Recall", "Accuracy", "F1", "Loss"])
        writer.writeheader()
        writer.writerow(test_metrics)
    print(f"\nSaved test metrics to: {test_log_path}")
    print(f"Best checkpoint: {best_ckpt_path}")
    print(f"Full per-epoch train/val log: {log_path}")


if __name__ == "__main__":
    main()
