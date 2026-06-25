"""
Train one model and save the best checkpoint by validation Dice.

Usage:
    python train.py --model attention_unet --data_dir ./data/training_set --epochs 100

Run this once per entry in models.MODEL_NAMES to fill out the comparison
table (same idea as the paper's Table I/II).
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader

from dataset import HC18Dataset, build_splits, get_train_transform
from models import build_model
from losses import DiceBCELoss
from metrics import segmentation_metrics


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, total_dice = 0.0, 0.0
    n = 0
    with torch.set_grad_enabled(train):
        for images, masks, _ in loader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            loss = criterion(preds, masks)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                m = segmentation_metrics(preds.cpu().numpy(), masks.cpu().numpy())
            total_loss += loss.item() * images.size(0)
            total_dice += m["dice"] * images.size(0)
            n += images.size(0)
    return total_loss / n, total_dice / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=[
        "unet_baseline", "attention_unet", "dilated_unet",
        "dense_unet", "ms_unet", "transformer_unet", "segformer"])
    parser.add_argument("--data_dir", default="./data/training_set")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--out_dir", default="./checkpoints")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    train_pairs, val_pairs, _ = build_splits(args.data_dir)
    train_ds = HC18Dataset(train_pairs, transform=get_train_transform())
    val_ds = HC18Dataset(val_pairs)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model(args.model).to(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5)

    best_dice, no_improve = 0.0, 0
    ckpt_path = os.path.join(args.out_dir, f"{args.model}_best.pth")

    for epoch in range(args.epochs):
        train_loss, train_dice = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_dice = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_dice)

        print(f"[{args.model}] epoch {epoch+1}/{args.epochs} "
              f"train_loss={train_loss:.4f} train_dice={train_dice:.4f} "
              f"val_loss={val_loss:.4f} val_dice={val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch+1}. Best val Dice: {best_dice:.4f}")
                break

    print(f"Saved best checkpoint to {ckpt_path} (val Dice {best_dice:.4f})")


if __name__ == "__main__":
    main()
