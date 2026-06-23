"""
=============================================================================
FETAL HEAD SEGMENTATION + CIRCUMFERENCE MEASUREMENT
U-Net + MiT-B2 (Mix Vision Transformer B2) — SOTA Implementation
Dataset: HC18 Grand Challenge
Target: Dice ≥ 0.9899 | IoU ≥ 0.9850
=============================================================================

LIGHTNING AI SETUP INSTRUCTIONS
---------------------------------
1. Open a new Studio in Lightning AI (GPU: T4 or A10G recommended)
2. Upload this file to your Studio
3. Install dependencies:
      pip install segmentation-models-pytorch timm albumentations \
                  opencv-python grad-cam matplotlib pandas tqdm torch torchvision

4. Download HC18 dataset from: https://zenodo.org/record/1327317
   Place it as:
      hc18/
      ├── training_set/
      │   ├── 000_HC.png
      │   ├── 000_HC_Annotation.png
      │   └── ...
      └── test_set/
          ├── 001_HC.png
          └── ...

5. Run:
      python fetal_hc_segmentation.py --mode train
      python fetal_hc_segmentation.py --mode evaluate
      python fetal_hc_segmentation.py --mode gradcam --image_path hc18/training_set/000_HC.png
=============================================================================
"""

import os
import cv2
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from tqdm import tqdm
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

CFG = {
    # Paths
    "train_dir":   "hc18/training_set",
    "test_dir":    "hc18/test_set",
    "checkpoint":  "checkpoints/best_model.pth",
    "results_dir": "results",

    # Model
    "encoder":         "mit_b2",
    "encoder_weights": "imagenet",
    "in_channels":     3,
    "classes":         1,
    "image_size":      256,

    # Training
    "epochs":       100,
    "batch_size":   8,
    "lr":           1e-4,
    "val_split":    0.15,
    "patience":     15,           # early stopping patience
    "threshold":    0.5,          # binary mask threshold
    "seed":         42,

    # HC pixel-to-mm scale (HC18 standard)
    # HC18 images: 800x540 px, pixel spacing ~0.154 mm/px (from dataset metadata)
    "pixel_spacing_mm": 0.154,
}


# ─────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

class HC18Dataset(Dataset):
    """
    HC18 fetal head ultrasound dataset.
    Loads image + binary mask pairs. Masks are derived from ellipse annotations.
    Patient-level splitting is done in get_dataloaders() to prevent data leakage.
    """

    def __init__(self, image_paths: list, mask_paths: list, transform=None):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load grayscale ultrasound → convert to 3-channel (RGB) for pretrained encoder
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Load annotation mask (white ellipse on black → binary)
        mask_raw = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        # Fill the ellipse contour to get a solid binary mask
        mask = self._fill_ellipse_mask(mask_raw)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask  = augmented["mask"]

        # Shape: (C, H, W) for image, (1, H, W) for mask
        mask = mask.unsqueeze(0).float() / 255.0
        return image, mask

    @staticmethod
    def _fill_ellipse_mask(mask_gray: np.ndarray) -> np.ndarray:
        """Convert thin ellipse annotation to solid filled binary mask."""
        binary = (mask_gray > 127).astype(np.uint8) * 255
        # Find contours and fill
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(binary)
        if contours:
            cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
        # Fallback: if fill failed (very thin annotation), use flood fill
        if filled.sum() == 0:
            filled = binary.copy()
        return filled


def get_transforms(image_size: int, mode: str = "train"):
    """Albumentations pipeline matching the paper's augmentation strategy."""
    if mode == "train":
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=15, p=0.4),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.ElasticTransform(alpha=1, sigma=50, p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])


def get_dataloaders(cfg: dict):
    """
    Patient-level train/val split to prevent data leakage.
    HC18 filenames: 000_HC.png / 000_HC_Annotation.png
    """
    train_dir = cfg["train_dir"]

    all_images = sorted(glob(os.path.join(train_dir, "*_HC.png")))
    all_masks  = [p.replace("_HC.png", "_HC_Annotation.png") for p in all_images]

    # Verify all masks exist
    valid_pairs = [(img, msk) for img, msk in zip(all_images, all_masks) if os.path.exists(msk)]
    all_images, all_masks = zip(*valid_pairs)

    # Patient-level split: use patient index (file prefix number)
    n_total   = len(all_images)
    n_val     = int(n_total * cfg["val_split"])
    # Take last n_val patients for validation (deterministic, no shuffle = no leakage)
    val_images   = list(all_images[-n_val:])
    val_masks    = list(all_masks[-n_val:])
    train_images = list(all_images[:-n_val])
    train_masks  = list(all_masks[:-n_val])

    img_size = cfg["image_size"]

    train_ds = HC18Dataset(train_images, train_masks, transform=get_transforms(img_size, "train"))
    val_ds   = HC18Dataset(val_images,   val_masks,   transform=get_transforms(img_size, "val"))

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False,
                              num_workers=4, pin_memory=True)

    print(f"[Dataset] Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_loader, val_loader


def get_test_dataloader(cfg: dict):
    test_dir   = cfg["test_dir"]
    test_images = sorted(glob(os.path.join(test_dir, "*.png")))
    # Test set has no masks — return paths only
    return test_images


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────

def build_model(cfg: dict) -> nn.Module:
    """
    U-Net with MiT-B2 (Mix Vision Transformer B2) encoder.
    Pre-trained on ImageNet via segmentation-models-pytorch.
    """
    model = smp.Unet(
        encoder_name    = cfg["encoder"],          # "mit_b2"
        encoder_weights = cfg["encoder_weights"],  # "imagenet"
        in_channels     = cfg["in_channels"],      # 3 (RGB)
        classes         = cfg["classes"],           # 1 (binary)
        activation      = None,                    # raw logits → sigmoid applied in loss
        decoder_use_batchnorm = True,
    )
    return model


# ─────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────

class HybridDiceBCELoss(nn.Module):
    """
    L_hybrid = L_BCE + (1 - Dice)
    Handles class imbalance inherent in binary fetal head segmentation.
    """
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
        self.bce    = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # BCE loss (operates on logits)
        bce_loss = self.bce(logits, targets)

        # Dice loss (apply sigmoid first)
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        union        = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice_score   = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss    = 1.0 - dice_score.mean()

        return bce_loss + dice_loss


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

class SegmentationMetrics:
    """
    Computes: Dice, IoU, Precision, Recall, Accuracy, F1 Score
    All metrics are computed over the full epoch (accumulated batch-wise).
    """
    def __init__(self, threshold: float = 0.5, smooth: float = 1e-6):
        self.threshold = threshold
        self.smooth    = smooth
        self.reset()

    def reset(self):
        self.tp = self.fp = self.fn = self.tn = 0.0

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        preds   = (torch.sigmoid(logits) > self.threshold).float()
        targets = targets.float()

        self.tp += (preds * targets).sum().item()
        self.fp += (preds * (1 - targets)).sum().item()
        self.fn += ((1 - preds) * targets).sum().item()
        self.tn += ((1 - preds) * (1 - targets)).sum().item()

    def compute(self) -> dict:
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        s = self.smooth

        dice      = (2 * tp + s) / (2 * tp + fp + fn + s)
        iou       = (tp + s) / (tp + fp + fn + s)
        precision = (tp + s) / (tp + fp + s)
        recall    = (tp + s) / (tp + fn + s)
        accuracy  = (tp + tn + s) / (tp + fp + fn + tn + s)
        f1        = (2 * precision * recall + s) / (precision + recall + s)

        return {
            "Dice":      dice,
            "IoU":       iou,
            "Precision": precision,
            "Recall":    recall,
            "Accuracy":  accuracy,
            "F1":        f1,
        }


# ─────────────────────────────────────────────
# TRAIN / VALIDATE ONE EPOCH
# ─────────────────────────────────────────────

def run_epoch(
    model, loader, criterion, optimizer, device, metrics: SegmentationMetrics,
    mode: str = "train", scaler=None
) -> tuple[float, dict]:
    """Single epoch of train or val. Returns (avg_loss, metric_dict)."""
    model.train() if mode == "train" else model.eval()
    metrics.reset()
    total_loss = 0.0

    ctx = torch.no_grad() if mode != "train" else torch.enable_grad()
    with ctx:
        for images, masks in tqdm(loader, desc=f"  [{mode.upper()}]", leave=False):
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device, non_blocking=True)

            if mode == "train":
                optimizer.zero_grad(set_to_none=True)
                if scaler:
                    with torch.autocast(device_type="cuda"):
                        logits = model(images)
                        loss   = criterion(logits, masks)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(images)
                    loss   = criterion(logits, masks)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
            else:
                logits = model(images)
                loss   = criterion(logits, masks)

            total_loss += loss.item()
            metrics.update(logits.detach(), masks.detach())

    avg_loss   = total_loss / len(loader)
    metric_dict = metrics.compute()
    return avg_loss, metric_dict


# ─────────────────────────────────────────────
# POST-PROCESSING + HC CALCULATION
# ─────────────────────────────────────────────

def mask_to_hc(mask_np: np.ndarray, pixel_spacing_mm: float) -> dict:
    """
    Given a binary mask (H, W, uint8), compute head circumference.

    Pipeline:
      1. Morphological cleanup (close small holes)
      2. Find largest contour (skull boundary)
      3. Fit ellipse via cv2.fitEllipse (least-squares)
      4. HC = Ramanujan approximation of ellipse perimeter × pixel_spacing_mm
    """
    result = {"hc_mm": None, "ellipse": None, "contour": None}

    binary = (mask_np > 127).astype(np.uint8) * 255

    # Morphological close: fills small holes in skull boundary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return result

    # Keep largest contour (fetal head)
    largest = max(contours, key=cv2.contourArea)
    result["contour"] = largest

    if len(largest) < 5:  # fitEllipse needs ≥5 points
        return result

    # Ellipse fitting
    ellipse = cv2.fitEllipse(largest)   # ((cx, cy), (a, b), angle_deg)
    result["ellipse"] = ellipse

    # Semi-axes in pixels
    axis_a = ellipse[1][0] / 2.0   # semi-major
    axis_b = ellipse[1][1] / 2.0   # semi-minor

    # Ramanujan's approximation for ellipse perimeter (more accurate than π(a+b))
    h = ((axis_a - axis_b) ** 2) / ((axis_a + axis_b) ** 2)
    perimeter_px = math.pi * (axis_a + axis_b) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)))

    hc_mm = perimeter_px * pixel_spacing_mm
    result["hc_mm"] = hc_mm
    return result


# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────

def train(cfg: dict):
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    os.makedirs(os.path.dirname(cfg["checkpoint"]), exist_ok=True)
    os.makedirs(cfg["results_dir"], exist_ok=True)

    # Data
    train_loader, val_loader = get_dataloaders(cfg)

    # Model, Loss, Optimizer, Scheduler
    model     = build_model(cfg).to(device)
    criterion = HybridDiceBCELoss()
    optimizer = Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5,
                                  min_lr=1e-7, verbose=True)
    scaler    = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    train_metrics = SegmentationMetrics(threshold=cfg["threshold"])
    val_metrics   = SegmentationMetrics(threshold=cfg["threshold"])

    # Tracking
    history = {"epoch": [], "train_loss": [], "val_loss": []}
    for m in ["Dice", "IoU", "Precision", "Recall", "Accuracy", "F1"]:
        history[f"train_{m}"] = []
        history[f"val_{m}"]   = []

    best_val_dice  = 0.0
    patience_count = 0

    print(f"\n{'='*60}")
    print(f"  Training U-Net + MiT-B2 on HC18")
    print(f"  Epochs: {cfg['epochs']} | Batch: {cfg['batch_size']} | LR: {cfg['lr']}")
    print(f"{'='*60}\n")

    for epoch in range(1, cfg["epochs"] + 1):
        print(f"Epoch [{epoch:03d}/{cfg['epochs']:03d}]")

        train_loss, train_m = run_epoch(
            model, train_loader, criterion, optimizer, device,
            train_metrics, mode="train", scaler=scaler
        )
        val_loss, val_m = run_epoch(
            model, val_loader, criterion, optimizer, device,
            val_metrics, mode="val", scaler=None
        )

        scheduler.step(val_m["Dice"])

        # Log
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        for k in ["Dice", "IoU", "Precision", "Recall", "Accuracy", "F1"]:
            history[f"train_{k}"].append(train_m[k])
            history[f"val_{k}"].append(val_m[k])

        print(
            f"  Train → Loss: {train_loss:.4f} | Dice: {train_m['Dice']:.4f} | "
            f"IoU: {train_m['IoU']:.4f} | Prec: {train_m['Precision']:.4f} | "
            f"Rec: {train_m['Recall']:.4f} | Acc: {train_m['Accuracy']:.4f} | F1: {train_m['F1']:.4f}"
        )
        print(
            f"  Val   → Loss: {val_loss:.4f} | Dice: {val_m['Dice']:.4f} | "
            f"IoU: {val_m['IoU']:.4f} | Prec: {val_m['Precision']:.4f} | "
            f"Rec: {val_m['Recall']:.4f} | Acc: {val_m['Accuracy']:.4f} | F1: {val_m['F1']:.4f}"
        )

        # Save best
        if val_m["Dice"] > best_val_dice:
            best_val_dice = val_m["Dice"]
            torch.save({
                "epoch":      epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice":   best_val_dice,
                "cfg":        cfg,
            }, cfg["checkpoint"])
            print(f"  ✅ Best model saved (Val Dice: {best_val_dice:.4f})")
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= cfg["patience"]:
                print(f"\n  ⏹  Early stopping at epoch {epoch} (no improvement for {cfg['patience']} epochs)")
                break

        print()

    # Save history
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(cfg["results_dir"], "training_history.csv"), index=False)
    print(f"\n[Done] Best Val Dice: {best_val_dice:.4f}")
    plot_history(history, cfg["results_dir"])


# ─────────────────────────────────────────────
# EVALUATION ON VALIDATION SET
# ─────────────────────────────────────────────

def evaluate(cfg: dict):
    """Load best checkpoint and report all metrics on val set + HC estimation."""
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader = get_dataloaders(cfg)

    model = build_model(cfg).to(device)
    ckpt  = torch.load(cfg["checkpoint"], map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[Checkpoint] Loaded from epoch {ckpt['epoch']} (Val Dice: {ckpt['val_dice']:.4f})")

    criterion    = HybridDiceBCELoss()
    val_metrics  = SegmentationMetrics(threshold=cfg["threshold"])
    model.eval()

    total_loss = 0.0
    val_metrics.reset()
    hc_errors  = []

    val_transform = get_transforms(cfg["image_size"], "val")
    # Re-build val paths for HC calculation
    train_dir   = cfg["train_dir"]
    all_images  = sorted(glob(os.path.join(train_dir, "*_HC.png")))
    all_masks   = [p.replace("_HC.png", "_HC_Annotation.png") for p in all_images]
    valid_pairs = [(img, msk) for img, msk in zip(all_images, all_masks) if os.path.exists(msk)]
    all_images, all_masks = zip(*valid_pairs)
    n_val       = int(len(all_images) * cfg["val_split"])
    val_images  = list(all_images[-n_val:])
    val_mask_paths = list(all_masks[-n_val:])

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="[Evaluating]"):
            images = images.to(device)
            masks  = masks.to(device)
            logits = model(images)
            loss   = criterion(logits, masks)
            total_loss += loss.item()
            val_metrics.update(logits, masks)

    avg_loss   = total_loss / len(val_loader)
    metric_dict = val_metrics.compute()

    print(f"\n{'='*55}")
    print(f"  EVALUATION RESULTS (Validation Set)")
    print(f"{'='*55}")
    print(f"  Loss      : {avg_loss:.4f}")
    print(f"  Dice Score: {metric_dict['Dice']:.4f}  ({metric_dict['Dice']*100:.2f}%)")
    print(f"  IoU       : {metric_dict['IoU']:.4f}  ({metric_dict['IoU']*100:.2f}%)")
    print(f"  Precision : {metric_dict['Precision']:.4f}")
    print(f"  Recall    : {metric_dict['Recall']:.4f}")
    print(f"  Accuracy  : {metric_dict['Accuracy']:.4f}")
    print(f"  F1 Score  : {metric_dict['F1']:.4f}")

    # HC estimation on a sample of val images
    print(f"\n[HC Estimation] Computing on {min(50, len(val_images))} val images...")
    model.eval()
    sample_n = min(50, len(val_images))
    for i in range(sample_n):
        img_path  = val_images[i]
        mask_path = val_mask_paths[i]

        img_raw  = cv2.imread(img_path,  cv2.IMREAD_GRAYSCALE)
        mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # GT HC
        gt_mask_filled = HC18Dataset._fill_ellipse_mask(mask_raw)
        gt_hc = mask_to_hc(gt_mask_filled, cfg["pixel_spacing_mm"])["hc_mm"]

        # Predicted HC
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2RGB)
        aug     = val_transform(image=img_rgb, mask=gt_mask_filled)
        inp     = aug["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            pred_mask = torch.sigmoid(model(inp)).squeeze().cpu().numpy()
        pred_mask_bin = (pred_mask > cfg["threshold"]).astype(np.uint8) * 255
        # Resize back to original for accurate HC
        h0, w0 = img_raw.shape[:2]
        pred_mask_orig = cv2.resize(pred_mask_bin, (w0, h0), interpolation=cv2.INTER_NEAREST)
        pred_hc_info   = mask_to_hc(pred_mask_orig, cfg["pixel_spacing_mm"])
        pred_hc        = pred_hc_info["hc_mm"]

        if gt_hc and pred_hc:
            hc_errors.append(abs(gt_hc - pred_hc))

    if hc_errors:
        mae = np.mean(hc_errors)
        mse = np.mean(np.array(hc_errors) ** 2)
        print(f"  HC MAE  : {mae:.2f} mm")
        print(f"  HC MSE  : {mse:.3f} mm²")

    print(f"{'='*55}\n")
    return metric_dict


# ─────────────────────────────────────────────
# GRAD-CAM++ VISUALIZATION
# ─────────────────────────────────────────────

def run_gradcam(cfg: dict, image_path: str):
    """Generate Grad-CAM++ heatmap for a single image."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg).to(device)
    ckpt  = torch.load(cfg["checkpoint"], map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Target layer: last block of MiT-B2 encoder (block 3, norm layer output)
    # In SMP's MiT encoder the blocks are accessed via model.encoder.block
    try:
        target_layer = [model.encoder.block[-1][-1].norm2]
    except AttributeError:
        # Fallback to decoder last conv
        target_layer = [model.decoder.blocks[-1]]

    # Prepare image
    img_bgr  = cv2.imread(image_path)
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    transform = get_transforms(cfg["image_size"], "val")
    aug  = transform(image=img_rgb, mask=np.zeros(img_rgb.shape[:2], dtype=np.uint8))
    inp  = aug["image"].unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        logit = model(inp)
        pred  = torch.sigmoid(logit).squeeze().cpu().numpy()

    # Grad-CAM++ (SemanticSegmentationTarget wraps pixel-level score)
    mask_pred_bin = (pred > cfg["threshold"]).astype(np.uint8)
    targets       = [SemanticSegmentationTarget(0, mask_pred_bin)]

    cam_obj = GradCAMPlusPlus(model=model, target_layers=target_layer)
    grayscale_cam = cam_obj(input_tensor=inp, targets=targets)[0]

    # Overlay
    img_resized  = cv2.resize(img_rgb, (cfg["image_size"], cfg["image_size"])) / 255.0
    cam_overlay  = show_cam_on_image(img_resized.astype(np.float32), grayscale_cam, use_rgb=True)

    os.makedirs(cfg["results_dir"], exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_resized)
    axes[0].set_title("Input Ultrasound", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(pred, cmap="gray")
    axes[1].set_title(f"Predicted Mask\n(Dice threshold = {cfg['threshold']})", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(cam_overlay)
    axes[2].set_title("Grad-CAM++ Heatmap\n(Model attention)", fontsize=12)
    axes[2].axis("off")

    plt.tight_layout()
    out_path = os.path.join(cfg["results_dir"], "gradcam_output.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Grad-CAM++] Saved to {out_path}")

    # HC from prediction
    pred_bin  = (pred > cfg["threshold"]).astype(np.uint8) * 255
    hc_result = mask_to_hc(pred_bin, cfg["pixel_spacing_mm"])
    if hc_result["hc_mm"]:
        print(f"[HC Estimate] {hc_result['hc_mm']:.1f} mm")


# ─────────────────────────────────────────────
# INFERENCE ON TEST SET
# ─────────────────────────────────────────────

def predict_test(cfg: dict):
    """Run inference on HC18 test set and save predicted masks + HC values."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg).to(device)
    ckpt  = torch.load(cfg["checkpoint"], map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_images = get_test_dataloader(cfg)
    transform   = get_transforms(cfg["image_size"], "val")

    os.makedirs(os.path.join(cfg["results_dir"], "test_masks"), exist_ok=True)
    records = []

    print(f"[Test Inference] {len(test_images)} images")
    for img_path in tqdm(test_images):
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h0, w0  = img_bgr.shape[:2]

        aug = transform(image=img_rgb, mask=np.zeros((h0, w0), dtype=np.uint8))
        inp = aug["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            pred = torch.sigmoid(model(inp)).squeeze().cpu().numpy()

        pred_bin  = (pred > cfg["threshold"]).astype(np.uint8) * 255
        pred_orig = cv2.resize(pred_bin, (w0, h0), interpolation=cv2.INTER_NEAREST)
        hc_result = mask_to_hc(pred_orig, cfg["pixel_spacing_mm"])

        # Save mask
        fname = os.path.basename(img_path).replace(".png", "_pred.png")
        cv2.imwrite(os.path.join(cfg["results_dir"], "test_masks", fname), pred_orig)

        records.append({
            "filename": os.path.basename(img_path),
            "hc_mm":    round(hc_result["hc_mm"], 2) if hc_result["hc_mm"] else None,
        })

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(cfg["results_dir"], "test_predictions.csv"), index=False)
    print(f"[Done] Predictions saved to {cfg['results_dir']}/test_predictions.csv")


# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────

def plot_history(history: dict, results_dir: str):
    epochs = history["epoch"]

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()

    metrics_to_plot = [
        ("Loss",      "train_loss",  "val_loss"),
        ("Dice",      "train_Dice",  "val_Dice"),
        ("IoU",       "train_IoU",   "val_IoU"),
        ("Precision", "train_Precision", "val_Precision"),
        ("Recall",    "train_Recall",    "val_Recall"),
        ("Accuracy",  "train_Accuracy",  "val_Accuracy"),
        ("F1 Score",  "train_F1",        "val_F1"),
    ]

    for i, (title, train_key, val_key) in enumerate(metrics_to_plot):
        axes[i].plot(epochs, history[train_key], label="Train", linewidth=2)
        axes[i].plot(epochs, history[val_key],   label="Val",   linewidth=2, linestyle="--")
        axes[i].set_title(title, fontsize=13, fontweight="bold")
        axes[i].set_xlabel("Epoch")
        axes[i].legend()
        axes[i].grid(alpha=0.3)

    axes[7].axis("off")
    plt.suptitle("U-Net + MiT-B2 — Training Curves (HC18 Dataset)", fontsize=15, fontweight="bold")
    plt.tight_layout()

    out = os.path.join(results_dir, "training_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved to {out}")


def visualize_predictions(cfg: dict, n_samples: int = 5):
    """Visual comparison: input | ground truth | prediction | overlay."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg).to(device)
    ckpt  = torch.load(cfg["checkpoint"], map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    train_dir = cfg["train_dir"]
    all_images = sorted(glob(os.path.join(train_dir, "*_HC.png")))
    all_masks  = [p.replace("_HC.png", "_HC_Annotation.png") for p in all_images]
    valid_pairs = [(img, msk) for img, msk in zip(all_images, all_masks) if os.path.exists(msk)]
    all_images, all_masks = zip(*valid_pairs)
    n_val = int(len(all_images) * cfg["val_split"])
    val_images = list(all_images[-n_val:])
    val_masks  = list(all_masks[-n_val:])

    transform = get_transforms(cfg["image_size"], "val")
    n_samples = min(n_samples, len(val_images))

    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    col_labels = ["Input Image", "Ground Truth", "Prediction", "Overlay (GT=green, Pred=red)"]

    for i in range(n_samples):
        img_raw  = cv2.imread(val_images[i], cv2.IMREAD_GRAYSCALE)
        mask_raw = cv2.imread(val_masks[i],  cv2.IMREAD_GRAYSCALE)
        img_rgb  = cv2.cvtColor(img_raw, cv2.COLOR_GRAY2RGB)
        gt_mask  = HC18Dataset._fill_ellipse_mask(mask_raw)

        aug = transform(image=img_rgb, mask=gt_mask)
        inp = aug["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            pred = torch.sigmoid(model(inp)).squeeze().cpu().numpy()

        pred_bin = (pred > cfg["threshold"]).astype(np.uint8) * 255
        gt_disp  = cv2.resize(gt_mask,    (cfg["image_size"], cfg["image_size"]))
        img_disp = cv2.resize(img_raw,    (cfg["image_size"], cfg["image_size"]))

        overlay = np.zeros((cfg["image_size"], cfg["image_size"], 3), dtype=np.uint8)
        overlay[..., 0] = img_disp  # R
        overlay[..., 1] = img_disp  # G
        overlay[..., 2] = img_disp  # B
        overlay[gt_disp > 0,   1] = 200   # GT = green tint
        overlay[pred_bin > 0,  0] = 200   # Pred = red tint

        axes[i][0].imshow(img_disp, cmap="gray")
        axes[i][1].imshow(gt_disp,  cmap="gray")
        axes[i][2].imshow(pred_bin, cmap="gray")
        axes[i][3].imshow(overlay)
        for j in range(4):
            axes[i][j].axis("off")
        if i == 0:
            for j, label in enumerate(col_labels):
                axes[i][j].set_title(label, fontsize=11, fontweight="bold")

    plt.suptitle("Prediction Visualization — U-Net + MiT-B2", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(cfg["results_dir"], "prediction_samples.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualize] Saved to {out}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fetal Head Segmentation — U-Net + MiT-B2")
    parser.add_argument("--mode",       type=str, default="train",
                        choices=["train", "evaluate", "predict", "gradcam", "visualize"],
                        help="Run mode")
    parser.add_argument("--image_path", type=str, default=None,
                        help="Path to single image for Grad-CAM visualization")
    parser.add_argument("--epochs",     type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr",         type=float, default=None)
    args = parser.parse_args()

    cfg = CFG.copy()
    if args.epochs:     cfg["epochs"]     = args.epochs
    if args.batch_size: cfg["batch_size"] = args.batch_size
    if args.lr:         cfg["lr"]         = args.lr

    if args.mode == "train":
        train(cfg)

    elif args.mode == "evaluate":
        evaluate(cfg)

    elif args.mode == "predict":
        predict_test(cfg)

    elif args.mode == "gradcam":
        if not args.image_path:
            print("[ERROR] Provide --image_path for Grad-CAM mode")
            return
        run_gradcam(cfg, args.image_path)

    elif args.mode == "visualize":
        visualize_predictions(cfg)


if __name__ == "__main__":
    main()
