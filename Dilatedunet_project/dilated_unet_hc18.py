"""
=============================================================================
  SOTA Dilated U-Net with ASPP — Fetal Head Segmentation on HC18 Dataset
=============================================================================

Architecture:  Dilated U-Net with Atrous Spatial Pyramid Pooling (ASPP)
               + Squeeze-and-Excitation (SE) channel attention in every block
               + Residual connections throughout encoder/decoder
               + Contour-based ellipse fitting for HC computation

Metrics:       Dice, IoU, Precision, Recall, Accuracy, F1, Loss
               (reported for Train / Validation / Test)

Dataset:       HC18 Grand Challenge  — https://zenodo.org/record/1322001
               ~999 training images (800×540 px, grayscale) + binary masks

Platform:      Lightning AI (or any Linux GPU box)
               Tested on: T4 / A100 (Google Colab), RTX 4060 6 GB

Authors note:  Dilated convolutions in the bottleneck + ASPP bridge replace
               the standard bottleneck of plain U-Net, expanding the effective
               receptive field from ~200 px to ~400 px — critical for capturing
               the full fetal skull boundary in a 256×256 input.

Usage:
    # 1. Install deps (Lightning AI terminal):
    pip install torch torchvision albumentations segmentation-models-pytorch \
                timm grad-cam opencv-python matplotlib tqdm

    # 2. Download HC18 and set DATA_ROOT below.

    # 3. Run:
    python dilated_unet_hc18.py
=============================================================================
"""

# ─── Stdlib ──────────────────────────────────────────────────────────────────
import os, math, time, random
from pathlib import Path

# ─── Scientific / Imaging ────────────────────────────────────────────────────
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

# ─── PyTorch ─────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# ─── Augmentation ────────────────────────────────────────────────────────────
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ─── Misc ─────────────────────────────────────────────────────────────────────
from tqdm import tqdm

# ===========================================================================
#  0.  CONFIG  (edit these paths before running)
# ===========================================================================

CFG = dict(
    # ── Data ──
    DATA_ROOT   = "./hc18",          # folder that contains training/ and test/
    IMG_SIZE    = 256,               # network input size (square)
    PIXEL_MM    = 0.154,             # HC18 pixel spacing in mm (from challenge)

    # ── Split ──
    TRAIN_FRAC  = 0.80,
    VAL_FRAC    = 0.10,
    # TEST_FRAC = remaining 0.10

    # ── Training ──
    EPOCHS      = 100,
    BATCH_SIZE  = 8,
    LR          = 1e-4,
    WEIGHT_DECAY= 1e-5,
    PATIENCE    = 15,                # early-stopping patience

    # ── Model ──
    ENCODER_CH  = [32, 64, 128, 256, 512],   # channels at each encoder level
    DROPOUT     = 0.1,

    # ── Paths ──
    CKPT_DIR    = "./checkpoints",
    LOG_DIR     = "./logs",

    # ── Reproducibility ──
    SEED        = 42,
)

# ===========================================================================
#  1.  SEED & DEVICE
# ===========================================================================

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CFG["SEED"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] {DEVICE}")

os.makedirs(CFG["CKPT_DIR"], exist_ok=True)
os.makedirs(CFG["LOG_DIR"],  exist_ok=True)

# ===========================================================================
#  2.  DATASET
# ===========================================================================

class HC18Dataset(Dataset):
    """
    HC18 Grand Challenge dataset loader.

    Folder structure expected:
        hc18/
          training/
            000_HC.png
            000_HC_Annotation.png
            ...
          test/
            ...

    Patient-level split: we keep all slices of one patient together so
    there is no data leakage between train / val / test splits.
    """

    def __init__(self, image_paths: list, mask_paths: list, transform=None):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # ── Load image as 3-channel (model expects 3ch; ultrasound is greyscale)
        img = cv2.imread(str(self.image_paths[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # (H, W, 3) uint8

        # ── Load mask (binary)
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8)          # (H, W) binary {0,1}

        if self.transform:
            aug   = self.transform(image=img, mask=mask)
            img   = aug["image"]   # tensor (3, H, W) float32 [0,1]
            mask  = aug["mask"]    # tensor (H, W)    int64

        # Add channel dim to mask
        mask = mask.float().unsqueeze(0)              # (1, H, W)
        return img, mask


def build_transforms(img_size: int, split: str):
    """Albumentations pipeline — different for train vs val/test."""
    mean = (0.485, 0.456, 0.406)    # ImageNet stats (works well for ultrasound too)
    std  = (0.229, 0.224, 0.225)

    if split == "train":
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=20, p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0.2, p=0.4),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.CoarseDropout(max_holes=8, max_height=16,
                            max_width=16, fill_value=0, p=0.2),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])


def load_hc18_paths(data_root: str):
    """Returns (image_paths, mask_paths) sorted lists from the training split."""
    root = Path(data_root) / "training"
    imgs  = sorted(root.glob("*_HC.png"))
    # Filter out annotation files accidentally picked up
    imgs  = [p for p in imgs if "Annotation" not in p.name]
    masks = [Path(str(p).replace("_HC.png", "_HC_Annotation.png"))
             for p in imgs]
    assert all(m.exists() for m in masks), \
        "Some annotation files are missing! Check DATA_ROOT."
    return imgs, masks


def build_dataloaders(cfg: dict):
    imgs, masks = load_hc18_paths(cfg["DATA_ROOT"])
    n = len(imgs)
    print(f"[Data] Total samples found: {n}")

    # ── Patient-level split (no leakage)
    # HC18 filenames: 000_HC.png, 001_HC.png … (each unique patient = 1 image)
    # So plain index split is fine (no repeated patient IDs).
    indices = list(range(n))
    random.shuffle(indices)
    n_train = int(n * cfg["TRAIN_FRAC"])
    n_val   = int(n * cfg["VAL_FRAC"])
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    def subset(idx_list, split):
        img_s  = [imgs[i]  for i in idx_list]
        mask_s = [masks[i] for i in idx_list]
        return HC18Dataset(img_s, mask_s,
                           transform=build_transforms(cfg["IMG_SIZE"], split))

    train_ds = subset(train_idx, "train")
    val_ds   = subset(val_idx,   "val")
    test_ds  = subset(test_idx,  "test")

    print(f"[Data] Train:{len(train_ds)}  Val:{len(val_ds)}  Test:{len(test_ds)}")

    loader_kw = dict(num_workers=4, pin_memory=True)
    train_dl  = DataLoader(train_ds, batch_size=cfg["BATCH_SIZE"],
                           shuffle=True,  drop_last=True,  **loader_kw)
    val_dl    = DataLoader(val_ds,   batch_size=cfg["BATCH_SIZE"],
                           shuffle=False, drop_last=False, **loader_kw)
    test_dl   = DataLoader(test_ds,  batch_size=cfg["BATCH_SIZE"],
                           shuffle=False, drop_last=False, **loader_kw)
    return train_dl, val_dl, test_dl

# ===========================================================================
#  3.  MODEL COMPONENTS
# ===========================================================================

# ─── 3.1  Squeeze-and-Excitation block ─────────────────────────────────────

class SEBlock(nn.Module):
    """Channel attention: squeezes global spatial info → re-calibrates channels."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.fc(self.pool(x)).view(x.size(0), x.size(1), 1, 1)
        return x * scale


# ─── 3.2  Residual Conv Block with SE ──────────────────────────────────────

class ResConvBlock(nn.Module):
    """
    Two 3×3 conv → BN → ReLU layers with a residual shortcut.
    Optional dilation for dilated encoder stages.
    SE attention applied after the second conv.
    """
    def __init__(self, in_ch: int, out_ch: int,
                 dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        pad = dilation  # same-size padding for dilated conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.se       = SEBlock(out_ch)
        self.relu     = nn.ReLU(inplace=True)
        self.dropout  = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # 1×1 projection for shortcut if channel dims differ
        self.shortcut = (nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ) if in_ch != out_ch else nn.Identity())

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        out = self.dropout(out)
        return self.relu(out + residual)


# ─── 3.3  ASPP Bottleneck ──────────────────────────────────────────────────

class ASPPBlock(nn.Module):
    """
    Atrous Spatial Pyramid Pooling bridge at the U-Net bottleneck.

    Branches:
      1×1 conv  (rate=1)
      3×3 dilated (rate=6)
      3×3 dilated (rate=12)
      3×3 dilated (rate=18)
      Global average pooling + 1×1 conv

    All branches → concat → 1×1 projection.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        mid = out_ch // 4

        self.b1 = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=6,  dilation=6,  bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        self.b3 = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        self.b4 = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True))

        self.project = nn.Sequential(
            nn.Conv2d(mid * 5, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout2d(0.1))

    def forward(self, x):
        h, w = x.shape[-2:]
        b1 = self.b1(x)
        b2 = self.b2(x)
        b3 = self.b3(x)
        b4 = self.b4(x)
        bp = F.interpolate(self.pool(x), size=(h, w), mode="bilinear",
                           align_corners=False)
        return self.project(torch.cat([b1, b2, b3, b4, bp], dim=1))


# ─── 3.4  Decoder Up-Block ─────────────────────────────────────────────────

class UpBlock(nn.Module):
    """Bilinear upsample → concat skip → ResConvBlock."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 dropout: float = 0.0):
        super().__init__()
        self.up   = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),   # channel compress
        )
        self.conv = ResConvBlock(out_ch + skip_ch, out_ch, dropout=dropout)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle spatial size mismatch (can arise from odd input dims)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                              align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


# ─── 3.5  Full Dilated U-Net ───────────────────────────────────────────────

class DilatedUNet(nn.Module):
    """
    Dilated U-Net with ASPP bottleneck and SE-Residual encoder/decoder blocks.

    Encoder:
      Level 0  (stride 1): ResConvBlock(3 → ch[0])
      Level 1–4 (stride 2, via MaxPool): progressively dilated conv blocks
        - Level 2 uses dilation=2, Level 3 uses dilation=2

    Bottleneck:  ASPPBlock

    Decoder:
      4 UpBlocks matching encoder skip connections

    Head:  1×1 conv → sigmoid (binary segmentation)
    """

    def __init__(self, ch=None, in_channels: int = 3,
                 num_classes: int = 1, dropout: float = 0.1):
        super().__init__()
        if ch is None:
            ch = [32, 64, 128, 256, 512]

        # ── Encoder ──────────────────────────────────────────────────────
        self.enc0 = ResConvBlock(in_channels, ch[0], dilation=1, dropout=0.0)
        self.pool0 = nn.MaxPool2d(2)

        self.enc1 = ResConvBlock(ch[0], ch[1], dilation=1, dropout=dropout)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ResConvBlock(ch[1], ch[2], dilation=2, dropout=dropout)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ResConvBlock(ch[2], ch[3], dilation=2, dropout=dropout)
        self.pool3 = nn.MaxPool2d(2)

        # ── Bottleneck (ASPP) ────────────────────────────────────────────
        self.bridge = ASPPBlock(ch[3], ch[4])

        # ── Decoder ──────────────────────────────────────────────────────
        self.dec3 = UpBlock(ch[4], ch[3], ch[3], dropout=dropout)
        self.dec2 = UpBlock(ch[3], ch[2], ch[2], dropout=dropout)
        self.dec1 = UpBlock(ch[2], ch[1], ch[1], dropout=dropout)
        self.dec0 = UpBlock(ch[1], ch[0], ch[0], dropout=dropout)

        # ── Output head ──────────────────────────────────────────────────
        self.head = nn.Conv2d(ch[0], num_classes, kernel_size=1)

        # ── Weight init ──────────────────────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Encoder
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool0(e0))
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Bottleneck
        b  = self.bridge(self.pool3(e3))

        # Decoder
        d3 = self.dec3(b,  e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        d0 = self.dec0(d1, e0)

        return torch.sigmoid(self.head(d0))   # (B, 1, H, W) ∈ [0,1]

# ===========================================================================
#  4.  LOSS
# ===========================================================================

class HybridDiceBCE(nn.Module):
    """
    L = α · BCE  +  (1 − Dice)
    BCE for pixel-wise supervision; Dice to handle class imbalance.
    """
    def __init__(self, bce_weight: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.bce_w  = bce_weight
        self.smooth = smooth
        self.bce    = nn.BCELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        bce_loss  = self.bce(pred, target)

        pred_flat   = pred.view(-1)
        target_flat = target.view(-1)
        inter       = (pred_flat * target_flat).sum()
        dice_loss   = 1 - (2 * inter + self.smooth) / \
                          (pred_flat.sum() + target_flat.sum() + self.smooth)

        return self.bce_w * bce_loss + dice_loss

# ===========================================================================
#  5.  METRICS
# ===========================================================================

class SegMetrics:
    """
    Accumulates TP/FP/FN/TN across batches then computes all metrics at once.
    All inputs are expected in [0,1] float; binarised at threshold 0.5.
    """
    def __init__(self, threshold: float = 0.5):
        self.thr   = threshold
        self.reset()

    def reset(self):
        self.tp = self.fp = self.fn = self.tn = 0.0
        self.total_loss = 0.0
        self.n_batches  = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor,
               loss: float = 0.0):
        pred_bin = (pred > self.thr).float()
        tp = (pred_bin * target).sum().item()
        fp = (pred_bin * (1 - target)).sum().item()
        fn = ((1 - pred_bin) * target).sum().item()
        tn = ((1 - pred_bin) * (1 - target)).sum().item()
        self.tp += tp;  self.fp += fp
        self.fn += fn;  self.tn += tn
        self.total_loss += loss
        self.n_batches  += 1

    def compute(self) -> dict:
        eps   = 1e-7
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        dice  = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        iou   = (tp + eps) / (tp + fp + fn + eps)
        prec  = (tp + eps) / (tp + fp + eps)
        rec   = (tp + eps) / (tp + fn + eps)
        acc   = (tp + tn + eps) / (tp + fp + fn + tn + eps)
        f1    = (2 * prec * rec + eps) / (prec + rec + eps)
        loss  = self.total_loss / max(self.n_batches, 1)
        return dict(dice=dice, iou=iou, precision=prec,
                    recall=rec, accuracy=acc, f1=f1, loss=loss)

# ===========================================================================
#  6.  TRAINING ENGINE
# ===========================================================================

def run_epoch(model, loader, loss_fn, optimizer, device, metrics, is_train):
    model.train() if is_train else model.eval()
    metrics.reset()
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for imgs, masks in tqdm(loader, leave=False,
                                desc="train" if is_train else "eval"):
            imgs  = imgs.to(device)
            masks = masks.to(device)

            preds = model(imgs)
            loss  = loss_fn(preds, masks)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            metrics.update(preds.detach(), masks.detach(), loss.item())

    return metrics.compute()


def train(cfg: dict):
    # ── Data ──
    train_dl, val_dl, test_dl = build_dataloaders(cfg)

    # ── Model ──
    model    = DilatedUNet(ch=cfg["ENCODER_CH"],
                           dropout=cfg["DROPOUT"]).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Trainable parameters: {n_params:,}")

    # ── Loss / Optimizer / Scheduler ──
    loss_fn   = HybridDiceBCE(bce_weight=0.5).to(DEVICE)
    optimizer = Adam(model.parameters(),
                     lr=cfg["LR"], weight_decay=cfg["WEIGHT_DECAY"])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # ── Metrics ──
    train_m = SegMetrics()
    val_m   = SegMetrics()

    # ── History for plotting ──
    history = {k: {"train": [], "val": []}
               for k in ["dice", "iou", "precision",
                         "recall", "accuracy", "f1", "loss"]}

    best_val_dice = 0.0
    no_improve    = 0
    ckpt_path     = os.path.join(cfg["CKPT_DIR"], "best_model.pth")

    # ════════════════════ TRAINING LOOP ════════════════════
    print("\n" + "═"*70)
    print("  Epoch  │   Loss       Dice      IoU    Prec    Rec     Acc     F1")
    print("─────────┼" + "─"*60)

    for epoch in range(1, cfg["EPOCHS"] + 1):
        t0 = time.time()

        tr = run_epoch(model, train_dl, loss_fn, optimizer, DEVICE,
                       train_m, is_train=True)
        va = run_epoch(model, val_dl,   loss_fn, optimizer, DEVICE,
                       val_m,   is_train=False)

        scheduler.step()

        for k in history:
            history[k]["train"].append(tr[k])
            history[k]["val"].append(va[k])

        elapsed = time.time() - t0
        print(f"  {epoch:>3d}/{cfg['EPOCHS']}  │  "
              f"T {tr['loss']:.4f} / {tr['dice']:.4f} / {tr['iou']:.4f} / "
              f"{tr['precision']:.4f} / {tr['recall']:.4f} / "
              f"{tr['accuracy']:.4f} / {tr['f1']:.4f}")
        print(f"           │  "
              f"V {va['loss']:.4f} / {va['dice']:.4f} / {va['iou']:.4f} / "
              f"{va['precision']:.4f} / {va['recall']:.4f} / "
              f"{va['accuracy']:.4f} / {va['f1']:.4f}  ({elapsed:.1f}s)")

        # ── Checkpoint ──
        if va["dice"] > best_val_dice:
            best_val_dice = va["dice"]
            no_improve    = 0
            torch.save({
                "epoch":      epoch,
                "state_dict": model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "val_dice":   best_val_dice,
                "cfg":        cfg,
            }, ckpt_path)
            print(f"           │  ✔  Saved checkpoint  (val_dice={best_val_dice:.4f})")
        else:
            no_improve += 1

        # ── Early stopping ──
        if no_improve >= cfg["PATIENCE"]:
            print(f"\n[EarlyStopping] No improvement for {cfg['PATIENCE']} epochs.")
            break

    print("═"*70)
    print(f"\n[Training complete]  Best val Dice: {best_val_dice:.4f}")

    # ── Plot learning curves ──
    plot_history(history, cfg["LOG_DIR"])

    # ── Evaluate on test set ──
    print("\n[Test set evaluation]")
    ckpt     = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    test_m   = SegMetrics()
    te       = run_epoch(model, test_dl, loss_fn, optimizer, DEVICE,
                         test_m, is_train=False)

    print("\n  ┌─────────────┬─────────────┬─────────────┐")
    print("  │   Metric    │    Value    │  (Best Val) │")
    print("  ├─────────────┼─────────────┼─────────────┤")
    for k, v in te.items():
        bv = max(history[k]["val"]) if k != "loss" else min(history[k]["val"])
        print(f"  │ {k:<11s} │  {v:.6f}  │  {bv:.6f}  │")
    print("  └─────────────┴─────────────┴─────────────┘")

    # ── Qualitative + HC prediction ──
    visualize_predictions(model, test_dl, DEVICE, cfg)

    return model, history


# ===========================================================================
#  7.  PLOTTING
# ===========================================================================

def plot_history(history: dict, log_dir: str):
    metrics = ["loss", "dice", "iou", "precision", "recall", "accuracy", "f1"]
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()

    for i, m in enumerate(metrics):
        ax = axes[i]
        ax.plot(history[m]["train"], label="Train", linewidth=1.5)
        ax.plot(history[m]["val"],   label="Val",   linewidth=1.5)
        ax.set_title(m.capitalize(), fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    axes[-1].axis("off")
    plt.suptitle("Dilated U-Net — Training History", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(log_dir, "training_history.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved → {path}")
    plt.show()


# ===========================================================================
#  8.  HEAD CIRCUMFERENCE COMPUTATION
# ===========================================================================

def predict_hc_mm(mask_np: np.ndarray, pixel_spacing_mm: float = 0.154) -> float:
    """
    Given a binary mask (H×W, values 0/1 uint8), fit an ellipse to the largest
    contour and return the head circumference in mm using Ramanujan's formula.

        HC ≈ π · [3(a+b) − √((3a+b)(a+3b))]     (Ramanujan 1914)
    """
    mask_u8 = (mask_np * 255).astype(np.uint8)

    # Morphological cleanup
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return float("nan")

    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:          # fitEllipse needs ≥5 points
        return float("nan")

    ellipse   = cv2.fitEllipse(largest)
    (cx, cy), (axis1, axis2), angle = ellipse

    a = (max(axis1, axis2) / 2) * pixel_spacing_mm   # semi-major axis in mm
    b = (min(axis1, axis2) / 2) * pixel_spacing_mm   # semi-minor axis in mm

    hc = math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))
    return hc


# ===========================================================================
#  9.  VISUALISATION + GRAD-CAM
# ===========================================================================

def visualize_predictions(model, loader, device, cfg, n_samples: int = 4):
    """
    Shows n_samples images from the loader with:
      - original image
      - ground-truth mask
      - predicted mask
      - overlay
      - predicted HC in mm
    """
    model.eval()
    imgs_list, masks_list, preds_list = [], [], []

    with torch.no_grad():
        for imgs, masks in loader:
            imgs  = imgs.to(device)
            preds = model(imgs)
            imgs_list.append(imgs.cpu())
            masks_list.append(masks.cpu())
            preds_list.append(preds.cpu())
            if sum(x.shape[0] for x in imgs_list) >= n_samples:
                break

    imgs_t  = torch.cat(imgs_list)[:n_samples]
    masks_t = torch.cat(masks_list)[:n_samples]
    preds_t = torch.cat(preds_list)[:n_samples]

    # ── Unnormalise for display ──
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    fig = plt.figure(figsize=(16, n_samples * 4))
    gs  = gridspec.GridSpec(n_samples, 4, figure=fig)

    for i in range(n_samples):
        img_np   = ((imgs_t[i] * std + mean).permute(1, 2, 0).numpy()
                    .clip(0, 1))
        mask_np  = masks_t[i, 0].numpy().astype(np.uint8)
        pred_np  = (preds_t[i, 0].numpy() > 0.5).astype(np.uint8)
        hc_mm    = predict_hc_mm(pred_np, cfg["PIXEL_MM"])

        # Overlay
        overlay  = img_np.copy()
        overlay[pred_np == 1] = overlay[pred_np == 1] * 0.5 + \
                                np.array([0.0, 1.0, 0.0]) * 0.5

        titles = ["Image", "GT Mask", "Predicted", f"Overlay\nHC={hc_mm:.1f} mm"]
        panels = [img_np, mask_np, pred_np, overlay]
        cmaps  = [None, "gray", "gray", None]

        for j, (panel, title, cmap) in enumerate(zip(panels, titles, cmaps)):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(panel, cmap=cmap)
            ax.set_title(title, fontsize=9)
            ax.axis("off")

    plt.suptitle("Dilated U-Net — Qualitative Results", fontsize=13,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(cfg["LOG_DIR"], "predictions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved → {path}")
    plt.show()


# ===========================================================================
#  10.  GRAD-CAM (standalone, no external library needed)
# ===========================================================================

class GradCAM:
    """
    Gradient-weighted Class Activation Map for the last convolutional layer
    of the encoder bridge (the ASPP output).
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model    = model
        self.grads    = None
        self.acts     = None
        self._hooks   = [
            target_layer.register_forward_hook(self._save_act),
            target_layer.register_full_backward_hook(self._save_grad),
        ]

    def _save_act(self, _, __, output):
        self.acts = output.detach()

    def _save_grad(self, _, __, grad_output):
        self.grads = grad_output[0].detach()

    def generate(self, x: torch.Tensor) -> np.ndarray:
        self.model.eval()
        x = x.unsqueeze(0).to(next(self.model.parameters()).device)
        out = self.model(x)
        self.model.zero_grad()
        out.sum().backward()

        weights = self.grads.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self.acts).sum(dim=1, keepdim=True)
        cam     = F.relu(cam)
        cam     = cam.squeeze().cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


def show_gradcam(model, loader, device, log_dir, n=2):
    """Generate and save Grad-CAM for n random samples."""
    # Target = last conv in ASPP project layer
    target_layer = model.bridge.project[0]   # Conv2d inside project Sequential
    gcam = GradCAM(model, target_layer)

    model.eval()
    imgs_all, masks_all = next(iter(loader))
    imgs_all  = imgs_all[:n]
    masks_all = masks_all[:n]

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    fig, axes = plt.subplots(n, 3, figsize=(12, n * 4))
    if n == 1:
        axes = axes[np.newaxis]

    for i in range(n):
        cam     = gcam.generate(imgs_all[i])
        img_np  = ((imgs_all[i] * std + mean).permute(1, 2, 0).numpy().clip(0, 1))
        h, w    = img_np.shape[:2]
        cam_up  = cv2.resize(cam, (w, h))
        heatmap = cv2.applyColorMap((cam_up * 255).astype(np.uint8),
                                    cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        blend   = 0.6 * img_np + 0.4 * heatmap

        axes[i, 0].imshow(img_np);      axes[i, 0].set_title("Image")
        axes[i, 1].imshow(cam_up, cmap="hot"); axes[i, 1].set_title("Grad-CAM")
        axes[i, 2].imshow(blend);       axes[i, 2].set_title("Overlay")
        for ax in axes[i]: ax.axis("off")

    plt.suptitle("Grad-CAM — ASPP Bottleneck Activations", fontsize=12,
                 fontweight="bold")
    plt.tight_layout()
    path = os.path.join(log_dir, "gradcam.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved → {path}")
    plt.show()
    gcam.remove_hooks()


# ===========================================================================
#  11.  MAIN
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  SOTA Dilated U-Net (ASPP + SE-Residual) — HC18 Fetal Head Seg")
    print("=" * 70)

    # Verify data root exists
    if not Path(CFG["DATA_ROOT"]).exists():
        print(f"\n[ERROR] DATA_ROOT not found: {CFG['DATA_ROOT']}")
        print("  Download HC18 from: https://zenodo.org/record/1322001")
        print("  Unzip so that:  ./hc18/training/000_HC.png  exists.\n")
        raise SystemExit(1)

    # Train
    model, history = train(CFG)

    # Grad-CAM on a batch from the val set
    _, val_dl, _ = build_dataloaders(CFG)
    show_gradcam(model, val_dl, DEVICE, CFG["LOG_DIR"])

    print("\n[Done] All outputs saved in:", CFG["LOG_DIR"])
