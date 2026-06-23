"""
dataset.py
HC18 dataset loading, patient-level (image-level) train/val/test split,
and albumentations-based augmentation pipeline.

HC18 folder layout expected (the official "training_set" zip):
    training_set/
        000_HC.png
        000_HC_Annotation.png
        001_HC.png
        001_HC_Annotation.png
        ...
        999_HC.png   (numbering may vary; we just glob *_Annotation.png)

The "_Annotation.png" files are pixel-thin ELLIPSE OUTLINES (not filled masks)
as drawn by a sonographer. The paper trains on binary head masks, so we fill
the ellipse outline into a solid mask at dataset-load time using OpenCV
contour fill -- this matches standard HC18 preprocessing used in the
reference implementations (see e.g. qubvel SMP-based fetal segmentation
repos that follow the same recipe).
"""

import os
import glob
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config


def _fill_ellipse_outline(annotation_img: np.ndarray) -> np.ndarray:
    """
    HC18 annotation PNGs contain a thin white ellipse OUTLINE on a black
    background. We convert that outline into a solid filled binary mask
    (1 = inside head, 0 = background) by finding the outline contour and
    filling it. This is the standard approach used across HC18 baselines.
    """
    # Binarize the outline (it's already near 0/255, but ensure it's clean)
    _, binary = cv2.threshold(annotation_img, config.MASK_BINARIZE_THRESHOLD, 255, cv2.THRESH_BINARY)
    binary = binary.astype(np.uint8)

    # Close small gaps in the outline so findContours yields one closed loop
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(binary)
    if len(contours) > 0:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(filled, [largest], -1, color=255, thickness=cv2.FILLED)
    else:
        # Fallback: if no contour found, just use the binarized outline itself
        filled = binary

    return (filled > 0).astype(np.uint8)  # {0, 1}


def build_split_lists(data_root: str, train_frac: float, val_frac: float,
                       test_frac: float, seed: int):
    """
    Scans data_root for all *_Annotation.png files, derives matching base
    images, shuffles with a fixed seed, and splits at the IMAGE level
    (each HC18 image is a distinct patient/scan, so this is equivalent to
    patient-level splitting and avoids data leakage, matching the paper's
    requirement of patient-level separation).
    """
    assert abs((train_frac + val_frac + test_frac) - 1.0) < 1e-6, \
        "train/val/test fractions must sum to 1.0"

    annotation_paths = sorted(glob.glob(os.path.join(data_root, "*_Annotation.png")))
    if len(annotation_paths) == 0:
        raise FileNotFoundError(
            f"No '*_Annotation.png' files found in {data_root}. "
            f"Check config.DATA_ROOT points to the HC18 'training_set' folder."
        )

    pairs = []
    for ann_path in annotation_paths:
        base = os.path.basename(ann_path).replace("_Annotation.png", ".png")
        img_path = os.path.join(data_root, base)
        if os.path.exists(img_path):
            pairs.append((img_path, ann_path))

    rng = random.Random(seed)
    rng.shuffle(pairs)

    n = len(pairs)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]

    return train_pairs, val_pairs, test_pairs


def get_train_transforms(image_size: int) -> A.Compose:
    """Training augmentations, matching the paper's augmentation pipeline."""
    return A.Compose([
        A.PadIfNeeded(min_height=image_size, min_width=image_size,
                      border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
        A.RandomCrop(height=image_size, width=image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.GaussNoise(std_range=(0.04, 0.2), p=0.3),  # std as fraction of [0,1] pixel range
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_eval_transforms(image_size: int) -> A.Compose:
    """Deterministic resize + normalize for validation/test (no augmentation)."""
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class HC18Dataset(Dataset):
    """
    Returns:
        image: FloatTensor [3, H, W], ImageNet-normalized
        mask:  FloatTensor [1, H, W], binary {0., 1.}
        meta:  dict with original image path (useful for HC post-processing)
    """

    def __init__(self, pairs, transforms: A.Compose):
        self.pairs = pairs
        self.transforms = transforms

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, ann_path = self.pairs[idx]

        # Read grayscale ultrasound image, replicate to 3 channels for the
        # ImageNet-pretrained MiT-B2 encoder.
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        annotation = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)
        if annotation is None:
            raise RuntimeError(f"Failed to read annotation: {ann_path}")
        mask = _fill_ellipse_outline(annotation)  # uint8 {0,1}, same H,W as image

        augmented = self.transforms(image=image, mask=mask)
        image_t = augmented["image"]                       # [3,H,W] float
        mask_t = augmented["mask"].float().unsqueeze(0)     # [1,H,W] float {0.,1.}

        return image_t, mask_t, {"image_path": img_path}


def build_dataloaders():
    """Convenience function used by train.py / evaluate.py."""
    from torch.utils.data import DataLoader

    train_pairs, val_pairs, test_pairs = build_split_lists(
        config.DATA_ROOT, config.TRAIN_FRAC, config.VAL_FRAC,
        config.TEST_FRAC, config.RANDOM_SEED
    )

    print(f"[Data] train={len(train_pairs)}  val={len(val_pairs)}  test={len(test_pairs)}")

    train_ds = HC18Dataset(train_pairs, get_train_transforms(config.IMAGE_SIZE))
    val_ds = HC18Dataset(val_pairs, get_eval_transforms(config.IMAGE_SIZE))
    test_ds = HC18Dataset(test_pairs, get_eval_transforms(config.IMAGE_SIZE))

    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True,
        persistent_workers=config.NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
        persistent_workers=config.NUM_WORKERS > 0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
        persistent_workers=config.NUM_WORKERS > 0,
    )

    return train_loader, val_loader, test_loader
