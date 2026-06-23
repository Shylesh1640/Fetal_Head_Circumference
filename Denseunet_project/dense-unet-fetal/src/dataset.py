"""
HC18 Dataset
============
Loads the HC18 Grand Challenge fetal-head ultrasound dataset
(https://zenodo.org/records/1327317) for binary segmentation.

Expected directory layout (exactly as distributed by the challenge):

    hc18_root/
      training_set/
        000_HC.png
        000_HC_Annotation.png
        001_HC.png
        001_HC_Annotation.png
        ...
      test_set/                      (no masks, optional, not used in training)
        ...

Patient-level splitting: each pair (image, mask) is keyed by its numeric
filename prefix. We split prefixes (not individual files) into
train/val/test so the same scan cannot leak across splits.
"""

from __future__ import annotations

import os
import re
import glob
import random
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


IMG_SUFFIX = "_HC.png"
MASK_SUFFIX = "_HC_Annotation.png"


def _find_pairs(root: str) -> List[Tuple[str, str]]:
    """Return list of (image_path, mask_path) sorted by patient id."""
    all_imgs = sorted(glob.glob(os.path.join(root, f"*{IMG_SUFFIX}")))
    pairs = []
    for img_path in all_imgs:
        if img_path.endswith(MASK_SUFFIX):
            continue
        mask_path = img_path.replace(IMG_SUFFIX, MASK_SUFFIX)
        if os.path.exists(mask_path):
            pairs.append((img_path, mask_path))
    if not pairs:
        raise FileNotFoundError(
            f"No (image, mask) pairs found under '{root}'. "
            f"Expected files like '000_HC.png' + '000_HC_Annotation.png'."
        )
    return pairs


def patient_level_split(
    pairs: List[Tuple[str, str]],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[List, List, List]:
    """Split pairs into train/val/test by patient id (filename prefix) so no
    patient's image appears in more than one split."""

    def patient_id(path: str) -> str:
        base = os.path.basename(path)
        m = re.match(r"(\d+)", base)
        return m.group(1) if m else base

    ids = sorted({patient_id(p[0]) for p in pairs})
    rng = random.Random(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_val = max(1, int(round(n * val_frac)))
    n_test = max(1, int(round(n * test_frac)))
    val_ids = set(ids[:n_val])
    test_ids = set(ids[n_val:n_val + n_test])
    train_ids = set(ids[n_val + n_test:])

    train_pairs = [p for p in pairs if patient_id(p[0]) in train_ids]
    val_pairs = [p for p in pairs if patient_id(p[0]) in val_ids]
    test_pairs = [p for p in pairs if patient_id(p[0]) in test_ids]
    return train_pairs, val_pairs, test_pairs


def get_train_transforms(image_size: int = 256) -> A.Compose:
    return A.Compose([
        A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
        A.RandomCrop(height=image_size, width=image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.GaussNoise(p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2(),
    ])


def get_eval_transforms(image_size: int = 256) -> A.Compose:
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=(0.5,), std=(0.5,)),
        ToTensorV2(),
    ])


class HC18Dataset(Dataset):
    """Binary fetal-head segmentation dataset.

    Returns:
        image: FloatTensor [1, H, W], normalized.
        mask:  FloatTensor [1, H, W], values in {0., 1.}.
    """

    def __init__(self, pairs: List[Tuple[str, str]], transforms: A.Compose):
        self.pairs = pairs
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise RuntimeError(f"Failed to read '{img_path}' or '{mask_path}'")

        # HC18 annotation masks are thin contour outlines; fill them to get
        # a solid binary region for standard segmentation training.
        mask = self._fill_contour(mask)

        augmented = self.transforms(image=image, mask=mask)
        image_t = augmented["image"].float()
        mask_t = augmented["mask"].float()
        mask_t = (mask_t > 0).float()
        if mask_t.ndim == 2:
            mask_t = mask_t.unsqueeze(0)
        return image_t, mask_t

    @staticmethod
    def _fill_contour(mask: np.ndarray) -> np.ndarray:
        binary = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return binary
        largest = max(contours, key=cv2.contourArea)
        filled = np.zeros_like(binary)
        cv2.drawContours(filled, [largest], -1, color=255, thickness=-1)
        return filled


def build_datasets(
    data_root: str,
    image_size: int = 256,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
):
    """Convenience builder used by the Lightning DataModule."""
    train_dir = os.path.join(data_root, "training_set")
    if not os.path.isdir(train_dir):
        # fall back: allow the user to point data_root directly at the
        # folder that contains the *_HC.png files.
        train_dir = data_root

    pairs = _find_pairs(train_dir)
    train_pairs, val_pairs, test_pairs = patient_level_split(
        pairs, val_frac=val_frac, test_frac=test_frac, seed=seed
    )

    train_ds = HC18Dataset(train_pairs, get_train_transforms(image_size))
    val_ds = HC18Dataset(val_pairs, get_eval_transforms(image_size))
    test_ds = HC18Dataset(test_pairs, get_eval_transforms(image_size))
    return train_ds, val_ds, test_ds
