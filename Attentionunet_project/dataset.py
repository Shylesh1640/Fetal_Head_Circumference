"""
HC18 Dataset loader.

Expects the Grand Challenge HC18 folder layout:

    hc18/
    ├── training_set/
    │   ├── 000_HC.png
    │   ├── 000_HC_Annotation.png
    │   ├── 001_HC.png
    │   ├── 001_HC_Annotation.png
    │   ...
    └── test_set/            (unlabeled in the original challenge — not used here
                               for held-out testing since it has no ground truth)

Because the official HC18 test_set has NO public masks, we use patient-level
splitting of the LABELED training_set into train / val / test ourselves
(70% / 15% / 15%) so all three of your requested splits have ground truth
to score Dice/IoU/etc. against. This also matches what the paper describes
("patient-level splitting ... to avoid data leakage").

Download:
    https://zenodo.org/records/3904280  (HC18 Grand Challenge, Zenodo)
"""

import os
import re
import glob
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_size: int = 256):
    return A.Compose(
        [
            A.PadIfNeeded(min_height=img_size, min_width=img_size,
                          border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
            A.RandomCrop(height=img_size, width=img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.GaussNoise(p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ]
    )


def get_eval_transforms(img_size: int = 256):
    return A.Compose(
        [
            A.PadIfNeeded(min_height=img_size, min_width=img_size,
                          border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
            A.CenterCrop(height=img_size, width=img_size),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2(),
        ]
    )


class HC18Dataset(Dataset):
    """
    Loads (image, mask) pairs from a list of base filenames (without the
    "_Annotation" suffix), e.g. "000_HC.png" -> image, "000_HC_Annotation.png" -> mask.
    """

    def __init__(self, root_dir: str, filenames: list, transform=None):
        self.root_dir = root_dir
        self.filenames = filenames
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        mask_name = img_name.replace(".png", "_Annotation.png")

        img_path = os.path.join(self.root_dir, img_name)
        mask_path = os.path.join(self.root_dir, mask_name)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # HC18 annotation masks are the head BOUNDARY (a thin ellipse outline),
        # not a filled region. Fill it to get a solid binary head mask.
        mask = self._fill_ellipse_outline(mask)

        mask = (mask > 0).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            image = torch.from_numpy(image).unsqueeze(0).float() / 255.0
            mask = torch.from_numpy(mask).float()

        mask = mask.unsqueeze(0) if mask.dim() == 2 else mask
        return image, mask, img_name

    @staticmethod
    def _fill_ellipse_outline(mask: np.ndarray) -> np.ndarray:
        """Fill the thin annotated ellipse boundary to produce a solid mask."""
        binary = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(binary)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(filled, [largest], -1, color=255, thickness=cv2.FILLED)
        else:
            filled = binary  # fallback: keep as-is if no contour found
        return filled


def _extract_patient_id(filename: str) -> str:
    """000_HC.png -> '000' (patient/case identifier used for leakage-free splitting)."""
    match = re.match(r"(\d+)_HC", filename)
    return match.group(1) if match else filename


def patient_level_split(root_dir: str, seed: int = 42, ratios=(0.70, 0.15, 0.15)):
    """
    Splits HC18 training_set images by patient ID (not randomly per-image) so
    the same case never leaks across train/val/test. Returns three filename lists.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1.0"

    all_images = sorted(
        f for f in glob.glob(os.path.join(root_dir, "*.png")) if "_Annotation" not in f
    )
    all_images = [os.path.basename(f) for f in all_images]

    patient_ids = sorted(set(_extract_patient_id(f) for f in all_images))
    rng = random.Random(seed)
    rng.shuffle(patient_ids)

    n = len(patient_ids)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    train_ids = set(patient_ids[:n_train])
    val_ids = set(patient_ids[n_train:n_train + n_val])
    test_ids = set(patient_ids[n_train + n_val:])

    train_files = [f for f in all_images if _extract_patient_id(f) in train_ids]
    val_files = [f for f in all_images if _extract_patient_id(f) in val_ids]
    test_files = [f for f in all_images if _extract_patient_id(f) in test_ids]

    return train_files, val_files, test_files
