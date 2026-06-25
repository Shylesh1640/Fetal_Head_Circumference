"""
HC18 dataset loader.

Expected folder layout after you download the dataset from
https://zenodo.org/records/1322001 and unzip it:

data/
  training_set/
    000_HC.png
    000_HC_Annotation.png
    001_HC.png
    001_HC_Annotation.png
    ...
  training_set_pixel_size_and_HC.csv   (optional, has true HC + pixel size)

We derive a "patient id" from the numeric prefix of the filename so that
augmented/duplicate frames from the same scan never leak across
train/val/test (the paper explicitly calls this out as important).
"""

import os
import glob
import re
import numpy as np
import cv2
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


def _patient_id(filename):
    m = re.match(r"(\d+)", os.path.basename(filename))
    return m.group(1) if m else os.path.basename(filename)


def build_splits(data_dir, val_size=0.15, test_size=0.15, seed=42):
    images = sorted(glob.glob(os.path.join(data_dir, "*_HC.png")))
    images = [f for f in images if "Annotation" not in f]
    pairs = []
    for img in images:
        mask = img.replace(".png", "_Annotation.png")
        if os.path.exists(mask):
            pairs.append((img, mask))

    patient_ids = sorted(set(_patient_id(p[0]) for p in pairs))
    train_ids, temp_ids = train_test_split(
        patient_ids, test_size=(val_size + test_size), random_state=seed
    )
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=test_size / (val_size + test_size), random_state=seed
    )

    def subset(ids):
        ids = set(ids)
        return [p for p in pairs if _patient_id(p[0]) in ids]

    return subset(train_ids), subset(val_ids), subset(test_ids)


class HC18Dataset(Dataset):
    def __init__(self, pairs, transform=None, img_size=256):
        self.pairs = pairs
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size),
                           interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype("float32")

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # 3ch for ImageNet encoders

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        image = image.astype("float32") / 255.0
        image = np.transpose(image, (2, 0, 1))
        mask = np.expand_dims(mask, 0)

        return image, mask, os.path.basename(img_path)


def get_train_transform():
    import albumentations as A
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.3),
    ])
