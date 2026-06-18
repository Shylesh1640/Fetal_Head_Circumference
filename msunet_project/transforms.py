"""
transforms.py
-------------
Augmentation pipelines built with Albumentations, matching the paper's
description (Section III-A):
    - Gaussian noise injection
    - Padding to 256 x 256
    - Random 90-degree rotations
    - Random cropping
    - Horizontal and vertical flipping
    - Brightness / contrast adjustment

Train transform applies these augmentations; validation/test transform only
resizes + normalizes (no stochastic augmentation), which is standard
practice and avoids leaking augmentation noise into evaluation metrics.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transform(img_size: int = 256):
    return A.Compose(
        [
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0),
            A.RandomCrop(height=img_size, width=img_size, pad_if_needed=True),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.GaussNoise(std_range=(0.02, 0.08), p=0.3),
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )


def get_val_transform(img_size: int = 256):
    return A.Compose(
        [
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0),
            A.CenterCrop(height=img_size, width=img_size),
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )
