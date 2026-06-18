"""
dataset.py
----------
Custom Dataset for the HC18 Grand Challenge fetal head ultrasound dataset.

HC18 layout (as distributed on the Grand Challenge / Zenodo / Kaggle mirror):

    training_set/
        000_HC.png                 <- grayscale ultrasound image
        000_HC_Annotation.png      <- 1px white ellipse contour on black bg
        001_HC.png
        001_HC_Annotation.png
        ...
    training_set_pixel_size_and_HC.csv
        columns: filename, pixel size(mm), head circumference (mm)

The "_Annotation" files contain only the ELLIPSE OUTLINE, not a filled
mask. This dataset class therefore:
    1. loads the contour annotation image,
    2. finds the contour with cv2.findContours,
    3. fills it with cv2.fillPoly / cv2.drawContours(thickness=-1)
       to obtain a solid binary mask of the fetal head.

If you are instead using a version of the dataset that already ships
filled binary masks (e.g. some Kaggle re-uploads), set
`already_filled=True` and the fill step will be skipped.
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def annotation_to_filled_mask(annotation_img: np.ndarray) -> np.ndarray:
    """
    Convert an HC18-style ellipse-contour annotation image into a solid
    (filled) binary mask.

    Parameters
    ----------
    annotation_img : np.ndarray
        Single-channel image containing a thin white ellipse contour
        on a black background (values in [0, 255]).

    Returns
    -------
    np.ndarray
        uint8 array, same shape as input, with values {0, 255}, where
        255 marks the interior + boundary of the fetal head ellipse.
    """
    if annotation_img.ndim == 3:
        annotation_img = cv2.cvtColor(annotation_img, cv2.COLOR_BGR2GRAY)

    # Binarize the thin contour line
    _, binary = cv2.threshold(annotation_img, 30, 255, cv2.THRESH_BINARY)

    # Dilate slightly to close any 1px gaps in the hand-drawn ellipse before
    # contour extraction -- this prevents leakage through small breaks.
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    mask = np.zeros_like(binary)
    if len(contours) == 0:
        # Fallback: nothing detected, return empty mask rather than crash
        return mask

    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(mask, [largest], -1, color=255, thickness=-1)
    return mask


class HC18Dataset(Dataset):
    """
    Returns (image_tensor, mask_tensor, meta) where:
        image_tensor : FloatTensor [3, H, W], normalized to [0, 1]
        mask_tensor  : FloatTensor [1, H, W], values in {0, 1}
        meta         : dict with filename, pixel_size_mm, true_hc_mm,
                        and the original (pre-resize) image size, needed
                        to convert predicted pixel-space HC back to mm.
    """

    def __init__(
        self,
        data_dir: str,
        csv_path: str,
        filenames,
        img_size: int = 256,
        transform=None,
        already_filled: bool = False,
    ):
        self.data_dir = data_dir
        self.img_size = img_size
        self.transform = transform
        self.already_filled = already_filled

        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        # Normalize expected column names across known CSV variants
        rename_map = {}
        for c in df.columns:
            cl = c.lower()
            if "pixel" in cl:
                rename_map[c] = "pixel_size_mm"
            elif "circumference" in cl or cl == "hc":
                rename_map[c] = "hc_mm"
            elif "filename" in cl:
                rename_map[c] = "filename"
        df = df.rename(columns=rename_map)
        df = df.set_index("filename")
        self.df = df

        self.filenames = list(filenames)

    def __len__(self):
        return len(self.filenames)

    def _load_image_path(self, fname: str) -> str:
        return os.path.join(self.data_dir, fname)

    def _annotation_path(self, fname: str) -> str:
        # e.g. 010_HC.png -> 010_HC_Annotation.png
        stem, ext = os.path.splitext(fname)
        return os.path.join(self.data_dir, f"{stem}_Annotation{ext}")

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        img_path = self._load_image_path(fname)
        ann_path = self._annotation_path(fname)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        orig_h, orig_w = image.shape[:2]

        annotation = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)
        if annotation is None:
            raise FileNotFoundError(f"Could not read annotation: {ann_path}")

        if self.already_filled:
            _, mask = cv2.threshold(annotation, 30, 255, cv2.THRESH_BINARY)
        else:
            mask = annotation_to_filled_mask(annotation)

        # Resize to a slightly larger canvas than the final network input so
        # that RandomCrop (applied in the train transform) has room to act
        # as a real spatial augmentation rather than a no-op. Validation/
        # test transforms use CenterCrop on the same canvas size, so all
        # splits still end up at exactly (img_size, img_size).
        resize_to = int(self.img_size * 1.15)
        image_rs = cv2.resize(image, (resize_to, resize_to), interpolation=cv2.INTER_LINEAR)
        mask_rs = cv2.resize(mask, (resize_to, resize_to), interpolation=cv2.INTER_NEAREST)

        # Convert grayscale -> 3-channel (SMP encoders pretrained on RGB ImageNet)
        image_rgb = cv2.cvtColor(image_rs, cv2.COLOR_GRAY2RGB)

        sample = {"image": image_rgb, "mask": mask_rs}

        if self.transform is not None:
            augmented = self.transform(image=sample["image"], mask=sample["mask"])
            image_out = augmented["image"]
            mask_out = augmented["mask"]
        else:
            # No transform supplied: just center-resize down to img_size and
            # normalize to [0, 1] manually.
            image_out = cv2.resize(image_rgb, (self.img_size, self.img_size))
            mask_out = cv2.resize(mask_rs, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # image_out may already be a tensor if albumentations ToTensorV2 was used
        if isinstance(image_out, np.ndarray):
            image_tensor = torch.from_numpy(image_out / 255.0).permute(2, 0, 1).float()
        else:
            image_tensor = image_out.float()
            if image_tensor.max() > 1.5:
                image_tensor = image_tensor / 255.0

        if isinstance(mask_out, np.ndarray):
            mask_tensor = torch.from_numpy((mask_out > 127).astype(np.float32)).unsqueeze(0)
        else:
            mask_tensor = mask_out.float()
            if mask_tensor.dim() == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
            mask_tensor = (mask_tensor > 0.5).float()

        row = self.df.loc[fname] if fname in self.df.index else None
        pixel_size_mm = float(row["pixel_size_mm"]) if row is not None else float("nan")
        hc_mm = float(row["hc_mm"]) if row is not None else float("nan")

        meta = {
            "filename": fname,
            "pixel_size_mm": pixel_size_mm,
            "true_hc_mm": hc_mm,
            "orig_h": orig_h,
            "orig_w": orig_w,
        }

        return image_tensor, mask_tensor, meta


def patient_level_split(filenames, val_split=0.15, test_split=0.10, seed=42):
    """
    HC18 filenames look like '010_HC.png' or '010_2HC.png' -- images sharing
    the same leading patient ID came from the same scan session. To avoid
    leakage, we split by patient ID rather than by individual file.
    """
    import re
    from collections import defaultdict

    rng = np.random.RandomState(seed)
    groups = defaultdict(list)
    for f in filenames:
        m = re.match(r"(\d+)", f)
        patient_id = m.group(1) if m else f
        groups[patient_id].append(f)

    patient_ids = list(groups.keys())
    rng.shuffle(patient_ids)

    n = len(patient_ids)
    n_test = max(1, int(n * test_split))
    n_val = max(1, int(n * val_split))

    test_ids = set(patient_ids[:n_test])
    val_ids = set(patient_ids[n_test:n_test + n_val])
    train_ids = set(patient_ids[n_test + n_val:])

    train_files = [f for pid in train_ids for f in groups[pid]]
    val_files = [f for pid in val_ids for f in groups[pid]]
    test_files = [f for pid in test_ids for f in groups[pid]]

    return train_files, val_files, test_files
