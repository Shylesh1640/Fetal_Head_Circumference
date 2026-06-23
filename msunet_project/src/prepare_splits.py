"""
prepare_splits.py
------------------
HC18 ships annotations as a THIN ELLIPSE BOUNDARY drawn on a black image
(e.g. "000_HC_Annotation.png"), not a filled mask. To train a segmentation
network we need a solid binary mask of the fetal head region.

This script:
  1. Reads every "<id>_HC.png" / "<id>_HC_Annotation.png" pair from
     config.RAW_IMAGE_DIR.
  2. Fills the ellipse boundary into a solid binary mask using a
     dilate -> binary_fill_holes -> erode pipeline (robust to small gaps
     in the drawn boundary).
  3. Saves filled masks to data/processed/masks/.
  4. Creates a reproducible train/val/test split (70/15/15) and writes it
     to data/processed/splits.csv together with pixel size (mm/px) and the
     ground-truth HC (mm) taken from training_set_pixel_size_and_HC.csv, so
     that the same CSV can later be used to validate the ellipse-fitting /
     HC-regression post-processing step.

Run:
    python src/prepare_splits.py
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config as cfg


def fill_ellipse_annotation(annotation_path: str, dilate_px: int = 2) -> np.ndarray:
    """Convert a thin ellipse-boundary annotation into a solid binary mask.

    HC18 annotations are a single-pixel-wide white ellipse outline on a
    black background. `binary_fill_holes` fills the interior IF the
    boundary forms a fully closed ring; a small dilation closes tiny gaps
    caused by anti-aliasing / rasterization, and we erode back afterwards
    so the mask is not artificially inflated.
    """
    ann = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)
    if ann is None:
        raise FileNotFoundError(f"Could not read annotation: {annotation_path}")

    binary = (ann > 30).astype(np.uint8)

    kernel = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)

    filled = binary_fill_holes(dilated.astype(bool)).astype(np.uint8)

    eroded = cv2.erode(filled, kernel, iterations=1)

    # Safety net: if fill_holes failed (e.g. boundary touches image edge and
    # never closes), fall back to convex hull of the largest contour so we
    # never emit an all-zero mask.
    if eroded.sum() < 50:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            largest = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(largest)
            fallback = np.zeros_like(binary)
            cv2.drawContours(fallback, [hull], -1, 1, thickness=cv2.FILLED)
            eroded = fallback

    return (eroded * 255).astype(np.uint8)


def main():
    if not os.path.isdir(cfg.RAW_IMAGE_DIR):
        raise FileNotFoundError(
            f"Could not find {cfg.RAW_IMAGE_DIR}.\n"
            "Download the HC18 dataset from https://zenodo.org/records/1327317, "
            "unzip it, and set HC18_DATA_ROOT to the folder that contains "
            "'training_set/' and 'training_set_pixel_size_and_HC.csv', e.g.\n"
            "  export HC18_DATA_ROOT=/teamspace/studios/this_studio/hc18_data"
        )

    csv_df = pd.read_csv(cfg.RAW_CSV_PATH)
    csv_df.columns = [c.strip() for c in csv_df.columns]
    # Expected columns: 'filename', 'pixel size(mm)', 'head circumference (mm)'
    filename_col = [c for c in csv_df.columns if "filename" in c.lower()][0]
    pixel_col = [c for c in csv_df.columns if "pixel" in c.lower()][0]
    hc_col = [c for c in csv_df.columns if "circumference" in c.lower()][0]

    mask_dir = os.path.join(cfg.PROCESSED_DIR, "masks")
    os.makedirs(mask_dir, exist_ok=True)

    records = []
    image_files = sorted(
        f for f in os.listdir(cfg.RAW_IMAGE_DIR)
        if f.endswith("_HC.png") and "Annotation" not in f
    )

    print(f"Found {len(image_files)} ultrasound images. Filling ellipse masks...")
    for fname in image_files:
        case_id = fname.replace("_HC.png", "")
        img_path = os.path.join(cfg.RAW_IMAGE_DIR, fname)
        ann_path = os.path.join(cfg.RAW_IMAGE_DIR, f"{case_id}_HC_Annotation.png")

        if not os.path.exists(ann_path):
            print(f"  [skip] no annotation for {fname}")
            continue

        mask = fill_ellipse_annotation(ann_path)
        mask_path = os.path.join(mask_dir, f"{case_id}_mask.png")
        cv2.imwrite(mask_path, mask)

        row = csv_df[csv_df[filename_col] == fname]
        pixel_size_mm = float(row[pixel_col].values[0]) if len(row) else np.nan
        hc_mm = float(row[hc_col].values[0]) if len(row) else np.nan

        records.append({
            "case_id": case_id,
            "image_path": os.path.relpath(img_path, cfg.PROJECT_ROOT),
            "mask_path": os.path.relpath(mask_path, cfg.PROJECT_ROOT),
            "pixel_size_mm": pixel_size_mm,
            "hc_mm_groundtruth": hc_mm,
        })

    df = pd.DataFrame(records)
    print(f"Prepared {len(df)} image/mask pairs.")

    train_df, temp_df = train_test_split(
        df, train_size=cfg.TRAIN_FRAC, random_state=cfg.SPLIT_SEED, shuffle=True
    )
    val_frac_of_remainder = cfg.VAL_FRAC / (cfg.VAL_FRAC + cfg.TEST_FRAC)
    val_df, test_df = train_test_split(
        temp_df, train_size=val_frac_of_remainder, random_state=cfg.SPLIT_SEED, shuffle=True
    )

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full_df.to_csv(cfg.SPLITS_CSV, index=False)

    print(f"Split sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    print(f"Saved split manifest to: {cfg.SPLITS_CSV}")


if __name__ == "__main__":
    main()
