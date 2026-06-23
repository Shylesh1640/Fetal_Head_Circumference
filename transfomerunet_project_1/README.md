# Fetal Head Segmentation — U-Net + MiT-B2
## SOTA Implementation on HC18 | Lightning AI Guide

**Target metrics:** Dice ≥ 0.9899 · IoU ≥ 0.9850 · Precision ≥ 0.9897 · Recall ≥ 0.9953

---

## Architecture

```
Ultrasound Image (256×256, 3ch)
        │
  ┌─────▼──────────────────────────────────┐
  │     MiT-B2 Encoder (ImageNet weights)  │
  │  Stage 1 → 64ch  @ H/4  × W/4         │
  │  Stage 2 → 128ch @ H/8  × W/8         │
  │  Stage 3 → 320ch @ H/16 × W/16        │
  │  Stage 4 → 512ch @ H/32 × W/32        │
  └─────────────────────────────────────────┘
        │  skip connections ×4
  ┌─────▼──────────────────────────────────┐
  │     U-Net Decoder (BatchNorm + ReLU)   │
  │  Upsample → 256 → 128 → 64 → 32 → 16  │
  └─────────────────────────────────────────┘
        │
  ┌─────▼──────────────────────────────────┐
  │   Segmentation Head (Conv1×1 + Sigmoid)│
  │   Binary Mask (1, H, W)                │
  └─────────────────────────────────────────┘
        │
  ┌─────▼──────────────────────────────────┐
  │   Post-Processing                      │
  │   Morph Cleanup → Contour → fitEllipse │
  │   Ramanujan HC = π(a+b)(1+3h/(10+√4-3h│
  └─────────────────────────────────────────┘
```

**Loss:** `L = L_BCE + (1 - Dice)` — hybrid handles class imbalance  
**Optimizer:** Adam, lr=1e-4, weight_decay=1e-5  
**Scheduler:** ReduceLROnPlateau (patience=5, factor=0.5)  
**AMP:** Mixed precision (FP16) via torch.cuda.amp — automatic on GPU  

---

## Lightning AI Setup

### 1. Create a Studio
- Go to [lightning.ai](https://lightning.ai)
- New Studio → Select **T4 GPU** (minimum) or **A10G** (recommended)
- Open Terminal

### 2. Install Dependencies
```bash
pip install segmentation-models-pytorch timm albumentations \
            opencv-python grad-cam matplotlib pandas tqdm \
            torch torchvision --upgrade
```

### 3. Get the HC18 Dataset
```bash
# Option A: Direct download via zenodo
pip install zenodo-get
zenodo_get 1327317   # downloads training_set.zip + test_set.zip

# Option B: Manual download
# Go to https://zenodo.org/record/1327317
# Download: training_set.zip and test_set.zip

unzip training_set.zip -d hc18/
unzip test_set.zip     -d hc18/
```

Expected folder structure:
```
hc18/
├── training_set/
│   ├── 000_HC.png              ← ultrasound image
│   ├── 000_HC_Annotation.png   ← ellipse mask annotation
│   ├── 001_HC.png
│   ├── 001_HC_Annotation.png
│   └── ... (999 pairs total)
└── test_set/
    ├── 001_HC.png
    └── ... (335 images, no masks)
```

### 4. Upload the Script
- Drag `fetal_hc_segmentation.py` into your Studio file tree
  OR
```bash
# If using Lightning AI CLI
lightning upload fetal_hc_segmentation.py
```

---

## Running the Code

### Train
```bash
python fetal_hc_segmentation.py --mode train
# Optional overrides:
python fetal_hc_segmentation.py --mode train --epochs 150 --batch_size 16 --lr 5e-5
```
Outputs:
- `checkpoints/best_model.pth` — best checkpoint (by Val Dice)
- `results/training_history.csv` — all metrics per epoch
- `results/training_curves.png` — 7-panel metric plots

### Evaluate (val set, all metrics)
```bash
python fetal_hc_segmentation.py --mode evaluate
```
Prints: Loss, Dice, IoU, Precision, Recall, Accuracy, F1, HC MAE (mm), HC MSE (mm²)

### Visualize Predictions
```bash
python fetal_hc_segmentation.py --mode visualize
```
Saves: `results/prediction_samples.png` — 5 samples showing input | GT | pred | overlay

### Grad-CAM++ on a Single Image
```bash
python fetal_hc_segmentation.py --mode gradcam \
    --image_path hc18/training_set/000_HC.png
```
Saves: `results/gradcam_output.png` — input | pred mask | attention heatmap

### Predict Test Set (HC18 submission)
```bash
python fetal_hc_segmentation.py --mode predict
```
Saves:
- `results/test_masks/` — binary predicted masks for each test image
- `results/test_predictions.csv` — filename + HC in mm

---

## Metrics Tracked (Train + Val per Epoch)

| Metric    | Formula                                | Paper Target |
|-----------|----------------------------------------|--------------|
| Dice      | 2TP / (2TP + FP + FN)                 | **0.9899**   |
| IoU       | TP / (TP + FP + FN)                   | **0.9850**   |
| Precision | TP / (TP + FP)                        | 0.9897       |
| Recall    | TP / (TP + FN)                        | 0.9953       |
| Accuracy  | (TP+TN) / (TP+FP+FN+TN)              | —            |
| F1 Score  | 2·P·R / (P+R)                         | 0.9899       |
| Loss      | BCE + (1 − Dice)                      | 35.35        |
| HC MAE    | mean\|pred_HC − gt_HC\| (mm)          | 0.54 mm      |
| HC MSE    | mean(pred_HC − gt_HC)² (mm²)          | 0.298 mm²    |

---

## Expected Training Timeline (Lightning AI)

| GPU      | Epochs | Time/Epoch | Total     |
|----------|--------|------------|-----------|
| T4 16GB  | 100    | ~3 min     | ~5 hours  |
| A10G 24GB| 100    | ~2 min     | ~3 hours  |
| A100 40GB| 100    | ~1 min     | ~1.5 hours|

Early stopping triggers at patience=15 (no Val Dice improvement), typically around epoch 60–80.

---

## Configuration (edit CFG dict in the script)

```python
CFG = {
    "train_dir":   "hc18/training_set",
    "test_dir":    "hc18/test_set",
    "checkpoint":  "checkpoints/best_model.pth",
    "results_dir": "results",

    "encoder":         "mit_b2",       # Change to "mit_b0" for lightweight
    "encoder_weights": "imagenet",
    "image_size":      256,            # Increase to 512 for better results (needs more VRAM)

    "epochs":       100,
    "batch_size":   8,                 # Reduce to 4 if OOM
    "lr":           1e-4,
    "val_split":    0.15,
    "patience":     15,
    "threshold":    0.5,

    "pixel_spacing_mm": 0.154,         # HC18 standard pixel spacing
}
```

---

## Key Design Decisions

**Why MiT-B2?** Transformer encoder captures long-range global context (skull curvature, overall head shape) that CNNs miss. Combined with U-Net's skip connections, it preserves fine local boundary detail simultaneously.

**Why Hybrid Loss?** BCE alone ignores spatial overlap; Dice alone is unstable early in training. Combined = fast convergence + good class-imbalance handling.

**Why Ramanujan's formula?** The fetal head is elliptical, not circular. Ramanujan's approximation error < 0.0001% for typical head eccentricities vs ~10% error from the naive π(a+b) formula.

**Why patient-level split?** Random splits can put the same patient in both train and val (since HC18 may have multiple images per patient). Patient-level split is the correct evaluation protocol.
