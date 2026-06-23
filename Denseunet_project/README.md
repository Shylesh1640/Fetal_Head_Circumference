# Dense U-Net — Fetal Head Segmentation (HC18)

A from-scratch PyTorch implementation of a **Dense U-Net** (FC-DenseNet /
"Tiramisu"-style encoder-decoder with densely-connected blocks and U-Net
skip connections), trained with PyTorch Lightning on the **HC18** fetal
head ultrasound dataset, with full **Dice, IoU, Precision, Recall,
Accuracy, F1, and Loss** tracking for train / validation / test.

```
src/
  dense_unet.py        # Dense U-Net model (DenseBlock + TransitionDown/Up)
  dataset.py            # HC18 Dataset, patient-level split, augmentations
  losses.py              # Hybrid Dice + BCE loss
  metrics_utils.py        # torchmetrics MetricCollection (Dice/IoU/P/R/Acc/F1)
  lightning_module.py     # LightningModule + LightningDataModule
  train.py                 # CLI training entry point
  inference.py               # Checkpoint -> mask -> ellipse fit -> HC (mm)
requirements.txt
```

## 1. Why this architecture

A "Dense U-Net" replaces every plain conv block of a standard U-Net with a
**DenseBlock**: a stack of `BN -> ReLU -> 3x3 Conv -> Dropout` layers whose
outputs are concatenated (not summed), so every layer reuses every previous
layer's features within the block. Between resolution levels we use:

- **TransitionDown**: `BN -> ReLU -> 1x1 Conv -> Dropout -> 2x2 MaxPool`
- **TransitionUp**: `3x3 stride-2 ConvTranspose2d`, followed by concatenating
  the corresponding encoder skip connection (the classic U-Net skip).

This is the standard 2D reference design behind the term "Dense U-Net" in
the segmentation literature (Jégou et al., *The One Hundred Layers
Tiramisu*, CVPR-W 2017), as distinct from the 3D *H-DenseUNet* used for CT
volumes. The default config in `dense_unet.py` is a compact ~1.4M-parameter,
5-level network — light enough to train comfortably on a single cloud GPU.

## 2. Dataset: HC18

Download the HC18 Grand Challenge dataset from Zenodo:
https://zenodo.org/records/1327317

Unzip it so you have:

```
hc18/
  training_set/
    000_HC.png
    000_HC_Annotation.png
    001_HC.png
    001_HC_Annotation.png
    ...
    training_set_pixel_size_and_HC.csv
```

`dataset.py` automatically:
1. Pairs every `*_HC.png` with its `*_HC_Annotation.png`.
2. Fills the thin annotation contour into a solid binary mask.
3. Splits patients (not individual files) into **train / val / test**
   (default 70% / 15% / 15%) so no scan leaks across splits.
4. Applies the paper's augmentations (pad/crop, flips, 90° rotation,
   Gaussian noise, brightness/contrast) to the training split only.

You only need to point `--data_root` at the `hc18/` folder (the script
looks for `training_set/` inside it automatically).

## 3. Running on Lightning AI (Lightning AI Studio)

### a) Create a Studio and upload the project
1. Go to https://lightning.ai → **New Studio** → choose a GPU machine
   (a single T4 or L4 is enough for the default config).
2. Open the Studio's terminal and either:
   - `git clone` your repo containing this project, **or**
   - drag-and-drop / upload this folder into the Studio's file browser
     (e.g. into `/teamspace/studios/this_studio/dense-unet-fetal`).

### b) Install dependencies
```bash
cd dense-unet-fetal
pip install -r requirements.txt
```

### c) Get the dataset onto the Studio
```bash
mkdir -p data
# Upload the unzipped HC18 folder into ./data/hc18, or download it directly:
# (HC18 requires a Zenodo download; use the Studio's file upload UI or
#  `curl`/`wget` the Zenodo record URL into ./data, then unzip.)
```

### d) Train
```bash
python src/train.py \
  --data_root ./data/hc18 \
  --image_size 256 \
  --batch_size 8 \
  --max_epochs 100 \
  --gpus 1 \
  --precision 16-mixed \
  --output_dir ./outputs \
  --run_name dense_unet_hc18
```

Useful flags:
- `--gpus 0` to force CPU (slow; only for smoke-testing).
- `--down_blocks 4 4 4 4 4 --up_blocks 4 4 4 4 4 --growth_rate 12` — default
  Dense U-Net depth/width; increase `growth_rate` or block counts for a
  larger-capacity model if you have spare VRAM.
- `--early_stop_patience 15` — stops training if `val_loss` stalls.

The script automatically:
- Logs `loss`, `Dice`, `IoU`, `Precision`, `Recall`, `Accuracy`, `F1` every
  epoch for **train** and **val** (CSV + TensorBoard, under `outputs/`).
- Saves the top-3 checkpoints by `val_loss` plus `last.ckpt`.
- After training, reloads the **best** checkpoint and runs a final
  `validate()` + `test()` pass, printing a clean metrics summary for both
  splits (train metrics are visible in the CSV/TensorBoard logs from the
  last training epoch).

### e) Monitor training
```bash
tensorboard --logdir outputs/tb_logs
```
Or inspect `outputs/logs/dense_unet_hc18/metrics.csv` directly (e.g. with
pandas) for per-epoch `train_*`, `val_*` columns.

### f) Run inference + HC measurement on a new image
```bash
python src/inference.py \
  --checkpoint outputs/checkpoints/<best>.ckpt \
  --image data/hc18/training_set/000_HC.png \
  --pixel_size_mm 0.143 \
  --save_mask predicted_mask.png
```
`--pixel_size_mm` should come from HC18's
`training_set_pixel_size_and_HC.csv` (per-image physical pixel size); this
converts the fitted-ellipse axes from pixels to millimetres before applying
Ramanujan's perimeter approximation for the final HC value.

## 4. Reported metrics — definitions

| Metric | Computed on | Formula |
|---|---|---|
| Loss | continuous logits | `BCEWithLogits + (1 - SoftDice)` |
| Dice Score | continuous probabilities | `2·Σ(p·t) / (Σp + Σt)` |
| IoU | thresholded @ 0.5 | `TP / (TP + FP + FN)` |
| Precision | thresholded @ 0.5 | `TP / (TP + FP)` |
| Recall | thresholded @ 0.5 | `TP / (TP + FN)` |
| Accuracy | thresholded @ 0.5 | `(TP + TN) / (TP+TN+FP+FN)` |
| F1 Score | thresholded @ 0.5 | `2·P·R / (P + R)` |

All metrics are accumulated per-epoch with `torchmetrics.MetricCollection`
(one independent instance per stage), so values reported for `train_*`,
`val_*`, `test_*` are true epoch-level aggregates, not batch averages.

## 5. Sanity-checking the model alone

```bash
python src/dense_unet.py
# Output shape: (2, 1, 256, 256)
# Total parameters: 1,372,465 (1.37 M)
```

## 6. Notes / things you may want to tune

- The default network is intentionally compact. If you want higher
  capacity (closer to the original Tiramisu paper, ~9M+ params), increase
  `--growth_rate` (e.g. 16) and/or block depths (e.g. `6 6 6 6 6`).
- Mixed precision (`--precision 16-mixed`) roughly halves VRAM usage and
  speeds up training on modern GPUs with negligible accuracy impact.
- For full reproducibility, `train.py` calls `pl.seed_everything(seed)`
  and the dataset split is also seeded — re-running with the same `--seed`
  reproduces the same train/val/test patients.
