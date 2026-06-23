# Fetal Head Segmentation — U-Net + MiT-B2 (SOTA Transformer-U-Net)

A clean, verified implementation of **U-Net with a MiT-B2 (SegFormer) transformer
encoder** for fetal head segmentation and head-circumference (HC) estimation on
the **HC18** ultrasound dataset, reproducing the architecture described in the
uploaded paper.

Built on top of the canonical, actively-maintained
[`segmentation_models_pytorch`](https://github.com/qubvel-org/segmentation_models.pytorch)
(SMP) library, which natively supports `mit_b0`–`mit_b5` (SegFormer / Mix
Vision Transformer) encoders pretrained on ImageNet as drop-in U-Net
backbones — this is exactly the "U-Net + MiT-B2" combination from the paper,
not a hand-rolled reimplementation, so the encoder weights and architecture
are correct and tested by a large open-source community.

---

## 1. Project structure

```
fetal_hc_project/
├── config.py          # all paths & hyperparameters in one place
├── dataset.py          # HC18 loading, mask-from-ellipse-outline filling, augmentation
├── model.py             # U-Net + MiT-B2 model builder (segmentation_models_pytorch)
├── losses.py             # Hybrid Dice + BCE loss
├── metrics.py             # Dice, IoU, Precision, Recall, Accuracy, F1 (SMP's official metrics API)
├── train.py                 # training loop, checkpointing, early stopping, CSV logging
├── evaluate.py                # test-set evaluation + training curve plots
├── postprocess.py               # mask cleanup -> contour -> ellipse fit -> Ramanujan HC
├── inference.py                   # single-image inference + Grad-CAM visualization
├── sanity_check.py                  # run BEFORE train.py to catch setup issues early
└── requirements.txt
```

---

## 2. Setup on Lightning AI

### Step 1 — Create a Studio
On [lightning.ai](https://lightning.ai), create a new **Studio** with a GPU
(an L4 or A10G is enough; the paper used an RTX 4060 6GB, so anything ≥8GB
VRAM will comfortably fit batch size 8–16 at 256×256).

### Step 2 — Upload the code
Upload this entire `fetal_hc_project/` folder into your Studio's file browser
(drag-and-drop works), or `git clone` it if you've pushed it to your own repo.

### Step 3 — Install dependencies
Open a Studio terminal:

```bash
cd fetal_hc_project
pip install -r requirements.txt
```

### Step 4 — Download HC18 dataset
The dataset is hosted on Zenodo (official source, free, no login required):

```bash
mkdir -p /teamspace/studios/this_studio/data
cd /teamspace/studios/this_studio/data

# Official HC18 dataset (van den Heuvel et al., 2018) — training_set.zip
# contains 999 images, 999 ellipse-outline annotation PNGs, and a CSV of
# pixel sizes / reference HC values.
wget https://zenodo.org/records/1327317/files/training_set.zip
unzip training_set.zip
```

This produces:
```
data/training_set/
    000_HC.png
    000_HC_Annotation.png
    001_HC.png
    001_HC_Annotation.png
    ...
    training_set_pixel_size_and_HC.csv
```

> **Important — only the 999 "training_set" images have ground-truth masks.**
> HC18's official 335-image "test_set" has NO public masks (it's used for
> the closed Grand Challenge leaderboard). This project therefore splits the
> **999 annotated images** itself into train/val/test (70/15/15) so that
> real Dice/IoU/Precision/Recall/Accuracy/F1 can be reported on all three
> splits — see `config.py` → `TRAIN_FRAC/VAL_FRAC/TEST_FRAC`.

### Step 5 — Point the config at your data
Open `config.py` and confirm `DATA_ROOT` matches where you unzipped the data:

```python
DATA_ROOT = "/teamspace/studios/this_studio/data/training_set"
```

### Step 6 — Run the sanity check (do this before training!)
```bash
python sanity_check.py
```
This verifies: package versions, GPU detection, dataset path, that the
ellipse-outline annotation masks fill correctly into solid head masks (saves
a visual to `outputs/plots/sanity_check_mask.png` — **open and eyeball it**),
and that one forward+backward pass through the real model runs cleanly.

### Step 7 — Train
```bash
python train.py
```
- Logs every epoch's `train/val` Loss, Dice, IoU, Precision, Recall,
  Accuracy, F1 to `outputs/logs/training_log.csv`
- Saves the best checkpoint (highest validation Dice) to
  `outputs/checkpoints/best_model.pth`
- Early-stops after 15 epochs without validation Dice improvement
- Uses automatic mixed precision (AMP) on GPU for speed/VRAM savings

A full 100-epoch run on an L4/A10G GPU at batch size 8, 256×256 typically
takes a few hours — exact time depends on the specific GPU tier.

### Step 8 — Evaluate on the held-out test set
```bash
python evaluate.py
```
This prints and saves final **Loss, Dice, IoU, Precision, Recall, Accuracy,
F1** on the test split (`outputs/logs/test_results.json`), and plots all
seven train-vs-val curves to `outputs/plots/training_curves.png`.

### Step 9 — Run inference + Grad-CAM on a single image
```bash
python inference.py --image data/training_set/000_HC.png --pixel_size_mm 0.123
```
(Look up the correct `pixel_size_mm` for your chosen image in
`training_set_pixel_size_and_HC.csv` if you want HC reported in millimetres
instead of pixels.) This saves a 3-panel figure (input / predicted mask +
fitted ellipse / Grad-CAM heatmap) to `outputs/predictions/`.

---

## 3. Metrics — how they're computed (and why they're correct)

All of **Dice Score, IoU, Precision, Recall, Accuracy, F1 Score** are computed
using `segmentation_models_pytorch.metrics`, SMP's own tested metrics module,
rather than hand-rolled formulas:

```python
tp, fp, fn, tn = smp.metrics.get_stats(probs, targets, mode="binary", threshold=0.5)
iou       = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
f1 (dice) = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro")
recall    = smp.metrics.recall(tp, fp, fn, tn, reduction="micro")
accuracy  = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")
```

- **Reduction = "micro"**: confusion-matrix counts are pooled across *every
  pixel in every image in the epoch* before the ratio is computed. This is
  the standard way segmentation papers report epoch-level metrics (as
  opposed to naively averaging per-image scores, which is more sensitive to
  outlier images).
- **Dice == F1** for binary segmentation — both are reported because the
  paper names them separately, but they will always be numerically
  identical; this is expected and correct, not a bug.
- **Loss** reported alongside metrics is the hybrid Dice-BCE training
  objective, not a metric per se, but tracked the same way for the standard
  train/val loss-curve diagnostic.

---

## 4. Architecture summary

```
Ultrasound image (256×256, replicated to 3ch)
        ↓
MiT-B2 encoder (SegFormer / Mix Vision Transformer, ImageNet-pretrained)
   — produces multi-scale hierarchical features via self-attention
        ↓
U-Net decoder (skip connections from each encoder stage)
   — upsamples back to full resolution, preserving spatial detail
        ↓
1×1 conv segmentation head → raw logits [B, 1, 256, 256]
        ↓ (sigmoid + threshold 0.5)
Binary head mask
        ↓
Morphological cleanup → largest external contour → cv2.fitEllipse
        ↓
Ramanujan's ellipse-perimeter approximation → Head Circumference (mm)
        ↓ (in parallel)
Grad-CAM on last MiT-B2 stage → interpretability heatmap
```

Loss: `0.5 × BCEWithLogitsLoss + 0.5 × DiceLoss` (hybrid Dice-BCE, matching
the paper, for stability under the class imbalance between head pixels and
background pixels).

Optimizer: Adam, lr=1e-4, weight_decay=1e-5, `ReduceLROnPlateau` scheduler
on validation Dice, gradient clipping at norm 1.0, up to 100 epochs with
early stopping (patience 15).

---

## 5. Common issues

| Symptom | Likely cause / fix |
|---|---|
| `FileNotFoundError` for `DATA_ROOT` | Check the zip extracted into a `training_set` *folder* — adjust `config.DATA_ROOT` to match the exact unzip path. |
| Sanity-check mask looks empty / wrong | Open `outputs/plots/sanity_check_mask.png`. If the filled mask doesn't look like a head, inspect a few raw `*_Annotation.png` files directly — they should be thin white ellipse outlines on black. |
| Out-of-memory on GPU | Lower `BATCH_SIZE` in `config.py` (e.g., 4), or reduce `IMAGE_SIZE` to 224. |
| Grad-CAM throws a layer-name error | `model.encoder.encoder.block4[-1].norm1` is MiT-B2's last transformer-stage norm layer as exposed via SMP/timm; if SMP's internal naming changes in a future version, run `print(model)` and pick a comparable late-stage layer. |
| Training Dice stuck near 0 early on | Normal for the first several epochs with a transformer encoder — MiT-B2 has more parameters than a CNN baseline and needs a short warm-up before Dice climbs sharply. |

---

## 6. References

- van den Heuvel et al., *"Automated measurement of fetal head circumference
  using 2D ultrasound images,"* PLoS ONE 13(8): e0200412, 2018 — HC18 dataset
  paper. Dataset: https://zenodo.org/records/1327317
- Yakubovskiy, P., *Segmentation Models PyTorch*, GitHub, 2019 —
  https://github.com/qubvel-org/segmentation_models.pytorch
- Xie et al., *"SegFormer: Simple and Efficient Design for Semantic
  Segmentation with Transformers,"* NeurIPS 2021 — source of the MiT
  (Mix Vision Transformer) encoder family.
- Ronneberger et al., *"U-Net: Convolutional Networks for Biomedical Image
  Segmentation,"* MICCAI 2015.
- Selvaraju et al., *"Grad-CAM: Visual Explanations from Deep Networks via
  Gradient-based Localization,"* ICCV 2017. Implementation:
  https://github.com/jacobgil/pytorch-grad-cam
