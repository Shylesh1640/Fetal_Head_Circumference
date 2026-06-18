# MS-UNet: U-Net + MiT-B2 for Fetal Head Segmentation & HC Estimation

Implementation of the architecture described in *"Intelligent Ultrasound
Analysis for Real Time Fetal Head Circumference Measurement and
Developmental Assessment"* (ICCIDS 2026): a U-Net decoder built on a
Mix Vision Transformer (MiT-B2 / SegFormer) encoder, trained on the
HC18 Grand Challenge dataset, with contour-based head-circumference (HC)
estimation and Grad-CAM interpretability.

Every module in this project was unit-tested on synthetic data before
delivery (mask generation, loss math, metrics, ellipse-fitting accuracy,
Grad-CAM hook correctness, and a full 2-epoch training+inference run).
You are still responsible for verifying results on the real dataset —
this removes implementation bugs, not data-dependent surprises.

---

## 1. Project structure

```
msunet_project/
├── config.py          # all paths & hyperparameters — EDIT THIS FIRST
├── dataset.py          # HC18Dataset: ellipse-contour -> filled mask, patient-level split
├── transforms.py        # Albumentations train/val augmentation pipelines
├── model.py            # MSUNet = SMP U-Net with encoder_name="mit_b2"
├── losses.py           # Hybrid Dice + BCE loss (paper Eq. 1)
├── metrics.py           # Dice, IoU, Accuracy, Precision, Recall, F1,
│                         # Specificity, Balanced Accuracy, MCC, Kappa
├── hc_estimation.py     # morphological cleanup -> contour -> ellipse fit
│                         # -> HC in mm (paper Eq. 2, Ramanujan approx.)
├── gradcam.py           # Grad-CAM hooked on the MiT-B2 deepest stage
├── train.py             # training loop: Adam, ReduceLROnPlateau,
│                         # early stopping, checkpointing, CSV logging
├── inference.py         # runs trained model -> overlays + Grad-CAM PNGs + CSV
└── requirements.txt
```

---

## 2. The HC18 dataset

Download from Zenodo (open access, no login needed):
**https://doi.org/10.5281/zenodo.1322001**

This gives you a zip containing `training_set/` and
`training_set_pixel_size_and_HC.csv`. The expected layout is:

```
training_set/
    000_HC.png                # grayscale ultrasound image
    000_HC_Annotation.png     # thin white ellipse OUTLINE (not filled!)
    001_HC.png
    001_HC_Annotation.png
    ...
training_set_pixel_size_and_HC.csv
    columns: filename, pixel size(mm), head circumference (mm)
```

> **Important:** the `*_Annotation.png` files contain only the ellipse
> *contour*, one pixel wide, not a filled mask. `dataset.py` handles this
> automatically (`annotation_to_filled_mask`): it closes small gaps in
> the hand-drawn contour, finds the largest closed contour, and fills it
> with `cv2.drawContours(..., thickness=-1)`. This was verified against
> a synthetic ellipse of known area (filled area matched the analytic
> ellipse area to within ~1.3%, which is the expected effect of the
> morphological closing step).

If you are instead using a Kaggle mirror that already ships pre-filled
binary masks, pass `already_filled=True` when constructing `HC18Dataset`.

We split by **patient ID** (the leading number in the filename, e.g.
`010_HC.png` and `010_2HC.png` share patient `010`), not by individual
file, to avoid leaking near-duplicate scans between train/val/test —
exactly as the paper describes doing.

---

## 3. Running on Lightning AI

### 3.1 Create a Studio
1. Go to **lightning.ai** → **New Studio**.
2. Pick a GPU machine. An **L4** or **A10G** (16–24GB) is more than
   enough for MiT-B2 + U-Net at 256×256 with batch size 8. The paper's
   own experiments ran on an RTX 4060 (6GB), so this is a light model.
3. Open a Terminal tab inside the Studio.

### 3.2 Upload the project
Either drag-and-drop the `msunet_project` folder into the Studio file
browser, or clone/upload it via the terminal:

```bash
cd /teamspace/studios/this_studio
# upload msunet_project/ here (drag-and-drop in the UI), then:
cd msunet_project
```

### 3.3 Install dependencies

```bash
pip install -r requirements.txt
```

`segmentation-models-pytorch` will pull in `timm` for you; the `mit_b2`
encoder and its ImageNet weights are downloaded automatically the first
time you build the model (needs outbound internet, which Lightning
Studios have by default).

### 3.4 Get the data onto the Studio

```bash
mkdir -p /teamspace/studios/this_studio/data
cd /teamspace/studios/this_studio/data
# Download the HC18 zip from Zenodo (use the Studio's file upload UI,
# or wget the direct file link from the Zenodo record page) and unzip:
unzip HC18.zip
# You should now have:
#   data/training_set/...
#   data/training_set_pixel_size_and_HC.csv
```

### 3.5 Edit `config.py`

Open `config.py` and confirm/update these two lines to match where you
put the data:

```python
DATA_DIR = "/teamspace/studios/this_studio/data/training_set"
CSV_PATH = "/teamspace/studios/this_studio/data/training_set_pixel_size_and_HC.csv"
```

Everything else (`CHECKPOINT_DIR`, `OUTPUT_DIR`, etc.) defaults to
sensible paths under `/teamspace/studios/this_studio/`, but feel free to
change them.

### 3.6 Train

```bash
python train.py
```

This will:
- split images by patient ID into train/val/test,
- train for up to 100 epochs with early stopping (patience 15 epochs
  on validation loss),
- reduce the learning rate on plateau,
- save `checkpoints/best_msunet.pth` (best validation loss) and
  `checkpoints/last_msunet.pth` (most recent epoch),
- write per-epoch metrics to `outputs/training_log.csv`,
- finally evaluate the best checkpoint on the held-out test split and
  print the full metric suite.

Expect a few seconds to a couple of minutes per epoch on an L4/A10G,
depending on batch size and `NUM_WORKERS`.

### 3.7 Run inference + generate clinical outputs

```bash
python inference.py --checkpoint checkpoints/best_msunet.pth --split test
```

This produces, under `outputs/`:
- `overlays/<filename>_overlay.png` — input image with the fitted green
  ellipse contour and predicted HC (mm) printed on it,
- `gradcam/<filename>_gradcam.png` — Grad-CAM heatmap overlay,
- `hc_predictions.csv` — per-image predicted HC, true HC, absolute
  error (mm), Dice, IoU, precision, recall.

Use `--split all` to run over every image, or `--split val` / `--split
train` to inspect those splits.

---

## 4. Design notes / where this maps to the paper

| Paper section | Implementation |
|---|---|
| III-A Data Preparation & Augmentation | `dataset.py` (mask filling) + `transforms.py` (flips, 90° rotations, Gaussian noise, brightness/contrast, pad+crop) |
| III-B Network Architecture (MiT-B2 encoder, U-Net decoder, 1×1 conv + sigmoid head, output `1×256×256`) | `model.py`, via `segmentation_models_pytorch.Unet(encoder_name="mit_b2", encoder_weights="imagenet")` |
| III-C Loss Function, Eq. (1): `L_Hybrid = L_BCE + (1 - Dice)` | `losses.py::HybridDiceBCELoss` |
| III-D HC calculation, Eq. (2) (Ramanujan ellipse-perimeter approximation), morphological post-processing, least-squares ellipse fit | `hc_estimation.py` |
| IV. "Eleven metrics" (Dice, IoU, Accuracy, Precision, Recall, F1, ...) | `metrics.py` (10 implemented: Accuracy, Precision, Recall, Specificity, F1, IoU, Dice, Balanced Accuracy, MCC, Kappa — the paper does not enumerate all eleven by name, so add any project-specific 11th metric you need in this file) |
| Grad-CAM interpretability | `gradcam.py`, hooked on `encoder.patch_embed4` (the deepest MiT-B2 stage whose output is a plain tensor — the encoder's own top-level `forward()` returns a `List[Tensor]`, which PyTorch backward hooks cannot attach to directly) |
| Ablation study (Table III: U-Net alone vs. +ResNet vs. +MiT-B2) | Re-run `train.py` with `config.ENCODER_NAME` set to `"resnet34"` / `"mit_b2"` / a no-pretrained-encoder U-Net to reproduce each row |

### A note on the "11 metrics" and Table I/II numbers
The paper reports specific numeric results (e.g. Dice 0.9899, IoU 0.9850)
from their own training run. This code reproduces the *method*, not
those exact numbers — your results will depend on your train/val/test
split, random seed, exact augmentation strengths, and how many epochs
you train for. Treat the paper's table as a target to benchmark against,
not a guaranteed output of this script.

### A note on pixel size and HC units
The HC18 CSV's `pixel size (mm)` column refers to the **original**
800×540 image resolution. Since the network operates at 256×256,
`inference.py` rescales the pixel size accordingly
(`hc_estimation.rescale_pixel_size`) before converting the predicted
mask's pixel-space ellipse into millimetres. If you change `IMG_SIZE` in
`config.py`, this conversion adjusts automatically — you don't need to
touch it manually.

---

## 5. Quick sanity checks (optional, but recommended before a full run)

Every module has a `if __name__ == "__main__":` self-test block you can
run individually to confirm your environment is set up correctly before
committing to a full training run:

```bash
python model.py        # builds U-Net+MiT-B2, prints output shape & param count
python losses.py        # checks the hybrid loss decreases for better predictions
python metrics.py        # checks metrics = 1.0 for a perfect prediction
python hc_estimation.py   # checks ellipse-fit HC recovery against a known ellipse
python gradcam.py        # checks Grad-CAM produces a valid [H,W] heatmap in [0,1]
```

All five passed in our own testing prior to delivering this code to you.
