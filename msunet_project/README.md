# Fetal Head Segmentation — U-Net + MiT-B2 (SegFormer Encoder)

Reproduction of the paper's proposed architecture (**U-Net decoder + MiT-B2
transformer encoder, pretrained on ImageNet**) for fetal head segmentation
and head-circumference (HC) measurement on the **HC18** ultrasound dataset.

Built on top of [`segmentation-models-pytorch`](https://github.com/qubvel-org/segmentation_models.pytorch)
(SMP), which is the actual library most public HC18 + MiT-B2 implementations
use — this keeps the encoder/decoder implementation itself battle-tested
rather than hand-rolling a transformer encoder.

Reports **Dice, IoU, Precision, Recall, Accuracy, F1, Loss** for **train,
validation, and test** every epoch, plus the downstream **HC MAE / HC MSE**
(mm) from the ellipse-fitting post-processing step.

---

## 1. Project structure

```
fetal_hc_unet_mitb2/
├── requirements.txt
├── README.md
├── src/
│   ├── config.py            # all paths & hyperparameters
│   ├── prepare_splits.py    # fills ellipse annotations -> masks, makes train/val/test split
│   ├── dataset.py           # HC18Dataset + albumentations pipelines
│   ├── model.py             # smp.Unet(encoder_name="mit_b2", ...)
│   ├── losses.py            # Hybrid Dice + BCE loss
│   ├── metrics.py           # Dice/IoU/Precision/Recall/Accuracy/F1 accumulator
│   ├── train.py             # full training loop + CSV logging + early stopping
│   ├── evaluate.py          # re-evaluate a checkpoint + HC MAE/MSE
│   ├── hc_postprocess.py    # morphology -> contour -> ellipse fit -> Ramanujan HC formula
│   └── plot_curves.py       # plots train/val curves from the log CSV
└── data/                     # created automatically (raw/ + processed/)
```

---

## 2. Get the HC18 dataset

The dataset is **not bundled** — download it yourself (it's a public
research dataset, ~250 MB):

1. Go to the HC18 Grand Challenge Zenodo record:
   `https://zenodo.org/records/1327317`
2. Download and unzip it. You should end up with a folder containing:
   ```
   training_set/                              # 999 images + 999 annotation pngs
   training_set_pixel_size_and_HC.csv         # pixel size (mm/px) + ground-truth HC (mm)
   ```
3. Note the **full path** to that folder — you'll point `HC18_DATA_ROOT` at it.

---

## 3. Run on Lightning AI (Studio)

### Step 1 — Create a Studio
Open [lightning.ai](https://lightning.ai) → **New Studio** → choose a GPU
machine (an **L4** or **T4** is enough; an **A10G/A100** will train faster).
A single L4/T4 is sufficient for `mit_b2` at batch size 8, 256×256 input.

### Step 2 — Upload the project
In the Studio terminal:
```bash
# Option A: clone if you've pushed this folder to your own GitHub repo
git clone <your-repo-url> fetal_hc_unet_mitb2
cd fetal_hc_unet_mitb2

# Option B: just drag-and-drop / upload the fetal_hc_unet_mitb2/ folder
# using the Studio file browser, then:
cd fetal_hc_unet_mitb2
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Upload the HC18 dataset
Upload the unzipped HC18 folder into the Studio (drag-and-drop in the file
browser is easiest), then point the project at it:
```bash
export HC18_DATA_ROOT=/teamspace/studios/this_studio/hc18_data
# (replace with wherever you actually placed the training_set/ folder)
```
Add that `export` line to `~/.bashrc` if you want it to persist across
terminal sessions in the same Studio.

### Step 5 — Build the processed dataset (fills ellipse masks + makes splits)
```bash
python src/prepare_splits.py
```
This creates `data/processed/masks/*.png` (solid binary masks) and
`data/processed/splits.csv` (the 70/15/15 train/val/test manifest).

### Step 6 — Train
```bash
python src/train.py
```
- Trains for up to 100 epochs with early stopping (patience = 15 on val Dice).
- Mixed precision (AMP) is enabled automatically on GPU.
- Per-epoch metrics for **train** and **val** (Dice, IoU, Precision, Recall,
  Accuracy, F1, Loss) are printed to the console AND written to
  `outputs/logs/training_log.csv`.
- The best checkpoint (highest val Dice) is saved to
  `outputs/checkpoints/best_model.pth`.
- At the end, it automatically reloads the best checkpoint and reports the
  same 7 metrics on the held-out **test** split, saved to
  `outputs/logs/test_results.csv`.

### Step 7 — Plot training curves (optional)
```bash
python src/plot_curves.py
```
Saves `outputs/logs/training_curves.png`.

### Step 8 — Full evaluation incl. head-circumference error (optional)
```bash
python src/evaluate.py --split test --ckpt outputs/checkpoints/best_model.pth
```
Prints the 7 segmentation metrics **and** HC MAE / HC MSE in millimetres
(computed via morphological cleanup → contour extraction → ellipse fitting →
Ramanujan's perimeter approximation, exactly as in the paper).

---

## 4. Running it as a single Lightning AI "Job" (non-interactive)

If you'd rather submit this as a background job instead of using the
interactive Studio terminal, from the Studio:

```bash
lightning run app train_job.py
```
or simply schedule the same commands (`prepare_splits.py` then `train.py`)
in a Lightning AI **Job** pointed at this repo with the same `pip install -r
requirements.txt` setup command and `HC18_DATA_ROOT` environment variable
set in the Job's environment-variables panel.

---

## 5. Key implementation notes (why the code is built this way)

- **HC18 annotations are NOT filled masks.** They're a 1px-wide ellipse
  *outline*. `prepare_splits.py` fills them with a
  `dilate → binary_fill_holes → erode` pipeline (robust to small gaps in the
  drawn boundary), with a convex-hull fallback if the fill ever fails. This
  matches how the HC18 dataset is handled in published fetal-head
  segmentation work.
- **Encoder = `mit_b2`** via SMP's `encoder_name="mit_b2"` — this is
  SegFormer's MiT-B2 backbone (~24–25M params), exactly as in the paper.
  `timm` is required and installed automatically as an SMP dependency.
- **Loss = 0.5·BCEWithLogits + 0.5·Dice**, matching the paper's hybrid
  Dice-BCE loss for class imbalance.
- **Metrics are accumulated over the whole epoch** (running TP/FP/FN/TN),
  not averaged per-batch — this is the numerically correct way to report
  Dice/IoU/Precision/Recall/Accuracy/F1 and avoids the bias that batch-wise
  averaging introduces for small/imbalanced masks.
- **Images are converted grayscale→3-channel** before feeding the encoder,
  since `mit_b2`'s ImageNet-pretrained weights expect 3-channel RGB-like
  input (this is also what the paper's pipeline and all public SMP+HC18
  examples do).

---

## 6. Expected results

With this exact setup (100 epochs, early stopping, 256×256, Adam 1e-4,
hybrid Dice+BCE loss) you should land in the same range the paper reports
on its held-out split:

| Metric | Paper (proposed model) |
|---|---|
| Dice | 0.9899 |
| IoU | 0.9850 |
| Precision | 0.9897 |
| Recall | 0.9953 |
| F1 | 0.9899 |

Exact numbers will vary slightly with your own train/val/test split (the
paper's split is not publicly released image-by-image), random seed, and
GPU — treat the table above as a target range, not a guarantee.
