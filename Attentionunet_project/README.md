# Attention U-Net for Fetal Head Segmentation (HC18)

A clean, from-scratch PyTorch implementation of **Attention U-Net** (Oktay et al.,
2018, *"Attention U-Net: Learning Where to Look for the Pancreas"*), applied to
fetal head segmentation and head-circumference (HC) measurement on the **HC18**
ultrasound dataset.

The architecture matches the canonical public reference implementation structure
(`conv_block` / `up_conv` / `Attention_block` / `AttU_Net`, channel progression
64→128→256→512→1024), the most widely cited open-source PyTorch version of this
paper (see [LeeJunHyun/Image_Segmentation](https://github.com/LeeJunHyun/Image_Segmentation)).

Tracks **Dice, IoU, Precision, Recall, Accuracy, F1, Loss** for **train / val / test**
every epoch.

```
attn_unet_hc18/
├── model.py        # AttU_Net architecture
├── losses.py        # Hybrid Dice + BCE loss
├── metrics.py       # Dice/IoU/Precision/Recall/Accuracy/F1 (epoch-accumulated)
├── dataset.py        # HC18 dataset + patient-level split + augmentations
├── train.py          # Full training loop, checkpointing, CSV logging
├── inference.py       # Run on one image, visualize mask + HC ellipse fit
├── requirements.txt
└── README.md
```

---

## 1. Set up on Lightning AI Studio

1. Create a new **Studio** (GPU — e.g. T4/L4/A10) at https://lightning.ai
2. Open a terminal in the Studio and clone/upload this project folder, e.g.:

```bash
mkdir -p /teamspace/studios/this_studio/attn_unet_hc18
cd /teamspace/studios/this_studio/attn_unet_hc18
# upload model.py, losses.py, metrics.py, dataset.py, train.py, inference.py,
# requirements.txt into this folder (drag-and-drop in the Studio file browser,
# or `scp` / `git clone` if you push this to your own repo first)
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 2. Download the HC18 dataset

The dataset is hosted on Zenodo (official HC18 Grand Challenge):
**https://zenodo.org/records/3904280**

```bash
cd /teamspace/studios/this_studio
wget -O hc18.zip "https://zenodo.org/records/3904280/files/training_set.zip?download=1"
unzip hc18.zip -d hc18
```

After extraction you should have:

```
hc18/
└── training_set/
    ├── 000_HC.png
    ├── 000_HC_Annotation.png
    ├── 001_HC.png
    ├── 001_HC_Annotation.png
    ├── ...
    └── training_set_pixel_size_and_HC.csv
```

> **Note on masks:** HC18 annotation files contain a thin **ellipse outline**
> (the boundary), not a filled mask. `dataset.py` automatically fills the
> contour into a solid binary mask before training — this is handled for you.

> **Note on splits:** The official HC18 `test_set` has **no public ground-truth
> masks** (it's for the challenge leaderboard only), so it can't be used to
> compute Dice/IoU/etc. locally. This code instead performs a **patient-level
> 70/15/15 split of `training_set`** into train/val/test — exactly the
> leakage-safe approach the paper describes — so all three splits you asked
> for (train/val/test) have ground truth to score against.

---

## 3. Train

```bash
python train.py \
  --data_root /teamspace/studios/this_studio/hc18/training_set \
  --output_dir ./outputs \
  --img_size 256 \
  --batch_size 16 \
  --epochs 100 \
  --lr 1e-4 \
  --patience 15
```

What this does:
- Splits patients into train/val/test (70/15/15), no patient overlap.
- Trains `AttU_Net` with hybrid Dice+BCE loss, Adam optimizer, `ReduceLROnPlateau` scheduler.
- Every epoch, computes **Dice, IoU, Precision, Recall, Accuracy, F1, Loss** on train and val.
- Saves the best checkpoint (by validation Dice) to `outputs/checkpoints/best_model.pth`.
- Early-stops if validation Dice doesn't improve for `--patience` epochs.
- After training, loads the best checkpoint and evaluates once on the held-out **test** split.
- Writes everything to `outputs/training_log.csv` (one row per split per epoch) and
  a human-readable `outputs/final_report.txt`.

Adjust `--batch_size` down (e.g. 8) if you hit GPU memory limits on a smaller Studio GPU.

---

## 4. Monitor results

```bash
# quick look at the metric log
column -s, -t outputs/training_log.csv | less -S
```

Or load it in Python/pandas for plotting:

```python
import pandas as pd
df = pd.read_csv("outputs/training_log.csv")
df[df.split == "val"].plot(x="epoch", y=["Dice", "IoU"])
```

---

## 5. Run inference + HC measurement on a new image

```bash
python inference.py \
  --checkpoint outputs/checkpoints/best_model.pth \
  --image hc18/training_set/010_HC.png \
  --pixel_size_mm 0.169   # get this value from training_set_pixel_size_and_HC.csv for that image
```

This saves `prediction.png` (input / predicted mask / fitted ellipse) and prints the
estimated head circumference in pixels (and mm, if `--pixel_size_mm` is given).

---

## Implementation notes

- **Architecture**: Faithful to the original Attention U-Net paper — encoder
  with 5 stages (64→1024 channels), decoder with attention-gated skip
  connections (`Attention_block` uses additive attention: `W_g·g + W_x·x → ReLU → ψ → sigmoid`).
- **Loss**: `BCEWithLogits + (1 - Dice)`, as used in the source paper, for
  stability under the class imbalance of small fetal-head regions.
- **Metrics**: Computed from epoch-accumulated TP/FP/FN/TN (not averaged
  per-batch means), which is the mathematically correct way to report Dice/IoU/F1
  over a whole split.
- **Single-channel input**: HC18 images are grayscale; `img_ch=1` avoids
  wastefully replicating channels. If you want to later swap in an ImageNet-pretrained
  encoder (e.g. MiT-B2, as in the original paper you uploaded), you'd switch to
  `segmentation_models_pytorch` with `in_channels=3` and replicate the grayscale
  channel — this script is the from-scratch Attention U-Net specifically.
- **Reproducibility**: All random seeds (`random`, `numpy`, `torch`, `cuda`) are
  fixed via `--seed` (default 42), and splitting is patient-level to prevent leakage.

## Expected results

Attention U-Net (no pretrained encoder) typically reaches **Dice ≈ 0.95–0.97** and
**IoU ≈ 0.91–0.95** on HC18 with this setup — strong, but below the MiT-B2 hybrid's
~0.989 reported in your uploaded paper, since that uses an ImageNet-pretrained
transformer encoder. If you need to match or exceed those numbers specifically,
let me know and I'll wire `segmentation_models_pytorch` (`encoder_name="mit_b2"`)
into this same training/metrics harness instead.
