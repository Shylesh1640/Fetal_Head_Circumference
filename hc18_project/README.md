# HC18 Architecture Comparison + Fusion Project

Builds on: *Intelligent Ultrasound Analysis for Real-Time Fetal Head
Circumference Measurement* (U-Net + MiT-B2, Dice 0.9899, IoU 0.9850).

Goal: test 6 architecture families under identical conditions, then fuse
the top 2 into a hybrid that beats the paper's numbers.

## 0. Environment

```bash
pip install torch torchvision --break-system-packages
pip install segmentation-models-pytorch timm albumentations \
    opencv-python pandas scikit-learn grad-cam --break-system-packages
```

## 1. Get the data

Download + unzip from https://zenodo.org/records/1322001 into
`./data/training_set/` so you have pairs like:
```
000_HC.png
000_HC_Annotation.png
```
(I can't fetch this myself — my sandbox network only allows package
registries, not zenodo.org. Do this step on Colab or your own machine.)

## 2. Train every model in the zoo

Run once per architecture (same epochs/loss/splits as the paper):

```bash
for m in unet_baseline attention_unet dilated_unet dense_unet ms_unet transformer_unet segformer; do
  python train.py --model $m --data_dir ./data/training_set --epochs 100
done
```

Each run saves `checkpoints/<model>_best.pth`, picked by best **validation
Dice**, with early stopping (patience=10) — mirrors the paper's training
protocol (Adam, lr=1e-4, ReduceLROnPlateau, patient-level split).

## 3. Build the leaderboard

```bash
python evaluate.py --data_dir ./data/training_set \
  --models unet_baseline attention_unet dilated_unet dense_unet ms_unet transformer_unet segformer \
  --out ./results/leaderboard.csv
```

This reproduces the paper's Table I/II style comparison (Acc, Precision,
IoU, Recall, F1, Dice) on your held-out test set. Identify your top 2.

## 4. Fuse the top 2

**Quick check (output averaging, no training):**
```bash
python ensemble.py --data_dir ./data/training_set \
  --model_a transformer_unet --model_b attention_unet --weight_a 0.6
```
Try a few `weight_a` values (0.4–0.7) and keep whichever beats both
single models on Dice/IoU on the *validation* set before reporting on
test.

**Stronger version (feature-level fusion):** use
`models.DualEncoderFusion` to merge the two best models' bottleneck
features before decoding, then fine-tune end-to-end for ~20–30 epochs.
This is more work but typically beats simple averaging because the two
encoders' representations interact before the decision is made, rather
than just blending two separate decisions.

## 5. Post-processing (unchanged from the paper)

`metrics.mask_to_hc_mm()` does morphological cleanup → largest contour →
`cv2.fitEllipse` → Ramanujan's perimeter formula, exactly like the
paper's Eq. 2. Use the `*_pixel_size_and_HC.csv` file from the dataset
download to convert pixels → mm and to compute HC MAE/MSE against ground
truth.

## Why this design should not regress below the paper

- Same loss (Dice+BCE), same optimizer, same patient-level split, same
  augmentations — so any score difference is attributable to
  architecture, not protocol drift.
- `transformer_unet` in the zoo *is* the paper's exact model, so it acts
  as your built-in sanity check: if it doesn't land near Dice 0.9899 on
  your run, something in data/training setup needs fixing before you
  trust the other 5 results.
- The fusion step only gets adopted if it beats both of its parent
  models on validation Dice/IoU — otherwise just report the best single
  model from step 3.

## File map

| File | Purpose |
|---|---|
| `dataset.py` | patient-level train/val/test split + HC18 Dataset |
| `models.py` | 7 architectures + `DualEncoderFusion` hybrid |
| `losses.py` | Dice+BCE hybrid loss (paper's Eq. 1) |
| `metrics.py` | Dice/IoU/Prec/Rec/F1 + ellipse-based HC calc (Eq. 2) |
| `train.py` | trains one model, saves best checkpoint |
| `evaluate.py` | builds the leaderboard CSV |
| `ensemble.py` | output-averaging fusion of top-2 models |
