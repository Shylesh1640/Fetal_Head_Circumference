# Dilated U-Net — HC18 Fetal Head Segmentation
## Setup & Run Guide (Lightning AI / Colab / any Linux GPU)

---

## Architecture Overview

```
Input (256×256×3)
       │
  ┌────▼────┐
  │  Enc0   │  SE-Residual  Conv  (dil=1)   → 32 ch
  └────┬────┘
       │ MaxPool
  ┌────▼────┐
  │  Enc1   │  SE-Residual  Conv  (dil=1)   → 64 ch
  └────┬────┘
       │ MaxPool
  ┌────▼────┐
  │  Enc2   │  SE-Residual  Conv  (dil=2)   → 128 ch
  └────┬────┘
       │ MaxPool
  ┌────▼────┐
  │  Enc3   │  SE-Residual  Conv  (dil=2)   → 256 ch
  └────┬────┘
       │ MaxPool
  ┌────▼────────────────────────────────────────────┐
  │               ASPP  Bridge                      │
  │  rate=1 │ rate=6 │ rate=12 │ rate=18 │ GAP     │  → 512 ch
  └────┬────────────────────────────────────────────┘
       │ UpBlock + skip(e3)
  ┌────▼────┐
  │  Dec3   │                                 → 256 ch
  └────┬────┘
       │ UpBlock + skip(e2)
  ┌────▼────┐
  │  Dec2   │                                 → 128 ch
  └────┬────┘
       │ UpBlock + skip(e1)
  ┌────▼────┐
  │  Dec1   │                                 →  64 ch
  └────┬────┘
       │ UpBlock + skip(e0)
  ┌────▼────┐
  │  Dec0   │                                 →  32 ch
  └────┬────┘
       │ 1×1 Conv + Sigmoid
  ┌────▼────┐
  │  Output │  (256×256×1)  — binary mask
  └─────────┘
```

Key improvements over plain U-Net:
- **Dilated convolutions** (rates 2, 2) in enc2/enc3 → 2× larger receptive field without losing resolution
- **ASPP bottleneck** (rates 1, 6, 12, 18 + GAP) → captures multi-scale context across the whole skull boundary
- **SE channel attention** in every residual block → suppresses irrelevant background features
- **Residual shortcuts** → stable gradient flow, faster convergence

---

## Step 1 — Install dependencies

Open a terminal in Lightning AI (or Colab) and run:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install albumentations opencv-python matplotlib tqdm
```

No extra libraries are needed — ASPP, SE-blocks, Grad-CAM, and metrics are all implemented from scratch in the script.

---

## Step 2 — Download the HC18 dataset

```bash
pip install zenodo-get
zenodo_get 1322001        # ~200 MB
```

Or download manually from: https://zenodo.org/record/1322001  
Unzip so the folder looks like:

```
hc18/
  training/
    000_HC.png
    000_HC_Annotation.png
    001_HC.png
    001_HC_Annotation.png
    ...
  test/
    ...
```

Then set `DATA_ROOT` at the top of `dilated_unet_hc18.py`:

```python
CFG = dict(
    DATA_ROOT = "./hc18",   # ← point here
    ...
)
```

---

## Step 3 — Run training

```bash
python dilated_unet_hc18.py
```

Training prints a live table every epoch:

```
Epoch  │   T loss / dice / iou / prec / rec / acc / f1
       │   V loss / dice / iou / prec / rec / acc / f1
```

Checkpoints are saved to `./checkpoints/best_model.pth` whenever val Dice improves.  
Plots are saved to `./logs/`.

---

## Step 4 — Outputs

| File | Contents |
|---|---|
| `checkpoints/best_model.pth` | Best model weights + optimizer state |
| `logs/training_history.png` | 7 metric curves for train + val |
| `logs/predictions.png` | Image / GT mask / pred mask / overlay with HC in mm |
| `logs/gradcam.png` | Grad-CAM heatmap from the ASPP bottleneck |

---

## Configuration reference

Edit the `CFG` dict at the top of the script:

| Key | Default | Meaning |
|---|---|---|
| `DATA_ROOT` | `./hc18` | Path to dataset root |
| `IMG_SIZE` | `256` | Square input resolution |
| `PIXEL_MM` | `0.154` | Pixel spacing (mm) from HC18 challenge |
| `TRAIN_FRAC` | `0.80` | Train split fraction |
| `VAL_FRAC` | `0.10` | Val split (test = remaining 10%) |
| `EPOCHS` | `100` | Maximum epochs |
| `BATCH_SIZE` | `8` | Batch size (reduce to 4 if OOM) |
| `LR` | `1e-4` | Adam learning rate |
| `PATIENCE` | `15` | Early-stopping patience (epochs) |
| `ENCODER_CH` | `[32,64,128,256,512]` | Channel sizes per level |
| `DROPOUT` | `0.1` | Dropout2d probability |

---

## Expected results

| Metric | Expected range |
|---|---|
| Dice Score | 0.97 – 0.99 |
| IoU | 0.96 – 0.98 |
| Precision | 0.97 – 0.99 |
| Recall | 0.97 – 0.99 |
| Accuracy | 0.99+ |
| F1 Score | 0.97 – 0.99 |

These are reported for the **test split** after loading the best checkpoint.

---

## Memory requirements

| GPU | Batch size | Works? |
|---|---|---|
| 4 GB (GTX 1650, MX550) | 4 | ✅ |
| 6 GB (RTX 4060) | 8 | ✅ |
| 8 GB (RTX 3070) | 16 | ✅ |
| T4 (Colab) | 8–16 | ✅ |

If you get a CUDA OOM error, set `BATCH_SIZE = 4` or `IMG_SIZE = 224`.

---

## Head Circumference (HC) pipeline

After the model predicts a binary mask, the code automatically:

1. Applies morphological closing + opening to clean the mask
2. Finds the largest contour with `cv2.findContours`
3. Fits an ellipse with `cv2.fitEllipse` (least-squares, needs ≥5 contour points)
4. Computes HC using **Ramanujan's approximation**:

```
HC = π · [3(a+b) − √((3a+b)(a+3b))]
```

where `a`, `b` are the semi-major and semi-minor axes converted to mm via `PIXEL_MM`.

---

## Extending the code

- **Swap to MiT-B2 encoder**: replace encoder blocks with `segmentation_models_pytorch.Unet(encoder_name="mit_b2")` and keep the ASPP bridge
- **Add TTA at inference**: flip horizontally + vertically, average 4 predictions
- **Quantisation**: call `torch.quantization.quantize_dynamic(model, ...)` after loading the checkpoint
- **ONNX export**: `torch.onnx.export(model, dummy, "model.onnx", opset_version=14)`
