# Intelligent Ultrasound Analysis for Automated Fetal Head Segmentation and Head Circumference Measurement

## Overview

This project develops an automated deep-learning pipeline for **fetal
head segmentation in ultrasound images** and subsequent **fetal head
circumference (HC) estimation**.

The overall objective is to reduce the dependence on manual fetal-head
delineation by combining:

-   Medical-image segmentation using encoder-decoder networks
-   nnU-Net-inspired convolutional design
-   Residual feature learning
-   Attention-gated skip connections
-   Multi-scale feature fusion at the bottleneck
-   Transformer-based alternatives for global-context modelling
-   Robust ultrasound preprocessing and augmentation
-   Contour extraction and ellipse-based HC estimation
-   Segmentation and biometric evaluation
-   Interpretability using attention/Grad-CAM-style visualisation

The project began with a U-Net + MiT-B2 approach described in the
associated IEEE ICCIDS 2026 paper and has subsequently been extended
into a research codebase containing multiple candidate architectures,
including residual-attention and multi-scale nnU-Net-style variants.

> **Research status:** The architecture implementations in the current
> codebase are research candidates. Their inclusion in the model library
> does not by itself mean that every architecture has been
> experimentally validated or that it outperforms nnU-Net or the
> published MiT-B2 baseline.

------------------------------------------------------------------------

## 1. Problem Statement

Fetal ultrasound is widely used for prenatal assessment, but automatic
fetal-head analysis is difficult because ultrasound images commonly
contain:

-   Speckle noise
-   Acoustic shadowing
-   Low contrast
-   Weak or incomplete skull boundaries
-   Variable probe orientation
-   Fetal motion and anatomical variation
-   Confusing maternal-tissue structures

Manual fetal-head contouring is time-consuming and operator-dependent.

The project therefore aims to learn a robust segmentation model that
can:

1.  Locate the fetal head automatically.
2.  Produce a clean binary segmentation mask.
3.  Preserve the skull boundary accurately.
4.  Extract the fetal-head contour.
5.  Estimate head circumference from the segmented geometry.
6.  Provide quantitative and visual outputs suitable for research and
    future clinical decision-support workflows.

The associated paper describes these challenges and motivates combining
U-Net's spatial localisation with stronger feature extraction and
global-context modelling.

------------------------------------------------------------------------

## 2. Main Objectives

### Primary objective

Develop a high-performance fetal-head ultrasound segmentation system
with strong:

-   Dice score
-   IoU
-   Precision
-   Recall
-   F1-score
-   Boundary accuracy
-   HD95 performance

### Secondary objective

Convert the segmentation output into a clinically meaningful biometric
measurement:

**Ultrasound image → fetal-head mask → contour → ellipse fitting → head
circumference**

### Research objective

Move beyond simply maximising Dice on HC18 and investigate improvements
in:

-   Boundary quality
-   Robustness to noisy and low-contrast scans
-   Generalisation to difficult cases
-   Computational efficiency
-   Interpretability
-   Reliability of HC measurement

------------------------------------------------------------------------

## 3. Dataset

The project uses the **HC18 fetal head ultrasound benchmark**.

The dataset contains paired ultrasound images and expert-annotated
fetal-head segmentation masks.

The project uses the images as grayscale inputs and produces a binary
fetal-head mask.

### Input

``` text
(B, 1, 256, 256)
```

### Output

``` text
(B, 1, 256, 256)
```

The current model-library implementations return **raw segmentation
logits**. Sigmoid should therefore be applied during metric
calculation/inference rather than inside the model.

------------------------------------------------------------------------

## 4. Overall Pipeline

``` text
                 FETAL ULTRASOUND IMAGE
                          │
                          ▼
                ┌───────────────────┐
                │   Pre-processing  │
                │ Normalisation etc.│
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │   Segmentation    │
                │      Network      │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ Raw segmentation  │
                │      logits       │
                └─────────┬─────────┘
                          │
                    sigmoid + threshold
                          │
                          ▼
                ┌───────────────────┐
                │ Binary head mask  │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ Post-processing / │
                │ contour extraction│
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ Ellipse fitting   │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ Head Circumference│
                │     estimation    │
                └───────────────────┘
```

The published baseline additionally used Grad-CAM visualisation for
interpretability.

------------------------------------------------------------------------

## 5. Published Baseline: U-Net + MiT-B2

The associated IEEE ICCIDS 2026 paper proposed a U-Net architecture with
a pretrained **MiT-B2 encoder**.

The design combines:

-   U-Net decoder and skip connections for spatial localisation
-   MiT-B2 for hierarchical/global feature representation
-   ImageNet pretraining
-   Dice + BCE hybrid loss
-   Ultrasound-specific augmentation
-   Morphological contour refinement
-   Ellipse fitting for HC estimation
-   Grad-CAM visualisation

The paper reports a validation Dice of **0.9899** and IoU of **0.9850**
for the proposed U-Net + MiT-B2 system.

The paper also reports training and validation comparisons against
U-Net, SegNet, DeepLabV3+, U-Net++, Attention U-Net, and U-Net + ResNet.

### Important experimental caveat

The same paper reports substantially lower IoU values for several
individual test examples than the headline validation IoU. Therefore,
those numbers should not be treated as interchangeable. Future
experiments in this project should report:

-   Per-image metrics
-   Dataset-level mean metrics
-   Confidence intervals
-   Independent test performance
-   HD95
-   HC measurement error

rather than relying on a single validation score.

------------------------------------------------------------------------

## 6. Current Research Codebase

The current model library contains several architectures:

``` text
models_library.py
│
├── nnUNet
├── ResAttnUNet
├── nnUNet_ResAttn
├── nnUNet_MSFFBridge_ResAttn
├── UNet_MiTB2
├── UNet_Hybrid
└── msunet
```

The implementations are designed around a common interface:

``` text
Input:
    (B, 1, 256, 256)

Output:
    (B, 1, 256, 256) raw logits
```

This common interface makes it possible to perform controlled model
comparisons.

------------------------------------------------------------------------

# 7. nnU-Net-Style Architecture

The code implements a practical 2D nnU-Net-inspired architecture rather
than embedding the complete official nnU-Net framework.

The basic convolutional block follows:

``` text
Conv2D
   ↓
Instance Normalisation
   ↓
LeakyReLU
   ↓
Conv2D
   ↓
Instance Normalisation
   ↓
LeakyReLU
```

The channel progression in the current implementations is:

``` text
Base channels = 32

Stage 1 → 32
Stage 2 → 64
Stage 3 → 128
Stage 4 → 256
Bottleneck → 512
```

The encoder progressively reduces spatial resolution while increasing
feature channels.

The decoder performs the reverse operation and combines decoder features
with encoder skip features.

------------------------------------------------------------------------

# 8. Residual Learning

The residual variants replace the standard convolutional block with a
residual convolutional block.

Conceptually:

``` text
Input
  │
  ├─────────────── Identity ───────────────┐
  │                                        │
  ▼                                        │
Conv → Norm → Activation → Conv → Norm    │
  │                                        │
  └──────────────── + ─────────────────────┘
                   │
                   ▼
                Output
```

### Motivation

Residual learning can improve:

-   Gradient propagation
-   Feature reuse
-   Optimisation of deeper networks
-   Stability of feature learning

This is particularly relevant for ultrasound because the network must
learn useful anatomical features despite noise and weak boundaries.

Residual learning itself is not considered novel; it is used here as an
engineering component within the segmentation system.

------------------------------------------------------------------------

# 9. Attention-Gated Skip Connections

The residual-attention models apply attention gates to decoder skip
connections.

Instead of blindly concatenating all encoder features, the attention
mechanism attempts to suppress irrelevant information and emphasise
features useful for reconstructing the fetal head.

Conceptually:

``` text
Encoder feature ───────┐
                       ▼
                 Attention Gate
                       │
                       ▼
                 Filtered skip
                       │
Decoder feature ───────┤
                       ▼
                  Concatenation
                       │
                       ▼
                 Decoder block
```

### Expected benefit

The fetal head occupies only part of the ultrasound image. Attention can
therefore help reduce the contribution of irrelevant background
structures.

### Research limitation

Attention gates are established technology and have already been used in
fetal-head segmentation literature. They should therefore be treated as
a justified architectural component rather than claimed as a standalone
novelty.

------------------------------------------------------------------------

# 10. Multi-Scale Feature Fusion Bridge

One of the current candidate architectures is:

``` text
nnUNet_MSFFBridge_ResAttn
```

It adds a **Multi-Scale Feature Fusion (MSFF) bridge** at the
bottleneck.

The bridge uses parallel dilated convolutions with dilation rates:

``` text
1
2
4
8
```

Conceptually:

``` text
                    Bottleneck
                         │
          ┌──────────────┼──────────────┐
          │              │              │
        d = 1          d = 2          d = 4          d = 8
          │              │              │              │
          └──────────────┴──────────────┴──────────────┘
                         │
                    Concatenate
                         │
                      1×1 Conv
                         │
                  InstanceNorm
                         │
                    LeakyReLU
                         │
                       + Input
                         │
                         ▼
                  Fused bottleneck
```

### Why multi-scale processing?

Fetal skull boundaries may contain both:

-   Fine local edge information
-   Larger-scale head-shape information

Dilated convolutions increase the effective receptive field without
requiring a proportional increase in spatial resolution.

### Expected effect

**Engineering hypothesis:** the MSFF bridge may improve robustness to
variable boundary quality and scale, particularly in difficult
ultrasound images.

There is currently no evidence in the supplied project files that this
specific implementation has already demonstrated a statistically
significant improvement over the strongest baseline. It must therefore
be validated experimentally.

------------------------------------------------------------------------

# 11. Complete Current Candidate Architecture

The main residual-attention + MSFF candidate can be represented as:

``` text
                    Input
                1 × 256 × 256
                       │
                       ▼
              Residual Encoder
                       │
          ┌────────────┼────────────┐
          │            │            │
         E1           E2           E3           E4
          │            │            │            │
          │            │            │            │
          └────────────┴────────────┴────────────┘
                       │
                       ▼
                Bottleneck
                   512 ch
                       │
                       ▼
               MSFF Bridge
            dilation 1/2/4/8
                       │
                       ▼
              Attention Decoder
                       │
             ┌─────────┴─────────┐
             │                   │
        Attention Gate      Residual Block
             │                   │
             └─────────┬─────────┘
                       │
                  repeated ×4
                       │
                       ▼
                  1×1 Conv
                       │
                       ▼
              Raw segmentation
                   logits
                       │
                       ▼
                   Sigmoid
                       │
                       ▼
                Binary mask
```

The encoder-decoder path retains skip connections from multiple
resolutions.

------------------------------------------------------------------------

# 12. Alternative Architectures

## 12.1 nnUNet

A baseline nnU-Net-inspired architecture.

Purpose:

-   Establish a strong convolutional baseline
-   Compare against residual and attention modifications
-   Determine whether additional architectural complexity is justified

------------------------------------------------------------------------

## 12.2 ResAttnUNet

Combines:

``` text
Residual blocks
+
Attention-gated skips
```

Purpose:

-   Improve optimisation through residual learning
-   Focus decoder reconstruction on relevant anatomical regions

------------------------------------------------------------------------

## 12.3 nnUNet_ResAttn

Uses the nnU-Net-style convolutional structure together with:

``` text
Residual blocks
+
Attention-gated decoder skips
```

This is a stronger controlled baseline for evaluating the effect of
residual and attention mechanisms.

------------------------------------------------------------------------

## 12.4 nnUNet_MSFFBridge_ResAttn

Adds:

``` text
Residual encoder/decoder
+
Attention-gated skips
+
Multi-scale bottleneck bridge
```

This is one of the main current research candidates.

------------------------------------------------------------------------

## 12.5 UNet_MiTB2

Uses:

``` text
U-Net decoder
+
SegFormer's MiT-B2 encoder
```

The implementation uses `segmentation_models_pytorch` with
ImageNet-pretrained MiT-B2 weights and no activation inside the model.

This corresponds closely to the published U-Net + MiT-B2 research
direction.

------------------------------------------------------------------------

## 12.6 UNet_Hybrid

Uses a CNN encoder-decoder with a lightweight transformer self-attention
block at the bottleneck.

Conceptually:

``` text
CNN Encoder
     ↓
CNN Bottleneck
     ↓
Transformer Self-Attention
     ↓
Feed-Forward Network
     ↓
CNN Decoder
```

This follows a CNN-transformer hybrid pattern.

------------------------------------------------------------------------

## 12.7 msunet

The multi-scale U-Net variant feeds downsampled versions of the original
image into multiple encoder stages.

Conceptually:

``` text
Original image
      │
      ├── Stage 1
      │
      ├── Downsample → Stage 2
      │
      ├── Downsample → Stage 3
      │
      └── Downsample → Stage 4
```

The intention is to preserve direct access to image information at
multiple scales.

------------------------------------------------------------------------

# 13. Data Augmentation

The associated paper describes augmentation intended to reproduce
realistic ultrasound variability.

Reported operations include:

-   Gaussian noise
-   Padding to 256 × 256
-   Random 90-degree rotation
-   Random cropping
-   Horizontal flipping
-   Vertical flipping
-   Brightness adjustment
-   Contrast adjustment

These transformations are intended to improve robustness to:

-   Probe orientation
-   Imaging-condition variation
-   Fetal movement
-   Noise
-   Contrast variation

### Important consideration

Augmentation should remain anatomically plausible. In particular,
transformations that unrealistically alter fetal anatomy should not be
used merely to increase dataset size.

------------------------------------------------------------------------

# 14. Loss Function

The published baseline uses a hybrid Dice + Binary Cross-Entropy
objective:

``` text
L = L_BCE + (1 - Dice)
```

### Dice component

Dice focuses on region overlap:

``` text
Dice = 2|P ∩ G| / (|P| + |G|)
```

where:

-   `P` = predicted foreground
-   `G` = ground-truth foreground

### BCE component

BCE provides pixel-level supervision and helps penalise incorrect
foreground/background classification.

### Important implementation rule

The current model library outputs raw logits.

Therefore:

``` text
logits
   ├──→ BCEWithLogitsLoss(logits, target)
   │
   └──→ sigmoid(logits) → Dice calculation
```

Do **not** apply sigmoid before `BCEWithLogitsLoss`.

------------------------------------------------------------------------

# 15. Training Strategy

The associated paper reports:

``` text
Optimizer: Adam
Initial learning rate: 1 × 10^-4
Training duration: up to 100 epochs
Early stopping: based on validation convergence
Learning-rate scheduling: used when performance plateaued
```

The paper reports experiments using an NVIDIA RTX 4060 with 6 GB GPU
memory.

The current research code should be evaluated using a consistent
training protocol across all candidate architectures.

For fair comparison:

-   Same train/validation split
-   Same preprocessing
-   Same augmentation
-   Same optimizer
-   Same learning-rate policy
-   Same loss
-   Same random seeds where possible
-   Same stopping criterion
-   Same test set
-   Same thresholding rule

------------------------------------------------------------------------

# 16. Evaluation Metrics

The project evaluates segmentation using multiple complementary metrics.

## Dice

Measures overlap between prediction and ground truth.

Higher is better.

``` text
Dice = 2TP / (2TP + FP + FN)
```

------------------------------------------------------------------------

## IoU

Measures intersection over union.

``` text
IoU = TP / (TP + FP + FN)
```

Higher is better.

------------------------------------------------------------------------

## Precision

Measures how much of the predicted foreground is correct.

``` text
Precision = TP / (TP + FP)
```

Higher precision means fewer false-positive regions.

------------------------------------------------------------------------

## Recall

Measures how much of the actual fetal head is recovered.

``` text
Recall = TP / (TP + FN)
```

Higher recall means fewer missed fetal-head pixels.

------------------------------------------------------------------------

## F1-score

Balances precision and recall.

``` text
F1 = 2 × Precision × Recall / (Precision + Recall)
```

------------------------------------------------------------------------

## HD95

Hausdorff Distance at the 95th percentile.

This is particularly important for fetal-head segmentation because a
model can achieve high Dice while still producing clinically relevant
boundary errors.

For this project, HD95 should be treated as an important research metric
rather than relying only on Dice/IoU.

------------------------------------------------------------------------

# 17. Head Circumference Estimation

After segmentation, the predicted mask is converted into a contour.

The published pipeline performs:

``` text
Predicted mask
      ↓
Morphological refinement
      ↓
Contour extraction
      ↓
Least-squares ellipse fitting
      ↓
Semi-major axis a
Semi-minor axis b
      ↓
Ramanujan approximation
      ↓
Head circumference
```

The approximation used in the paper is:

``` text
HC ≈ π [ 3(a + b)
        - √((3a + b)(a + 3b)) ]
```

where:

-   `a` = semi-major axis
-   `b` = semi-minor axis

This converts the segmented fetal-head geometry into an estimated
circumference.

------------------------------------------------------------------------

# 18. End-to-End Output

For one ultrasound image, the system can produce:

``` text
Original ultrasound
        │
        ├── Predicted segmentation mask
        │
        ├── Fetal-head contour
        │
        ├── Ellipse fitted to contour
        │
        ├── Head circumference
        │
        └── Optional interpretability map
```

A complete application can therefore expose both:

### Segmentation output

``` text
Binary fetal-head mask
```

and

### Biometric output

``` text
Estimated fetal head circumference
```

------------------------------------------------------------------------

# 19. Interpretability

The published system uses Grad-CAM to visualise image regions
contributing to the model prediction.

The intended interpretation is:

``` text
Ultrasound
    ↓
Segmentation network
    ↓
Prediction
    ↓
Grad-CAM
    ↓
Highlighted anatomical regions
```

The paper reports that the visualisations emphasised the cranial region
and skull outline.

Interpretability should be used as a diagnostic tool for understanding
model behaviour, not as proof that the model is clinically correct.

------------------------------------------------------------------------

# 20. Why This Project Matters

The clinical motivation is straightforward:

Manual fetal-head measurement is:

-   Time-consuming
-   Operator-dependent
-   Sensitive to image quality
-   Difficult to standardise

An automated system can potentially provide:

-   Faster segmentation
-   Reproducible contours
-   Automated HC estimation
-   Consistent quantitative evaluation
-   Decision-support information
-   Potential real-time integration into ultrasound workflows

However, this project is a research system and should not be considered
a clinically validated diagnostic device without prospective and
multicentre validation.

------------------------------------------------------------------------

# 21. Research Direction

A key lesson from the project literature review is that **simply
combining more well-known modules is not sufficient to establish
novelty**.

Components such as:

-   Residual blocks
-   Attention gates
-   nnU-Net
-   Transformer blocks
-   Vision Mamba
-   Multi-scale feature fusion
-   Ellipse fitting
-   Deep supervision

all have substantial prior art.

Therefore, the research should focus on measurable problems where HC18
and related methods still have meaningful weaknesses.

Priority areas include:

1.  Boundary accuracy and HD95
2.  Difficult low-contrast scans
3.  Acoustic-shadow robustness
4.  First-trimester or difficult subsets
5.  Cross-scanner/cross-site generalisation
6.  HC measurement error
7.  Measurement uncertainty/calibration
8.  Inference efficiency

------------------------------------------------------------------------

# 22. Current Research Hypothesis

The strongest current architectural hypothesis is:

``` text
nnU-Net-style encoder
        +
Residual feature learning
        +
Attention-gated skip connections
        +
Multi-scale bottleneck feature fusion
        =
Improved robustness and boundary preservation
```

### Expected behaviour

The reasoning is:

-   Residual blocks should make feature optimisation more stable.
-   Attention gates should reduce irrelevant skip information.
-   Multi-scale dilation should increase contextual coverage at the
    bottleneck.
-   U-Net-style skip connections should preserve fine spatial detail.

### Scientific status

This is an **engineering hypothesis**, not a demonstrated result.

The model should only be claimed to be better if controlled experiments
show statistically meaningful improvement over appropriate baselines.

------------------------------------------------------------------------

# 23. Novelty and Prior-Art Position

The project should distinguish between:

### Established components

These are not novel by themselves:

-   U-Net
-   nnU-Net-style blocks
-   Residual learning
-   Attention U-Net-style gates
-   Transformer bottlenecks
-   Multi-scale feature fusion
-   Ellipse fitting
-   Grad-CAM
-   Dice+BCE loss

### Potential research contribution

The contribution should be framed around a **specific mechanism and
measurable clinical/research benefit**, rather than simply stating:

> "We combined several existing architectures."

The accompanying research report specifically warns that HC18
performance is already very high and that Dice alone may have limited
headroom.

Therefore, a strong paper should emphasise robustness, boundary quality,
difficult-case performance, measurement reliability, or calibrated
uncertainty in addition to conventional Dice/IoU.

------------------------------------------------------------------------

# 24. Recommended Experimental Design

A scientifically defensible experiment should include:

## Baselines

``` text
1. Standard U-Net
2. nnU-Net-style baseline
3. U-Net + ResNet
4. U-Net + MiT-B2
```

## Proposed ablations

``` text
5. nnU-Net + Residual
6. nnU-Net + Attention
7. nnU-Net + Residual + Attention
8. nnU-Net + Residual + Attention + MSFF
```

This isolates the contribution of each component.

### Example ablation table

  ----------------------------------------------------------------------------------------
  Model         Residual   Attention     MSFF     Dice      IoU     HD95   Params    FLOPs
  ----------- ---------- ----------- -------- -------- -------- -------- -------- --------
  U-Net               No          No       No      ---      ---      ---      ---      ---

  nnU-Net             No          No       No      ---      ---      ---      ---      ---

  nnU-Net +          Yes          No       No      ---      ---      ---      ---      ---
  Res                                                                             

  nnU-Net +          Yes         Yes       No      ---      ---      ---      ---      ---
  ResAttn                                                                         

  nnU-Net +          Yes         Yes      Yes      ---      ---      ---      ---      ---
  ResAttn +                                                                       
  MSFF                                                                            
  ----------------------------------------------------------------------------------------

The dashes should be replaced only with experimentally measured values.

------------------------------------------------------------------------

# 25. Statistical Validation

For a publishable study, report more than the best single score.

Recommended reporting:

-   Mean
-   Standard deviation
-   95% confidence interval
-   Per-image Dice distribution
-   Per-image IoU distribution
-   HD95 distribution
-   HC MAE
-   HC RMSE
-   Inference time
-   Parameter count
-   FLOPs

For paired model comparisons, use the same test images for all models.

Bootstrap confidence intervals or appropriate paired statistical tests
can be used to quantify uncertainty.

------------------------------------------------------------------------

# 26. Failure-Case Analysis

The project should explicitly inspect difficult cases such as:

``` text
Low contrast
     ↓
Weak skull boundary

Heavy speckle
     ↓
False structures

Acoustic shadow
     ↓
Incomplete contour

Partial skull visibility
     ↓
Under-segmentation

Maternal tissue
     ↓
False-positive segmentation
```

A model should not be considered successful merely because its average
Dice is high.

A strong research result should show how the model behaves when the
image quality deteriorates.

------------------------------------------------------------------------

# 27. Computational Considerations

The current architectures have different computational characteristics.

### CNN / nnU-Net-style models

Advantages:

-   Relatively efficient
-   Strong local feature extraction
-   Straightforward optimisation
-   Suitable for 256 × 256 segmentation

### Attention-gated models

Advantages:

-   Potentially better skip-feature selection

Cost:

-   Additional convolutional operations
-   Slight increase in memory and computation

### MSFF bridge

Advantages:

-   Larger effective receptive field
-   Multi-scale contextual processing

Cost:

-   Multiple parallel dilated convolutions
-   Increased bottleneck computation

### Transformer models

Advantages:

-   Global/contextual modelling

Cost:

-   Higher memory and computation
-   More sensitivity to implementation and training configuration

The exact computational cost should be measured using the same hardware
and input size for every model.

------------------------------------------------------------------------

# 28. Project Structure

A recommended repository structure is:

``` text
project/
│
├── README.md
│
├── models/
│   └── models_library.py
│
├── notebooks/
│   └── training_and_evaluation.ipynb
│
├── data/
│   ├── images/
│   └── masks/
│
├── outputs/
│   ├── predictions/
│   ├── contours/
│   ├── hc_measurements/
│   └── visualisations/
│
├── checkpoints/
│
├── results/
│   ├── metrics.csv
│   ├── comparison.csv
│   └── ablation.csv
│
└── docs/
    ├── methodology.md
    ├── experiments.md
    └── literature_review.md
```

The exact directory names can be adapted to the actual training
environment.

------------------------------------------------------------------------

# 29. Basic Model Usage

The current model library exposes a registry of candidate architectures.

Conceptually:

``` python
from models_library import MODEL_REGISTRY

model = MODEL_REGISTRY["nnUNet_MSFFBridge_ResAttn"](
    in_ch=1,
    out_ch=1,
    base=32
)
```

Input:

``` python
x.shape
# (B, 1, 256, 256)
```

Output:

``` python
logits = model(x)

logits.shape
# (B, 1, 256, 256)
```

For inference:

``` python
probability = torch.sigmoid(logits)
mask = probability > 0.5
```

The threshold can be evaluated separately on a validation set if
threshold optimisation is part of the experimental protocol.

------------------------------------------------------------------------

# 30. Important Training Implementation Rule

For binary segmentation, the model should output logits.

Correct:

``` text
Model
  ↓
Raw logits
  ├──────────────→ BCEWithLogitsLoss
  │
  └→ sigmoid → Dice
```

Incorrect:

``` text
Model
  ↓
sigmoid
  ↓
BCEWithLogitsLoss
```

Applying sigmoid before `BCEWithLogitsLoss` changes the expected input
domain of the loss and can negatively affect optimisation.

------------------------------------------------------------------------

# 31. Expected Research Contributions

The project is ultimately targeting a system with three layers of
contribution:

### Contribution 1 --- Segmentation

Accurate automatic fetal-head segmentation.

### Contribution 2 --- Biometric estimation

Automatic extraction of the fetal-head contour and HC measurement.

### Contribution 3 --- Robustness and reliability

Evaluation of model behaviour under difficult ultrasound conditions,
with emphasis on boundary accuracy and measurement reliability.

The third contribution is particularly important because very high Dice
values on HC18 can make small improvements in average overlap less
meaningful scientifically.

------------------------------------------------------------------------

# 32. Limitations

Current limitations include:

-   HC18 is a relatively small benchmark for deep-learning research.
-   Dataset-specific performance may not translate directly to clinical
    deployment.
-   Ultrasound scanners and acquisition protocols can differ.
-   Operator variability can affect image appearance.
-   High validation Dice does not guarantee low boundary error.
-   High segmentation accuracy does not automatically guarantee accurate
    HC measurement.
-   The current candidate architectures require controlled ablation
    studies before claiming superiority.
-   The supplied project evidence does not establish multicentre
    clinical validation.
-   The supplied project evidence does not establish that the current
    residual-attention-MSFF candidate is superior to official nnU-Net.

------------------------------------------------------------------------

# 33. Roadmap

## Phase 1 --- Pipeline validation

-   Verify dataset loading
-   Verify mask alignment
-   Verify preprocessing
-   Verify augmentation
-   Verify metric implementation
-   Verify loss implementation
-   Verify train/validation/test separation

## Phase 2 --- Strong baselines

-   Train U-Net
-   Train nnU-Net-style baseline
-   Train U-Net + MiT-B2
-   Reproduce baseline metrics

## Phase 3 --- Architecture ablation

-   Add residual blocks
-   Add attention gates
-   Add MSFF bridge
-   Measure parameter/FLOP overhead

## Phase 4 --- Robustness

Evaluate performance on:

-   Low-contrast images
-   High-noise images
-   Acoustic-shadow cases
-   Poorly defined boundaries
-   Different anatomical orientations

## Phase 5 --- Biometric validation

Evaluate:

-   HC MAE
-   HC RMSE
-   HC error distribution
-   Correlation with reference measurements

## Phase 6 --- Research contribution

Select the mechanism that provides the strongest evidence for:

-   Accuracy
-   Boundary quality
-   Robustness
-   Measurement reliability
-   Efficiency

------------------------------------------------------------------------

# 34. Key References

The project documentation and associated paper cite the following
important works:

1.  Ronneberger, Fischer, Brox --- **U-Net: Convolutional Networks for
    Biomedical Image Segmentation**, MICCAI, 2015.
2.  He et al. --- **Deep Residual Learning for Image Recognition**,
    CVPR, 2016.
3.  Badrinarayanan et al. --- **SegNet**, TPAMI, 2017.
4.  Oktay et al. --- **Attention U-Net**, 2018.
5.  Zhou et al. --- **U-Net++**, 2018.
6.  Chen et al. --- **DeepLabV3+**, ECCV, 2018.
7.  Zeng et al. --- **Fetal Ultrasound Image Segmentation for Automatic
    Head Circumference Biometry Using Deeply Supervised Attention-Gated
    V-Net**, Journal of Digital Imaging, 2021.
8.  Chen et al. --- **TransUNet: Transformers Make Strong Encoders for
    Medical Image Segmentation**, 2021 preprint.
9.  Płotka et al. --- **FetalNet: Multi-task Deep Learning Framework for
    Fetal Ultrasound Biometric Measurements**, 2021.
10. Zhang, Petitjean, Ainouz --- HC18 benchmark work on
    segmentation-based versus regression-based fetal HC estimation.
11. Xie et al. --- **SegFormer: Simple and Efficient Design for Semantic
    Segmentation with Transformers**, 2021.
12. Isensee et al. --- **nnU-Net: a self-configuring method for deep
    learning-based biomedical image segmentation**, 2021.

The associated IEEE ICCIDS 2026 paper in this project reports the
U-Net + MiT-B2 fetal-head segmentation pipeline and its HC estimation
workflow.

------------------------------------------------------------------------

# 35. Citation / Project Source

The project is documented in the supplied paper:

**"Intelligent Ultrasound Analysis for Real Time Fetal Head
Circumference Measurement and Developmental Assessment"**

The paper describes the U-Net + MiT-B2 baseline, HC18 experiments,
Dice+BCE training, contour/ellipse-based HC estimation, Grad-CAM
interpretation, and comparative evaluation.

The current `models_library.py` extends the project with additional
research architectures, including residual-attention and MSFF variants.

------------------------------------------------------------------------

# 36. Final Summary

This project is an end-to-end AI pipeline for:

``` text
Fetal Ultrasound
      ↓
Pre-processing
      ↓
Deep Learning Segmentation
      ↓
Fetal Head Mask
      ↓
Contour Extraction
      ↓
Ellipse Fitting
      ↓
Head Circumference
      ↓
Quantitative + Visual Analysis
```

The research progression is:

``` text
U-Net
   ↓
U-Net + MiT-B2
   ↓
nnU-Net-style baseline
   ↓
Residual nnU-Net
   ↓
Residual + Attention nnU-Net
   ↓
Residual + Attention + MSFF nnU-Net
   ↓
Robustness / boundary / biometric validation
```

The central scientific principle of the project is:

> **Do not claim that architectural complexity improves performance
> until controlled experiments demonstrate the improvement.**

The ultimate goal is not simply to obtain a higher Dice score on one
benchmark, but to develop a segmentation and biometric-analysis system
that is accurate, robust, computationally practical, interpretable, and
scientifically defensible.
