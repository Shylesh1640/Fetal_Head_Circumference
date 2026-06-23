"""
config.py
Central configuration for the U-Net + MiT-B2 fetal head segmentation project.
Edit DATA_ROOT to point at your unzipped HC18 'training_set' folder.
"""

import os
import torch

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
# This must point to the folder that directly contains:
#   000_HC.png, 000_HC_Annotation.png, 001_HC.png, 001_HC_Annotation.png, ...
# i.e. the HC18 "training_set" directory (999 annotated images + masks).
# Download from: https://zenodo.org/records/1327317  (HC18 dataset, v2)
DATA_ROOT = "/teamspace/studios/this_studio/data/training_set"

OUTPUT_DIR = "/teamspace/studios/this_studio/outputs"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")

for d in [OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR, PLOT_DIR, PRED_DIR]:
    os.makedirs(d, exist_ok=True)

# ----------------------------------------------------------------------
# Data split (paper-style: patient/image-level split, no leakage)
# 999 annotated images -> 70% train / 15% val / 15% test
# ----------------------------------------------------------------------
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
RANDOM_SEED = 42

# ----------------------------------------------------------------------
# Image / model settings
# ----------------------------------------------------------------------
IMAGE_SIZE = 256            # paper uses 256x256
IN_CHANNELS = 3             # MiT-B2 / ImageNet weights expect 3-channel input
NUM_CLASSES = 1             # binary segmentation (head vs background)

ENCODER_NAME = "mit_b2"     # Mix Vision Transformer B2 (SegFormer encoder), via timm
ENCODER_WEIGHTS = "imagenet"
ACTIVATION = None           # raw logits; sigmoid applied in loss/metrics, not in model

# ----------------------------------------------------------------------
# Training hyperparameters (matches paper's reported configuration)
# ----------------------------------------------------------------------
BATCH_SIZE = 8               # raise to 16 if VRAM allows
NUM_WORKERS = 4
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 15
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.5
GRAD_CLIP_NORM = 1.0

# Loss weighting for hybrid Dice-BCE loss
BCE_WEIGHT = 0.5
DICE_WEIGHT = 0.5

# ----------------------------------------------------------------------
# Misc
# ----------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_ENABLED = torch.cuda.is_available()   # mixed precision only makes sense on GPU
PIXEL_THRESHOLD = 0.5                     # sigmoid threshold for binarizing predictions
MASK_BINARIZE_THRESHOLD = 127             # for raw 0-255 PNG masks -> {0,1}
