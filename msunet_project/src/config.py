"""
config.py
Central configuration for the Fetal Head Segmentation project
(U-Net encoder-decoder + MiT-B2 / SegFormer transformer encoder).

Edit DATA_ROOT to point at the folder that contains the HC18 'training_set'
folder and the 'training_set_pixel_size_and_HC.csv' file, downloaded from:
https://zenodo.org/records/1327317  (HC18 Grand Challenge dataset)
"""

import os
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Folder that contains: training_set/  and training_set_pixel_size_and_HC.csv
DATA_ROOT = os.environ.get("HC18_DATA_ROOT", os.path.join(PROJECT_ROOT, "data", "raw"))
RAW_IMAGE_DIR = os.path.join(DATA_ROOT, "training_set")
RAW_CSV_PATH = os.path.join(DATA_ROOT, "training_set_pixel_size_and_HC.csv")

# Where the prepared (split + filled-mask) dataset is written
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
SPLITS_CSV = os.path.join(PROCESSED_DIR, "splits.csv")

# Where checkpoints / logs / predictions go
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")

for d in [PROCESSED_DIR, OUTPUT_DIR, CKPT_DIR, LOG_DIR, PRED_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# Data split (patient/image level — HC18 has one image per case, so this is
# a simple random split seeded for reproducibility, stratified is not needed)
# ---------------------------------------------------------------------------
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
SPLIT_SEED = 42

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
ENCODER_NAME = "mit_b2"          # SegFormer / Mix Vision Transformer encoder
ENCODER_WEIGHTS = "imagenet"
IN_CHANNELS = 3                  # SMP encoders expect 3 channels (grayscale repeated)
NUM_CLASSES = 1                  # binary segmentation
ACTIVATION = None                # we apply sigmoid manually (logits out of the model)

IMAGE_SIZE = 256                 # paper uses 256x256

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_WORKERS = 4
EPOCHS = 100
LR = 1e-4
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 15
LR_SCHEDULER_PATIENCE = 6
LR_SCHEDULER_FACTOR = 0.5
GRAD_CLIP_NORM = 5.0
THRESHOLD = 0.5                  # binarization threshold for predicted mask
SEED = 42

# Hybrid loss weighting:  L = bce_weight * BCE + dice_weight * Dice
BCE_WEIGHT = 0.5
DICE_WEIGHT = 0.5

# AMP (mixed precision) — speeds up training on most GPUs (T4/A100/RTX), safe to
# disable by setting to False if you hit numerical issues with a given GPU.
USE_AMP = True
