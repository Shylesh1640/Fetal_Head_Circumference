"""
config.py
---------
Central configuration for the MS-UNet (U-Net + MiT-B2) fetal head
segmentation and head-circumference (HC) estimation pipeline.

Edit the paths under DATA section to point at your local copy of the
HC18 Grand Challenge dataset before running train.py.
"""

import torch

# --------------------------------------------------------------------------
# DATA
# --------------------------------------------------------------------------
# Folder that directly contains the *_HC.png and *_HC_Annotation.png files
# and the training_set_pixel_size_and_HC.csv file (standard HC18 layout).
DATA_DIR = "/teamspace/studios/this_studio/data/training_set"
CSV_PATH = "/teamspace/studios/this_studio/data/training_set_pixel_size_and_HC.csv"

IMG_SIZE = 256          # network input/output resolution (paper uses 256x256)
VAL_SPLIT = 0.15         # fraction of patients held out for validation
TEST_SPLIT = 0.10        # fraction of patients held out for testing
RANDOM_SEED = 42

# --------------------------------------------------------------------------
# MODEL
# --------------------------------------------------------------------------
ENCODER_NAME = "mit_b2"        # Mix Vision Transformer encoder (SegFormer backbone)
ENCODER_WEIGHTS = "imagenet"   # pretrained weights
IN_CHANNELS = 3                 # SMP encoders expect 3-channel input
NUM_CLASSES = 1                 # binary segmentation (head vs background)

# --------------------------------------------------------------------------
# TRAINING
# --------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EARLY_STOPPING_PATIENCE = 15
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.5
NUM_WORKERS = 4
GRAD_CLIP_NORM = 5.0

CHECKPOINT_DIR = "/teamspace/studios/this_studio/checkpoints"
BEST_MODEL_PATH = f"{CHECKPOINT_DIR}/best_msunet.pth"
LAST_MODEL_PATH = f"{CHECKPOINT_DIR}/last_msunet.pth"

# --------------------------------------------------------------------------
# OUTPUTS
# --------------------------------------------------------------------------
OUTPUT_DIR = "/teamspace/studios/this_studio/outputs"
GRADCAM_DIR = f"{OUTPUT_DIR}/gradcam"
OVERLAY_DIR = f"{OUTPUT_DIR}/overlays"
RESULTS_CSV = f"{OUTPUT_DIR}/hc_predictions.csv"
METRICS_LOG_CSV = f"{OUTPUT_DIR}/training_log.csv"
