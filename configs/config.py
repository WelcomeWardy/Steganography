import os

# ─── Dataset Paths ────────────────────────────────────────────────────────────
DATASET_ROOT = "/mnt/c/Users/jange/Downloads/archive/tiny-imagenet-200"
TRAIN_DIR    = "/mnt/c/Users/jange/Downloads/archive/tiny-imagenet-200/train"
TEST_DIR     = "/mnt/c/Users/jange/Downloads/archive/tiny-imagenet-200/val/images"

# ─── Data ─────────────────────────────────────────────────────────────────────
IMAGE_SIZE     = (64, 64, 3)          # H × W × C
NUM_TRAIN      = 1000                 # images loaded for training
NUM_TEST       = 1000                 # images loaded for testing
IMAGES_PER_CAT = 10                  # random images sampled per category
NUM_CATEGORIES = 200                  # Tiny-ImageNet categories (max 200 used partially)

# ─── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE     = 32
EPOCHS         = 1000
BETA           = 2.0                  # weight for rev_loss

# Learning-rate schedule break-points
LR_SCHEDULE = [
    (0,   200, 0.001),
    (200, 400, 0.0003),
    (400, 600, 0.0001),
    (600, None, 0.00003),
]

# ─── Paths ─────────────────────────────────────────────────────────────────────
MODEL_SAVE_DIR      = "saved_models"
ENCODER_SAVE_PATH   = os.path.join(MODEL_SAVE_DIR, "encoder.h5")
DECODER_SAVE_PATH   = os.path.join(MODEL_SAVE_DIR, "decoder.h5")
FULL_MODEL_SAVE_PATH= os.path.join(MODEL_SAVE_DIR, "deep_stegan.h5")
LOSS_HISTORY_PATH   = os.path.join(MODEL_SAVE_DIR, "loss_history.npy")