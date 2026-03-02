import os

# ─── Dataset Paths ─────────────────────────────────────────────────────────────
DATASET_ROOT = "/mnt/c/Users/jange/Downloads/archive/tiny-imagenet-200"
TRAIN_DIR    = "/mnt/c/Users/jange/Downloads/archive/tiny-imagenet-200/train"
TEST_DIR     = "/mnt/c/Users/jange/Downloads/archive/tiny-imagenet-200/val/images"

# ─── Data ──────────────────────────────────────────────────────────────────────
IMAGE_SIZE     = (64, 64, 3)
# Now loading 4000 training images total → 2000 cover + 2000 secret
NUM_TRAIN      = 4000
NUM_TEST       = 1000
IMAGES_PER_CAT = 20          # 20 images per category × 200 categories = 4000
NUM_CATEGORIES = 200

# ─── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE        = 32
EPOCHS            = 1000
BETA              = 2.0
JPEG_AUG_PROB     = 0.5      # probability of applying JPEG to a batch during training
JPEG_QUALITY_RANGE = (20, 90) # random quality drawn from this range during training

# ─── Learning rate schedule ────────────────────────────────────────────────────
LR_SCHEDULE = [
    (0,   200, 0.001),
    (200, 400, 0.0003),
    (400, 600, 0.0001),
    (600, None, 0.00003),
]

# ─── JPEG evaluation quality factors ──────────────────────────────────────────
JPEG_EVAL_QUALITIES = [10, 20, 30, 50, 70, 90]

# ─── Paths ─────────────────────────────────────────────────────────────────────
MODEL_SAVE_DIR       = "saved_models"
ENCODER_SAVE_PATH    = os.path.join(MODEL_SAVE_DIR, "encoder.h5")
DECODER_SAVE_PATH    = os.path.join(MODEL_SAVE_DIR, "decoder.h5")
FULL_MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "deep_stegan.h5")
LOSS_HISTORY_PATH    = os.path.join(MODEL_SAVE_DIR, "loss_history.npy")
METRICS_SAVE_PATH    = os.path.join(MODEL_SAVE_DIR, "eval_metrics.npy")