"""
preprocessing/preprocess.py
Loads Tiny-ImageNet images, normalises pixel values to [0,1] and splits
the training array into cover (input_C) and secret (input_S) halves.
"""

import os
import numpy as np
import cv2

from configs.config import (
    TRAIN_DIR, TEST_DIR,
    IMAGE_SIZE, NUM_TRAIN, NUM_TEST, IMAGES_PER_CAT,
)


# ─────────────────────────────────────────────────────────────────────────────
def _load_training_images() -> np.ndarray:
    """
    Iterates over Tiny-ImageNet training categories and randomly samples
    IMAGES_PER_CAT images per category until NUM_TRAIN images are collected.
    Returns a uint8 numpy array of shape (NUM_TRAIN, H, W, C).
    """
    files = sorted(os.listdir(TRAIN_DIR))
    h, w, c = IMAGE_SIZE
    x_train = np.empty((NUM_TRAIN, h, w, c), dtype="uint8")
    a = 0

    for i in range(NUM_CATEGORIES := len(files)):
        if a >= NUM_TRAIN:
            break
        rand_indices = np.random.randint(0, 500, IMAGES_PER_CAT)
        img_dir = os.path.join(TRAIN_DIR, files[i], "images")
        if not os.path.isdir(img_dir):
            continue
        img_files = sorted(os.listdir(img_dir))
        for j in rand_indices:
            if a >= NUM_TRAIN:
                break
            idx = int(j) % len(img_files)
            img_path = os.path.join(img_dir, img_files[idx])
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (w, h))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x_train[a] = img
            a += 1

    return x_train[:a]


def _load_test_images() -> np.ndarray:
    """
    Loads up to NUM_TEST images from the validation / test directory.
    Returns a uint8 numpy array of shape (NUM_TEST, H, W, C).
    """
    files_te = sorted(os.listdir(TEST_DIR))
    h, w, c = IMAGE_SIZE
    x_test = np.empty((NUM_TEST, h, w, c), dtype="uint8")

    for a in range(min(NUM_TEST, len(files_te))):
        img_path = os.path.join(TEST_DIR, files_te[a])
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x_test[a] = img

    return x_test


# ─────────────────────────────────────────────────────────────────────────────
def load_and_preprocess():
    """
    Public entry point.

    Returns
    -------
    input_S  : np.ndarray float64, shape (N, H, W, C) – secret images
    input_C  : np.ndarray float64, shape (N, H, W, C) – cover  images
    x_test   : np.ndarray float64, shape (M, H, W, C) – test   images (normalised)
    """
    print("[preprocess] Loading training images …")
    x_train = _load_training_images()
    print(f"[preprocess] Loaded {len(x_train)} training images.")

    print("[preprocess] Loading test images …")
    x_test  = _load_test_images()
    print(f"[preprocess] Loaded {len(x_test)} test images.")

    # ── Normalise to [0, 1] ──────────────────────────────────────────────────
    half = len(x_train) // 2
    input_C = x_train[:half]   / 255.0
    input_S = x_train[half:]   / 255.0
    x_test  = x_test            / 255.0

    # ── Cast to float64 ──────────────────────────────────────────────────────
    input_C = input_C.astype("float64")
    input_S = input_S.astype("float64")
    x_test  = x_test.astype("float64")

    print(f"[preprocess] input_C shape: {input_C.shape}")
    print(f"[preprocess] input_S shape: {input_S.shape}")
    print(f"[preprocess] x_test  shape: {x_test.shape}")

    return input_S, input_C, x_test