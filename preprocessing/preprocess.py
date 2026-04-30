"""
preprocessing/preprocess.py
Loads 4000 training images (2000 cover + 2000 secret) from Tiny-ImageNet.
"""

import os
import numpy as np
import cv2

from configs.config import (
    TRAIN_DIR, TEST_DIR,
    IMAGE_SIZE, NUM_TRAIN, NUM_TEST, IMAGES_PER_CAT,
)


def _load_training_images() -> np.ndarray:
    """
    Load NUM_TRAIN images (default 4000) from training categories.
    Samples IMAGES_PER_CAT (default 20) random images per category.
    200 categories × 20 images = 4000 images.
    """
    files = sorted(os.listdir(TRAIN_DIR))
    h, w, c = IMAGE_SIZE
    x_train = np.empty((NUM_TRAIN, h, w, c), dtype="uint8")
    a = 0

    for i in range(len(files)):
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
            idx      = int(j) % len(img_files)
            img_path = os.path.join(img_dir, img_files[idx])
            img      = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (w, h))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x_train[a] = img
            a += 1

    print(f"[preprocess] Loaded {a} training images")
    return x_train[:a]


def _load_test_images() -> np.ndarray:
    files_te = sorted(os.listdir(TEST_DIR))
    h, w, c  = IMAGE_SIZE
    x_test   = np.empty((NUM_TEST, h, w, c), dtype="uint8")

    for a in range(min(NUM_TEST, len(files_te))):
        img_path = os.path.join(TEST_DIR, files_te[a])
        img      = cv2.imread(img_path)
        if img is None:
            continue
        img       = cv2.resize(img, (w, h))
        img       = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x_test[a] = img

    print(f"[preprocess] Loaded {min(NUM_TEST, len(files_te))} test images")
    return x_test


def load_and_preprocess():
    """
    Returns
    -------
    input_S : (2000, 64, 64, 3) float64  secret images
    input_C : (2000, 64, 64, 3) float64  cover  images
    x_test  : (1000, 64, 64, 3) float64  test   images
    """
    print("[preprocess] Loading training images …")
    x_train = _load_training_images()

    print("[preprocess] Loading test images …")
    x_test  = _load_test_images()

    # Split 4000 images → 2000 cover + 2000 secret
    half    = len(x_train) // 2
    input_C = (x_train[:half]  / 255.0).astype("float64")
    input_S = (x_train[half:]  / 255.0).astype("float64")
    x_test  = (x_test           / 255.0).astype("float64")

    print(f"[preprocess] input_C (cover):  {input_C.shape}")
    print(f"[preprocess] input_S (secret): {input_S.shape}")
    print(f"[preprocess] x_test:           {x_test.shape}")

    return input_S, input_C, x_test