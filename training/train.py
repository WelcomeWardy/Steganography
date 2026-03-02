"""
training/train.py
Training loop with hypergraph JPEG augmentation.

Strategy per batch:
  ① Encode: encoder produces container images
  ② With probability JPEG_AUG_PROB (50%), apply hypergraph JPEG compression
     to the container images with a random quality factor
  ③ Set jpeg_flag = 1.0 for compressed batches, 0.0 for clean batches
  ④ Train full model: encoder + decoder learn together end-to-end
  ⑤ Train decoder 2 extra steps on the (possibly compressed) containers
"""

import os
import numpy as np
import tensorflow as tf

from configs.config import (
    BATCH_SIZE, EPOCHS, LR_SCHEDULE,
    MODEL_SAVE_DIR, ENCODER_SAVE_PATH,
    DECODER_SAVE_PATH, FULL_MODEL_SAVE_PATH,
    LOSS_HISTORY_PATH, JPEG_AUG_PROB, JPEG_QUALITY_RANGE,
)
from jpeg_layer.hypergraph_jpeg import compress_batch_random_quality


# ── Learning rate schedule ─────────────────────────────────────────────────────

def lr_schedule(epoch_idx: int) -> float:
    for start, end, lr in LR_SCHEDULE:
        if end is None or epoch_idx < end:
            return lr
    return LR_SCHEDULE[-1][2]


# ── Training loop ──────────────────────────────────────────────────────────────

def train(deep_stegan, encoder, decoder,
          input_S: np.ndarray, input_C: np.ndarray):
    """
    Parameters
    ----------
    deep_stegan : compiled combined Keras model
    encoder     : hide network
    decoder     : reveal network (JPEG-aware)
    input_S     : (N, H, W, 3) float64 — secret images
    input_C     : (N, H, W, 3) float64 — cover  images
    """
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    m              = len(input_S)
    full_loss_hist = []
    rev_loss_hist  = []
    jpeg_applied   = 0     # counter for logging

    print(f"[train] Training on {m} image pairs")
    print(f"[train] JPEG augmentation probability: {JPEG_AUG_PROB}")
    print(f"[train] JPEG quality range: {JPEG_QUALITY_RANGE}")

    for epoch in range(EPOCHS):

        # ── Shuffle ────────────────────────────────────────────────────────────
        perm   = np.random.permutation(m)
        s_shuf = input_S[perm]
        c_shuf = input_C[perm]

        itera       = m // BATCH_SIZE
        f_loss_mean = 0.0
        r_loss_mean = 0.0
        jpeg_count  = 0

        for i in range(itera):
            bs = i * BATCH_SIZE
            be = min((i + 1) * BATCH_SIZE, m)

            batch_message = s_shuf[bs:be]
            batch_cover   = c_shuf[bs:be]
            n_batch       = len(batch_message)

            # ── ① Encoder forward: generate containers ─────────────────────
            container = encoder.predict(
                [batch_message, batch_cover], verbose=0
            )

            # ── ② Randomly apply JPEG to the container ─────────────────────
            apply_jpeg = np.random.random() < JPEG_AUG_PROB

            if apply_jpeg:
                container_input = compress_batch_random_quality(
                    container, JPEG_QUALITY_RANGE
                )
                jpeg_flag_val = np.ones((n_batch, 1), dtype=np.float64)
                jpeg_count += 1
            else:
                container_input = container.copy()
                jpeg_flag_val   = np.zeros((n_batch, 1), dtype=np.float64)

            # ── ③ Train full model end-to-end ──────────────────────────────
            # Target: recover original secret AND keep cover unchanged
            y_true = np.concatenate([batch_message, batch_cover], axis=-1)

            f_loss = deep_stegan.train_on_batch(
                x=[batch_message, batch_cover, jpeg_flag_val],
                y=y_true,
            )

            # ── ④ Extra decoder training (2 steps) ─────────────────────────
            # Use the (possibly compressed) container with the flag
            for _ in range(2):
                r_loss = decoder.train_on_batch(
                    x=[container_input, jpeg_flag_val],
                    y=batch_message,
                )

            f_loss_mean += f_loss
            r_loss_mean += r_loss

            if (i + 1) % 10 == 0:
                print(
                    f"  Epoch {epoch+1:04d} | Batch {i+1:03d}/{itera} | "
                    f"full={f_loss:.4f}  rev={r_loss:.4f}  "
                    f"jpeg={'yes' if apply_jpeg else 'no '}"
                )

        # ── Epoch summary ──────────────────────────────────────────────────────
        f_loss_mean /= itera
        r_loss_mean /= itera
        full_loss_hist.append(f_loss_mean)
        rev_loss_hist.append(r_loss_mean)

        current_lr = lr_schedule(epoch)
        deep_stegan.optimizer.learning_rate.assign(current_lr)
        decoder.optimizer.learning_rate.assign(current_lr)

        print(
            f"Epoch {epoch+1:04d}/{EPOCHS} | "
            f"full={f_loss_mean:.4f} | rev={r_loss_mean:.4f} | "
            f"LR={current_lr} | JPEG batches={jpeg_count}/{itera}"
        )

        # ── Checkpoint every 100 epochs ────────────────────────────────────────
        if (epoch + 1) % 100 == 0:
            deep_stegan.save(FULL_MODEL_SAVE_PATH)
            encoder.save(ENCODER_SAVE_PATH)
            decoder.save(DECODER_SAVE_PATH)
            print(f"[train] Checkpoint saved at epoch {epoch+1}")

    # ── Final save ─────────────────────────────────────────────────────────────
    deep_stegan.save(FULL_MODEL_SAVE_PATH)
    encoder.save(ENCODER_SAVE_PATH)
    decoder.save(DECODER_SAVE_PATH)
    np.save(LOSS_HISTORY_PATH,
            {"full_loss": full_loss_hist, "reveal_loss": rev_loss_hist})

    print(f"\n[train] Done. Models saved to '{MODEL_SAVE_DIR}/'")
    return full_loss_hist, rev_loss_hist