"""
training/train.py
Custom training loop for the deep steganography model.

Key fixes vs original:
  - Decoder is no longer frozen — trains end-to-end with encoder
  - Decoder also gets extra dedicated training steps per batch
  - Better logging
"""

import os
import numpy as np
import tensorflow as tf

from configs.config import (
    BATCH_SIZE, EPOCHS, LR_SCHEDULE,
    MODEL_SAVE_DIR, ENCODER_SAVE_PATH,
    DECODER_SAVE_PATH, FULL_MODEL_SAVE_PATH,
    LOSS_HISTORY_PATH,
)


# ─── Learning-rate schedule ───────────────────────────────────────────────────

def lr_schedule(epoch_idx: int) -> float:
    for start, end, lr in LR_SCHEDULE:
        if end is None or epoch_idx < end:
            return lr
    return LR_SCHEDULE[-1][2]


# ─── Training loop ────────────────────────────────────────────────────────────

def train(deep_stegan, encoder, decoder, input_S: np.ndarray, input_C: np.ndarray):
    """
    Training strategy:
      Step A — Train the full model (encoder + decoder end-to-end) on full_loss
               This teaches the encoder to hide AND the decoder to reveal together
      Step B — Train the decoder alone again on the container output
               This gives the decoder extra gradient signal to improve extraction
    """
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    m              = len(input_S)
    batch_size     = BATCH_SIZE
    full_loss_hist = []
    rev_loss_hist  = []

    for epoch in range(EPOCHS):

        # ── Shuffle ───────────────────────────────────────────────────────────
        perm   = np.random.permutation(m)
        s_shuf = input_S[perm]
        c_shuf = input_C[perm]

        itera       = m // batch_size
        f_loss_mean = 0.0
        r_loss_mean = 0.0

        for i in range(itera):
            bs = i * batch_size
            be = min((i + 1) * batch_size, m)

            batch_message = s_shuf[bs:be]
            batch_cover   = c_shuf[bs:be]

            # ── Step A: Train full model end-to-end ───────────────────────────
            # y_true = [secret, cover] concatenated — what we want the model to output
            y_true = np.concatenate([batch_message, batch_cover], axis=-1)
            f_loss = deep_stegan.train_on_batch(
                x=[batch_message, batch_cover],
                y=y_true,
            )

            # ── Step B: Extra decoder training ────────────────────────────────
            # Generate containers using encoder, then train decoder to extract secret
            # Do this 2x per batch to give decoder more training signal
            container = encoder.predict([batch_message, batch_cover], verbose=0)
            for _ in range(2):
                r_loss = decoder.train_on_batch(
                    x=container,
                    y=batch_message,
                )

            f_loss_mean += f_loss
            r_loss_mean += r_loss

            if (i + 1) % 10 == 0:
                print(
                    f"  Epoch {epoch+1:04d} | Batch {i+1:03d}/{itera} | "
                    f"full_loss={f_loss:.4f}  rev_loss={r_loss:.4f}"
                )

        # ── Epoch summary ─────────────────────────────────────────────────────
        f_loss_mean /= itera
        r_loss_mean /= itera
        full_loss_hist.append(f_loss_mean)
        rev_loss_hist.append(r_loss_mean)

        # ── Update learning rates ─────────────────────────────────────────────
        current_lr = lr_schedule(epoch)
        deep_stegan.optimizer.learning_rate.assign(current_lr)
        decoder.optimizer.learning_rate.assign(current_lr)

        print(
            f"Epoch {epoch+1:04d}/{EPOCHS} | "
            f"Mean full_loss={f_loss_mean:.4f} | "
            f"Mean reveal_loss={r_loss_mean:.4f} | "
            f"LR={current_lr}"
        )

        # ── Save checkpoint every 100 epochs ─────────────────────────────────
        if (epoch + 1) % 100 == 0:
            deep_stegan.save(FULL_MODEL_SAVE_PATH)
            encoder.save(ENCODER_SAVE_PATH)
            decoder.save(DECODER_SAVE_PATH)
            print(f"[train] Checkpoint saved at epoch {epoch+1}")

    # ── Final save ────────────────────────────────────────────────────────────
    deep_stegan.save(FULL_MODEL_SAVE_PATH)
    encoder.save(ENCODER_SAVE_PATH)
    decoder.save(DECODER_SAVE_PATH)

    np.save(
        LOSS_HISTORY_PATH,
        {"full_loss": full_loss_hist, "reveal_loss": rev_loss_hist},
    )

    print(f"\n[train] Models saved to '{MODEL_SAVE_DIR}/'")
    return full_loss_hist, rev_loss_hist