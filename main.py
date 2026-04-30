"""
main.py — Deep Steganography v2 with JPEG augmentation

Usage
-----
    python main.py                  # full run (preprocess → train → evaluate)
    python main.py --eval-only      # load saved models and evaluate
    python main.py --no-eval        # skip evaluation after training
    python main.py --eval-v1        # evaluate OLD v1 model with new metrics
"""

import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description="Deep Steganography v2")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--no-eval",   action="store_true")
    parser.add_argument("--eval-v1",   action="store_true",
                        help="Evaluate old v1 model (no JPEG layer) with new metrics")
    return parser.parse_args()


def main():
    args = parse_args()

    from configs.config import (
        IMAGE_SIZE, FULL_MODEL_SAVE_PATH,
        ENCODER_SAVE_PATH, DECODER_SAVE_PATH, LOSS_HISTORY_PATH,
    )

    # ── STEP 1: Pre-processing ─────────────────────────────────────────────────
    from preprocessing.preprocess import load_and_preprocess
    print("=" * 60)
    print("STEP 1 — Data Pre-Processing  (4000 images)")
    print("=" * 60)
    input_S, input_C, x_test = load_and_preprocess()

    # ── Evaluate OLD v1 model with new metrics ─────────────────────────────────
    if args.eval_v1:
        import tensorflow as tf

        def rev_loss_v1(t, p):
            return 2.0 * tf.reduce_sum(tf.square(t - p))

        def full_loss_v1(t, p):
            mt, ct = t[..., :3], t[..., 3:6]
            mp, cp = p[..., :3], p[..., 3:6]
            return 2.0 * rev_loss_v1(mt, mp) + tf.reduce_sum(tf.square(ct - cp))

        co = {"rev_loss": rev_loss_v1, "full_loss": full_loss_v1}

        print("[main] Loading v1 saved models…")
        encoder = tf.keras.models.load_model(
            ENCODER_SAVE_PATH, custom_objects=co, compile=False)
        decoder = tf.keras.models.load_model(
            DECODER_SAVE_PATH, custom_objects=co, compile=False)
        print("[main] v1 models loaded.")

        full_loss_hist = rev_loss_hist = None
        if os.path.exists(LOSS_HISTORY_PATH):
            h = np.load(LOSS_HISTORY_PATH, allow_pickle=True).item()
            full_loss_hist = h.get("full_loss")
            rev_loss_hist  = h.get("reveal_loss")

        from evaluation.evaluate2 import evaluate_v1
        evaluate_v1(encoder, decoder,
                    input_S[:200], input_C[:200],
                    full_loss_hist, rev_loss_hist,
                    output_dir="evaluation_outputs_v1")
        print("\n[main] v1 evaluation done.")
        return

    # ── STEP 2: Build / Load v2 model ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2 — Model Build / Load")
    print("=" * 60)

    if args.eval_only:
        import tensorflow as tf
        from models.model import rev_loss, full_loss
        co = {"rev_loss": rev_loss, "full_loss": full_loss}

        print("[main] Loading saved v2 models…")
        for path in [FULL_MODEL_SAVE_PATH, ENCODER_SAVE_PATH, DECODER_SAVE_PATH]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Model not found: {path}\n"
                    "Run 'python main.py' to train first."
                )

        deep_stegan = tf.keras.models.load_model(
            FULL_MODEL_SAVE_PATH, custom_objects=co, compile=False)
        encoder = tf.keras.models.load_model(
            ENCODER_SAVE_PATH, custom_objects=co, compile=False)
        decoder = tf.keras.models.load_model(
            DECODER_SAVE_PATH, custom_objects=co, compile=False)
        print("[main] Loaded v2 models successfully.")

    else:
        from models.model import build_deep_steganography_model
        deep_stegan, encoder, decoder = build_deep_steganography_model(IMAGE_SIZE)
        deep_stegan.summary()

    # ── STEP 3: Training ───────────────────────────────────────────────────────
    full_loss_hist = rev_loss_hist = None

    if not args.eval_only:
        from training.train import train
        print("\n" + "=" * 60)
        print("STEP 3 — Training  (with JPEG augmentation)")
        print("=" * 60)
        full_loss_hist, rev_loss_hist = train(
            deep_stegan, encoder, decoder, input_S, input_C
        )
    else:
        if os.path.exists(LOSS_HISTORY_PATH):
            h = np.load(LOSS_HISTORY_PATH, allow_pickle=True).item()
            full_loss_hist = h.get("full_loss")
            rev_loss_hist  = h.get("reveal_loss")

    # ── STEP 4: Evaluation ─────────────────────────────────────────────────────
    if not args.no_eval:
        from evaluation.evaluate import evaluate
        print("\n" + "=" * 60)
        print("STEP 4 — Evaluation  (all metrics + JPEG robustness)")
        print("=" * 60)
        evaluate(
            encoder, decoder,
            input_S[:200], input_C[:200],
            full_loss_hist=full_loss_hist,
            rev_loss_hist=rev_loss_hist,
            output_dir="evaluation_outputs",
        )

    print("\n[main] Done.")


if __name__ == "__main__":
    main()