"""
main.py
Entry point for the deep steganography pipeline.

Usage
-----
    python main.py                  # full run (preprocess → train → evaluate)
    python main.py --eval-only      # evaluate a previously saved model
    python main.py --no-eval        # skip evaluation after training

Set DATASET_ROOT in configs/config.py to point at your local Tiny-ImageNet-200
folder before running.
"""

import argparse
import sys
import os

# ── Make sure project root is on the Python path ─────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description="Deep Image Steganography")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training; load saved models and evaluate.")
    parser.add_argument("--no-eval",   action="store_true",
                        help="Skip evaluation after training.")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── 1. Pre-processing ─────────────────────────────────────────────────────
    from preprocessing.preprocess import load_and_preprocess
    print("=" * 60)
    print("STEP 1 – Data Pre-Processing")
    print("=" * 60)
    input_S, input_C, x_test = load_and_preprocess()

    # ── 2. Build / Load model ─────────────────────────────────────────────────
    from configs.config import (
        IMAGE_SIZE, FULL_MODEL_SAVE_PATH,
        ENCODER_SAVE_PATH, DECODER_SAVE_PATH, LOSS_HISTORY_PATH,
    )
    print("\n" + "=" * 60)
    print("STEP 2 – Model Build / Load")
    print("=" * 60)

    if args.eval_only:
        import tensorflow as tf
        from models.model import rev_loss, full_loss

        custom_objs = {"rev_loss": rev_loss, "full_loss": full_loss}

        print("[main] Loading saved models...")

        # Check files exist before trying to load
        for path, name in [
            (FULL_MODEL_SAVE_PATH, "deep_stegan"),
            (ENCODER_SAVE_PATH,    "encoder"),
            (DECODER_SAVE_PATH,    "decoder"),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"[main] Could not find saved model: {path}\n"
                    "Run 'python main.py' (without --eval-only) to train first."
                )

        deep_stegan = tf.keras.models.load_model(
            FULL_MODEL_SAVE_PATH,
            custom_objects=custom_objs,
            compile=False,
        )
        encoder = tf.keras.models.load_model(
            ENCODER_SAVE_PATH,
            custom_objects=custom_objs,
            compile=False,
        )
        decoder = tf.keras.models.load_model(
            DECODER_SAVE_PATH,
            custom_objects=custom_objs,
            compile=False,
        )
        print("[main] Loaded saved models successfully.")

    else:
        from models.model import build_deep_steganography_model
        deep_stegan, encoder, decoder = build_deep_steganography_model(IMAGE_SIZE)
        deep_stegan.summary()

    # ── 3. Training ───────────────────────────────────────────────────────────
    full_loss_hist = rev_loss_hist = None

    if not args.eval_only:
        from training.train import train
        print("\n" + "=" * 60)
        print("STEP 3 – Training")
        print("=" * 60)
        full_loss_hist, rev_loss_hist = train(
            deep_stegan, encoder, decoder, input_S, input_C
        )
    else:
        # Try to load saved loss history for plotting
        import numpy as np
        if os.path.exists(LOSS_HISTORY_PATH):
            hist = np.load(LOSS_HISTORY_PATH, allow_pickle=True).item()
            full_loss_hist = hist.get("full_loss")
            rev_loss_hist  = hist.get("reveal_loss")
            print("[main] Loaded loss history.")
        else:
            print("[main] No loss history found — skipping loss plot.")

    # ── 4. Evaluation ─────────────────────────────────────────────────────────
    if not args.no_eval:
        from evaluation.evaluate import evaluate
        print("\n" + "=" * 60)
        print("STEP 4 – Evaluation")
        print("=" * 60)
        results = evaluate(
            encoder, decoder,
            input_S[:200], input_C[:200],
            full_loss_hist=full_loss_hist,
            rev_loss_hist =rev_loss_hist,
            output_dir    ="evaluation_outputs",
        )
        print(f"\n[main] Secret RMSE : {results['rmse_S']:.4f}")
        print(f"[main] Cover  RMSE : {results['rmse_C']:.4f}")

    print("\n[main] Done.")


if __name__ == "__main__":
    main()