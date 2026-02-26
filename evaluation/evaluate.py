"""
evaluation/evaluate.py
Visual and quantitative evaluation of the trained deep steganography model.

Produces:
  • Training-loss curve
  • Error distribution plots (KDE) for cover and secret images
  • Side-by-side image grid: Cover | Secret | Encoded Cover | Decoded Secret
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _rgb2gray(rgb: np.ndarray) -> np.ndarray:
    """Luminance-weighted greyscale conversion."""
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def _show_image(img: np.ndarray, ax, title: str = "", gray: bool = False):
    ax.set_xticks([])
    ax.set_yticks([])
    if gray:
        ax.imshow(_rgb2gray(img), cmap=plt.get_cmap("gray"))
    else:
        ax.imshow(np.clip(img, 0, 1))
    ax.set_title(title, fontsize=8)


# ─── Loss curve ───────────────────────────────────────────────────────────────

def plot_loss(full_loss_hist: list, rev_loss_hist: list, save_path: str = None):
    """Plot full-loss and reveal-loss vs. epoch."""
    plt.figure(figsize=(10, 4))
    plt.plot(full_loss_hist,  label="Full Loss",   color="steelblue")
    plt.plot(rev_loss_hist,   label="Reveal Loss", color="darkorange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[evaluate] Loss plot saved → {save_path}")
    plt.show()


# ─── Pixel-error statistics ───────────────────────────────────────────────────

def pixel_errors(input_S, input_C, decoded_S, decoded_C):
    """
    Compute per-pixel RMSE arrays for secret and cover images.

    Returns
    -------
    see_Spixel : 1-D float array  – pixel errors for the secret image
    see_Cpixel : 1-D float array  – pixel errors for the cover  image
    """
    see_Spixel = np.sqrt(np.mean(np.square(255 * (input_S - decoded_S))))
    see_Cpixel = np.sqrt(np.mean(np.square(255 * (input_C - decoded_C))))
    return see_Spixel, see_Cpixel


def plot_error_distribution(diff_S: np.ndarray, diff_C: np.ndarray,
                             save_path: str = None):
    """
    Histogram + KDE overlay of per-pixel errors for the secret and cover images.

    Parameters
    ----------
    diff_S : 2-D array   (H, W) or (N, H, W) absolute differences for secret
    diff_C : 2-D array   (H, W) or (N, H, W) absolute differences for cover
    """
    diff_Sflat = (diff_S * 255).flatten()
    diff_Cflat = (diff_C * 255).flatten()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, label in zip(
        axes,
        [diff_Cflat, diff_Sflat],
        ["Distribution of error in the Cover image",
         "Distribution of errors in the Secret image"],
    ):
        counts, bins, _ = ax.hist(data, bins=100, alpha=0.5,
                                  facecolor="red", density=True)

        # KDE overlay
        if len(data) > 1:
            kde = gaussian_kde(data)
            xs  = np.linspace(data.min(), data.max(), 500)
            ax.plot(xs, kde(xs), color="darkred", linewidth=1.5)

        ax.set_xlim([0, 250])
        ax.set_ylim([0, 0.2])
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Pixel error (0-255)")
        ax.set_ylabel("Density")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[evaluate] Error distribution plot saved → {save_path}")
    plt.show()


# ─── Image grid ───────────────────────────────────────────────────────────────

def show_image_grid(input_C, input_S, decoded_C, decoded_S,
                    n: int = 4, show_diff: bool = False,
                    save_path: str = None):
    """
    Display a grid of n randomly chosen samples showing:
        Cover | Secret | Encoded Cover | Decoded Secret
    (and optionally the diff images).

    Parameters
    ----------
    input_C   : (N, H, W, C) – original cover images
    input_S   : (N, H, W, C) – original secret images
    decoded_C : (N, H, W, C) – encoder output  (steganographic container)
    decoded_S : (N, H, W, C) – decoder output  (recovered secret)
    n         : number of random samples to display
    show_diff : whether to show difference images as an extra column
    save_path : optional path to save the figure
    """
    rand_idx = np.random.randint(0, len(input_C), n)
    n_col    = 6 if show_diff else 4
    col_titles = ["Cover", "Secret", "Encoded Cover", "Decoded Secret"]
    if show_diff:
        col_titles += ["Δ Cover", "Δ Secret"]

    fig = plt.figure(figsize=(3 * n_col, 3 * n))
    gs  = gridspec.GridSpec(n, n_col, figure=fig, hspace=0.4)

    for row, idx in enumerate(rand_idx):
        images = [
            input_C[idx],
            input_S[idx],
            decoded_C[idx],
            decoded_S[idx],
        ]
        if show_diff:
            images.append(np.abs(input_C[idx] - decoded_C[idx]))
            images.append(np.abs(input_S[idx] - decoded_S[idx]))

        for col, (img, title) in enumerate(zip(images, col_titles)):
            ax = fig.add_subplot(gs[row, col])
            _show_image(img, ax, title=(title if row == 0 else ""))

    plt.suptitle("Deep Steganography – Effectiveness & Fidelity", fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[evaluate] Image grid saved → {save_path}")
    plt.show()


# ─── Full evaluation pipeline ─────────────────────────────────────────────────

def evaluate(encoder, decoder, input_S, input_C,
             full_loss_hist=None, rev_loss_hist=None,
             output_dir: str = "evaluation_outputs"):
    """
    Run all evaluations:
      1. Loss curve (if history provided)
      2. Pixel-error statistics
      3. Error-distribution plots
      4. Image grid

    Parameters
    ----------
    encoder          : trained keras encoder model
    decoder          : trained keras decoder model
    input_S          : (N, H, W, C) secret  images (normalised)
    input_C          : (N, H, W, C) cover   images (normalised)
    full_loss_hist   : list of full-loss values per epoch
    rev_loss_hist    : list of reveal-loss values per epoch
    output_dir       : directory to save output plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # ── Generate decoded images ────────────────────────────────────────────────
    print("[evaluate] Generating container images via encoder …")
    decoded_C = encoder.predict([input_S, input_C], verbose=1)  # stego containers
    print("[evaluate] Recovering secret images via decoder …")
    decoded_S = decoder.predict(decoded_C, verbose=1)            # recovered secrets

    # ── Loss curve ────────────────────────────────────────────────────────────
    if full_loss_hist is not None and rev_loss_hist is not None:
        plot_loss(
            full_loss_hist, rev_loss_hist,
            save_path=os.path.join(output_dir, "training_loss.png"),
        )

    # ── Pixel errors ──────────────────────────────────────────────────────────
    rmse_S, rmse_C = pixel_errors(input_S, input_C, decoded_S, decoded_C)
    print(f"[evaluate] Secret  RMSE: {rmse_S:.4f}")
    print(f"[evaluate] Cover   RMSE: {rmse_C:.4f}")

    # ── Error distribution ────────────────────────────────────────────────────
    diff_S = np.abs(input_S - decoded_S)
    diff_C = np.abs(input_C - decoded_C)
    plot_error_distribution(
        diff_S, diff_C,
        save_path=os.path.join(output_dir, "error_distribution.png"),
    )

    # ── Visual grid ───────────────────────────────────────────────────────────
    show_image_grid(
        input_C, input_S, decoded_C, decoded_S,
        n=4, show_diff=False,
        save_path=os.path.join(output_dir, "image_grid.png"),
    )

    return {
        "decoded_C" : decoded_C,
        "decoded_S" : decoded_S,
        "rmse_S"    : rmse_S,
        "rmse_C"    : rmse_C,
    }