"""
evaluation/evaluate.py
Full evaluation for the JPEG-aware steganography model.

Metrics computed:
  • RMSE  (secret and cover)
  • PSNR  (secret and cover)
  • SSIM  (secret and cover)
  • BER   (Bit Error Rate — secret and cover)

All metrics computed PER IMAGE and stored in numpy arrays.
Summary statistics (mean, std, min, max) printed at the end.

JPEG robustness evaluation:
  • Runs inference at each quality factor in JPEG_EVAL_QUALITIES
  • Produces consolidated table: quality → PSNR/SSIM for secret and cover
  • Plots BER vs JPEG quality factor

Also produces:
  • Training loss curve
  • Error distribution plots (KDE)
  • Visual image grid
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde

from configs.config import JPEG_EVAL_QUALITIES, METRICS_SAVE_PATH
from jpeg_layer.hypergraph_jpeg import compress_batch


# ══════════════════════════════════════════════════════════════════════════════
# METRIC FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def compute_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Peak Signal-to-Noise Ratio in dB.
    Higher is better. >30 dB is generally good quality.

    PSNR = 10 × log10(MAX² / MSE)
    For images in [0,1]: MAX = 1.0
    """
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10(1.0 / mse)


def compute_ssim(img1: np.ndarray, img2: np.ndarray,
                  data_range: float = 1.0) -> float:
    """
    Structural Similarity Index Measure.
    Range: [-1, 1]. Closer to 1 = more similar.

    SSIM considers luminance, contrast, and structure simultaneously.
    Much closer to human perception than MSE or PSNR.

    Formula:
      SSIM(x,y) = (2μxμy + C1)(2σxy + C2)
                  ─────────────────────────────────────
                  (μx² + μy² + C1)(σx² + σy² + C2)
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # Compute over the whole image (global SSIM)
    mu1    = np.mean(img1)
    mu2    = np.mean(img2)
    sigma1 = np.var(img1)
    sigma2 = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    numerator   = (2*mu1*mu2 + C1) * (2*sigma12 + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (sigma1 + sigma2 + C2)

    return float(numerator / denominator)


def compute_ber(original: np.ndarray, reconstructed: np.ndarray,
                threshold: float = 0.5) -> float:
    """
    Bit Error Rate — fraction of bits that differ between original and
    reconstructed image when both are binarised at 'threshold'.

    Process:
      1. Clip both images to [0,1]
      2. Binarise: pixel > threshold → bit=1, else bit=0
      3. XOR the two binary images
      4. BER = number of differing bits / total bits

    Range: [0, 1]. 0 = perfect match, 0.5 = random (worst).
    """
    orig_bits  = (np.clip(original,       0, 1) > threshold).astype(np.uint8)
    recon_bits = (np.clip(reconstructed,  0, 1) > threshold).astype(np.uint8)
    differing  = np.sum(orig_bits != recon_bits)
    total      = orig_bits.size
    return float(differing / total)


def compute_rmse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Root Mean Square Error on pixel scale [0, 255]."""
    return float(np.sqrt(np.mean(np.square(255 * (original - reconstructed)))))


# ══════════════════════════════════════════════════════════════════════════════
# PER-IMAGE METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_all_metrics_per_image(input_S: np.ndarray,
                                   input_C: np.ndarray,
                                   decoded_S: np.ndarray,
                                   decoded_C: np.ndarray) -> dict:
    """
    Compute all metrics for every test image individually.

    Returns a dict of numpy arrays, one value per image:
    {
      "psnr_secret":  (N,),
      "psnr_cover":   (N,),
      "ssim_secret":  (N,),
      "ssim_cover":   (N,),
      "ber_secret":   (N,),
      "ber_cover":    (N,),
      "rmse_secret":  (N,),
      "rmse_cover":   (N,),
    }
    """
    N = len(input_S)
    metrics = {k: np.zeros(N) for k in [
        "psnr_secret", "psnr_cover",
        "ssim_secret", "ssim_cover",
        "ber_secret",  "ber_cover",
        "rmse_secret", "rmse_cover",
    ]}

    for i in range(N):
        metrics["psnr_secret"][i] = compute_psnr(input_S[i], decoded_S[i])
        metrics["psnr_cover"][i]  = compute_psnr(input_C[i], decoded_C[i])
        metrics["ssim_secret"][i] = compute_ssim(input_S[i], decoded_S[i])
        metrics["ssim_cover"][i]  = compute_ssim(input_C[i], decoded_C[i])
        metrics["ber_secret"][i]  = compute_ber(input_S[i],  decoded_S[i])
        metrics["ber_cover"][i]   = compute_ber(input_C[i],  decoded_C[i])
        metrics["rmse_secret"][i] = compute_rmse(input_S[i], decoded_S[i])
        metrics["rmse_cover"][i]  = compute_rmse(input_C[i], decoded_C[i])

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{N}] PSNR_S={metrics['psnr_secret'][i]:.2f} "
                  f"PSNR_C={metrics['psnr_cover'][i]:.2f} "
                  f"SSIM_S={metrics['ssim_secret'][i]:.4f} "
                  f"BER_S={metrics['ber_secret'][i]:.4f}")

    return metrics


def print_metrics_summary(metrics: dict, label: str = ""):
    """Print mean ± std and min/max for every metric."""
    print(f"\n{'='*60}")
    print(f"METRICS SUMMARY  {label}")
    print(f"{'='*60}")
    header = f"{'Metric':20s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}"
    print(header)
    print("-" * 62)
    for key, arr in metrics.items():
        finite = arr[np.isfinite(arr)]
        if len(finite) == 0:
            continue
        print(f"{key:20s} {finite.mean():10.4f} {finite.std():10.4f} "
              f"{finite.min():10.4f} {finite.max():10.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# JPEG ROBUSTNESS TABLE
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_jpeg_robustness(encoder, decoder,
                              input_S: np.ndarray,
                              input_C: np.ndarray,
                              qualities: list = None,
                              output_dir: str = "evaluation_outputs") -> dict:
    """
    For each JPEG quality factor:
      1. Encode the images (encoder)
      2. Apply hypergraph JPEG at that quality
      3. Decode with jpeg_flag=1.0
      4. Compute all metrics

    Returns dict: { quality: { metric_name: mean_value } }
    Also prints a consolidated table and saves a BER vs quality plot.
    """
    if qualities is None:
        qualities = JPEG_EVAL_QUALITIES

    results = {}
    print(f"\n{'='*60}")
    print("JPEG ROBUSTNESS EVALUATION")
    print(f"{'='*60}")

    for q in qualities:
        print(f"\n  Quality factor: {q}")

        # Encode
        containers = encoder.predict([input_S, input_C], verbose=0)

        # Apply JPEG at this quality
        compressed = compress_batch(containers, quality=q)

        # Decode with flag=1 (decoder knows JPEG was applied)
        n          = len(compressed)
        flag_ones  = np.ones((n, 1), dtype=np.float64)
        decoded_S  = decoder.predict([compressed, flag_ones], verbose=0)

        # Also get decoded cover (the compressed container vs original cover)
        decoded_C  = compressed   # container IS the cover output

        # Compute metrics
        m = compute_all_metrics_per_image(input_S, input_C, decoded_S, decoded_C)
        results[q] = {k: float(np.mean(v[np.isfinite(v)])) for k, v in m.items()}
        print(f"    PSNR_S={results[q]['psnr_secret']:.2f} dB  "
              f"SSIM_S={results[q]['ssim_secret']:.4f}  "
              f"BER_S={results[q]['ber_secret']:.4f}")
        print(f"    PSNR_C={results[q]['psnr_cover']:.2f} dB  "
              f"SSIM_C={results[q]['ssim_cover']:.4f}  "
              f"BER_C={results[q]['ber_cover']:.4f}")

    # ── Consolidated table ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("CONSOLIDATED JPEG QUALITY TABLE")
    print(f"{'='*70}")
    print(f"{'Quality':>8} {'PSNR_S':>10} {'SSIM_S':>10} "
          f"{'PSNR_C':>10} {'SSIM_C':>10} {'BER_S':>10} {'BER_C':>10}")
    print("-" * 70)
    for q in qualities:
        r = results[q]
        print(f"{q:>8d} {r['psnr_secret']:>10.2f} {r['ssim_secret']:>10.4f} "
              f"{r['psnr_cover']:>10.2f} {r['ssim_cover']:>10.4f} "
              f"{r['ber_secret']:>10.4f} {r['ber_cover']:>10.4f}")

    # ── BER vs quality plot ────────────────────────────────────────────────────
    _plot_ber_vs_quality(results, qualities, output_dir)

    return results


def _plot_ber_vs_quality(results: dict, qualities: list, output_dir: str):
    """Plot BER (secret and cover) vs JPEG quality factor."""
    ber_s = [results[q]["ber_secret"] for q in qualities]
    ber_c = [results[q]["ber_cover"]  for q in qualities]

    plt.figure(figsize=(9, 5))
    plt.plot(qualities, ber_s, "o-", color="steelblue",   linewidth=2,
             markersize=7, label="Secret BER")
    plt.plot(qualities, ber_c, "s-", color="darkorange",  linewidth=2,
             markersize=7, label="Cover BER")
    plt.xlabel("JPEG Quality Factor", fontsize=13)
    plt.ylabel("Bit Error Rate (BER)", fontsize=13)
    plt.title("BER vs JPEG Quality Factor", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(qualities)
    plt.tight_layout()

    path = os.path.join(output_dir, "ber_vs_jpeg_quality.png")
    plt.savefig(path, dpi=150)
    print(f"\n[evaluate] BER vs quality plot saved → {path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def plot_loss(full_loss_hist: list, rev_loss_hist: list,
              save_path: str = None):
    plt.figure(figsize=(10, 4))
    plt.plot(full_loss_hist, label="Full Loss",   color="steelblue")
    plt.plot(rev_loss_hist,  label="Reveal Loss", color="darkorange")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Training Loss"); plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[evaluate] Loss plot → {save_path}")
    plt.show()


def plot_error_distribution(diff_S: np.ndarray, diff_C: np.ndarray,
                             save_path: str = None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, label in zip(
        axes,
        [(diff_C * 255).flatten(), (diff_S * 255).flatten()],
        ["Cover image error distribution", "Secret image error distribution"],
    ):
        ax.hist(data, bins=100, alpha=0.5, facecolor="red", density=True)
        if len(data) > 1:
            kde = gaussian_kde(data)
            xs  = np.linspace(data.min(), data.max(), 500)
            ax.plot(xs, kde(xs), color="darkred", linewidth=1.5)
        ax.set_xlim([0, 250]); ax.set_ylim([0, 0.2])
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Pixel error (0-255)"); ax.set_ylabel("Density")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[evaluate] Error distribution → {save_path}")
    plt.show()


def show_image_grid(input_C, input_S, decoded_C, decoded_S,
                    n: int = 4, save_path: str = None):
    rand_idx   = np.random.randint(0, len(input_C), n)
    col_titles = ["Cover", "Secret", "Encoded Cover", "Decoded Secret"]
    fig = plt.figure(figsize=(12, 3 * n))
    gs  = gridspec.GridSpec(n, 4, figure=fig, hspace=0.4)

    for row, idx in enumerate(rand_idx):
        images = [input_C[idx], input_S[idx], decoded_C[idx], decoded_S[idx]]
        for col, (img, title) in enumerate(zip(images, col_titles)):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(np.clip(img, 0, 1))
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize=9)

    plt.suptitle("Deep Steganography — Effectiveness & Fidelity", fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[evaluate] Image grid → {save_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(encoder, decoder,
             input_S: np.ndarray,
             input_C: np.ndarray,
             full_loss_hist=None,
             rev_loss_hist=None,
             output_dir: str = "evaluation_outputs") -> dict:
    """
    Full evaluation pipeline for the JPEG-aware model.

    1. Encode images
    2. Decode WITHOUT JPEG (flag=0) — baseline
    3. Compute all per-image metrics and summary
    4. JPEG robustness evaluation across quality factors
    5. Generate all plots

    Returns dict with all results.
    """
    os.makedirs(output_dir, exist_ok=True)
    N = len(input_S)

    # ── ① Encode ──────────────────────────────────────────────────────────────
    print("\n[evaluate] Generating container images (encoder)…")
    decoded_C = encoder.predict([input_S, input_C], verbose=1)

    # ── ② Decode — no JPEG (baseline) ─────────────────────────────────────────
    print("[evaluate] Recovering secrets (decoder, no JPEG)…")
    flag_zeros = np.zeros((N, 1), dtype=np.float64)
    decoded_S  = decoder.predict([decoded_C, flag_zeros], verbose=1)

    # ── ③ Per-image metrics ───────────────────────────────────────────────────
    print("\n[evaluate] Computing per-image metrics…")
    per_image_metrics = compute_all_metrics_per_image(
        input_S, input_C, decoded_S, decoded_C
    )

    # Print every image's values
    print(f"\n{'─'*85}")
    print(f"{'Img':>4} {'PSNR_S':>8} {'PSNR_C':>8} {'SSIM_S':>8} "
          f"{'SSIM_C':>8} {'BER_S':>8} {'BER_C':>8} {'RMSE_S':>8} {'RMSE_C':>8}")
    print(f"{'─'*85}")
    for i in range(N):
        print(
            f"{i:>4d} "
            f"{per_image_metrics['psnr_secret'][i]:>8.2f} "
            f"{per_image_metrics['psnr_cover'][i]:>8.2f} "
            f"{per_image_metrics['ssim_secret'][i]:>8.4f} "
            f"{per_image_metrics['ssim_cover'][i]:>8.4f} "
            f"{per_image_metrics['ber_secret'][i]:>8.4f} "
            f"{per_image_metrics['ber_cover'][i]:>8.4f} "
            f"{per_image_metrics['rmse_secret'][i]:>8.2f} "
            f"{per_image_metrics['rmse_cover'][i]:>8.2f}"
        )

    # Print summary
    print_metrics_summary(per_image_metrics, label="(No JPEG — baseline)")

    # ── ④ JPEG robustness ─────────────────────────────────────────────────────
    jpeg_results = evaluate_jpeg_robustness(
        encoder, decoder, input_S, input_C,
        qualities=JPEG_EVAL_QUALITIES,
        output_dir=output_dir,
    )

    # ── ⑤ Save all metrics ────────────────────────────────────────────────────
    save_data = {
        "per_image":   per_image_metrics,
        "jpeg_table":  jpeg_results,
    }
    np.save(METRICS_SAVE_PATH, save_data, allow_pickle=True)
    print(f"\n[evaluate] All metrics saved → {METRICS_SAVE_PATH}")

    # ── ⑥ Plots ───────────────────────────────────────────────────────────────
    if full_loss_hist is not None:
        plot_loss(full_loss_hist, rev_loss_hist,
                  save_path=os.path.join(output_dir, "training_loss.png"))

    diff_S = np.abs(input_S - decoded_S)
    diff_C = np.abs(input_C - decoded_C)
    plot_error_distribution(
        diff_S, diff_C,
        save_path=os.path.join(output_dir, "error_distribution.png")
    )
    show_image_grid(
        input_C, input_S, decoded_C, decoded_S,
        save_path=os.path.join(output_dir, "image_grid.png")
    )

    return {
        "decoded_C":       decoded_C,
        "decoded_S":       decoded_S,
        "per_image":       per_image_metrics,
        "jpeg_robustness": jpeg_results,
    }