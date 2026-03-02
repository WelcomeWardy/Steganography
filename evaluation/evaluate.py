"""
evaluation/evaluate2.py
Same full metrics as evaluate.py BUT for the OLD model
(encoder → decoder, no JPEG layer, no jpeg_flag input).

Use this to evaluate models trained with the original v1 architecture.

Metrics:
  • RMSE  (secret and cover)
  • PSNR  (secret and cover)
  • SSIM  (secret and cover)
  • BER   (secret and cover)

All computed per image + full summary.
JPEG robustness table across quality factors.
BER vs quality plot.

Usage:
    from evaluation.evaluate2 import evaluate_v1
    results = evaluate_v1(encoder, decoder, input_S, input_C)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde

from configs.config import JPEG_EVAL_QUALITIES, METRICS_SAVE_PATH
from jpeg_layer.hypergraph_jpeg import compress_batch


# ══════════════════════════════════════════════════════════════════════════════
# METRIC FUNCTIONS  (identical to evaluate.py — standalone so no circular deps)
# ══════════════════════════════════════════════════════════════════════════════

def compute_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10(1.0 / mse)


def compute_ssim(img1: np.ndarray, img2: np.ndarray,
                  data_range: float = 1.0) -> float:
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    mu1    = np.mean(img1);  mu2    = np.mean(img2)
    sigma1 = np.var(img1);   sigma2 = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    num = (2*mu1*mu2 + C1) * (2*sigma12 + C2)
    den = (mu1**2 + mu2**2 + C1) * (sigma1 + sigma2 + C2)
    return float(num / den)


def compute_ber(original: np.ndarray, reconstructed: np.ndarray,
                threshold: float = 0.5) -> float:
    orig_bits  = (np.clip(original,      0, 1) > threshold).astype(np.uint8)
    recon_bits = (np.clip(reconstructed, 0, 1) > threshold).astype(np.uint8)
    return float(np.sum(orig_bits != recon_bits) / orig_bits.size)


def compute_rmse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(255 * (original - reconstructed)))))


# ══════════════════════════════════════════════════════════════════════════════
# PER-IMAGE METRICS
# ══════════════════════════════════════════════════════════════════════════════

def _compute_all_metrics(input_S, input_C, decoded_S, decoded_C) -> dict:
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
            print(f"  [{i+1}/{N}] "
                  f"PSNR_S={metrics['psnr_secret'][i]:.2f}  "
                  f"SSIM_S={metrics['ssim_secret'][i]:.4f}  "
                  f"BER_S={metrics['ber_secret'][i]:.4f}")
    return metrics


def _print_summary(metrics: dict, label: str = ""):
    print(f"\n{'='*62}")
    print(f"METRICS SUMMARY  {label}")
    print(f"{'='*62}")
    print(f"{'Metric':20s} {'Mean':>10s} {'Std':>10s} {'Min':>10s} {'Max':>10s}")
    print("-" * 62)
    for key, arr in metrics.items():
        finite = arr[np.isfinite(arr)]
        if len(finite) == 0:
            continue
        print(f"{key:20s} {finite.mean():10.4f} {finite.std():10.4f} "
              f"{finite.min():10.4f} {finite.max():10.4f}")


def _print_per_image_table(metrics: dict):
    N = len(next(iter(metrics.values())))
    print(f"\n{'─'*85}")
    print(f"{'Img':>4} {'PSNR_S':>8} {'PSNR_C':>8} {'SSIM_S':>8} "
          f"{'SSIM_C':>8} {'BER_S':>8} {'BER_C':>8} {'RMSE_S':>8} {'RMSE_C':>8}")
    print(f"{'─'*85}")
    for i in range(N):
        print(
            f"{i:>4d} "
            f"{metrics['psnr_secret'][i]:>8.2f} "
            f"{metrics['psnr_cover'][i]:>8.2f} "
            f"{metrics['ssim_secret'][i]:>8.4f} "
            f"{metrics['ssim_cover'][i]:>8.4f} "
            f"{metrics['ber_secret'][i]:>8.4f} "
            f"{metrics['ber_cover'][i]:>8.4f} "
            f"{metrics['rmse_secret'][i]:>8.2f} "
            f"{metrics['rmse_cover'][i]:>8.2f}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# JPEG ROBUSTNESS  (v1 model — no jpeg_flag)
# ══════════════════════════════════════════════════════════════════════════════

def _jpeg_robustness_v1(encoder, decoder,
                         input_S, input_C,
                         qualities, output_dir) -> dict:
    """
    For each quality:
      encode → JPEG compress → decode (old model has no flag input)
    """
    results = {}
    print(f"\n{'='*60}")
    print("JPEG ROBUSTNESS EVALUATION  (v1 model)")
    print(f"{'='*60}")

    for q in qualities:
        print(f"\n  Quality factor: {q}")
        containers = encoder.predict([input_S, input_C], verbose=0)
        compressed = compress_batch(containers, quality=q)

        # Old decoder: single input, no jpeg_flag
        decoded_S = decoder.predict(compressed, verbose=0)
        decoded_C = compressed

        m = _compute_all_metrics(input_S, input_C, decoded_S, decoded_C)
        results[q] = {k: float(np.mean(v[np.isfinite(v)])) for k, v in m.items()}
        print(f"    PSNR_S={results[q]['psnr_secret']:.2f}  "
              f"SSIM_S={results[q]['ssim_secret']:.4f}  "
              f"BER_S={results[q]['ber_secret']:.4f}")
        print(f"    PSNR_C={results[q]['psnr_cover']:.2f}  "
              f"SSIM_C={results[q]['ssim_cover']:.4f}  "
              f"BER_C={results[q]['ber_cover']:.4f}")

    # Consolidated table
    print(f"\n{'='*70}")
    print("CONSOLIDATED JPEG QUALITY TABLE  (v1 model)")
    print(f"{'='*70}")
    print(f"{'Quality':>8} {'PSNR_S':>10} {'SSIM_S':>10} "
          f"{'PSNR_C':>10} {'SSIM_C':>10} {'BER_S':>10} {'BER_C':>10}")
    print("-" * 70)
    for q in qualities:
        r = results[q]
        print(f"{q:>8d} {r['psnr_secret']:>10.2f} {r['ssim_secret']:>10.4f} "
              f"{r['psnr_cover']:>10.2f} {r['ssim_cover']:>10.4f} "
              f"{r['ber_secret']:>10.4f} {r['ber_cover']:>10.4f}")

    # BER vs quality plot
    ber_s = [results[q]["ber_secret"] for q in qualities]
    ber_c = [results[q]["ber_cover"]  for q in qualities]
    plt.figure(figsize=(9, 5))
    plt.plot(qualities, ber_s, "o-", color="steelblue",  linewidth=2,
             markersize=7, label="Secret BER")
    plt.plot(qualities, ber_c, "s-", color="darkorange", linewidth=2,
             markersize=7, label="Cover BER")
    plt.xlabel("JPEG Quality Factor", fontsize=13)
    plt.ylabel("Bit Error Rate (BER)", fontsize=13)
    plt.title("BER vs JPEG Quality Factor  (v1 model — no JPEG training)", fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(qualities)
    plt.tight_layout()
    path = os.path.join(output_dir, "ber_vs_quality_v1.png")
    plt.savefig(path, dpi=150)
    print(f"\n[evaluate2] BER vs quality plot → {path}")
    plt.show()

    return results


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def _plot_loss(full_loss_hist, rev_loss_hist, save_path):
    plt.figure(figsize=(10, 4))
    plt.plot(full_loss_hist, label="Full Loss",   color="steelblue")
    plt.plot(rev_loss_hist,  label="Reveal Loss", color="darkorange")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Training Loss (v1 model)"); plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def _plot_errors(diff_S, diff_C, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, label in zip(
        axes,
        [(diff_C*255).flatten(), (diff_S*255).flatten()],
        ["Cover error distribution", "Secret error distribution"]
    ):
        ax.hist(data, bins=100, alpha=0.5, facecolor="red", density=True)
        if len(data) > 1:
            kde = gaussian_kde(data)
            xs  = np.linspace(data.min(), data.max(), 500)
            ax.plot(xs, kde(xs), color="darkred", linewidth=1.5)
        ax.set_xlim([0, 250]); ax.set_ylim([0, 0.2])
        ax.set_title(label, fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def _show_grid(input_C, input_S, decoded_C, decoded_S, save_path):
    rand_idx   = np.random.randint(0, len(input_C), 4)
    col_titles = ["Cover", "Secret", "Encoded Cover", "Decoded Secret"]
    fig = plt.figure(figsize=(12, 12))
    gs  = gridspec.GridSpec(4, 4, figure=fig, hspace=0.4)
    for row, idx in enumerate(rand_idx):
        for col, (img, title) in enumerate(
            zip([input_C[idx], input_S[idx], decoded_C[idx], decoded_S[idx]],
                col_titles)
        ):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(np.clip(img, 0, 1))
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(title, fontsize=9)
    plt.suptitle("v1 Model — Effectiveness & Fidelity", fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_v1(encoder, decoder,
                input_S: np.ndarray,
                input_C: np.ndarray,
                full_loss_hist=None,
                rev_loss_hist=None,
                output_dir: str = "evaluation_outputs_v1") -> dict:
    """
    Full evaluation for the OLD model (v1 — no JPEG layer, no jpeg_flag).

    Parameters
    ----------
    encoder        : trained v1 encoder model
    decoder        : trained v1 decoder model (single input)
    input_S        : (N, H, W, 3) float64 — secret images
    input_C        : (N, H, W, 3) float64 — cover  images
    full_loss_hist : list of floats (optional, for loss plot)
    rev_loss_hist  : list of floats (optional, for loss plot)
    output_dir     : where to save plots

    Returns
    -------
    dict with per_image metrics and jpeg_robustness results
    """
    os.makedirs(output_dir, exist_ok=True)
    N = len(input_S)

    print("\n[evaluate2] ── V1 Model Evaluation ──")

    # ── Encode ────────────────────────────────────────────────────────────────
    print("[evaluate2] Generating containers…")
    decoded_C = encoder.predict([input_S, input_C], verbose=1)

    # ── Decode ────────────────────────────────────────────────────────────────
    print("[evaluate2] Recovering secrets…")
    decoded_S = decoder.predict(decoded_C, verbose=1)

    # ── Per-image metrics ─────────────────────────────────────────────────────
    print("\n[evaluate2] Computing per-image metrics…")
    metrics = _compute_all_metrics(input_S, input_C, decoded_S, decoded_C)

    _print_per_image_table(metrics)
    _print_summary(metrics, label="(v1 model — no JPEG)")

    # ── JPEG robustness ───────────────────────────────────────────────────────
    jpeg_results = _jpeg_robustness_v1(
        encoder, decoder, input_S, input_C,
        JPEG_EVAL_QUALITIES, output_dir
    )

    # ── Save metrics ──────────────────────────────────────────────────────────
    save_path = os.path.join(output_dir, "eval_metrics_v1.npy")
    np.save(save_path,
            {"per_image": metrics, "jpeg_table": jpeg_results},
            allow_pickle=True)
    print(f"\n[evaluate2] Metrics saved → {save_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if full_loss_hist is not None:
        _plot_loss(full_loss_hist, rev_loss_hist,
                   os.path.join(output_dir, "training_loss_v1.png"))

    _plot_errors(
        np.abs(input_S - decoded_S),
        np.abs(input_C - decoded_C),
        os.path.join(output_dir, "error_distribution_v1.png")
    )
    _show_grid(input_C, input_S, decoded_C, decoded_S,
               os.path.join(output_dir, "image_grid_v1.png"))

    return {
        "decoded_C":       decoded_C,
        "decoded_S":       decoded_S,
        "per_image":       metrics,
        "jpeg_robustness": jpeg_results,
    }