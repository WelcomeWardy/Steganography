"""
jpeg_layer/hypergraph_jpeg.py

Hypergraph-based JPEG compression used as a training augmentation layer.
Applied to container images AFTER encoding and BEFORE decoding during training.

Key difference from standard JPEG: the compression pipeline is modelled
as a hypergraph where pixels are nodes and processing groups are hyperedges.
This makes the relationships between pixels and processing stages explicit.

Public API:
    compress_batch(batch, quality)  →  compressed batch (numpy)
    compress_single(image, quality) →  compressed image  (numpy)
"""

import numpy as np
from collections import defaultdict


# ══════════════════════════════════════════════════════════════════════════════
# QUANTISATION TABLES
# ══════════════════════════════════════════════════════════════════════════════

_QUANT_Y = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68,109,103, 77],
    [24, 35, 55, 64, 81,104,113, 92],
    [49, 64, 78, 87,103,121,120,101],
    [72, 92, 95, 98,112,100,103, 99],
], dtype=np.float64)

_QUANT_C = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# HYPERGRAPH STRUCTURE
# ══════════════════════════════════════════════════════════════════════════════

class _Hypergraph:
    """
    Lightweight hypergraph used internally during compression.
    Nodes = pixel positions (row, col)
    Hyperedges = processing groups (blocks, frequency bands, etc.)
    """

    def __init__(self):
        self.hyperedges = {}   # edge_id → set of (r, c) nodes
        self.metadata   = {}   # edge_id → dict

    def add_edge(self, edge_id: str, nodes: set, meta: dict = None):
        self.hyperedges[edge_id] = nodes
        self.metadata[edge_id]   = meta or {}

    def nodes_of(self, edge_id: str) -> set:
        return self.hyperedges.get(edge_id, set())


# ══════════════════════════════════════════════════════════════════════════════
# DCT / IDCT  (pure numpy — no scipy dependency)
# ══════════════════════════════════════════════════════════════════════════════

def _build_dct_matrix() -> np.ndarray:
    """
    Pre-compute the 8×8 DCT-II transform matrix D such that
    DCT(block) = D @ block @ D.T
    This vectorised form is ~200× faster than the nested-loop version.
    """
    N = 8
    D = np.zeros((N, N))
    for u in range(N):
        for x in range(N):
            if u == 0:
                D[u, x] = np.sqrt(1 / N)
            else:
                D[u, x] = np.sqrt(2 / N) * np.cos((2*x + 1)*u*np.pi / (2*N))
    return D


_DCT_MATRIX  = _build_dct_matrix()
_IDCT_MATRIX = _DCT_MATRIX.T          # inverse DCT matrix = transpose


def _dct2(block: np.ndarray) -> np.ndarray:
    """2D DCT-II of an 8×8 block using the pre-computed matrix."""
    return _DCT_MATRIX @ block @ _DCT_MATRIX.T


def _idct2(block: np.ndarray) -> np.ndarray:
    """2D inverse DCT of an 8×8 block."""
    return _IDCT_MATRIX @ block @ _IDCT_MATRIX.T


# ══════════════════════════════════════════════════════════════════════════════
# COLOUR CONVERSION
# ══════════════════════════════════════════════════════════════════════════════

def _rgb_to_ycbcr(img: np.ndarray) -> np.ndarray:
    """
    RGB [0,255] → YCbCr [0,255]
    Hypergraph representation: three channel hyperedges each connecting
    ALL pixels, with different weights per channel.
    """
    hg = _Hypergraph()
    H, W = img.shape[:2]
    all_pixels = {(r, c) for r in range(H) for c in range(W)}

    # Each channel hyperedge encodes the transformation weights
    hg.add_edge("Y",  all_pixels, {"weights": [ 0.299,  0.587,  0.114], "offset":   0})
    hg.add_edge("Cb", all_pixels, {"weights": [-0.169, -0.331,  0.500], "offset": 128})
    hg.add_edge("Cr", all_pixels, {"weights": [ 0.500, -0.419, -0.081], "offset": 128})

    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
    out = np.zeros_like(img, dtype=np.float64)

    for i, ch in enumerate(["Y", "Cb", "Cr"]):
        w      = hg.metadata[ch]["weights"]
        offset = hg.metadata[ch]["offset"]
        out[:,:,i] = w[0]*R + w[1]*G + w[2]*B + offset

    return np.clip(out, 0, 255)


def _ycbcr_to_rgb(img: np.ndarray) -> np.ndarray:
    """YCbCr [0,255] → RGB [0,255]"""
    Y  = img[:,:,0]
    Cb = img[:,:,1]
    Cr = img[:,:,2]

    R = Y + 1.40200  * (Cr - 128)
    G = Y - 0.34414  * (Cb - 128) - 0.71414 * (Cr - 128)
    B = Y + 1.77200  * (Cb - 128)

    return np.clip(np.stack([R, G, B], axis=-1), 0, 255)


# ══════════════════════════════════════════════════════════════════════════════
# CHROMA SUBSAMPLING
# ══════════════════════════════════════════════════════════════════════════════

def _subsample_420(ycbcr: np.ndarray) -> dict:
    """
    4:2:0 chroma subsampling.
    Hypergraph: each 2×2 block of pixels forms a hyperedge sharing one Cb/Cr value.
    """
    hg = _Hypergraph()
    H, W = ycbcr.shape[:2]

    Y_full = ycbcr[:,:,0].copy()
    Cb_sub = np.zeros((H//2, W//2))
    Cr_sub = np.zeros((H//2, W//2))

    for r in range(0, H-1, 2):
        for c in range(0, W-1, 2):
            # Hyperedge: 4 pixels share one chroma value
            hg.add_edge(f"cs_{r}_{c}",
                        {(r,c),(r,c+1),(r+1,c),(r+1,c+1)},
                        {"sub_r": r//2, "sub_c": c//2})
            Cb_sub[r//2, c//2] = ycbcr[r:r+2, c:c+2, 1].mean()
            Cr_sub[r//2, c//2] = ycbcr[r:r+2, c:c+2, 2].mean()

    return {"Y": Y_full, "Cb": Cb_sub, "Cr": Cr_sub, "_hg": hg}


def _upsample_420(channels: dict, H: int, W: int) -> np.ndarray:
    """Upsample Cb/Cr back to full resolution."""
    Y  = channels["Y"]
    Cb = np.repeat(np.repeat(channels["Cb"], 2, axis=0), 2, axis=1)[:H, :W]
    Cr = np.repeat(np.repeat(channels["Cr"], 2, axis=0), 2, axis=1)[:H, :W]
    return np.stack([Y, Cb, Cr], axis=-1)


# ══════════════════════════════════════════════════════════════════════════════
# BLOCK PROCESSING  (Step 3 + 4 + 5 + reverse)
# ══════════════════════════════════════════════════════════════════════════════

def _scale_quant_table(base: np.ndarray, quality: int) -> np.ndarray:
    """Scale a quantisation table based on quality (1–95)."""
    quality = max(1, min(95, quality))
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality
    q = np.floor((base * scale + 50) / 100)
    return np.clip(q, 1, 255)


def _compress_channel(channel: np.ndarray,
                       quant_table: np.ndarray) -> np.ndarray:
    """
    Compress one image channel through the full block pipeline:
    Split → Level-shift → DCT → Quantise → Dequantise → IDCT → Reassemble

    Hypergraph role: each 8×8 block is a hyperedge connecting 64 pixel nodes.
    For each block, 64 frequency hyperedges connect those same pixels with
    different DCT basis weights.
    """
    H, W  = channel.shape
    out   = np.zeros_like(channel)
    hg    = _Hypergraph()

    # Pad to multiple of 8
    pH = (8 - H % 8) % 8
    pW = (8 - W % 8) % 8
    padded = np.pad(channel, ((0, pH), (0, pW)), mode='edge')
    PH, PW = padded.shape

    for br in range(0, PH, 8):
        for bc in range(0, PW, 8):
            block = padded[br:br+8, bc:bc+8].copy()

            # ── Block hyperedge: 64 pixels processed together ──────────────
            pixels = {(br+r, bc+c) for r in range(8) for c in range(8)}
            hg.add_edge(f"blk_{br}_{bc}", pixels,
                        {"origin": (br, bc), "quant": quant_table})

            # ── Level shift ────────────────────────────────────────────────
            block = block - 128.0

            # ── DCT  (frequency hyperedges implicit in the matrix multiply) ─
            dct_block = _dct2(block)

            # ── Quantise ───────────────────────────────────────────────────
            q_block = np.round(dct_block / quant_table)

            # ── Dequantise ─────────────────────────────────────────────────
            dq_block = q_block * quant_table

            # ── IDCT ───────────────────────────────────────────────────────
            spatial = _idct2(dq_block) + 128.0

            # ── Place back ─────────────────────────────────────────────────
            r_end = min(br+8, H)
            c_end = min(bc+8, W)
            out[br:r_end, bc:c_end] = spatial[:r_end-br, :c_end-bc]

    return np.clip(out, 0, 255)


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def compress_single(image: np.ndarray, quality: int = 50) -> np.ndarray:
    """
    Apply hypergraph-based JPEG compression to a single image.

    Parameters
    ----------
    image   : np.ndarray  float64 in [0,1], shape (H, W, 3)
    quality : int         JPEG quality factor 1–95

    Returns
    -------
    np.ndarray  float64 in [0,1], same shape — compressed image
    """
    # Scale to [0, 255] for processing
    img255 = (image * 255.0).astype(np.float64)

    # Step 1: RGB → YCbCr  (colour channel hyperedges)
    ycbcr = _rgb_to_ycbcr(img255)

    # Step 2: Chroma subsampling  (2×2 block hyperedges)
    H, W  = img255.shape[:2]
    channels = _subsample_420(ycbcr)

    # Step 3–5: Block split + DCT + Quantise for each channel
    qY = _scale_quant_table(_QUANT_Y, quality)
    qC = _scale_quant_table(_QUANT_C, quality)

    channels["Y"]  = _compress_channel(channels["Y"],  qY)
    channels["Cb"] = _compress_channel(channels["Cb"], qC)
    channels["Cr"] = _compress_channel(channels["Cr"], qC)

    # Upsample Cb/Cr + convert back to RGB
    ycbcr_out = _upsample_420(channels, H, W)
    rgb_out   = _ycbcr_to_rgb(ycbcr_out)

    # Return in [0, 1]
    return np.clip(rgb_out / 255.0, 0.0, 1.0)


def compress_batch(batch: np.ndarray, quality: int = 50) -> np.ndarray:
    """
    Apply hypergraph JPEG compression to a batch of images.

    Parameters
    ----------
    batch   : np.ndarray  shape (N, H, W, 3), float64 [0,1]
    quality : int         JPEG quality factor 1–95

    Returns
    -------
    np.ndarray  same shape, compressed
    """
    return np.stack([compress_single(img, quality) for img in batch])


def compress_batch_random_quality(batch: np.ndarray,
                                   quality_range: tuple = (20, 90)) -> np.ndarray:
    """
    Compress each image in a batch with a DIFFERENT random quality factor.
    Used during training for augmentation diversity.

    Parameters
    ----------
    batch         : (N, H, W, 3) float64 [0,1]
    quality_range : (min_q, max_q) inclusive

    Returns
    -------
    compressed batch, same shape
    """
    result = np.zeros_like(batch)
    for i, img in enumerate(batch):
        q = np.random.randint(quality_range[0], quality_range[1] + 1)
        result[i] = compress_single(img, q)
    return result