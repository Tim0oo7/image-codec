"""Simple binary-mask coders, for comparison with the context model.

All return the payload size in bits (no header): the same header is added to every method in the
comparison. Exact sizes, not estimates, except where noted.

raw          1 bit per pixel.
rle_gamma    Runs of equal pixels in raster order (the first run is transparent and may be
             empty), each length + 1 written in Elias-gamma code. A plain, classic RLE.
rle_adaptive Best-effort RLE: the same runs, but each run length is binarised (gamma: unary
             exponent + mantissa bits) and every binary decision is coded with an adaptive
             estimator, separately for transparent and opaque runs. Ideal arithmetic-coding cost.
positions    The positions of the opaque pixels, at the theoretical minimum for an unstructured
             set: log2(N + 1) bits for the count k, plus log2(C(N, k)) for which k of the N
             pixels (enumerative coding; no real list of coordinates can do better).
row_changes  "Bitmappy": per row, the number of changes (gamma), then the gaps between change
             positions (gamma). No use of the row above.
g4           CCITT Group 4 (fax, T.6) via TIFF: per row it codes where the colour changes relative
             to the changes in the row above. Real files; the TIFF container is removed by
             subtracting the size of an empty image with the same dimensions.
"""

import io
import math

import numpy as np
from PIL import Image


def raw(m):
    return float(m.size)


def gamma_len(v):
    """Bits of the Elias-gamma code of v >= 1."""
    return 2 * int(v).bit_length() - 1


def runs(m):
    x = m.ravel()
    change = np.flatnonzero(x[1:] != x[:-1]) + 1
    bounds = np.concatenate(([0], change, [x.size]))
    lengths = np.diff(bounds)
    if x[0] == 1:
        lengths = np.concatenate(([0], lengths))  # leading empty transparent run
    return lengths


def rle_gamma(m):
    return float(sum(gamma_len(n + 1) for n in runs(m)))


def rle_adaptive(m, alpha=0.5):
    """Adaptive binary coding of gamma-binarised run lengths, one model per run colour."""
    counts = {}
    total = 0.0

    def code(key, bit):
        nonlocal total
        c = counts.setdefault(key, [0, 0])
        p = (c[bit] + alpha) / (c[0] + c[1] + 2 * alpha)
        total -= math.log2(p)
        c[bit] += 1

    for i, n in enumerate(runs(m)):
        colour = i % 2
        v = int(n) + 1
        e = v.bit_length() - 1
        for k in range(e):
            code((colour, "exp", k), 1)
        code((colour, "exp", e), 0)
        for k in range(e - 1, -1, -1):
            code((colour, "mant", e, k), (v >> k) & 1)
    return total


def positions(m):
    n = m.size
    k = int(m.sum())
    lc = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    return math.log2(n + 1) + lc / math.log(2)


def row_changes(m):
    bits = 0
    for row in m:
        padded = np.concatenate(([0], row))
        ch = np.flatnonzero(padded[1:] != padded[:-1])
        bits += gamma_len(len(ch) + 1)
        prev = -1
        for c in ch:
            bits += gamma_len(int(c) - prev)
            prev = int(c)
    return float(bits)


def _tiff_g4_size(m):
    b = io.BytesIO()
    Image.fromarray((m * 255).astype(np.uint8)).convert("1").save(b, "TIFF", compression="group4")
    return len(b.getvalue())


def g4(m):
    empty = _tiff_g4_size(np.zeros_like(m))
    return float(8 * max(_tiff_g4_size(m) - empty, 0))
