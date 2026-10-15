"""Exact ideal code lengths (in bits) of sequential adaptive models for a binary image.

The Dirichlet/KT estimator gives a sequence probability that depends only on the final
per-context counts, and so does context-tree weighting (CTW) built on it. The cost of coding a
whole image adaptively, pixel by pixel in raster order, can therefore be computed exactly from
counts. No arithmetic coder is needed: a real coder adds only a few bytes on top.
"""

import math

import numpy as np

LN2 = math.log(2)


def causal_offsets(n):
    """First n causal neighbours (dy, dx), nearest first (Euclidean), ties broken N/W first."""
    cand = []
    for dy in range(-8, 1):
        for dx in range(-8, 9):
            if dy == 0 and dx >= 0:
                continue
            cand.append((dy * dy + dx * dx, abs(dx), -dy, dy, dx))
    cand.sort()
    return [(c[3], c[4]) for c in cand[:n]]


def shifted(m, dy, dx):
    """Value of the neighbour at (y + dy, x + dx) for every pixel, 0 outside the image."""
    h, w = m.shape
    out = np.zeros_like(m)
    ys, yd = (slice(0, h + dy), slice(-dy, h)) if dy < 0 else (slice(0, h), slice(0, h))
    if dx < 0:
        xs, xd = slice(0, w + dx), slice(-dx, w)
    elif dx > 0:
        xs, xd = slice(dx, w), slice(0, w - dx)
    else:
        xs, xd = slice(0, w), slice(0, w)
    out[yd, xd] = m[ys, xs]
    return out


def context_bits(m, depth, extra=()):
    """Per-pixel context bit arrays: the `extra` bits first, then the nearest neighbours."""
    bits = [np.asarray(e, dtype=np.uint8).ravel() for e in extra]
    for dy, dx in causal_offsets(depth - len(bits)):
        bits.append(shifted(m, dy, dx).ravel())
    return bits


def _lgamma(x):
    vals, inv = np.unique(x, return_inverse=True)
    return np.array([math.lgamma(v) for v in vals])[inv]


def log_pe(n0, n1, alpha=0.5):
    """ln P of a binary sequence with counts (n0, n1) under the Dirichlet(alpha) estimator.

    alpha = 1/2 is the Krichevsky-Trofimov (KT) estimator.
    """
    n0 = np.asarray(n0, dtype=np.float64)
    n1 = np.asarray(n1, dtype=np.float64)
    return (
        _lgamma(n0 + alpha)
        + _lgamma(n1 + alpha)
        - _lgamma(n0 + n1 + 2 * alpha)
        + math.lgamma(2 * alpha)
        - 2 * math.lgamma(alpha)
    )


def _counts(ctx, x):
    key = ctx * 2 + x
    u, c = np.unique(key, return_counts=True)
    ids = u >> 1
    n1 = np.where(u & 1, c, 0)
    n0 = np.where(u & 1, 0, c)
    uid, start = np.unique(ids, return_index=True)
    return uid, np.add.reduceat(n0, start), np.add.reduceat(n1, start)


def prefix_contexts(bits, depth):
    ctx = np.zeros(bits[0].shape if bits else 0, dtype=np.int64)
    for b in bits[:depth]:
        ctx = ctx * 2 + b
    return ctx


def kt_bits(x, bits, depth, alpha=0.5):
    """Adaptive cost of one fixed template (the first `depth` context bits)."""
    _, n0, n1 = _counts(prefix_contexts(bits, depth), x)
    return -log_pe(n0, n1, alpha).sum() / LN2


def oracle_bits(x, bits, depth):
    """Static cost if the per-context probabilities were known in advance (hindsight)."""
    _, n0, n1 = _counts(prefix_contexts(bits, depth), x)
    n = n0 + n1
    with np.errstate(divide="ignore", invalid="ignore"):
        h0 = np.where(n0 > 0, n0 * np.log2(n0 / n), 0)
        h1 = np.where(n1 > 0, n1 * np.log2(n1 / n), 0)
    return float(-(h0 + h1).sum())


def ctw_bits(x, bits, depth, alpha=0.5):
    """Context-tree weighting over the given bit order, maximum depth `depth`."""
    ids, n0, n1 = _counts(prefix_contexts(bits, depth), x)
    lw = log_pe(n0, n1, alpha)  # leaves: Pw = Pe
    for _ in range(depth):
        parent = ids >> 1
        pid, start = np.unique(parent, return_index=True)
        p0 = np.add.reduceat(n0, start)
        p1 = np.add.reduceat(n1, start)
        children = np.add.reduceat(lw, start)  # ln(Pw(c0) * Pw(c1)); a missing child has Pw = 1
        lw = np.logaddexp(log_pe(p0, p1, alpha), children) - LN2
        ids, n0, n1 = pid, p0, p1
    return float(-lw.sum() / LN2)
