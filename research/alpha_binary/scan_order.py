"""Does the scan order matter? Raster vs Hilbert vs coarse-to-fine, with the same window.

Every order uses the same context window: the 16 nearest pixels in ALL directions. For each
window pixel the context has two bits, [already decoded] and [value if decoded], nearest first,
so every order sees exactly what it has decoded and nothing else. The ideal adaptive cost is
CTW over those 32 bits, computed from counts (valid for any order).
"""

import numpy as np
from idealcost import ctw_bits

WINDOW = 16


def window_offsets(n):
    cand = [
        (dy * dy + dx * dx, abs(dx), dy, dx)
        for dy in range(-4, 5)
        for dx in range(-4, 5)
        if (dy, dx) != (0, 0)
    ]
    cand.sort()
    return [(c[2], c[3]) for c in cand[:n]]


def order_cost(m, rank, alpha=1 / 4):
    """rank[y, x] = time at which (y, x) is coded."""
    h, w = m.shape
    pm = np.pad(m, 4)
    pr = np.pad(rank, 4, constant_values=np.iinfo(np.int64).max)
    bits = []
    for dy, dx in window_offsets(WINDOW):
        r = pr[4 + dy : 4 + dy + h, 4 + dx : 4 + dx + w]
        v = pm[4 + dy : 4 + dy + h, 4 + dx : 4 + dx + w]
        known = (r < rank).astype(np.uint8)
        bits += [known.ravel(), (known & v).ravel()]
    # sequence order = coding order
    seq = np.argsort(rank.ravel(), kind="stable")
    return ctw_bits(m.ravel()[seq], [b[seq] for b in bits], len(bits), alpha)


def raster_rank(h, w):
    return np.arange(h * w, dtype=np.int64).reshape(h, w)


def _hilbert_d2xy(order, d):
    x = y = 0
    s, t = 1, d
    while s < (1 << order):
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        if ry == 0:
            if rx == 1:
                x, y = s - 1 - x, s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y


def hilbert_rank(h, w):
    order = max(1, int(np.ceil(np.log2(max(h, w)))))
    rank = np.zeros((h, w), dtype=np.int64)
    t = 0
    for d in range(1 << (2 * order)):
        x, y = _hilbert_d2xy(order, d)
        if x < w and y < h:
            rank[y, x] = t
            t += 1
    return rank


def interlaced_rank(h, w):
    """Coarse to fine: pixels on a 2^k grid first (k decreasing), raster order within a level."""
    level = np.zeros((h, w), dtype=np.int64)
    yy, xx = np.mgrid[0:h, 0:w]
    top = int(np.ceil(np.log2(max(h, w))))
    for k in range(top, -1, -1):
        on = ((yy % (1 << k)) == 0) & ((xx % (1 << k)) == 0) & (level == 0)
        level[on] = top + 1 - k
    key = level * (h * w) + yy * w + xx
    return np.argsort(np.argsort(key.ravel(), kind="stable"), kind="stable").reshape(h, w)


def raster(m):
    return order_cost(m, raster_rank(*m.shape))


def hilbert(m):
    return order_cost(m, hilbert_rank(*m.shape))


def interlaced(m):
    return order_cost(m, interlaced_rank(*m.shape))
