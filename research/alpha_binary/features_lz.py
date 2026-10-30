"""LZ-style copy predictor: keep a copy offset while it keeps working.

State (all from decoded pixels): a copy offset d (in raster positions) and a streak, the number of
consecutive pixels it predicted correctly. For pixel i:

    prediction = x[i - d]                      (if an offset is set)
    after decoding x[i]: if correct, streak += 1
                         else streak = 0 and d = i' - j, where j is the most recent earlier
                         pixel with the same n-neighbour context as pixel i' = i + 1

So a long repeat (sprite frames, tiles, rows) is followed with growing confidence, and a new
offset is picked from the 2D context match as soon as the old one fails. Bits:
[offset set], [predicted value], [streak >= 8], [streak >= 64].
"""

import numpy as np
from features_match import context_hash
from numba import njit


def previous_same_context(m, n):
    """prev[i] = most recent j < i with the same n-neighbour context, or -1."""
    h = context_hash(m, n)
    order = np.lexsort((np.arange(h.size), h))
    prev = np.full(h.size, -1, dtype=np.int64)
    same = h[order[1:]] == h[order[:-1]]
    prev[order[1:][same]] = order[:-1][same]
    return prev


@njit(cache=True)
def _run(x, prev):
    n = x.size
    has = np.zeros(n, np.uint8)
    val = np.zeros(n, np.uint8)
    s8 = np.zeros(n, np.uint8)
    s64 = np.zeros(n, np.uint8)
    d = 0
    streak = 0
    for i in range(n):
        if d == 0 and prev[i] >= 0:  # no offset yet: take one from the context match
            d = i - prev[i]
            streak = 0
        if d > 0:
            has[i] = 1
            val[i] = x[i - d]
            s8[i] = streak >= 8
            s64[i] = streak >= 64
            if x[i] == x[i - d]:
                streak += 1
            else:
                streak = 0
                d = 0
                if i + 1 < n and prev[i + 1] >= 0:
                    d = i + 1 - prev[i + 1]
    return has, val, s8, s64


def lz_planes(m, n=48):
    prev = previous_same_context(m, n)
    has, val, s8, s64 = _run(m.ravel().astype(np.uint8), prev)
    return [a.reshape(m.shape) for a in (has, val, s8, s64)]


def lz48(m):
    return lz_planes(m, 48)


def lz32(m):
    return lz_planes(m, 32)
