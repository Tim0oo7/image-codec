"""2D match feature: what followed the same large neighbourhood last time?

For every pixel, take the context of its n nearest causal neighbours (a larger window than the
CTW template). The most recent earlier pixel with exactly the same context gives a predicted
value. Bits: [an earlier occurrence exists], [its value]. This copies repeated tiles, sprites
and glyphs without any stored offsets, like LZ77 but per pixel.
"""

import numpy as np
from harness import neighbour_bits

_P = np.int64(0x9E3779B97F4A7C15 & 0x7FFFFFFFFFFFFFFF)


def context_hash(m, n):
    h = np.zeros(m.size, dtype=np.int64)
    for k, b in enumerate(neighbour_bits(m, n)):
        if k and k % 60 == 0:
            h = h * _P  # mix before the next 60 bits (wraps, fine for hashing)
        h = (h << np.int64(1)) ^ b.astype(np.int64)
    return h


def last_occurrence(m, n):
    """(exists, value) of the previous pixel with the same n-neighbour context."""
    x = m.ravel()
    h = context_hash(m, n)
    order = np.lexsort((np.arange(x.size), h))
    hs = order[1:]
    same = h[order[1:]] == h[order[:-1]]
    exists = np.zeros(x.size, dtype=np.uint8)
    value = np.zeros(x.size, dtype=np.uint8)
    exists[hs[same]] = 1
    value[hs[same]] = x[order[:-1][same]]
    return exists.reshape(m.shape), value.reshape(m.shape)


def match32(m):
    return list(last_occurrence(m, 32))


def match48(m):
    return list(last_occurrence(m, 48))


def match80(m):
    return list(last_occurrence(m, 80))
