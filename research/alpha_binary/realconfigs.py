"""Harness cost functions that return REAL file sizes (bits = 8 x bytes).

File = image dimensions field (docs/format-spec.md) + alpha header (2-bit alpha mode + the two
8-bit values) + arithmetic-coded payload, rounded up to whole bytes.
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import features_combo  # noqa: E402
import realcoder  # noqa: E402
from harness import neighbour_bits  # noqa: E402

from image_codec.dimensions import dimension_bits  # noqa: E402

ALPHA_HEADER_BITS = 2 + 8 + 8


def _file_bits(m, bits, alpha):
    payload = realcoder.encode(bits, m.ravel(), table_bits=24, alpha=alpha)
    h, w = m.shape
    total = dimension_bits(w, h) + ALPHA_HEADER_BITS + 8 * len(payload)
    return 8 * math.ceil(total / 8)


def real_v2(m):
    return _file_bits(m, neighbour_bits(m, 24), 1 / 16)


def real_v3(m):
    nb = neighbour_bits(m, 24)
    f = [np.asarray(p, dtype=np.uint8).ravel() for p in features_combo.line_match48(m)]
    return _file_bits(m, nb[:2] + f + nb[2:], 1 / 4)


def v4_bits(m):
    """v4 context: 2 neighbours, line + match bits, 12 neighbours, copy (LZ) bits, 10 neighbours."""
    import features_lz

    nb = neighbour_bits(m, 24)
    f = [np.asarray(p, dtype=np.uint8).ravel() for p in features_combo.line_match48(m)]
    lz = [np.asarray(p, dtype=np.uint8).ravel() for p in features_lz.lz48(m)]
    return nb[:2] + f + nb[2:14] + lz + nb[14:]


def real_v4(m):
    return _file_bits(m, v4_bits(m), 1 / 4)


def real_mix(m):
    """v3 and v3-with-copy-bits CTWs, mixed by one online logistic weight pair."""
    x = m.ravel()
    nb = neighbour_bits(m, 24)
    f = [np.asarray(p, dtype=np.uint8).ravel() for p in features_combo.line_match48(m)]
    pa = realcoder.probabilities(nb[:2] + f + nb[2:], x, 24, 1 / 4)
    pb = realcoder.probabilities(v4_bits(m), x, 24, 1 / 4)
    payload = realcoder.encode_mixed(x, pa, pb)
    h, w = m.shape
    total = dimension_bits(w, h) + ALPHA_HEADER_BITS + 8 * len(payload)
    return 8 * math.ceil(total / 8)
