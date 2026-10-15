"""Check the count-based CTW against a straightforward pixel-by-pixel CTW."""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from idealcost import context_bits, ctw_bits  # noqa: E402


def sequential_ctw(x, bits, depth, alpha=0.5):
    nodes = {}  # (depth, prefix) -> [n0, n1, ln Pe, ln Pw]
    total = 0.0
    for i, sym in enumerate(int(v) for v in x):
        path = [(0, 0)]
        p = 0
        for d in range(depth):
            p = p * 2 + int(bits[d][i])
            path.append((d + 1, p))
        before = nodes.get((0, 0), [0, 0, 0.0, 0.0])[3]
        for d, pre in reversed(path):
            nd = nodes.setdefault((d, pre), [0, 0, 0.0, 0.0])
            nd[2] += math.log((nd[sym] + alpha) / (nd[0] + nd[1] + 2 * alpha))
            nd[sym] += 1
            if d == depth:
                nd[3] = nd[2]
            else:
                c0 = nodes.get((d + 1, pre * 2), [0, 0, 0.0, 0.0])[3]
                c1 = nodes.get((d + 1, pre * 2 + 1), [0, 0, 0.0, 0.0])[3]
                nd[3] = np.logaddexp(nd[2], c0 + c1) - math.log(2)
        total -= (nodes[(0, 0)][3] - before) / math.log(2)
    return total


def test_ctw_matches_sequential():
    rng = np.random.default_rng(1)
    m = (rng.random((24, 30)) < 0.3).astype(np.uint8)
    m[5:15, 4:20] = 1
    for depth in (0, 1, 3, 6):
        for alpha in (0.5, 1 / 16):
            bits = context_bits(m, max(depth, 1))
            fast = ctw_bits(m.ravel(), bits, depth, alpha)
            slow = sequential_ctw(m.ravel(), bits, depth, alpha)
            assert abs(fast - slow) < 1e-6
