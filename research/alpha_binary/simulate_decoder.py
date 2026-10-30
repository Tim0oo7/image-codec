"""End-to-end check: decode pixel by pixel and compare with the count-based cost.

At pixel i the simulated decoder only has pixels 0..i-1: everything from i on is zeroed. The
feature planes are recomputed from that partial image and only their value at i is used. A plain
sequential CTW then assigns the probability of the true pixel. If the total equals
harness.cost(...) on the full image, the features are causal and the measurement is exact.

Usage: python simulate_decoder.py MANIFEST.json [N_IMAGES]
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import features_combo  # noqa: E402
from harness import cost, load, neighbour_bits  # noqa: E402

FEATURES = features_combo.line_match48
NEIGHBOURS, AT, ALPHA = 24, 2, 1 / 4


def sequential_bits(m):
    h, w = m.shape
    n = m.size
    x = m.ravel()
    decoded = np.zeros_like(m)
    nodes = {}  # (depth, prefix) -> [n0, n1, ln Pe, ln Pw]
    total = 0.0
    for i in range(n):
        y, xx = divmod(i, w)
        partial = decoded.copy()
        nb = [b[i] for b in neighbour_bits(partial, NEIGHBOURS)]
        feats = [int(np.asarray(p)[y, xx]) for p in FEATURES(partial)]
        bits = nb[:AT] + feats + nb[AT:]
        depth = len(bits)
        path, p = [(0, 0)], 0
        for d in range(depth):
            p = p * 2 + int(bits[d])
            path.append((d + 1, p))
        before = nodes.get((0, 0), [0, 0, 0.0, 0.0])[3]
        sym = int(x[i])
        for d, pre in reversed(path):
            nd = nodes.setdefault((d, pre), [0, 0, 0.0, 0.0])
            nd[2] += math.log((nd[sym] + ALPHA) / (nd[0] + nd[1] + 2 * ALPHA))
            nd[sym] += 1
            if d == depth:
                nd[3] = nd[2]
            else:
                c0 = nodes.get((d + 1, pre * 2), [0, 0, 0.0, 0.0])[3]
                c1 = nodes.get((d + 1, pre * 2 + 1), [0, 0, 0.0, 0.0])[3]
                nd[3] = float(np.logaddexp(nd[2], c0 + c1)) - math.log(2)
        total -= (nodes[(0, 0)][3] - before) / math.log(2)
        decoded.ravel()[i] = x[i]
    return total


def main():
    items = json.load(open(sys.argv[1]))
    want = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    seen = set()
    done = 0
    for it in items:
        if it["w"] * it["h"] > 48 * 48 or it["category"] in seen:
            continue
        seen.add(it["category"])
        m, _ = load(it)
        fast = cost(m, FEATURES(m), NEIGHBOURS, alpha=ALPHA, at=AT)
        slow = sequential_bits(m)
        ok = abs(fast - slow) < 1e-6 * max(1.0, fast)
        print(
            f"{it['category']:12s} {it['w']}x{it['h']}  count-based {fast:10.4f}  "
            f"decoder {slow:10.4f}  {'OK' if ok else 'MISMATCH'}",
            flush=True,
        )
        done += 1
        if done >= want:
            break


if __name__ == "__main__":
    main()
