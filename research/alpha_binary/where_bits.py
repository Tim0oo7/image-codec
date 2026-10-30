"""Where do the bits go? Per-pixel adaptive cost split by local geometry.

Usage: python where_bits.py MANIFEST.json

Per-pixel costs come from an adaptive KT model over the 16 nearest causal neighbours (a proxy for
the CTW baseline; its per-pixel cost is exact and vectorisable). Pixels are classified by their
full 3x3 neighbourhood (analysis only, not causal) into geometry types.
"""

import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from harness import load, neighbour_bits  # noqa: E402

TYPES = ["flat", "straight_h", "straight_v", "diagonal", "corner", "thin", "isolated", "other"]


def per_pixel_cost(m, n=16, alpha=0.5):
    """Exact sequential KT cost (bits) of each pixel under the n-neighbour template."""
    x = m.ravel().astype(np.int64)
    ctx = np.zeros(x.size, dtype=np.int64)
    for b in neighbour_bits(m, n):
        ctx = ctx * 2 + b
    order = np.lexsort((np.arange(x.size), ctx))  # by context, then raster position
    cs, xs = ctx[order], x[order]
    start = np.r_[True, cs[1:] != cs[:-1]]
    group = np.cumsum(start) - 1
    first = np.flatnonzero(start)
    ones = np.cumsum(xs) - xs  # ones before, globally
    seen = np.arange(xs.size) - first[group]
    ones_before = ones - (np.cumsum(xs) - xs)[first[group]]
    p1 = (ones_before + alpha) / (seen + 2 * alpha)
    p = np.where(xs == 1, p1, 1 - p1)
    out = np.empty(x.size)
    out[order] = -np.log2(p)
    return out.reshape(m.shape)


def classify(m):
    """Geometry type of each pixel from its 3x3 neighbourhood (non-causal, analysis only)."""
    p = np.pad(m, 1)
    h, w = m.shape
    nb = {
        (dy, dx): p[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
    }
    c = nb[(0, 0)]
    same = sum((nb[k] == c).astype(np.int8) for k in nb if k != (0, 0))
    n4 = [nb[(-1, 0)], nb[(1, 0)], nb[(0, -1)], nb[(0, 1)]]
    diff4 = sum((v != c).astype(np.int8) for v in n4)
    t = np.full(m.shape, 7, dtype=np.int8)  # other
    t[same == 8] = 0
    vert_edge = (nb[(-1, 0)] == c) & (nb[(1, 0)] == c) & (nb[(0, -1)] != nb[(0, 1)])
    horiz_edge = (nb[(0, -1)] == c) & (nb[(0, 1)] == c) & (nb[(-1, 0)] != nb[(1, 0)])
    rows_const = (nb[(-1, -1)] == nb[(-1, 0)]) & (nb[(-1, 0)] == nb[(-1, 1)])
    rows_const &= (nb[(1, -1)] == nb[(1, 0)]) & (nb[(1, 0)] == nb[(1, 1)])
    cols_const = (nb[(-1, -1)] == nb[(0, -1)]) & (nb[(0, -1)] == nb[(1, -1)])
    cols_const &= (nb[(-1, 1)] == nb[(0, 1)]) & (nb[(0, 1)] == nb[(1, 1)])
    t[(same < 8) & horiz_edge & rows_const] = 1
    t[(same < 8) & vert_edge & cols_const] = 2
    t[(t == 7) & (diff4 == 1)] = 3  # a step: one 4-neighbour differs
    t[(t == 7) & (diff4 == 2) & ~((n4[0] != c) & (n4[1] != c)) & ~((n4[2] != c) & (n4[3] != c))] = 4
    t[(t == 7) & (diff4 >= 2)] = 5  # 1-px lines and spikes
    t[same == 0] = 6
    return t


def work(item):
    try:
        m, _ = load(item)
    except Exception:
        return None
    cost = per_pixel_cost(m)
    t = classify(m)
    res = {"category": item["category"], "px": m.size}
    for k, name in enumerate(TYPES):
        sel = t == k
        res[name] = (float(cost[sel].sum()), int(sel.sum()))
    return res


def main():
    items = json.load(open(sys.argv[1]))
    with mp.Pool(8) as pool:
        rows = [r for r in pool.imap_unordered(work, items, chunksize=8) if r]
    cats = sorted({r["category"] for r in rows})
    print("share of bits (share of pixels) per geometry type\n")
    print("| category | " + " | ".join(TYPES) + " | bits/px |")
    print("|---|" + "---|" * (len(TYPES) + 1))
    for cat in cats + ["ALL"]:
        g = [r for r in rows if cat == "ALL" or r["category"] == cat]
        bits = {t: sum(r[t][0] for r in g) for t in TYPES}
        px = {t: sum(r[t][1] for r in g) for t in TYPES}
        tb, tp = sum(bits.values()), sum(px.values())
        cells = [f"{bits[t] / tb:.0%} ({px[t] / tp:.1%})" for t in TYPES]
        print(f"| {cat} | " + " | ".join(cells) + f" | {tb / tp:.4f} |")
    print("\nbits per pixel of each type (ALL):")
    for t in TYPES:
        b = sum(r[t][0] for r in rows)
        p = sum(r[t][1] for r in rows)
        print(f"  {t:11s} {b / max(p, 1):.3f} bits/px over {p:,} px")


if __name__ == "__main__":
    main()
