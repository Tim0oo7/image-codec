"""How much would a fixed starting table (trained once, stored in the spec) save?

Usage: python prior_table.py MANIFEST.json N_NEIGHBOURS

Images are split in two by hash. A per-context probability table is trained on one half and used
as a Beta prior with strength s for adaptive coding of the other half (and vice versa). The cost of
Beta-binomial coding depends only on counts, so it is exact:
    bits(context) = -log2 [ B(n1 + a, n0 + b) / B(a, b) ],  a = s * p, b = s * (1 - p)
"""

import json
import math
import multiprocessing as mp
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from harness import load, neighbour_bits  # noqa: E402

STRENGTHS = [0, 0.5, 2, 8, 32]


def contexts(m, n):
    ctx = np.zeros(m.size, dtype=np.int64)
    for b in neighbour_bits(m, n):
        ctx = ctx * 2 + b
    return ctx


def counts(item_n):
    item, n = item_n
    try:
        m, _ = load(item)
    except Exception:
        return None
    ctx = contexts(m, n)
    x = m.ravel().astype(np.int64)
    size = 1 << n
    c1 = np.bincount(ctx, weights=x, minlength=size)
    c0 = np.bincount(ctx, minlength=size) - c1
    return item, c0, c1


def lbeta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def image_bits(c0, c1, p, s, alpha=0.5):
    used = np.flatnonzero(c0 + c1)
    total = 0.0
    for c in used:
        if s == 0:
            a = b = alpha
        else:
            a = s * p[c] + alpha * 0.01
            b = s * (1 - p[c]) + alpha * 0.01
        total -= lbeta(c1[c] + a, c0[c] + b) - lbeta(a, b)
    return total / math.log(2)


def main():
    items = json.load(open(sys.argv[1]))
    n = int(sys.argv[2])
    with mp.Pool(8) as pool:
        res = [r for r in pool.imap_unordered(counts, [(it, n) for it in items], chunksize=8) if r]
    halves = [[r for r in res if int(r[0]["hash"][:2], 16) % 2 == k] for k in (0, 1)]
    out = {}
    for k in (0, 1):
        train, test = halves[1 - k], halves[k]
        t0 = sum(r[1] for r in train)
        t1 = sum(r[2] for r in train)
        p = (t1 + 0.5) / (t0 + t1 + 1)  # training-half probability per context
        for item, c0, c1 in test:
            cat = item["category"]
            row = out.setdefault(cat, {s: 0.0 for s in STRENGTHS})
            for s in STRENGTHS:
                row[s] += image_bits(c0, c1, p, s)
    print(
        f"total bytes per category, {n}-neighbour template, prior strength s (0 = KT, no table)\n"
    )
    print("| category | " + " | ".join(f"s={s}" for s in STRENGTHS) + " |")
    print("|---|" + "---|" * len(STRENGTHS))
    allrow = {s: 0.0 for s in STRENGTHS}
    for cat in sorted(out):
        base = out[cat][0]
        cells = [f"{out[cat][s] / 8:,.0f} ({out[cat][s] / base:.3f})" for s in STRENGTHS]
        print(f"| {cat} | " + " | ".join(cells) + " |")
        for s in STRENGTHS:
            allrow[s] += out[cat][s]
    cells = [f"{allrow[s] / 8:,.0f} ({allrow[s] / allrow[0]:.3f})" for s in STRENGTHS]
    print("| ALL | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
