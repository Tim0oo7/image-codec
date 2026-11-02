"""Size classes: a different starting state per image size, chosen from the dimensions header.

The decoder knows W x H before the first pixel, so picking a model by size costs no bits.
For each class a table of P(opaque) per context prefix (the first T context bits) is trained on
the dev split and used as the starting state (a Beta prior of strength s) of every CTW node at
depth >= T. Strength 0 = the current model (start from nothing). Costs are exact ideal code
lengths (count-based, see ctw_prior.py).

Usage: python size_classes.py DEV.json TEST.json
"""

import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import features_combo  # noqa: E402
from ctw_prior import ctw_prior_bits  # noqa: E402
from harness import load, neighbour_bits  # noqa: E402
from idealcost import prefix_contexts  # noqa: E402

CLASSES = [("tiny", 0, 256), ("small", 257, 65536), ("large", 65537, 10**12)]
T = 12
STRENGTHS = [0, 1, 4, 16, 64]
ALPHA = 1 / 4


def size_class(w, h):
    n = w * h
    return next(name for name, lo, hi in CLASSES if lo <= n <= hi)


def layout(m):
    nb = neighbour_bits(m, 24)
    f = [np.asarray(p, dtype=np.uint8).ravel() for p in features_combo.line_match48(m)]
    return nb[:2] + f + nb[2:]


def _counts(item):
    try:
        m, _ = load(item)
    except Exception:
        return None
    ctx = prefix_contexts(layout(m), T)
    x = m.ravel().astype(np.float64)
    size = 1 << T
    return size_class(*m.shape[::-1]), np.bincount(ctx, x, size), np.bincount(ctx, minlength=size)


def train(items):
    tables = {}
    with mp.Pool(8) as pool:
        for res in pool.imap_unordered(_counts, items, chunksize=8):
            if res is None:
                continue
            cls, ones, tot = res
            t = tables.setdefault(cls, [np.zeros(1 << T), np.zeros(1 << T)])
            t[0] += ones
            t[1] += tot
    return {c: (o + 0.5) / (n + 1) for c, (o, n) in tables.items()}


def _costs(args):
    item, tables = args
    try:
        m, _ = load(item)
    except Exception:
        return None
    cls = size_class(*m.shape[::-1])
    bits = layout(m)
    x = m.ravel()
    table = tables.get(cls)
    row = {"path": item["path"], "category": item["category"], "class": cls, "bits": {}}
    for s in STRENGTHS:
        if s and table is None:
            continue
        row["bits"][s] = ctw_prior_bits(x, bits, table if s else np.full(1 << T, 0.5), T, s, ALPHA)
    return row


def evaluate(items, tables):
    with mp.Pool(8) as pool:
        return [r for r in pool.imap_unordered(_costs, [(it, tables) for it in items], 8) if r]


def summary(rows, title):
    print(f"\n{title}: total bytes per size class and starting-table strength s (s=0: current)")
    print("| class | masks | " + " | ".join(f"s={s}" for s in STRENGTHS) + " |")
    print("|---|---|" + "---|" * len(STRENGTHS))
    for name, _, _ in CLASSES:
        g = [r for r in rows if r["class"] == name]
        if not g:
            continue
        base = sum(r["bits"][0] for r in g)
        cells = [
            f"{sum(r['bits'][s] for r in g) / 8:,.0f} ({sum(r['bits'][s] for r in g) / base:.3f})"
            for s in STRENGTHS
        ]
        print(f"| {name} | {len(g)} | " + " | ".join(cells) + " |")


def main():
    dev = json.load(open(sys.argv[1]))
    test = json.load(open(sys.argv[2]))
    # 1) choose s on dev: train on half of dev, evaluate on the other half
    half_a = [it for it in dev if int(it["hash"][:2], 16) % 2 == 0]
    half_b = [it for it in dev if int(it["hash"][:2], 16) % 2 == 1]
    summary(evaluate(half_b, train(half_a)), "dev (train on dev half A, evaluate on dev half B)")
    # 2) final: train on all of dev, evaluate on test
    rows = evaluate(test, train(dev))
    summary(rows, "test (train on all of dev)")
    json.dump(rows, open(sys.argv[3] if len(sys.argv) > 3 else "size_classes_test.json", "w"))


if __name__ == "__main__":
    main()
