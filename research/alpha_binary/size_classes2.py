"""Size-dependent choices, measured per size bucket (dev for choosing, test for reporting).

1. Starting tables at every CTW depth: node at depth d starts from the trained P(opaque) of its
   d-bit prefix (d <= T), or of its first T bits (d > T), with prior strength s.
2. Context layouts: v3 (neighbours + line + match bits), neighbours only (24), neighbours (12).
3. Termination: bit-exact end of the arithmetic code (about ideal + 2 bits) packed right after
   the header, versus a payload of whole bytes after the header.

Usage: python size_classes2.py DEV_ALL.json TEST.json
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
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import features_combo  # noqa: E402
from ctw_prior import log_beta_seq  # noqa: E402
from harness import load, neighbour_bits  # noqa: E402
from idealcost import LN2, _counts, log_pe, prefix_contexts  # noqa: E402

from image_codec.dimensions import dimension_bits  # noqa: E402

BUCKETS = [
    (0, 64),
    (65, 256),
    (257, 1024),
    (1025, 4096),
    (4097, 16384),
    (16385, 65536),
    (65537, 262144),
    (262145, 10**12),
]
T = 12
ALPHA = 1 / 4


def bucket(n):
    return next(i for i, (lo, hi) in enumerate(BUCKETS) if lo <= n <= hi)


def layouts(m):
    nb = neighbour_bits(m, 24)
    f = [np.asarray(p, dtype=np.uint8).ravel() for p in features_combo.line_match48(m)]
    return {"v3": nb[:2] + f + nb[2:], "nb24": nb, "nb12": nb[:12]}


def ctw_tables(x, bits, tables, s, alpha=ALPHA):
    depth = len(bits)
    ids, n0, n1 = _counts(prefix_contexts(bits, depth), x)

    def le(ids, n0, n1, d):
        if s == 0 or tables is None:
            return log_pe(n0, n1, alpha)
        p = tables[d][ids] if d <= T else tables[T][ids >> (d - T)]
        return log_beta_seq(
            n0.astype(np.float64), n1.astype(np.float64), s * p + 1e-3, s * (1 - p) + 1e-3
        )

    lw = le(ids, n0, n1, depth)
    for d in range(depth - 1, -1, -1):
        parent = ids >> 1
        pid, start = np.unique(parent, return_index=True)
        p0 = np.add.reduceat(n0, start)
        p1 = np.add.reduceat(n1, start)
        children = np.add.reduceat(lw, start)
        lw = np.logaddexp(le(pid, p0, p1, d), children) - LN2
        ids, n0, n1 = pid, p0, p1
    return float(-lw.sum() / LN2)


def _train_one(item):
    try:
        m, _ = load(item)
    except Exception:
        return None
    bits = layouts(m)["v3"]
    x = m.ravel().astype(np.float64)
    out = []
    for d in range(T + 1):
        ctx = prefix_contexts(bits, d) if d else np.zeros(m.size, dtype=np.int64)
        out.append((np.bincount(ctx, x, 1 << d), np.bincount(ctx, minlength=1 << d)))
    return bucket(m.size), out


def train(items, by_bucket):
    acc = {}
    with mp.Pool(8) as pool:
        for res in pool.imap_unordered(_train_one, items, 8):
            if res is None:
                continue
            key = res[0] if by_bucket else 0
            a = acc.setdefault(key, [[np.zeros(1 << d), np.zeros(1 << d)] for d in range(T + 1)])
            for d, (o, n) in enumerate(res[1]):
                a[d][0] += o
                a[d][1] += n
    return {k: [(o + 0.5) / (n + 1) for o, n in a] for k, a in acc.items()}


def _eval(args):
    item, tab_all = args
    try:
        m, _ = load(item)
    except Exception:
        return None
    lay = layouts(m)
    x = m.ravel()
    t = tab_all.get(0)
    r = {"n": m.size, "b": bucket(m.size), "w": m.shape[1], "h": m.shape[0], "bits": {}}
    r["bits"]["v3"] = ctw_tables(x, lay["v3"], None, 0)
    r["bits"]["nb24"] = ctw_tables(x, lay["nb24"], None, 0)
    r["bits"]["nb12"] = ctw_tables(x, lay["nb12"], None, 0)
    for s in (2, 8, 32):
        r["bits"][f"v3+table s={s}"] = ctw_tables(x, lay["v3"], t, s)
    r["header"] = dimension_bits(m.shape[1], m.shape[0]) + 3
    return r


def run(items, tables):
    with mp.Pool(8) as pool:
        return [r for r in pool.imap_unordered(_eval, [(it, tables) for it in items], 8) if r]


def report(rows, title):
    names = list(rows[0]["bits"])
    print(f"\n{title}: payload bytes (ideal) per size bucket, ratio to v3")
    print("| pixels | masks | " + " | ".join(names) + " |")
    print("|---|---|" + "---|" * len(names))
    for i, (lo, hi) in enumerate(BUCKETS):
        g = [r for r in rows if r["b"] == i]
        if not g:
            continue
        base = sum(r["bits"]["v3"] for r in g)
        cells = [
            f"{sum(r['bits'][n] for r in g) / 8:,.0f} ({sum(r['bits'][n] for r in g) / base:.3f})"
            for n in names
        ]
        print(f"| {lo}-{hi} | {len(g)} | " + " | ".join(cells) + " |")
    print(f"\n{title}: whole file bytes, v3 model, by termination style")
    print("| pixels | masks | byte payload after header | bit-exact end | saving |")
    print("|---|---|---|---|---|")
    for i, (lo, hi) in enumerate(BUCKETS):
        g = [r for r in rows if r["b"] == i]
        if not g:
            continue
        # byte payload: arithmetic code ends on a byte boundary (ideal + up to 8 bits), then the
        # file is header + payload rounded up; bit-exact: ideal + 2 bits right after the header
        bytewise = sum(
            math.ceil((r["header"] + 8 * math.ceil((r["bits"]["v3"] + 2) / 8)) / 8) for r in g
        )
        bitexact = sum(math.ceil((r["header"] + math.ceil(r["bits"]["v3"]) + 2) / 8) for r in g)
        saving = 1 - bitexact / bytewise
        print(f"| {lo}-{hi} | {len(g)} | {bytewise:,} | {bitexact:,} | {saving:.1%} |")


def main():
    dev = json.load(open(sys.argv[1]))
    test = json.load(open(sys.argv[2]))
    half_a = [it for it in dev if int(it["hash"][:2], 16) % 2 == 0]
    half_b = [it for it in dev if int(it["hash"][:2], 16) % 2 == 1]
    report(run(half_b, train(half_a, False)), "DEV half B (table trained on dev half A)")
    rows = run(test, train(dev, False))
    report(rows, "TEST (table trained on all dev)")
    json.dump(rows, open(sys.argv[3], "w"))


if __name__ == "__main__":
    main()
