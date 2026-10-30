"""CTW whose nodes start from a trained table instead of an uninformed estimator.

Nodes at depth d >= T (they know the first T context bits) use a Beta prior centred on
table[first T bits] with strength s; shallower nodes use the plain Dirichlet(alpha) estimator.
The sequence probability still depends only on counts, so the cost is exact.

Usage (train): python ctw_prior.py train MANIFEST.json TABLE.npy T FEATURE_SPEC
  Trains on the images whose hash starts with an even byte. FEATURE_SPEC as in harness
  ('none@24', 'features_line:line@24/2', ...).
Cost functions for harness.py are made with make_cost(...) below (see prior_configs.py).
"""

import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from harness import load, neighbour_bits, resolve  # noqa: E402
from idealcost import LN2, _counts, _lgamma, log_pe, prefix_contexts  # noqa: E402


def order_bits(m, spec):
    _, fn, (n, at) = resolve(spec)
    nb = neighbour_bits(m, n)
    feats = [np.asarray(p, dtype=np.uint8).ravel() for p in (fn(m) if fn else [])]
    return nb[:at] + feats + nb[at:]


def log_beta_seq(n0, n1, a, b):
    """ln P of counts (n0, n1) under a Beta(a, b) prior (Beta-binomial sequence probability)."""
    return (
        _lgamma(n1 + a)
        + _lgamma(n0 + b)
        - _lgamma(n0 + n1 + a + b)
        - (_lgamma(a) + _lgamma(b) - _lgamma(a + b))
    )


def ctw_prior_bits(x, bits, table, T, s, alpha=1 / 16):
    depth = len(bits)
    ids, n0, n1 = _counts(prefix_contexts(bits, depth), x)

    def le(ids, n0, n1, d):
        if d < T or s == 0:
            return log_pe(n0, n1, alpha)
        p = table[ids >> (d - T)]
        a = s * p + 1e-3
        b = s * (1 - p) + 1e-3
        return log_beta_seq(n0.astype(np.float64), n1.astype(np.float64), a, b)

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


def _train_one(args):
    item, spec, T = args
    try:
        m, _ = load(item)
    except Exception:
        return None
    bits = order_bits(m, spec)
    ctx = prefix_contexts(bits, T)
    x = m.ravel().astype(np.float64)
    size = 1 << T
    return np.bincount(ctx, weights=x, minlength=size), np.bincount(ctx, minlength=size)


def train(manifest, table_path, T, spec):
    items = [it for it in json.load(open(manifest)) if int(it["hash"][:2], 16) % 2 == 0]
    with mp.Pool(8) as pool:
        res = [r for r in pool.imap_unordered(_train_one, [(it, spec, T) for it in items], 8) if r]
    ones = sum(r[0] for r in res)
    tot = sum(r[1] for r in res)
    np.save(table_path, (ones + 0.5) / (tot + 1))
    print("trained on", len(res), "images,", T, "bits")


_TABLES = {}


def cost(m, spec, table_path, T, s):
    if table_path not in _TABLES:
        _TABLES[table_path] = np.load(table_path)
    return ctw_prior_bits(m.ravel(), order_bits(m, spec), _TABLES[table_path], T, s)


if __name__ == "__main__" and sys.argv[1] == "train":
    train(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5])
