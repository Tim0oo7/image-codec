"""Shared harness for measuring causal context features on binary masks.

A feature is a function `f(m) -> list[np.ndarray]` that takes a binary mask (H x W, uint8 0/1,
raster order, pixels outside the image read as 0) and returns bit planes (H x W, uint8 0/1).
The bit at (y, x) may depend only on pixels strictly before (y, x) in raster order. The model
under test is CTW over [feature bits..., nearest causal neighbours...] in that order, and its
exact ideal cost is computed from counts (see idealcost.py).

Usage:
    python harness.py build MANIFEST.json SCAN.json [SCAN.json ...]   # choose the eval set
    python harness.py run MANIFEST.json OUT.json CONFIG [CONFIG ...]
        CONFIG = name=module:function[@neighbours]
    python harness.py report OUT.json [BASELINE_NAME]

The special config `name=none@24` means no features, 24 neighbours (the current baseline).
"""

import hashlib
import importlib
import json
import multiprocessing as mp
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # one BLAS thread per worker process

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from idealcost import causal_offsets, ctw_bits, shifted  # noqa: E402

warnings.simplefilter("ignore")
Image.MAX_IMAGE_PIXELS = 50_000_000

PER_CATEGORY = 400  # images per category in the eval set
MIN_CATEGORY = 20  # smaller categories are pooled into "misc"
MAX_PIXELS = 2_100_000
DEFAULT_WORKERS = 8  # CTW on large masks needs a few hundred MB per worker
ALPHA = 1 / 16


# ---------- masks ----------
def load(item):
    """Return (mask, rgba) for a manifest/scan item."""
    with Image.open(item["path"]) as im:
        rgba = np.asarray(im.convert("RGBA"))
    if "threshold" in item:  # derived mask from anti-aliased alpha
        m = (rgba[..., 3] >= item["threshold"]).astype(np.uint8)
    else:
        m = (rgba[..., 3] == item["values"][1]).astype(np.uint8)
    return m, rgba


def neighbour_bits(m, n):
    return [shifted(m, dy, dx).ravel() for dy, dx in causal_offsets(n)]


def boundary(m):
    return int((m[1:] != m[:-1]).sum() + (m[:, 1:] != m[:, :-1]).sum())


# ---------- cost of one configuration ----------
def cost(m, planes, neighbours=24, alpha=ALPHA, at=0):
    """Ideal adaptive cost in bits of CTW over the context bit order:
    the `at` nearest neighbours, then the feature planes, then the remaining neighbours."""
    nb = neighbour_bits(m, neighbours)
    feats = [np.asarray(p, dtype=np.uint8).ravel() for p in planes]
    bits = nb[:at] + feats + nb[at:]
    if not bits:
        bits = [np.zeros(m.size, dtype=np.uint8)]
    return ctw_bits(m.ravel(), bits, len(bits), alpha)


def check_causal(fn, m, trials=24, seed=0):
    """Raise if a feature at or before position p changes when pixels from p onward change.

    The future is replaced in four ways (random, all 0, all 1, inverted) so that aggregate
    leaks such as "any 1 later in this row" are caught too.
    """
    rng = np.random.default_rng(seed)
    base = [np.asarray(p).ravel() for p in fn(m)]
    n = m.size
    for t in range(trials):
        if t % 8 < 4:
            p = int(rng.integers(0, n))
        else:  # start of a row: catches whole-row leaks
            p = int(rng.integers(0, m.shape[0])) * m.shape[1]
        m2 = m.copy().ravel()
        mode = t % 4
        if mode == 0:
            m2[p:] = rng.integers(0, 2, size=n - p)
        elif mode == 1:
            m2[p:] = 0
        elif mode == 2:
            m2[p:] = 1
        else:
            m2[p:] = 1 - m2[p:]
        other = [np.asarray(q).ravel() for q in fn(m2.reshape(m.shape))]
        if len(other) != len(base):
            raise AssertionError("number of planes depends on future pixels")
        for k, (a, b) in enumerate(zip(base, other, strict=True)):
            if not np.array_equal(a[: p + 1], b[: p + 1]):
                bad = int(np.flatnonzero(a[: p + 1] != b[: p + 1])[0])
                raise AssertionError(f"plane {k} at index {bad} depends on index >= {p}")


def check_planes(planes, m):
    for k, p in enumerate(planes):
        p = np.asarray(p)
        if p.shape != m.shape:
            raise AssertionError(f"plane {k} has shape {p.shape}, expected {m.shape}")
        if p.dtype != np.uint8 or p.max(initial=0) > 1:
            raise AssertionError(f"plane {k} must be uint8 with values 0/1")


# ---------- eval set ----------
def build(manifest_path, scan_paths, per_category=PER_CATEGORY):
    per_category = int(os.environ.get("HARNESS_PER_CATEGORY", per_category))
    items = []
    for sp in scan_paths:
        items += json.load(open(sp))["binary"]
    by_cat = {}
    seen = set()
    for it in items:
        if it["hash"] in seen or min(it["w"], it["h"]) < 8 or it["w"] * it["h"] > MAX_PIXELS:
            continue
        seen.add(it["hash"])
        by_cat.setdefault(it["category"], []).append(it)
    for cat in [c for c, g in by_cat.items() if len(g) < MIN_CATEGORY]:
        for it in by_cat.pop(cat):
            it["category"] = "misc"
            by_cat.setdefault("misc", []).append(it)
    chosen = []
    for cat, group in sorted(by_cat.items()):
        group.sort(key=lambda it: hashlib.blake2b(it["hash"].encode(), digest_size=8).digest())
        chosen += group[:per_category]
        print(f"{cat}: {len(group)} available, {min(len(group), per_category)} chosen")
    json.dump(chosen, open(manifest_path, "w"), indent=1)


# ---------- running ----------
def resolve(spec):
    """Parse a config spec.

    'module:function@24'      feature function, then 24 neighbours (default 24)
    'module:function@24/4'    the same, with the features after the 4 nearest neighbours
    'none@24'                 no features
    'cost:module:function'    custom cost function f(m) -> bits (for estimator experiments)
    Returns (kind, function or None, neighbours).
    """
    if spec.startswith("cost:"):
        mod, _, fn = spec[5:].partition(":")
        return "cost", getattr(importlib.import_module(mod), fn), None
    target, _, n = spec.partition("@")
    n, _, at = n.partition("/")
    n = (int(n) if n else 24, int(at) if at else 0)
    if target in ("", "none"):
        return "features", None, n
    mod, _, fn = target.partition(":")
    return "features", getattr(importlib.import_module(mod), fn), n


def _work(task):
    item, configs = task
    try:
        m, _ = load(item)
    except Exception:
        return None
    row = {k: item[k] for k in ("path", "category", "hash", "w", "h")}
    row["boundary"] = boundary(m)
    row["bits"] = {}
    row["sec"] = {}
    for name, spec in configs.items():
        kind, fn, n = resolve(spec)
        t0 = time.time()
        if kind == "cost":
            row["bits"][name] = float(fn(m))
        else:
            planes = fn(m) if fn else []
            check_planes(planes, m)
            row["bits"][name] = cost(m, planes, n[0], at=n[1])
        row["sec"][name] = round(time.time() - t0, 3)
    return row


def run(manifest_path, out_path, config_args, workers=None):
    configs = dict(c.split("=", 1) for c in config_args)
    items = json.load(open(manifest_path))
    workers = workers or int(os.environ.get("HARNESS_WORKERS", DEFAULT_WORKERS))
    rows = []
    with mp.Pool(workers) as pool:
        for row in pool.imap_unordered(_work, [(it, configs) for it in items], chunksize=4):
            if row is not None:
                rows.append(row)
    json.dump(rows, open(out_path, "w"), indent=1)
    return rows


def report(rows, baseline=None):
    """Per-category totals in bytes and ratio to the baseline config."""
    names = list(rows[0]["bits"])
    baseline = baseline or names[0]
    cats = sorted({r["category"] for r in rows})
    lines = ["| category | images | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    ratios = {n: [] for n in names}
    for cat in cats + ["ALL"]:
        g = [r for r in rows if cat == "ALL" or r["category"] == cat]
        base = sum(r["bits"][baseline] for r in g)
        cells = []
        for n in names:
            tot = sum(r["bits"][n] for r in g)
            cells.append(f"{tot / 8:,.0f} ({tot / base:.3f})")
            if cat != "ALL":
                ratios[n].append(tot / base)
        lines.append(f"| {cat} | {len(g)} | " + " | ".join(cells) + " |")
    lines.append(
        "| category-balanced ratio | | "
        + " | ".join(f"{np.mean(ratios[n]):.3f}" for n in names)
        + " |"
    )
    return "\n".join(lines)


def main():
    cmd = sys.argv[1]
    if cmd == "build":
        build(sys.argv[2], sys.argv[3:])
    elif cmd == "run":
        rows = run(sys.argv[2], sys.argv[3], sys.argv[4:])
        print(report(rows))
    elif cmd == "report":
        rows = json.load(open(sys.argv[2]))
        print(report(rows, sys.argv[3] if len(sys.argv) > 3 else None))


if __name__ == "__main__":
    main()
