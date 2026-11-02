"""The reference codec (src/image_codec/alpha_binary.py), compiled with numba for measurements.

load() runs the module's source in a fresh namespace, optionally with changed constants, and
replaces its kernel functions by numba-compiled versions. They run the same integer arithmetic
on numpy arrays instead of lists, so the output is identical (test_fixedcodec.py checks that).

Usage (from this folder):
    python fixedcodec.py MANIFEST.json OUT.json NAME[:CONST=VALUE,...] [NAME:...]
    python fixedcodec.py report OUT.json REFERENCE.json [REFERENCE_KEY]

Each configuration records, per mask, the payload size in bits after dropping trailing zero bits.
`report` compares configurations with a reference file from harness.py (e.g. mix_real).
"""

import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "image_codec" / "alpha_binary.py"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from image_codec.dimensions import dimension_bits  # noqa: E402

KERNELS = (
    "_log2",
    "_squash",
    "_stretch",
    "_ctw_predict",
    "_ctw_update",
    "next_step",
    "_advance",
    "_nearest",
    "_build_row",
    "_code",
)
DTYPES = {8: np.uint8, 32: np.int32, 64: np.int64}
ALPHA_HEADER_BITS = 18  # what realconfigs counts for the alpha values


def zeros(n, bits):
    return np.zeros(n, DTYPES[bits])


class Codec:
    """The codec's namespace with compiled kernels."""

    def __init__(self, **constants):
        ns = {"__name__": "image_codec._alpha_binary_jit", "__package__": "image_codec"}
        exec(compile(SRC.read_text(encoding="utf-8"), str(SRC), "exec"), ns)
        for name, value in constants.items():
            if name not in ns:
                raise KeyError(name)
            ns[name] = value
        for name in KERNELS:
            ns[name] = njit(ns[name])
        self.ns = ns
        self.log = np.array(ns["LOG"], np.int64)
        self.squash = np.array(ns["SQUASH"], np.int64)

    def _run(self, width, height, decode, bits, data, out, trace):
        img, offs, a, b, match, line, scratch, tbits, mbits = self.ns["_buffers"](
            width, height, zeros
        )
        return self.ns["_code"](
            width,
            height,
            decode,
            bits,
            data,
            out,
            trace,
            img,
            offs,
            a,
            b,
            match,
            line,
            scratch,
            self.log,
            self.squash,
            tbits,
            mbits,
        )

    def encode_mask(self, m, trace=False):
        h, w = m.shape
        bits = np.ascontiguousarray(m.ravel(), dtype=np.uint8)
        out = np.zeros(2 * m.size + 8, np.uint8)
        tr = np.zeros(self.ns["FEATURES"] * m.size if trace else 0, np.uint8)
        n = self._run(w, h, 0, bits, np.zeros(0, np.uint8), out, tr)
        data = out[:n].tobytes()
        if trace:
            return data, tr.reshape(m.size, self.ns["FEATURES"]).T
        return data

    def decode_mask(self, data, width, height):
        bits = np.zeros(width * height, np.uint8)
        self._run(
            width,
            height,
            1,
            bits,
            np.frombuffer(data, np.uint8),
            np.zeros(0, np.uint8),
            np.zeros(0, np.uint8),
        )
        return bits.reshape(height, width)


def payload_bits(data):
    """Length in bits up to the last 1 bit (the stream drops trailing zeros)."""
    data = data.rstrip(b"\0")
    if not data:
        return 0
    return 8 * len(data) - (data[-1] & -data[-1]).bit_length() + 1


def file_bits(m, data, header=ALPHA_HEADER_BITS, whole_bytes=False):
    """File size in bits. whole_bytes: end the payload on a byte, as realcoder does."""
    h, w = m.shape
    n = payload_bits(data)
    if whole_bytes:
        n = 8 * math.ceil(n / 8)
    return 8 * math.ceil((dimension_bits(w, h) + header + n) / 8)


# ---------- corpus runs ----------
_CODECS = {}


def parse(spec):
    name, _, rest = spec.partition(":")
    consts = {}
    for item in filter(None, rest.split(",")):
        k, _, v = item.partition("=")
        consts[k] = int(v, 0)
    return name, consts


def _work(task):
    from harness import load

    item, specs = task
    try:
        m, _ = load(item)
    except Exception:
        return None
    row = {k: item[k] for k in ("path", "category", "hash", "w", "h")}
    row["bits"], row["sec"] = {}, {}
    for spec in specs:
        name, consts = parse(spec)
        if spec not in _CODECS:
            _CODECS[spec] = Codec(**consts)
        t0 = time.time()
        data = _CODECS[spec].encode_mask(m)
        row["sec"][name] = round(time.time() - t0, 3)
        row["bits"][name] = file_bits(m, data)
        row["bits"][name + "_bytes"] = file_bits(m, data, whole_bytes=True)
    return row


def run(manifest, out_path, specs):
    items = json.load(open(manifest))
    items.sort(key=lambda it: -it["w"] * it["h"])  # big ones first, for load balance
    workers = int(os.environ.get("HARNESS_WORKERS", 6))
    rows = []
    with mp.Pool(workers) as pool:
        for row in pool.imap_unordered(_work, [(it, specs) for it in items]):
            if row is not None:
                rows.append(row)
    json.dump(rows, open(out_path, "w"), indent=1)
    return rows


def report(rows, ref_rows, ref_key="mix_real"):
    ref = {r["hash"]: r["bits"][ref_key] for r in ref_rows}
    rows = [r for r in rows if r["hash"] in ref]
    names = list(rows[0]["bits"])
    cats = sorted({r["category"] for r in rows})
    lines = [
        f"| category | masks | {ref_key} (bytes) | " + " | ".join(names) + " |",
        "|---|---|---|" + "---|" * len(names),
    ]
    for cat in cats + ["ALL"]:
        g = [r for r in rows if cat == "ALL" or r["category"] == cat]
        base = sum(ref[r["hash"]] for r in g)
        cells = [f"{sum(r['bits'][n] for r in g) / base:.4f}" for n in names]
        lines.append(f"| {cat} | {len(g)} | {base / 8:,.0f} | " + " | ".join(cells) + " |")
    worst = {n: max(rows, key=lambda r: r["bits"][n] / ref[r["hash"]]) for n in names}
    lines.append(
        "| worst mask | | | "
        + " | ".join(f"{worst[n]['bits'][n] / ref[worst[n]['hash']]:.3f}" for n in names)
        + " |"
    )
    return "\n".join(lines)


def main():
    if sys.argv[1] == "report":
        rows = json.load(open(sys.argv[2]))
        ref = json.load(open(sys.argv[3]))
        print(report(rows, ref, *sys.argv[4:]))
        return
    rows = run(sys.argv[1], sys.argv[2], sys.argv[3:])
    print(f"{len(rows)} masks")


if __name__ == "__main__":
    main()
