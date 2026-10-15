"""Summarise measure.py output as markdown tables.

Usage: python analyze.py results.json
"""

import json
import sys
from collections import Counter

import numpy as np

HDR = 4  # bytes assumed for our own alpha header (mode, values, flags)


def jxl(r):
    return min(r["jxl_e9"], r.get("jxl_e10", r["jxl_e9"]))


def ours(r, key="ctw24"):
    return r[key] + HDR


def n(v):
    return f"{v:,.0f}"


def table(header, rows):
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in rows:
        print("| " + " | ".join(row) + " |")


def pct(a, q):
    return float(np.percentile(a, q))


def synthetic(rows):
    print("## Synthetic masks\n")
    header = ["mask", "size", "boundary px", "CTW-24", "KT-10", "KT-16", "JXL", "WebP", "PNG"]
    header += ["AVIF", "JXL / CTW"]
    body = []
    for r in rows:
        body.append(
            [r["name"], f"{r['w']}x{r['h']}", n(r["boundary"]), n(ours(r)), n(r["kt10"])]
            + [n(r["kt16"]), n(jxl(r)), n(r["webp"]), n(r["png"]), n(r["avif"])]
            + [f"{jxl(r) / ours(r):.2f}"]
        )
    table(header, body)


def real(rows):
    print("\n## Real binary-alpha images\n")
    px = np.array([r["w"] * r["h"] for r in rows])
    print(f"images {len(rows)}, pixels median {np.median(px):.0f}, max {px.max()}")
    print("source modes:", dict(Counter(r["mode"] for r in rows)))
    print("alpha pairs:", Counter(tuple(r["values"]) for r in rows).most_common(5))
    const = [r for r in rows if r["dominant_frac"] == 1.0]
    keyed = [r for r in rows if r["colour_key"]]
    print(f"constant colour under alpha=0: {len(const)}; exact colour key: {len(keyed)}")
    print("constant colours:", Counter(r["dominant_colour"] for r in const).most_common(5))

    groups = {
        "all": rows,
        ">= 128x128 px": [r for r in rows if r["w"] * r["h"] >= 128 * 128],
        "< 128x128 px": [r for r in rows if r["w"] * r["h"] < 128 * 128],
        "no colour key": [r for r in rows if not r["colour_key"]],
    }
    print()
    header = ["set", "images", "CTW-24", "CTW-24 + colour", "+ colour-key mode", "JXL", "WebP"]
    header += ["PNG", "AVIF", "JXL RGBA - RGB", "WebP RGBA - RGB"]
    body = []
    for name, g in groups.items():
        if not g:
            continue
        keymode = sum(HDR if r["colour_key"] else ours(r, "ctw24_colour") for r in g)
        body.append(
            [
                name,
                str(len(g)),
                n(sum(ours(r) for r in g)),
                n(sum(ours(r, "ctw24_colour") for r in g)),
            ]
            + [n(keymode), n(sum(jxl(r) for r in g)), n(sum(r["webp"] for r in g))]
            + [n(sum(r["png"] for r in g)), n(sum(r["avif"] for r in g))]
            + [n(sum(r["jxl_e9_rgba"] - r["jxl_e9_rgb"] for r in g))]
            + [n(sum(r["webp_rgba"] - r["webp_rgb"] for r in g))]
        )
    table(header, body)

    print(
        "\nPayload only (each codec's size for an all-opaque mask of the same size subtracted):\n"
    )
    body = []
    for name, g in groups.items():
        cols = [name]
        for c in ("jxl_e9", "webp", "png", "avif"):
            cols.append(n(sum(max(r[c] - r["const"][c], 0) for r in g)))
        cols.append(n(sum(r["ctw24"] for r in g)))
        body.append(cols)
    table(["set", "JXL e9", "WebP", "PNG", "AVIF", "CTW-24 (no header)"], body)

    ratio = np.array([jxl(r) / ours(r) for r in rows])
    best = np.array([min(jxl(r), r["webp"], r["png"], r["avif"]) for r in rows])
    mine = np.array([ours(r) for r in rows])
    print(f"\nCTW-24 smaller than every baseline: {(mine < best).sum()}/{len(rows)}")
    print(f"CTW-24 smaller than JXL: {(ratio > 1).sum()}/{len(rows)}")
    med, p10, p90 = np.median(ratio), pct(ratio, 10), pct(ratio, 90)
    print(f"JXL / CTW-24: median {med:.2f}, p10 {p10:.2f}, p90 {p90:.2f}")
    cg = np.array([(r["ctw24_colour"] + 1) / (r["ctw24"] + 1) for r in rows if not r["colour_key"]])
    print(f"no-key images, CTW+colour / CTW: median {np.median(cg):.3f}, p10 {pct(cg, 10):.3f}")
    bpb = np.array([r["ctw24"] * 8 / max(r["boundary"], 1) for r in rows])
    print(f"CTW-24 bits per boundary pixel: median {np.median(bpb):.3f}, p90 {pct(bpb, 90):.3f}")

    print("\nLargest 12 images:\n")
    header = ["image", "size", "boundary", "CTW-24", "CTW-24 + colour", "key", "JXL", "WebP"]
    header += ["PNG", "AVIF", "JXL RGBA - RGB"]
    body = []
    for r in sorted(rows, key=lambda r: -r["w"] * r["h"])[:12]:
        name = r["path"].replace("\\", "/").split("/")[-1][:36]
        body.append(
            [name, f"{r['w']}x{r['h']}", n(r["boundary"]), n(ours(r)), n(ours(r, "ctw24_colour"))]
            + ["yes" if r["colour_key"] else "no", n(jxl(r)), n(r["webp"]), n(r["png"])]
            + [n(r["avif"]), n(r["jxl_e9_rgba"] - r["jxl_e9_rgb"])]
        )
    table(header, body)


def main():
    rows = json.load(open(sys.argv[1]))
    synth = sorted((r for r in rows if r["kind"] == "synthetic"), key=lambda r: r["name"])
    synthetic(synth)
    found = [r for r in rows if r["kind"] == "real"]
    if found:
        real(found)


if __name__ == "__main__":
    main()
