"""Scan folders for PNG/GIF/WebP images, classify their alpha, and collect binary-alpha ones.

Usage: python scan_alpha.py OUT.json LABEL=DIR [LABEL=DIR ...] [--exclude DIR ...]

Each source directory gets a category label. Images with identical pixel data are counted once
(the first label wins). Only paths and statistics are written: no pixel data is copied.
"""

import hashlib
import json
import os
import sys
import warnings

import numpy as np
from PIL import Image

warnings.simplefilter("ignore")
Image.MAX_IMAGE_PIXELS = 50_000_000
EXTS = (".png", ".gif", ".webp")


def parse_args(argv):
    out, sources, exclude = argv[0], [], []
    rest = argv[1:]
    if "--exclude" in rest:
        i = rest.index("--exclude")
        rest, exclude = rest[:i], rest[i + 1 :]
    for item in rest:
        label, _, path = item.partition("=")
        sources.append((label, os.path.abspath(path)))
    return out, sources, [os.path.abspath(e) for e in exclude]


def binary_stats(rgba, a, values):
    lo, hi = int(values[0]), int(values[1])
    t = a == lo
    rgb = rgba[..., :3].reshape(-1, 3).astype(np.uint32)
    key = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]
    kt, ko = key[t.ravel()], key[~t.ravel()]
    u, c = np.unique(kt, return_counts=True)
    dom = int(u[np.argmax(c)])
    return {
        "values": [lo, hi],
        "frac_transparent": float(t.mean()),
        "boundary": int((a[1:] != a[:-1]).sum() + (a[:, 1:] != a[:, :-1]).sum()),
        "transparent_colours": int(len(u)),
        "dominant_colour": f"{dom:06x}",
        "dominant_frac": float(c.max() / len(kt)),
        "dominant_in_opaque": int((ko == dom).sum()),
    }


def main():
    out_path, sources, exclude = parse_args(sys.argv[1:])
    stats = {}
    seen = set()
    binary = []
    for label, root in sources:
        s = stats.setdefault(
            label,
            dict.fromkeys(
                ["files", "errors", "no_alpha", "dup", "constant", "binary", "general"], 0
            ),
        )
        # an exclusion only applies to sources that are not themselves inside it
        skip = [e for e in exclude if not root.startswith(e)]
        for dirpath, dirs, files in os.walk(root):
            dirs[:] = [
                d for d in dirs if not any(os.path.join(dirpath, d).startswith(e) for e in skip)
            ]
            for name in files:
                if not name.lower().endswith(EXTS):
                    continue
                path = os.path.join(dirpath, name)
                s["files"] += 1
                try:
                    if os.path.getsize(path) > 30_000_000:
                        continue
                    with Image.open(path) as im:
                        has_alpha = im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info
                        if not has_alpha:
                            s["no_alpha"] += 1
                            continue
                        mode = im.mode
                        rgba = np.asarray(im.convert("RGBA"))
                except Exception:
                    s["errors"] += 1
                    continue
                digest = hashlib.blake2b(rgba.tobytes(), digest_size=16).hexdigest()
                if digest in seen:
                    s["dup"] += 1
                    continue
                seen.add(digest)
                a = rgba[..., 3]
                values = np.flatnonzero(np.bincount(a.ravel(), minlength=256))
                if len(values) == 1:
                    s["constant"] += 1
                    continue
                if len(values) > 2:
                    s["general"] += 1
                    continue
                s["binary"] += 1
                item = {"path": path, "category": label, "hash": digest, "mode": mode}
                item.update({"w": int(a.shape[1]), "h": int(a.shape[0])})
                item.update(binary_stats(rgba, a, values))
                binary.append(item)
        print(label, json.dumps(s), flush=True)
    with open(out_path, "w") as f:
        json.dump({"stats": stats, "binary": binary}, f, indent=1)


if __name__ == "__main__":
    main()
