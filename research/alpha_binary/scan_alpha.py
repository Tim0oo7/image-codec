"""Scan folders for PNG/GIF images, classify their alpha channel, and collect binary-alpha ones.

Usage: python scan_alpha.py OUT.json DIR [DIR ...]
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

out_path, roots = sys.argv[1], sys.argv[2:]
stats = {"files": 0, "errors": 0, "no_alpha": 0, "constant": 0, "binary": 0, "general": 0}
seen = set()
binary = []

for root in roots:
    for dirpath, _, files in os.walk(root):
        for name in files:
            if not name.lower().endswith((".png", ".gif")):
                continue
            path = os.path.join(dirpath, name)
            stats["files"] += 1
            try:
                if os.path.getsize(path) > 30_000_000:
                    continue
                with Image.open(path) as im:
                    has_alpha = im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info
                    if not has_alpha:
                        stats["no_alpha"] += 1
                        continue
                    src_mode = im.mode
                    rgba = np.asarray(im.convert("RGBA"))
            except Exception:
                stats["errors"] += 1
                continue
            digest = hashlib.blake2b(rgba.tobytes(), digest_size=16).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            a = rgba[..., 3]
            counts = np.bincount(a.ravel(), minlength=256)
            values = np.flatnonzero(counts)
            if len(values) == 1:
                stats["constant"] += 1
                continue
            if len(values) > 2:
                stats["general"] += 1
                continue
            stats["binary"] += 1
            lo, hi = int(values[0]), int(values[1])
            t = a == lo
            rgb = rgba[..., :3].reshape(-1, 3)
            key = (
                (rgb[:, 0].astype(np.uint32) << 16) | (rgb[:, 1].astype(np.uint32) << 8) | rgb[:, 2]
            )
            kt, ko = key[t.ravel()], key[~t.ravel()]
            u, c = np.unique(kt, return_counts=True)
            dom = int(u[np.argmax(c)])
            binary.append(
                {
                    "path": path,
                    "mode": src_mode,
                    "w": int(a.shape[1]),
                    "h": int(a.shape[0]),
                    "values": [lo, hi],
                    "frac_transparent": float(t.mean()),
                    "boundary": int((a[1:, :] != a[:-1, :]).sum() + (a[:, 1:] != a[:, :-1]).sum()),
                    "transparent_colours": int(len(u)),
                    "dominant_colour": f"{dom:06x}",
                    "dominant_frac": float(c.max() / len(kt)),
                    "dominant_in_opaque": int((ko == dom).sum()),
                }
            )

with open(out_path, "w") as f:
    json.dump({"stats": stats, "unique_alpha_images": len(seen), "binary": binary}, f, indent=1)
print(json.dumps(stats), "unique alpha images:", len(seen))
