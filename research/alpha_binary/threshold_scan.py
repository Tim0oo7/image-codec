"""List anti-aliased alpha images as *derived* binary masks (alpha >= 128).

Usage: python threshold_scan.py OUT.json LABEL=DIR [LABEL=DIR ...]

Thresholding coverage-based anti-aliased alpha at 50 % gives, for straight edges, about the same
pixels as an aliased rasterisation of the same vector shape. These masks stand in for vector
cutouts and are always reported as a separate, derived category.
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
THRESHOLD = 128


def main():
    out_path, items, seen = sys.argv[1], [], set()
    for spec in sys.argv[2:]:
        label, _, root = spec.partition("=")
        for dirpath, _, files in os.walk(root):
            for name in sorted(files):
                if not name.lower().endswith((".png", ".gif", ".webp")):
                    continue
                path = os.path.join(dirpath, name)
                try:
                    with Image.open(path) as im:
                        if im.mode not in ("RGBA", "LA", "PA") and "transparency" not in im.info:
                            continue
                        a = np.asarray(im.convert("RGBA"))[..., 3]
                except Exception:
                    continue
                if len(np.unique(a)) <= 2:
                    continue  # already constant or binary: not derived
                m = a >= THRESHOLD
                if m.all() or not m.any():
                    continue
                digest = hashlib.blake2b(np.packbits(m).tobytes(), digest_size=16).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                items.append(
                    {
                        "path": path,
                        "category": label,
                        "hash": digest,
                        "w": int(a.shape[1]),
                        "h": int(a.shape[0]),
                        "threshold": THRESHOLD,
                        "values": [0, 255],
                    }
                )
        print(label, sum(it["category"] == label for it in items), flush=True)
    json.dump({"binary": items}, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
