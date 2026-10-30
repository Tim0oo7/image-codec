"""Split the corpus into dev and test by *source object*, not by file.

Usage: python split.py OUT_PREFIX SCAN.json [SCAN.json ...]
Writes OUT_PREFIX_dev.json and OUT_PREFIX_test.json.

Related files share a group and always land in the same split:
  glyphs       font + codepoint (the 24, 64 and 192 px renderings belong together)
  pixelart     the whole Kenney pack (tiles and the tile sheets made from them)
  icons_thr    icon name (icon_64 and icon_512 are the same icons)
  system, user the folder (sprite-strip variants, icon sizes of one app)
  others       the file name
A group goes to test if a hash of its key is odd. Each split keeps up to PER_SPLIT masks per
category, chosen by hash.
"""

import hashlib
import json
import os
import sys

MIN_SIDE, MAX_PIXELS, PER_SPLIT = 8, 2_100_000, 300


def group_key(it):
    parts = os.path.normpath(it["path"]).split(os.sep)
    cat = it["category"]
    if cat == "glyphs":
        return f"glyph:{parts[-3]}:{parts[-1]}"
    if cat == "pixelart":
        return "kenney:" + parts[parts.index("kenney") + 1]
    if cat == "icons_thr":
        return "icon:" + os.path.splitext(parts[-1])[0]
    if cat in ("system", "user"):
        return "dir:" + os.sep.join(parts[:-1]).lower()
    return "file:" + parts[-1].lower()


def h(s):
    return hashlib.blake2b(s.encode(), digest_size=8).digest()


def main():
    prefix, scans = sys.argv[1], sys.argv[2:]
    items, seen = [], set()
    for sp in scans:
        for it in json.load(open(sp))["binary"]:
            if it["hash"] in seen or min(it["w"], it["h"]) < MIN_SIDE:
                continue
            if it["w"] * it["h"] > MAX_PIXELS:
                continue
            seen.add(it["hash"])
            if it["category"] in ("qoi_cutouts", "qoi_textures"):
                it["category"] = "misc"
            items.append(it)
    out = {"dev": {}, "test": {}}
    for it in items:
        split = "test" if h(group_key(it))[0] % 2 else "dev"
        out[split].setdefault(it["category"], []).append(it)
    for split, cats in out.items():
        chosen = []
        for cat, group in sorted(cats.items()):
            group.sort(key=lambda it: h(it["hash"]))
            chosen += group[:PER_SPLIT]
            groups = len({group_key(it) for it in group})
            kept = min(len(group), PER_SPLIT)
            print(f"{split:4s} {cat:12s} {len(group):5d} masks in {groups:4d} groups, {kept} kept")
        json.dump(chosen, open(f"{prefix}_{split}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
