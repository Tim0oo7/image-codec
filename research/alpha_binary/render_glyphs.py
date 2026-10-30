"""Render icon-font glyphs without anti-aliasing: binary "vector cutout" masks.

Usage: python render_glyphs.py OUTDIR FONT.ttf [FONT.ttf ...]

Every glyph in the Private Use Area (where icon fonts keep their icons) is rendered at a few
sizes and saved as an RGBA PNG whose alpha is the glyph and whose colour is black everywhere,
so colour carries no information about alpha. Glyphs are sampled with a fixed stride to keep
the set balanced between fonts.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SIZES = (24, 64, 192)
PER_FONT = 240  # glyphs sampled per font (per size)


def glyphs(font_path):
    probe = ImageFont.truetype(font_path, 64)
    seen, found = set(), []
    for cp in range(0xE000, 0xF900):
        mask = probe.getmask(chr(cp), mode="1")
        if mask.size[0] == 0 or mask.size[1] == 0:
            continue
        digest = hashlib.blake2b(bytes(mask), digest_size=12).digest()
        if digest in seen:  # .notdef boxes and duplicates
            continue
        seen.add(digest)
        found.append(cp)
    step = max(1, len(found) // PER_FONT)
    return found[::step][:PER_FONT]


def render(font_path, cp, size):
    font = ImageFont.truetype(font_path, size)
    canvas = Image.new("L", (size * 2, size * 2), 0)
    draw = ImageDraw.Draw(canvas)
    draw.fontmode = "1"
    draw.text((size // 2, size // 2), chr(cp), fill=255, font=font)
    box = canvas.getbbox()
    if box is None:
        return None
    pad = max(1, size // 16)
    left, top, right, bottom = box
    alpha = np.asarray(canvas.crop((left - pad, top - pad, right + pad, bottom + pad)))
    rgba = np.zeros(alpha.shape + (4,), dtype=np.uint8)
    rgba[..., 3] = np.where(alpha > 0, 255, 0)
    return Image.fromarray(rgba, "RGBA")


def main():
    out = Path(sys.argv[1])
    for font_path in sys.argv[2:]:
        name = Path(font_path).stem
        cps = glyphs(font_path)
        for size in SIZES:
            d = out / name / str(size)
            d.mkdir(parents=True, exist_ok=True)
            for cp in cps:
                im = render(font_path, cp, size)
                if im is not None:
                    im.save(d / f"{cp:04x}.png")
        print(name, len(cps), "glyphs x", len(SIZES), "sizes")


if __name__ == "__main__":
    main()
