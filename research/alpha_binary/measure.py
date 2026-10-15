"""Compare reference codecs with exact ideal code lengths on binary masks.

Usage: python measure.py scan.json results.json [workers]

The synthetic masks are generated here. The real ones come from scan_alpha.py output.
Each mask is encoded on its own as a greyscale image with JPEG XL, WebP, PNG and AVIF
(all lossless), and the ideal adaptive cost of several context models is computed exactly.
"""

import io
import json
import multiprocessing as mp
import sys
import time
import warnings
from pathlib import Path

import imagecodecs
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from idealcost import context_bits, ctw_bits, kt_bits, oracle_bits, shifted  # noqa: E402

warnings.simplefilter("ignore")
MAX_E10_PIXELS = 1_100_000  # JPEG XL effort 10 is slow, skip it above this size


def synth():
    rng = np.random.default_rng(12345)
    out = {}
    yy, xx = np.mgrid[0:1024, 0:1024]
    c, s = np.cos(0.5), np.sin(0.5)
    u, v = (xx - 500) * c + (yy - 520) * s, -(xx - 500) * s + (yy - 520) * c
    out["ellipse_1024"] = ((u / 420) ** 2 + (v / 250) ** 2 < 1).astype(np.uint8)

    im = Image.new("1", (1024, 1024), 0)
    d = ImageDraw.Draw(im)
    for _ in range(6):
        k = rng.integers(3, 8)
        pts = [tuple(p) for p in rng.integers(0, 1024, size=(k, 2)).tolist()]
        d.polygon(pts, fill=int(rng.integers(0, 2)))
    out["polygons_1024"] = np.asarray(im, dtype=np.uint8)

    im = Image.new("1", (1024, 512), 0)
    d = ImageDraw.Draw(im)
    d.fontmode = "1"
    font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 36)
    text = "The quick brown fox jumps over the lazy dog 0123456789"
    for i in range(12):
        d.text((10, 8 + i * 41), text[i:] + text[:i], fill=1, font=font)
    out["text_1024x512"] = np.asarray(im, dtype=np.uint8)

    small = Image.fromarray((rng.random((64, 64)) < 0.5).astype(np.uint8) * 255)
    small = small.filter(ImageFilter.GaussianBlur(2)).point(lambda p: 255 if p > 127 else 0)
    big = small.resize((512, 512), Image.NEAREST)
    out["pixelart_x8_512"] = (np.asarray(big) > 0).astype(np.uint8)

    yy, xx = np.mgrid[0:512, 0:512]
    grad = np.clip(255 - np.hypot(xx - 256, yy - 256), 0, 255).astype(np.uint8)
    out["fs_dither_512"] = (np.asarray(Image.fromarray(grad).convert("1")) > 0).astype(np.uint8)
    bayer = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]])
    out["bayer_dither_512"] = (grad / 16 > np.tile(bayer, (128, 128))).astype(np.uint8)

    out["noise_256"] = (rng.random((256, 256)) < 0.5).astype(np.uint8)

    yy, xx = np.mgrid[0:32, 0:32]
    disc = ((xx - 15.5) ** 2 + (yy - 15.5) ** 2) < 14**2
    eye = ((xx - 10) ** 2 + (yy - 12) ** 2) < 9
    mouth = (xx > 12) & (xx < 20) & (yy > 20) & (yy < 24)
    sprite = disc & ~eye & ~mouth
    tiles = [sprite, sprite[:, ::-1], sprite[::-1, :], np.zeros_like(sprite)]
    sheet = np.block([[tiles[rng.integers(0, 4)] for _ in range(16)] for _ in range(16)])
    out["spritesheet_512"] = sheet.astype(np.uint8)

    field = Image.fromarray((rng.random((1024, 1024)) * 255).astype(np.uint8))
    field = np.asarray(field.filter(ImageFilter.GaussianBlur(24)), dtype=np.float32)
    out["blobs_1024"] = (field > np.median(field)).astype(np.uint8)
    return out


def enc_sizes(m, effort10):
    """Sizes in bytes of the mask coded on its own by the reference codecs."""
    g = (m * 255).astype(np.uint8)
    r = {
        "jxl_e9": len(
            imagecodecs.jpegxl_encode(m, lossless=True, effort=9, bitspersample=1, numthreads=1)
        )
    }
    if effort10:
        r["jxl_e10"] = len(
            imagecodecs.jpegxl_encode(m, lossless=True, effort=10, bitspersample=1, numthreads=1)
        )
    b = io.BytesIO()
    Image.fromarray(g).save(b, "WEBP", lossless=True, quality=100, method=6, exact=True)
    r["webp"] = len(b.getvalue())
    b = io.BytesIO()
    Image.fromarray(g).convert("1").save(b, "PNG", optimize=True)
    r["png"] = len(b.getvalue())
    r["avif"] = len(imagecodecs.avif_encode(g, level=100, speed=4, numthreads=1))
    return r


def ideal(m, extra=()):
    """Ideal adaptive costs in bytes. `extra` bits are put first in the CTW context order."""
    x = m.ravel()
    k = len(extra)
    bits = context_bits(m, 24 + k, extra)
    r = {
        "ctw24": ctw_bits(x, bits, 24 + k, 1 / 16) / 8,
        "ctw24_kt": ctw_bits(x, bits, 24 + k, 0.5) / 8,
    }
    if not extra:
        r["kt10"] = kt_bits(x, bits, 10) / 8
        r["kt16"] = kt_bits(x, bits, 16) / 8
        r["oracle16"] = oracle_bits(x, bits, 16) / 8
    return r


def colour_bits(rgba, dom):
    """Context bits available to the alpha coder if colour is decoded first."""
    rgb = rgba[..., :3].astype(np.uint32)
    key = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
    is_key = (key == dom).astype(np.uint8)
    eq_w = (key == shifted(key, 0, -1)).astype(np.uint8)
    eq_n = (key == shifted(key, -1, 0)).astype(np.uint8)
    return [is_key, eq_w, eq_n]


def rgba_sizes(rgba):
    """Whole-image sizes, to see what alpha adds when the codec sees colour too."""
    rgb = np.ascontiguousarray(rgba[..., :3])
    r = {
        "jxl_e9_rgba": len(imagecodecs.jpegxl_encode(rgba, lossless=True, effort=9, numthreads=1)),
        "jxl_e9_rgb": len(imagecodecs.jpegxl_encode(rgb, lossless=True, effort=9, numthreads=1)),
    }
    for name, arr in (("webp_rgba", rgba), ("webp_rgb", rgb)):
        b = io.BytesIO()
        Image.fromarray(arr).save(b, "WEBP", lossless=True, quality=100, method=6, exact=True)
        r[name] = len(b.getvalue())
    return r


def boundary(m):
    return int((m[1:] != m[:-1]).sum() + (m[:, 1:] != m[:, :-1]).sum())


_SYNTH = None


def run(task):
    global _SYNTH
    kind, ref = task
    t0 = time.time()
    if kind == "synthetic":
        if _SYNTH is None:
            _SYNTH = synth()
        m = _SYNTH[ref]
        row = {"name": ref, "w": m.shape[1], "h": m.shape[0]}
    else:
        try:
            with Image.open(ref["path"]) as im:
                rgba = np.asarray(im.convert("RGBA"))
        except Exception:
            return None
        m = (rgba[..., 3] == ref["values"][1]).astype(np.uint8)
        row = dict(ref)
        row["colour_key"] = ref["dominant_frac"] == 1.0 and ref["dominant_in_opaque"] == 0
        cb = colour_bits(rgba, int(ref["dominant_colour"], 16))
        row.update({k + "_colour": v for k, v in ideal(m, cb).items()})
        row.update(rgba_sizes(rgba))
    row["kind"] = kind
    row["boundary"] = boundary(m)
    row.update(ideal(m))
    row.update(enc_sizes(m, m.size <= MAX_E10_PIXELS))
    row["const"] = enc_sizes(np.ones_like(m), False)
    row["sec"] = round(time.time() - t0, 2)
    return row


def main():
    scan_path, out_path = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else max(1, mp.cpu_count() - 2)
    tasks = [("synthetic", name) for name in synth()]
    for item in json.load(open(scan_path))["binary"]:
        if min(item["w"], item["h"]) >= 16 and item["w"] * item["h"] <= 4_200_000:
            tasks.append(("real", item))
    results = []
    with mp.Pool(workers) as pool:
        for i, row in enumerate(pool.imap_unordered(run, tasks, chunksize=2)):
            if row is not None:
                results.append(row)
            if i % 100 == 0:
                print(f"{i}/{len(tasks)}", flush=True)
    json.dump(results, open(out_path, "w"), indent=1)
    print("done", len(results), flush=True)


if __name__ == "__main__":
    main()
