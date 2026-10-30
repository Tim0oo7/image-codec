"""Reference codecs as harness cost functions (bits = 8 x file size, mask coded on its own)."""

import io

import imagecodecs
import numpy as np
from PIL import Image


def jxl_e9(m):
    return 8 * len(
        imagecodecs.jpegxl_encode(m, lossless=True, effort=9, bitspersample=1, numthreads=1)
    )


def webp(m):
    b = io.BytesIO()
    Image.fromarray((m * 255).astype(np.uint8)).save(
        b, "WEBP", lossless=True, quality=100, method=6, exact=True
    )
    return 8 * len(b.getvalue())


def png(m):
    b = io.BytesIO()
    Image.fromarray((m * 255).astype(np.uint8)).convert("1").save(b, "PNG", optimize=True)
    return 8 * len(b.getvalue())
