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


# JBIG2 generic region, template 0: 16 pixels incl. the 4 adaptive pixels at their nominal
# positions. Coded here with an ideal adaptive KT estimator, which is never worse than the real
# MQ coder, so this is an optimistic (lower) estimate of a JBIG2 generic-region file.
JBIG2_T0 = [(-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2)]
JBIG2_T0 += [(-1, -3), (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2), (-1, 3)]
JBIG2_T0 += [(0, -4), (0, -3), (0, -2), (0, -1)]


def jbig2_t0_ideal(m):
    from harness import shifted
    from idealcost import kt_bits

    bits = [shifted(m, dy, dx).ravel() for dy, dx in JBIG2_T0]
    return kt_bits(m.ravel(), bits, len(bits))


def avif(m):
    g = (m * 255).astype(np.uint8)
    return 8 * len(imagecodecs.avif_encode(g, level=100, speed=4, numthreads=1))
