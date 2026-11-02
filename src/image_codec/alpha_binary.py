"""Binary alpha: the reference codec. Specified in docs/format-spec.md, "Binary alpha".

A binary alpha channel has exactly two values, lo and hi. It is stored as the dimensions field,
the alpha-values code, and the mask (1 where alpha = hi), arithmetic-coded pixel by pixel in
raster order. The probability of each pixel comes from two context-tree-weighting (CTW) models,
combined by a two-weight logistic mixer. The context bits are the nearest decoded neighbours and
three small predictors (line, match, copy) that also read decoded pixels only, so the stream
carries no side information. All arithmetic is on integers below 2^63, so the result is the same
on every platform.

The pixel loop runs on plain lists of ints and is slow (a few thousand pixels per second).
research/alpha_binary/fixedcodec.py compiles the same functions with numba for corpus
measurements; this module itself does not need numba.
"""

import math

from .bitio import BitReader, BitWriter
from .dimensions import read_dimensions, write_dimensions

# ---------- context ----------


def _causal_offsets(n):
    """The n nearest decoded neighbours (dy, dx): by distance, then |dx|, then the left one."""
    cand = [(dy * dy + dx * dx, abs(dx), dx, dy) for dy in range(-8, 1) for dx in range(-8, 9)]
    return [(dy, dx) for _, _, dx, dy in sorted(c for c in cand if c[3] < 0 or c[2] < 0)[:n]]


NEIGHBOURS = _causal_offsets(48)  # the match key; the first 24 are CTW context bits
CTW_NEIGHBOURS = 24
PAD = 5  # no neighbour is further away than 5 rows or columns
HISTORY = 24  # rows of edge-track history used by the line predictor
RING = HISTORY + 1  # rows of transitions kept
DEPTH_A = 30  # context bits of model A
DEPTH_B = 34  # model B: A plus the 4 copy bits after the 14th neighbour
FEATURES = 10  # here, past, near, certain, exists, value, copy: set, value, streak 8, 64
SMALL = 1024  # masks of at most this many pixels use one CTW over the 24 neighbours only

# ---------- fixed point ----------

ONE = 1 << 32  # CTW probabilities are P(1) * 2^32, always in [1, ONE - 1]
COUNT_LIMIT = 65535  # node counts are halved when n0 + n1 reaches this (so each fits 16 bits)
WEIGHT_LIMIT = 24 << 16  # |log2 beta| of a CTW node, in 2^-16 bit
STRETCH_LIMIT = 20 << 16  # |stretch| of a mixer input, in 2^-16 bit
MIX_SHIFT = 23  # mixer learning rate: 2^-23 per (error * stretch)
MIX_LIMIT = 1 << 20  # |mixer weight|, in 2^-16
NODE_MULT = 0x2C1B3C6D
MATCH_MULT = (0x9E3779B1, 0x85EBCA77)

# Node and match table sizes grow with the image, up to a limit.
TABLE_EXTRA, TABLE_MIN, TABLE_MAX = 5, 12, 24
MATCH_EXTRA, MATCH_MIN, MATCH_MAX = 3, 8, 22


def _log_table():
    """LOG[i] = 2^16 * log2(1 + i / 4096), i = 0..4096, rounded to nearest."""
    return [round(65536 * math.log2(1 + i / 4096)) for i in range(4097)]


def _squash_table():
    """SQUASH[i + 4096] = 2^16 / (1 + 2^(-i / 256)), i = -4096..4096, rounded, in [1, 65535]."""
    return [min(max(round(65536 / (1 + 2 ** (-i / 256))), 1), 65535) for i in range(-4096, 4097)]


LOG = _log_table()
SQUASH = _squash_table()


def table_bits(width, height):
    """(node table bits, match table bits) for an image of this size."""
    n = (width * height).bit_length()
    return (
        min(TABLE_MAX, max(TABLE_MIN, n + TABLE_EXTRA)),
        min(MATCH_MAX, max(MATCH_MIN, n + MATCH_EXTRA)),
    )


def _log2(p, log):
    """2^16 * log2(p / 2^32) for 1 <= p < 2^32: table lookup with linear interpolation."""
    e = 0
    v = p
    if v >> 16:
        v >>= 16
        e += 16
    if v >> 8:
        v >>= 8
        e += 8
    if v >> 4:
        v >>= 4
        e += 4
    if v >> 2:
        v >>= 2
        e += 2
    if v >> 1:
        e += 1
    m = p << (31 - e)  # normalised: 2^31 <= m < 2^32
    i = (m >> 19) - 4096
    f = m & 0x7FFFF
    return (e - 32) * 65536 + log[i] + (((log[i + 1] - log[i]) * f) >> 19)


def _squash(t, squash):
    """2^16 / (1 + 2^(-t / 2^16)), from the table at 1/256-bit steps."""
    i = (t + 128) >> 8
    if i > 4096:
        i = 4096
    elif i < -4096:
        i = -4096
    return squash[i + 4096]


def _stretch(p, log):
    """log2(p / (1 - p)) in 2^-16 bit, clamped, for a CTW probability p * 2^32."""
    s = _log2(p, log) - _log2(ONE - p, log)
    return min(max(s, -STRETCH_LIMIT), STRETCH_LIMIT)


# ---------- CTW ----------


def _ctw_predict(ctx, depth, tkey, tn0, tn1, tlb, tbits, path, pe, pw, squash):
    """P(1) * 2^32 of one CTW model over the context bits ctx[0 .. depth - 1].

    Node d of the path is the context's first d bits; its key is those bits after a leading 1.
    It lives in a direct-mapped table at a slot given by a hash of the path; a slot holding
    another key is reset. A node holds the counts n0, n1 and lb = log2 beta (in 2^-16 bit), where
    beta = Pe(node) / (Pw(child 0) * Pw(child 1)). Fills path, pe (the node's own estimate) and
    pw (the weighted probability) for _ctw_update.
    """
    shift = 32 - tbits
    h = 0
    key = 1
    for d in range(depth + 1):
        s = h >> shift
        if tkey[s] != key:
            tkey[s] = key
            tn0[s] = 0
            tn1[s] = 0
            tlb[s] = 0
        path[d] = s
        if d < depth:
            b = ctx[d]
            key = 2 * key + b
            h = (((h ^ (h >> 15)) + b + 1) * NODE_MULT) & 0xFFFFFFFF
    # bottom-up: Pw(1) = (beta Pe(1) + Pw_child(1)) / (beta + 1)
    p = 0
    for d in range(depth, -1, -1):
        s = path[d]
        n1 = tn1[s]
        e = ((4 * n1 + 1) << 32) // (4 * (tn0[s] + n1) + 2)  # Dirichlet, alpha = 1/4
        pe[d] = e
        if d == depth:
            p = e
        else:
            p += ((e - p) * _squash(tlb[s], squash)) >> 16
        pw[d] = p
    return p


def _ctw_update(bit, depth, tn0, tn1, tlb, path, pe, pw, log):
    """Count the coded bit in every node of the path and update beta."""
    for d in range(depth + 1):
        s = path[d]
        if d < depth:
            if bit:
                g = _log2(pe[d], log) - _log2(pw[d + 1], log)
            else:
                g = _log2(ONE - pe[d], log) - _log2(ONE - pw[d + 1], log)
            tlb[s] = min(max(tlb[s] + g, -WEIGHT_LIMIT), WEIGHT_LIMIT)
        if bit:
            tn1[s] += 1
        else:
            tn0[s] += 1
        if tn0[s] + tn1[s] >= COUNT_LIMIT:
            tn0[s] = (tn0[s] + 1) >> 1
            tn1[s] = (tn1[s] + 1) >> 1


# ---------- line predictor ----------


def next_step(deltas, n):
    """(step, certain) from the steps deltas[0 .. n - 1] of an edge track, oldest first.

    docs/alpha-binary-line-predictor.md, next_step: a digital straight line steps by a or a + 1,
    one of them isolated, the runs of the other L or L + 1 long.
    """
    if n == 0:
        return 0, 0
    last = deltas[n - 1]
    lo = last
    hi = last
    for k in range(n):
        lo = min(lo, deltas[k])
        hi = max(hi, deltas[k])
    if hi - lo != 1:  # one value, or not a straight line
        return last, 0
    other = hi if last == lo else lo
    first = 1  # length of the first run, which may be cut off
    while first < n and deltas[first] == deltas[0]:
        first += 1
    cur = 1  # length of the current (last) run
    while cur < n and deltas[n - 1 - cur] == last:
        cur += 1
    # complete runs: everything between the first and the last run
    count_lo = count_hi = 0
    min_lo = min_hi = n
    max_lo = max_hi = 0
    k = first
    while k < n - cur:
        j = k
        while deltas[j] == deltas[k]:
            j += 1
        if deltas[k] == lo:
            count_lo += 1
            min_lo = min(min_lo, j - k)
            max_lo = max(max_lo, j - k)
        else:
            count_hi += 1
            min_hi = min(min_hi, j - k)
            max_hi = max(max_hi, j - k)
        k = j
    iso_lo = count_lo > 0 and max_lo == 1
    iso_hi = count_hi > 0 and max_hi == 1
    if iso_lo == iso_hi:  # no isolated value
        if iso_lo:  # a, a + 1, a, a + 1, ...
            return other, 1 if cur == 1 else 0
        return last, 0
    if (last == lo) == iso_lo:  # the isolated step was just taken
        return other, 1 if cur == 1 else 0
    count, short, long = (count_lo, min_lo, max_lo) if last == lo else (count_hi, min_hi, max_hi)
    if count == 0:
        return last, 0
    if cur < short:
        return last, 1
    if cur >= long:
        return other, 1 if long - short <= 1 else 0
    return last, 0


def _advance(tpos, tkind, base, count, ptr, p, last):
    """Move a scan of one row's transitions past position p; last[k] = last one of kind k."""
    while ptr < count and tpos[base + ptr] <= p:
        last[tkind[base + ptr]] = ptr
        ptr += 1
    return ptr


def _nearest(tpos, tnext, tfirst, base, slot, kind, p, last):
    """Index of the transition of this kind nearest to p (the left one on a tie), or -1."""
    left = last[kind]
    right = tnext[base + left] if left >= 0 else tfirst[3 * slot + kind]
    if left < 0 or right < 0:
        return left if right < 0 else right
    return left if p - tpos[base + left] <= tpos[base + right] - p else right


def _build_row(r, width, img, stride, line, last, hist, deltas):
    """Transitions of the decoded row r, their links to row r - 1, and the predicted next
    position of every edge track (tpred, tcert)."""
    tpos, tkind, tnext, tback, tpred, tcert, tcount, tfirst = line
    cap = width + 1
    slot = r % RING
    base = slot * cap
    pix = (r + PAD) * stride + PAD
    prev = img[pix]
    tpos[base] = 0
    tkind[base] = prev
    n = 1
    for t in range(1, width):
        v = img[pix + t]
        if v != prev:
            tpos[base + n] = t
            tkind[base + n] = v
            n += 1
            prev = v
    tpos[base + n] = width
    tkind[base + n] = 2
    n += 1
    tcount[slot] = n
    for k in range(3):
        last[k] = -1
    for j in range(n - 1, -1, -1):
        k = tkind[base + j]
        tnext[base + j] = last[k]
        last[k] = j
    for k in range(3):
        tfirst[3 * slot + k] = last[k]
    # links to the nearest transition of the same kind in row r - 1
    if r == 0:
        for j in range(n):
            tback[base + j] = -1
    else:
        up = (r - 1) % RING
        ub = up * cap
        for k in range(3):
            last[k] = -1
        ptr = 0
        for j in range(n):
            p = tpos[base + j]
            ptr = _advance(tpos, tkind, ub, tcount[up], ptr, p, last)
            tback[base + j] = _nearest(tpos, tnext, tfirst, ub, up, tkind[base + j], p, last)
    # follow each track upward and predict its next step
    for j in range(n):
        if tkind[base + j] == 2:
            tpred[base + j] = width
            tcert[base + j] = 0
            continue
        hist[0] = tpos[base + j]
        m = 1
        rr = r
        jj = j
        while rr > 0 and m <= HISTORY:
            jj = tback[(rr % RING) * cap + jj]
            if jj < 0:
                break
            rr -= 1
            hist[m] = tpos[(rr % RING) * cap + jj]
            m += 1
        for k in range(m - 1):
            deltas[k] = hist[m - 2 - k] - hist[m - 1 - k]
        step, certain = next_step(deltas, m - 1)
        tpred[base + j] = tpos[base + j] + step
        tcert[base + j] = certain


# ---------- the pixel loop ----------


def _code(
    width,
    height,
    decode,
    bits,
    data,
    out,
    trace,
    img,
    offs,
    model_a,
    model_b,
    match,
    line,
    scratch,
    log,
    squash,
    tbits,
    mbits,
):
    """Code (decode = 0) or decode (decode = 1) the mask. Encoding reads bits[i] and writes
    bytes to out; decoding reads data and writes bits[i]. Returns the number of bytes written."""
    akey, an0, an1, alb, apath, ape, apw, actx = model_a
    bkey, bn0, bn1, blb, bpath, bpe, bpw, bctx = model_b
    mkey, mpos = match
    tpos, tkind, tnext, tback, tpred, tcert, tcount, tfirst = line
    last, hist, deltas = scratch
    stride = width + 2 * PAD
    cap = width + 1
    mshift = 32 - mbits
    tracing = len(trace) > 0
    # arithmetic coder: the interval [x1, x2], and for decoding the code value xv
    x1 = 0
    x2 = 0xFFFFFFFF
    xv = 0
    pos = 0
    nout = 0
    if decode:
        for _ in range(4):
            xv = (xv << 8) | (data[pos] if pos < len(data) else 0)
            pos += 1
    small = width * height <= SMALL
    w0 = 65536  # mixer weights in 2^-16: start with model A only
    w1 = 0
    sa = sb = 0
    dist = 0  # copy predictor: offset (0 = none) and streak of correct predictions
    streak = 0
    known = 0  # line predictor: is the end E of the current run predicted, and certain?
    end = 0
    certain = 0
    ptr = 0
    up = 0
    i = 0
    for y in range(height):
        if y > 0:
            _build_row(y - 1, width, img, stride, line, last, hist, deltas)
            up = (y - 1) % RING
            for k in range(3):
                last[k] = -1
            ptr = 0
            known = 0
        row = (y + PAD) * stride + PAD
        for x in range(width):
            pix = row + x
            key = 1  # the 48 neighbours after a leading 1, so that 0 means "empty"
            for k in range(48):
                key = 2 * key + img[pix + offs[k]]
            here = past = near = cert = 0
            if known and x > 0:
                here = 1 if x == end else 0
                past = 1 if x > end else 0
                near = 1 if x - end == 1 or end - x == 1 else 0
                cert = certain
            ms = ((key & 0xFFFFFF) * MATCH_MULT[0] + (key >> 24) * MATCH_MULT[1]) & 0xFFFFFFFF
            ms = (((ms ^ (ms >> 16)) * NODE_MULT) & 0xFFFFFFFF) >> mshift
            exists = 1 if mkey[ms] == key else 0
            value = 0
            if exists:
                j = mpos[ms]
                value = img[(j // width + PAD) * stride + PAD + j % width]
                if dist == 0:
                    dist = i - j
                    streak = 0
            copy = copied = s8 = s64 = 0
            if dist > 0:
                j = i - dist
                copy = 1
                copied = img[(j // width + PAD) * stride + PAD + j % width]
                s8 = 1 if streak >= 8 else 0
                s64 = 1 if streak >= 64 else 0
            # contexts: 2 neighbours, line, match, then neighbours (B: copy bits after the 14th)
            for c in (actx, bctx):
                c[0] = (key >> 47) & 1
                c[1] = (key >> 46) & 1
                c[2] = here
                c[3] = past
                c[4] = near
                c[5] = cert
                c[6] = exists
                c[7] = value
            for k in range(2, CTW_NEIGHBOURS):
                nb = (key >> (47 - k)) & 1
                actx[k + 6] = nb
                bctx[k + 6 if k < 14 else k + 10] = nb
            bctx[20] = copy
            bctx[21] = copied
            bctx[22] = s8
            bctx[23] = s64
            if tracing:
                f = (here, past, near, cert, exists, value, copy, copied, s8, s64)
                for k in range(FEATURES):
                    trace[FEATURES * i + k] = f[k]
            if small:  # one CTW over the 24 neighbours, no mixer
                for k in range(CTW_NEIGHBOURS):
                    actx[k] = (key >> (47 - k)) & 1
                pa = _ctw_predict(
                    actx, CTW_NEIGHBOURS, akey, an0, an1, alb, tbits, apath, ape, apw, squash
                )
                p = min(max((pa + 32768) >> 16, 1), 65535)
            else:
                pa = _ctw_predict(
                    actx, DEPTH_A, akey, an0, an1, alb, tbits, apath, ape, apw, squash
                )
                pb = _ctw_predict(
                    bctx, DEPTH_B, bkey, bn0, bn1, blb, tbits, bpath, bpe, bpw, squash
                )
                sa = _stretch(pa, log)
                sb = _stretch(pb, log)
                p = _squash((w0 * sa + w1 * sb) >> 16, squash)  # P(1) * 2^16, 1..65535
            # binary arithmetic coding of the pixel
            xmid = x1 + (((x2 - x1) * p) >> 16)
            if decode:
                bit = 1 if xv <= xmid else 0
                bits[i] = bit
            else:
                bit = int(bits[i])
            if bit:
                x2 = xmid
            else:
                x1 = xmid + 1
            while ((x1 ^ x2) & 0xFF000000) == 0:
                if decode:
                    xv = ((xv << 8) & 0xFFFFFFFF) | (data[pos] if pos < len(data) else 0)
                    pos += 1
                else:
                    out[nout] = x2 >> 24
                    nout += 1
                x1 = (x1 << 8) & 0xFFFFFFFF
                x2 = ((x2 << 8) & 0xFFFFFFFF) | 255
            img[pix] = bit
            # learn
            if small:
                _ctw_update(bit, CTW_NEIGHBOURS, an0, an1, alb, apath, ape, apw, log)
            else:
                _ctw_update(bit, DEPTH_A, an0, an1, alb, apath, ape, apw, log)
                _ctw_update(bit, DEPTH_B, bn0, bn1, blb, bpath, bpe, bpw, log)
                err = (bit << 16) - p
                w0 = min(max(w0 + ((err * sa) >> MIX_SHIFT), -MIX_LIMIT), MIX_LIMIT)
                w1 = min(max(w1 + ((err * sb) >> MIX_SHIFT), -MIX_LIMIT), MIX_LIMIT)
            mkey[ms] = key
            mpos[ms] = i
            if dist > 0:
                if bit == copied:
                    streak += 1
                else:
                    streak = 0
                    dist = 0
            if y > 0 and (x == 0 or bit != img[pix - 1]):  # a run starts at x
                ptr = _advance(tpos, tkind, up * cap, tcount[up], ptr, x, last)
                s1 = _nearest(tpos, tnext, tfirst, up * cap, up, bit, x, last)
                known = 0
                if s1 >= 0:
                    end = tpred[up * cap + s1 + 1]
                    certain = tcert[up * cap + s1 + 1]
                    known = 1 if end >= 0 else 0  # a track predicted left of the row: no bits
            i += 1
    if not decode:
        # end with the value in [x1, x2] that has the most trailing zero bits
        k = 32
        while ((x2 >> k) << k) < x1:
            k -= 1
        v = (x2 >> k) << k
        for j in range(4):
            out[nout] = (v >> (24 - 8 * j)) & 255
            nout += 1
    return nout


def _buffers(width, height, zeros):
    """Everything _code works on. zeros(n, bits) makes n zero ints that need `bits` bits."""
    tbits, mbits = table_bits(width, height)
    img = zeros((height + PAD) * (width + 2 * PAD), 8)
    offs = zeros(48, 32)
    for k, (dy, dx) in enumerate(NEIGHBOURS):
        offs[k] = dy * (width + 2 * PAD) + dx

    def model(depth):
        n = 1 << tbits
        return (
            zeros(n, 64),
            zeros(n, 32),
            zeros(n, 32),
            zeros(n, 32),
            zeros(depth + 1, 64),
            zeros(depth + 1, 64),
            zeros(depth + 1, 64),
            zeros(depth, 8),
        )

    match = (zeros(1 << mbits, 64), zeros(1 << mbits, 64))
    cap = width + 1
    line = (
        zeros(RING * cap, 64),
        zeros(RING * cap, 8),
        zeros(RING * cap, 64),
        zeros(RING * cap, 64),
        zeros(RING * cap, 64),
        zeros(RING * cap, 8),
        zeros(RING, 64),
        zeros(3 * RING, 64),
    )
    scratch = (zeros(3, 64), zeros(HISTORY + 1, 64), zeros(HISTORY, 64))
    return img, offs, model(DEPTH_A), model(DEPTH_B), match, line, scratch, tbits, mbits


def _zeros(n, bits):
    return [0] * n


def encode_mask(mask, trace=None):
    """Arithmetic-coded payload of a 0/1 mask (a list of rows, or a 2-D array)."""
    height, width = len(mask), len(mask[0])
    bits = [int(v) for row in mask for v in row]
    if any(b not in (0, 1) for b in bits):
        raise ValueError("mask values must be 0 or 1")
    img, offs, a, b, match, line, scratch, tbits, mbits = _buffers(width, height, _zeros)
    out = bytearray(2 * width * height + 8)  # a pixel never costs more than 16 bits
    trace = trace if trace is not None else []
    n = _code(
        width,
        height,
        0,
        bits,
        b"",
        out,
        trace,
        img,
        offs,
        a,
        b,
        match,
        line,
        scratch,
        LOG,
        SQUASH,
        tbits,
        mbits,
    )
    return bytes(out[:n])


def decode_mask(data, width, height):
    """The mask (a list of rows) from its payload. Bytes past the end of data read as 0."""
    img, offs, a, b, match, line, scratch, tbits, mbits = _buffers(width, height, _zeros)
    bits = [0] * (width * height)
    _code(
        width,
        height,
        1,
        bits,
        data,
        b"",
        [],
        img,
        offs,
        a,
        b,
        match,
        line,
        scratch,
        LOG,
        SQUASH,
        tbits,
        mbits,
    )
    return [bits[y * width : (y + 1) * width] for y in range(height)]


# ---------- the stream ----------


def write_values(writer, lo, hi):
    """The alpha-values code: 1 = (0, 255), 00 hi = (0, hi), 010 lo = (lo, 255), 011 lo hi."""
    if (lo, hi) == (0, 255):
        writer.write(1, 1)
    elif lo == 0:
        writer.write(0b00, 2)
        writer.write(hi, 8)
    elif hi == 255:
        writer.write(0b010, 3)
        writer.write(lo, 8)
    else:
        writer.write(0b011, 3)
        writer.write(lo, 8)
        writer.write(hi, 8)


def read_values(reader):
    if reader.read(1):
        return 0, 255
    if reader.read(1) == 0:
        return 0, reader.read(8)
    if reader.read(1) == 0:
        return reader.read(8), 255
    lo = reader.read(8)
    return lo, reader.read(8)


def binary_values(alpha):
    """(lo, hi) of an alpha channel with at most two values; one value v is coded as (0, v),
    or (0, 255) with an empty mask when v = 0."""
    values = sorted({int(v) for row in alpha for v in row})
    if len(values) > 2 or not 0 <= values[0] <= values[-1] <= 255:
        raise ValueError("binary alpha needs at most two 8-bit values")
    if len(values) == 1:
        return (0, 255) if values[0] == 0 else (0, values[0])
    return values[0], values[1]


def encode(alpha):
    """A binary alpha channel (rows of 8-bit values, at most two distinct) as a stream:
    dimensions, alpha values, payload. Trailing zero bits are dropped."""
    height, width = len(alpha), len(alpha[0])
    lo, hi = binary_values(alpha)
    writer = BitWriter()
    write_dimensions(writer, width, height)
    write_values(writer, lo, hi)
    for byte in encode_mask([[1 if v == hi else 0 for v in row] for row in alpha]):
        writer.write(byte, 8)
    while writer.bits and writer.bits[-1] == 0:
        writer.bits.pop()
    return writer.to_bytes()


def decode(data):
    """Rows of alpha values from a stream made by encode."""
    reader = BitReader(data + bytes(16))  # the stream may end in the header's last zero bits
    width, height = read_dimensions(reader)
    lo, hi = read_values(reader)
    payload = bytearray()
    while reader.pos < 8 * len(data):
        payload.append(reader.read(8))
    mask = decode_mask(bytes(payload), width, height)
    return [[hi if v else lo for v in row] for row in mask]
