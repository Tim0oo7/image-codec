"""A real binary-alpha coder: sequential CTW in a bounded hash table + arithmetic coding.

This turns ideal code lengths into actual bytes:
- CTW nodes live in a fixed-size, direct-mapped table (2^TABLE_BITS entries). A collision
  resets the slot, so statistics are really lost, as in a real decoder.
- The probability of a 1 is rounded to 16 bits (1..65535) before coding.
- A 32-bit binary arithmetic coder (carry-less, LPAQ-style) writes the bytes, and the stream
  ends with the fewest bytes that still decode correctly.
- decode() runs the mirror image and must reproduce every pixel.

Model arithmetic is still floating point (fine for this measurement, not for a spec).
The context bits are passed in: their causality is verified separately (simulate_decoder.py).
"""

import math

import numpy as np
from numba import njit

TABLE_BITS = 22
ALPHA = 0.25
LN2 = math.log(2.0)


@njit(cache=True)
def _slot(depth, prefix, mask):
    key = (np.int64(depth) << np.int64(40)) | np.int64(prefix)
    hsh = (np.uint64(key) * np.uint64(0x9E3779B97F4A7C15)) >> np.uint64(64 - 30)
    return np.int64(hsh) & mask, key


@njit(cache=True)
def _find(tkey, tn0, tn1, tpe, tpw, depth, prefix, mask, create):
    s, key = _slot(depth, prefix, mask)
    if tkey[s] != key:
        if not create:
            return -1
        tkey[s] = key
        tn0[s] = 0
        tn1[s] = 0
        tpe[s] = 0.0
        tpw[s] = 0.0
    return s


@njit(cache=True)
def _code(bits, x, table_bits, alpha, decode, data):
    n, depth = bits.shape
    size = 1 << table_bits
    mask = np.int64(size - 1)
    tkey = np.full(size, -1, np.int64)
    tn0 = np.zeros(size, np.int32)
    tn1 = np.zeros(size, np.int32)
    tpe = np.zeros(size, np.float64)
    tpw = np.zeros(size, np.float64)
    out = np.zeros(n // 4 + 64, np.uint8)
    nout = 0
    x1 = np.uint32(0)
    x2 = np.uint32(0xFFFFFFFF)
    xv = np.uint32(0)
    pos = 0
    if decode:
        for _ in range(4):
            b = data[pos] if pos < data.size else 0
            pos += 1
            xv = np.uint32((xv << np.uint32(8)) | np.uint32(b))
    decoded = np.zeros(n, np.uint8)
    path = np.zeros(depth + 1, np.int64)
    prefix = np.zeros(depth + 1, np.int64)
    sib = np.zeros(depth + 1, np.int64)
    pw1 = np.zeros(depth + 1, np.float64)
    for i in range(n):
        p = np.int64(0)
        prefix[0] = 0
        for d in range(depth):
            p = p * 2 + bits[i, d]
            prefix[d + 1] = p
        for d in range(depth + 1):
            path[d] = _find(tkey, tn0, tn1, tpe, tpw, d, prefix[d], mask, True)
        for d in range(1, depth + 1):
            sib[d] = _find(tkey, tn0, tn1, tpe, tpw, d, prefix[d] ^ 1, mask, False)
        # conditional probability of a 1, bottom-up
        s = path[depth]
        pw1[depth] = (tn1[s] + alpha) / (tn0[s] + tn1[s] + 2 * alpha)
        for d in range(depth - 1, -1, -1):
            s = path[d]
            pe1 = (tn1[s] + alpha) / (tn0[s] + tn1[s] + 2 * alpha)
            lsib = tpw[sib[d + 1]] if sib[d + 1] >= 0 else 0.0
            t = tpw[path[d + 1]] + lsib - tpe[s]  # ln(Pw children / Pe)
            if t > 50.0:
                w = 0.0
            elif t < -50.0:
                w = 1.0
            else:
                w = 1.0 / (1.0 + math.exp(t))
            pw1[d] = w * pe1 + (1.0 - w) * pw1[d + 1]
        q = np.int64(pw1[0] * 65536.0 + 0.5)
        q = min(max(q, 1), 65535)
        # arithmetic coding of the bit
        rng = np.uint64(x2 - x1)
        xmid = np.uint32(np.uint64(x1) + ((rng * np.uint64(q)) >> np.uint64(16)))
        if decode:
            sym = 1 if xv <= xmid else 0
        else:
            sym = x[i]
        decoded[i] = sym
        if sym:
            x2 = xmid
        else:
            x1 = np.uint32(xmid + np.uint32(1))
        while ((x1 ^ x2) & np.uint32(0xFF000000)) == 0:
            if decode:
                b = data[pos] if pos < data.size else 0
                pos += 1
                xv = np.uint32((xv << np.uint32(8)) | np.uint32(b))
            else:
                if nout >= out.size:
                    out = np.concatenate((out, np.zeros(out.size, np.uint8)))
                out[nout] = np.uint8(x2 >> np.uint32(24))
                nout += 1
            x1 = np.uint32(x1 << np.uint32(8))
            x2 = np.uint32((x2 << np.uint32(8)) | np.uint32(255))
        # update counts and weighted probabilities, bottom-up
        for d in range(depth, -1, -1):
            s = path[d]
            n0, n1 = tn0[s], tn1[s]
            if sym:
                tpe[s] += math.log((n1 + alpha) / (n0 + n1 + 2 * alpha))
                tn1[s] = n1 + 1
            else:
                tpe[s] += math.log((n0 + alpha) / (n0 + n1 + 2 * alpha))
                tn0[s] = n0 + 1
            if d == depth:
                tpw[s] = tpe[s]
            else:
                lsib = tpw[sib[d + 1]] if sib[d + 1] >= 0 else 0.0
                a = tpe[s]
                b = tpw[path[d + 1]] + lsib
                m = max(a, b)
                tpw[s] = m + math.log(math.exp(a - m) + math.exp(b - m)) - LN2
    if not decode:
        # shortest flush: k bytes v with x1 <= v (followed by zeros) <= x2
        for k in range(1, 5):
            unit = np.uint64(1) << np.uint64(32 - 8 * k)
            v = ((np.uint64(x1) + unit - np.uint64(1)) // unit) * unit
            if v <= np.uint64(x2):
                for j in range(k):
                    if nout >= out.size:
                        out = np.concatenate((out, np.zeros(out.size, np.uint8)))
                    out[nout] = np.uint8((v >> np.uint64(24 - 8 * j)) & np.uint64(255))
                    nout += 1
                break
    return out[:nout], decoded


def encode(bits, x, table_bits=TABLE_BITS, alpha=ALPHA):
    b = np.ascontiguousarray(np.stack(bits, axis=1).astype(np.int64))
    data, _ = _code(b, x.astype(np.uint8), table_bits, alpha, False, np.zeros(1, np.uint8))
    return data


def decode(bits, n, data, table_bits=TABLE_BITS, alpha=ALPHA):
    b = np.ascontiguousarray(np.stack(bits, axis=1).astype(np.int64))
    _, dec = _code(b, np.zeros(n, np.uint8), table_bits, alpha, True, data)
    return dec
