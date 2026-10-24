"""Image dimensions header field. Specified in docs/format-spec.md, "Image dimensions".

The long side L is coded as its length n (number of binary digits of L - 1) and those digits,
the short side S as a shape: square, one of eight aspect ratios, or free.
"""

MAX_SIDE = 1 << 32
SMALL_N = 8  # n <= 8 means L <= 256

# (name, p, q, has_exact_bit). The short side is S = round(L * q / p), halves rounded up.
RATIOS = [
    ("4:3", 4, 3, True),
    ("3:2", 3, 2, False),
    ("16:9", 16, 9, True),
    ("16:10", 16, 10, True),
    ("5:4", 5, 4, True),
    ("2:1", 2, 1, False),
    ("A4", 99, 70, False),
    ("20:9", 20, 9, True),
]

# Kind codes as (value, nbits), for small (L <= 256) and large images.
KINDS = {
    True: {"square": (0b0, 1), "free": (0b10, 2), "ratio": (0b11, 2)},
    False: {"ratio": (0b0, 1), "free": (0b10, 2), "square": (0b11, 2)},
}


def rounded_short_side(L, p, q):
    """round(L * q / p) with halves rounded up, in integer arithmetic."""
    return (2 * L * q + p) // (2 * p)


def multiples(n, p):
    """(k_min, count): the k for which p * k is a long side of length n."""
    lo, hi = (1 << (n - 1)) + 1, 1 << n
    k_min = -(-lo // p)
    return k_min, hi // p - k_min + 1


def truncated_binary(x, m):
    """Fields coding x in [0, m) with floor(log2 m) or ceil(log2 m) bits."""
    if m == 1:
        return []
    k = (m - 1).bit_length()
    u = (1 << k) - m
    return [(x, k - 1)] if x < u else [(x + u, k)]


def read_truncated_binary(reader, m):
    if m == 1:
        return 0
    k = (m - 1).bit_length()
    u = (1 << k) - m
    x = reader.read(k - 1)
    return x if x < u else ((x << 1) | reader.read(1)) - u


def encodings(width, height):
    """Every valid encoding of (width, height), each as a list of (value, nbits) fields."""
    if not (1 <= width <= MAX_SIDE and 1 <= height <= MAX_SIDE):
        raise ValueError(f"dimensions {width}x{height} out of range")
    L, S = max(width, height), min(width, height)
    portrait = int(height > width)
    n = (L - 1).bit_length()
    head = [(n, 4)] if n < 15 else [(15, 4), (n - 15, 5)]
    if L == 1:
        return [head]
    kind = KINDS[n <= SMALL_N]
    digits = [(L - 1 - (1 << (n - 1)), n - 1)]
    if L == S:
        return [head + [kind["square"]] + digits]
    out = []
    for index, (_, p, q, has_exact) in enumerate(RATIOS):
        if rounded_short_side(L, p, q) != S:
            continue
        shape = head + [kind["ratio"], (index, 3), (portrait, 1)]
        if not has_exact:
            out.append(shape + digits)
        elif L % p:
            out.append(shape + [(0, 1)] + digits)
        else:
            k_min, count = multiples(n, p)
            out.append(shape + [(1, 1)] + truncated_binary(L // p - k_min, count))
    out.append(head + [kind["free"], (portrait, 1)] + digits + truncated_binary(S - 1, L - 1))
    return out


def shortest(width, height):
    """The shortest encoding (the first one on ties)."""
    return min(encodings(width, height), key=lambda fields: sum(nbits for _, nbits in fields))


def dimension_bits(width, height):
    return sum(nbits for _, nbits in shortest(width, height))


def write_dimensions(writer, width, height):
    for value, nbits in shortest(width, height):
        writer.write(value, nbits)


def read_dimensions(reader):
    """Decode (width, height)."""
    n = reader.read(4)
    if n == 15:
        n += reader.read(5)
        if n > 32:
            raise ValueError("long side exceeds 2**32")
    if n == 0:
        return 1, 1
    small = n <= SMALL_N
    if reader.read(1) == 0:
        kind = "square" if small else "ratio"
    elif reader.read(1) == 0:
        kind = "free"
    else:
        kind = "ratio" if small else "square"
    lo = (1 << (n - 1)) + 1
    if kind == "square":
        L = lo + reader.read(n - 1)
        return L, L
    if kind == "ratio":
        _, p, q, has_exact = RATIOS[reader.read(3)]
        portrait = reader.read(1)
        if has_exact and reader.read(1):
            k_min, count = multiples(n, p)
            if count < 1:
                raise ValueError(f"no multiple of {p} has length {n}")
            k = k_min + read_truncated_binary(reader, count)
            L, S = p * k, q * k
        else:
            L = lo + reader.read(n - 1)
            S = rounded_short_side(L, p, q)
    else:
        portrait = reader.read(1)
        L = lo + reader.read(n - 1)
        S = read_truncated_binary(reader, L - 1) + 1
    return (S, L) if portrait else (L, S)
