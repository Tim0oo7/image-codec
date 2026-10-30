"""Symmetry and row-repetition features, all inferred by the decoder from decoded rows.

hmirror  If a recent row is mirror-symmetric (its boundaries pair up around an axis), assume the
         current row shares that axis. Once x is past the axis, the mirror pixel is decoded.
         Bits: [mirror pixel available], [its value].
vmirror  If the last rows mirror earlier rows (row y-1 = row j, row y-2 = row j+1, ...), predict
         row y from row j-1. Bits: [available], [value].
rowcopy  If the last rows repeat earlier rows (row y-1 = row j, row y-2 = row j-1, ...), predict
         row y from row j+1. Bits: [available], [value].
"""

import numpy as np

AXIS_LOOKBACK = 8  # rows searched for a symmetric row


def boundaries(row):
    """Pixel-boundary positions where the value changes, with the image border counting as 0."""
    padded = np.concatenate(([0], row, [0]))
    return np.flatnonzero(padded[1:] != padded[:-1])  # boundary t lies between pixels t-1 and t


def axis_sum(row):
    """S such that boundary t mirrors to S - t, if the row is symmetric with >= 2 boundaries."""
    t = boundaries(row)
    if t.size < 2:
        return None
    s = t + t[::-1]
    return int(s[0]) if np.all(s == s[0]) else None


def hmirror(m):
    h, w = m.shape
    avail = np.zeros_like(m)
    value = np.zeros_like(m)
    axes = [axis_sum(m[r]) for r in range(h)]
    for y in range(1, h):
        S = None
        for r in range(y - 1, max(-1, y - 1 - AXIS_LOOKBACK), -1):
            if axes[r] is not None:
                S = axes[r]
                break
        if S is None:
            continue
        x = np.arange(w)
        xm = S - 1 - x
        ok = (xm >= 0) & (xm < x)
        avail[y, ok] = 1
        value[y, ok] = m[y, xm[ok]]
    return [avail, value]


def _row_ids(m):
    _, ids = np.unique(m, axis=0, return_inverse=True)
    return ids.ravel()


def _row_predict(m, mirror):
    h, w = m.shape
    avail = np.zeros_like(m)
    value = np.zeros_like(m)
    ids = _row_ids(m)
    where = {}
    for y in range(h):
        if y >= 3:
            best, best_len = None, 1
            for j in reversed(where.get(ids[y - 1], [])[-64:]):
                if j > y - 3:
                    continue
                src = j - 1 if mirror else j + 1
                if src < 0 or src >= y:
                    continue
                # length of the agreement going back from y-1
                k = 1
                while k < 16 and y - 1 - k >= 0:
                    jj = j + k if mirror else j - k
                    if jj < 0 or jj >= y - 1 - k or ids[y - 1 - k] != ids[jj]:
                        break
                    k += 1
                if k >= 2 and (k > best_len or (k == best_len and best is not None and j > best)):
                    best, best_len = j, k
            if best is not None:
                src = best - 1 if mirror else best + 1
                avail[y] = 1
                value[y] = m[src]
        where.setdefault(ids[y], []).append(y)
    return [avail, value]


def vmirror(m):
    return _row_predict(m, True)


def rowcopy(m):
    return _row_predict(m, False)
