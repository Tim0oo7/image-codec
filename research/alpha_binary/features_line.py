"""Digital-line ("Sturmian") edge prediction.

A straight edge, followed row by row, moves by a step Δ that takes only two neighbouring values
a and a + 1, in a balanced pattern: one of the two values is isolated, and the runs of the other
have lengths L or L + 1 only. So once a track has shown a couple of complete runs, the next step
is usually forced:

    current run of the frequent step shorter than the shortest run seen  -> continue
    current run as long as the longest run seen                          -> switch
    the isolated step was just taken                                     -> switch back
    otherwise                                                            -> ambiguous

Tracks are built as in features_edge: transitions matched to the nearest same-polarity
transition in the row above. The prediction is for the transition that ends the current run.
"""

import numpy as np
from features_edge import nearest_same, transitions

HISTORY = 24  # rows of track history used


def next_step(deltas):
    """deltas: steps of one track, oldest first. Returns (predicted step, certain) or None."""
    if not deltas:
        return None
    last = deltas[-1]
    vals = set(deltas)
    if len(vals) == 1:
        return last, False  # one value so far: probably continues, not certain
    lo, hi = min(vals), max(vals)
    if hi - lo != 1 or len(vals) > 2:
        return last, False  # not a digital straight line (curve or corner)
    other = hi if last == lo else lo
    # run-length encode, newest last
    runs = []
    for d in deltas:
        if runs and runs[-1][0] == d:
            runs[-1][1] += 1
        else:
            runs.append([d, 1])
    # which value is isolated? the one whose complete runs all have length 1
    complete = runs[1:-1]  # first and last runs may be cut off
    iso = None
    for v in (lo, hi):
        lens = [n for d, n in complete if d == v]
        if lens and max(lens) == 1:
            iso = v if iso is None else None  # both length 1: alternating, no isolated value
    cur_len = runs[-1][1]
    if iso is None:
        # alternating a, a+1 (slope ~ a + 1/2): switch
        if all(n == 1 for _, n in complete) and complete:
            return other, cur_len == 1
        return last, False
    if last == iso:
        return other, cur_len == 1
    lens = [n for d, n in complete if d == last]
    if not lens:
        return last, False
    if cur_len < min(lens):
        return last, True
    if cur_len >= max(lens):
        return other, cur_len >= max(lens) and max(lens) - min(lens) <= 1
    return last, False  # cur_len == min < max: ambiguous


def predictions(m):
    """E[y, x], C[y, x]: predicted end of the current run and whether it is certain."""
    h, w = m.shape
    E = np.full((h, w), -1, dtype=np.int64)
    C = np.zeros((h, w), dtype=np.uint8)
    rows = [transitions(m[r]) for r in range(h)]
    back = [None] + [nearest_same(*rows[r], *rows[r - 1]) for r in range(1, h)]
    for y in range(1, h):
        pos1, pol1 = rows[y - 1]
        pred = pos1.copy()
        cert = np.zeros(pos1.size, dtype=np.uint8)
        for i in range(pos1.size):
            if pol1[i] == 2:
                continue
            hist = [int(pos1[i])]
            r, j = y - 1, i
            while r > 0 and len(hist) <= HISTORY:
                j = back[r][j]
                if j < 0:
                    break
                r -= 1
                hist.append(int(rows[r][0][j]))
            deltas = [hist[k] - hist[k + 1] for k in range(len(hist) - 1)][::-1]
            res = next_step(deltas)
            if res is not None:
                pred[i] = pos1[i] + res[0]
                cert[i] = res[1]
        pos, pol = transitions(m[y])
        xs = np.arange(1, w)
        idx = np.searchsorted(pos, xs - 1, side="right") - 1
        j = nearest_same(pos[idx], pol[idx], pos1, pol1)
        valid = (j >= 0) & (j + 1 < pos1.size)
        e = np.full(xs.size, -1, dtype=np.int64)
        c = np.zeros(xs.size, dtype=np.uint8)
        e[valid] = pred[j[valid] + 1]
        c[valid] = cert[j[valid] + 1]
        E[y, 1:] = e
        C[y, 1:] = c
    return E, C


def line_planes(m, certain_bit=True, near=True):
    E, C = predictions(m)
    x = np.arange(m.shape[1])[None, :]
    known = E >= 0
    planes = [(known & (x == E)).astype(np.uint8), (known & (x > E)).astype(np.uint8)]
    if near:
        planes.append((known & (np.abs(x - E) == 1)).astype(np.uint8))
    if certain_bit:
        planes.append(C)
    return planes


def line(m):
    return line_planes(m)


def line_nocert(m):
    return line_planes(m, certain_bit=False)
