"""Edge-track features: predict where the current run ends from the boundary's slope above.

Every row r has a list of transitions T_r: position 0 (row start, polarity m[r, 0]), every t with
m[r, t] != m[r, t - 1] (polarity m[r, t]), and the sentinel W (polarity 2). A transition is
matched to the nearest transition with the same polarity in the row above, which links
transitions into edge tracks.

For pixel (y, x) the current run in row y started at s, the last transition before x. It is
matched to its counterpart s1 in row y - 1. The transition after s1 in row y - 1 ends that run; its
track over the last k rows gives a slope, and extrapolating one row gives E, the predicted end of
the current run. Everything used is decoded before (y, x).
"""

import numpy as np


def transitions(row):
    w = row.size
    t = np.flatnonzero(row[1:] != row[:-1]) + 1
    pos = np.concatenate(([0], t, [w]))
    pol = np.concatenate(([row[0]], row[t], [2])).astype(np.int8)
    return pos, pol


def nearest_same(pos_a, pol_a, pos_b, pol_b):
    """For each transition in a, index of the nearest same-polarity transition in b (-1: none)."""
    out = np.full(pos_a.size, -1, dtype=np.int64)
    for p in (0, 1, 2):
        ia = np.flatnonzero(pol_a == p)
        ib = np.flatnonzero(pol_b == p)
        if ia.size == 0 or ib.size == 0:
            continue
        cand = pos_b[ib]
        k = np.searchsorted(cand, pos_a[ia])
        left = np.clip(k - 1, 0, cand.size - 1)
        right = np.clip(k, 0, cand.size - 1)
        pick = np.where(
            np.abs(cand[left] - pos_a[ia]) <= np.abs(cand[right] - pos_a[ia]), left, right
        )
        out[ia] = ib[pick]
    return out


def predicted_ends(m, k):
    """E[y, x]: predicted end of the run containing (y, x - 1); -1 where unknown."""
    h, w = m.shape
    E = np.full((h, w), -1, dtype=np.int64)
    rows = [transitions(m[r]) for r in range(h)]
    # back[r][i]: index in row r - 1 of the track predecessor of transition i of row r
    back = [None] + [nearest_same(*rows[r], *rows[r - 1]) for r in range(1, h)]
    for y in range(1, h):
        pos1, pol1 = rows[y - 1]
        # extrapolated position in row y of every transition of row y - 1
        pred = pos1.astype(np.float64).copy()
        cur = np.arange(pos1.size)
        depth = np.zeros(pos1.size, dtype=np.int64)
        anc = pos1.astype(np.float64).copy()
        for step in range(1, k + 1):
            r = y - 1 - step
            if r < 0:
                break
            ok = cur >= 0
            nxt = np.full_like(cur, -1)
            nxt[ok] = back[r + 1][cur[ok]]
            ok &= nxt >= 0
            anc[ok] = rows[r][0][nxt[ok]]
            depth[ok] = step
            cur = np.where(ok, nxt, -1)
        has = depth > 0
        pred[has] = pos1[has] + (pos1[has] - anc[has]) / depth[has]
        pred = np.floor(pred + 0.5).astype(np.int64)
        pred[pol1 == 2] = w  # the row end stays the row end
        # current row: run start s for each x >= 1, matched into row y - 1
        pos, pol = transitions(m[y])
        xs = np.arange(1, w)
        idx = np.searchsorted(pos, xs - 1, side="right") - 1
        s_pos, s_pol = pos[idx], pol[idx]
        j = nearest_same(s_pos, s_pol, pos1, pol1)
        valid = (j >= 0) & (j + 1 < pos1.size)
        e = np.full(xs.size, -1, dtype=np.int64)
        e[valid] = pred[j[valid] + 1]
        E[y, 1:] = e
    return E


def _planes(m, k, near):
    E = predicted_ends(m, k)
    x = np.arange(m.shape[1])[None, :]
    known = E >= 0
    here = (known & (x == E)).astype(np.uint8)
    past = (known & (x > E)).astype(np.uint8)
    planes = [here, past]
    if near:
        planes.append((known & (np.abs(x - E) == 1)).astype(np.uint8))
    return planes


def edge_k1(m):
    return _planes(m, 1, False)


def edge_k4(m):
    return _planes(m, 4, False)


def edge_k8(m):
    return _planes(m, 8, False)


def edge_k4_near(m):
    return _planes(m, 4, True)


def edge_k1_near(m):
    return _planes(m, 1, True)
