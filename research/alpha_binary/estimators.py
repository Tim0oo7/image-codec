"""Cost functions for estimator experiments (Dirichlet alpha of CTW)."""

import features_combo
import features_line
import features_match
from harness import cost


def lm48_a32(m):
    return cost(m, features_combo.line_match48(m), 24, alpha=1 / 32, at=2)


def lm48_a8(m):
    return cost(m, features_combo.line_match48(m), 24, alpha=1 / 8, at=2)


def lm48_a4(m):
    return cost(m, features_combo.line_match48(m), 24, alpha=1 / 4, at=2)


def lm48_a2(m):
    return cost(m, features_combo.line_match48(m), 24, alpha=1 / 2, at=2)


def base_a4(m):
    return cost(m, [], 24, alpha=1 / 4)


def line_a4(m):
    return cost(m, features_line.line(m), 24, alpha=1 / 4, at=2)


def m48_a4(m):
    return cost(m, features_match.match48(m), 24, alpha=1 / 4, at=2)


# ---- v4 candidates: current predictor (line + match48) plus extra decoder-inferred bits ----
import features_sym  # noqa: E402


def _v3(m):
    return features_combo.line_match48(m)


def v3_h(m):
    return cost(m, _v3(m) + features_sym.hmirror(m), 24, alpha=1 / 4, at=2)


def v3_v(m):
    return cost(m, _v3(m) + features_sym.vmirror(m), 24, alpha=1 / 4, at=2)


def v3_c(m):
    return cost(m, _v3(m) + features_sym.rowcopy(m), 24, alpha=1 / 4, at=2)


def v3_hv(m):
    return cost(
        m, _v3(m) + features_sym.hmirror(m) + features_sym.vmirror(m), 24, alpha=1 / 4, at=2
    )


import features_lz  # noqa: E402


def v3_lz(m):
    return cost(m, _v3(m) + features_lz.lz48(m), 24, alpha=1 / 4, at=2)


def v3_lz_h(m):
    return cost(m, _v3(m) + features_lz.lz48(m) + features_sym.hmirror(m), 24, alpha=1 / 4, at=2)


def v3_lz32(m):
    return cost(m, _v3(m) + features_lz.lz32(m), 24, alpha=1 / 4, at=2)


# ---- explicit context layouts: neighbours and feature planes interleaved ----
from harness import neighbour_bits  # noqa: E402
from idealcost import ctw_bits  # noqa: E402


def layout_cost(m, layout, alpha=1 / 4):
    """layout: list of int (take that many next-nearest neighbours) or lists of feature planes."""
    nb = neighbour_bits(m, 64)
    bits, k = [], 0
    for part in layout:
        if isinstance(part, int):
            bits += nb[k : k + part]
            k += part
        else:
            bits += [np.asarray(p, dtype=np.uint8).ravel() for p in part]
    return ctw_bits(m.ravel(), bits, len(bits), alpha)


import numpy as np  # noqa: E402


def _lz_compact(m):
    has, val, s8, _ = features_lz.lz48(m)
    return [s8, val]


def lzc_at2(m):
    return layout_cost(m, [2, _v3(m) + _lz_compact(m), 22])


def lzc_at8(m):
    return layout_cost(m, [2, _v3(m), 6, _lz_compact(m), 16])


def lz_at8(m):
    return layout_cost(m, [2, _v3(m), 6, features_lz.lz48(m), 16])


def lz_at14(m):
    return layout_cost(m, [2, _v3(m), 12, features_lz.lz48(m), 10])


def _fused(m):
    """Line bits + 2 copy bits: the LZ copy when it is on a streak, else the context match."""
    line = features_combo.features_line.line(m)
    mex, mval = features_combo.features_match.match48(m)
    has, val, s8, _ = features_lz.lz48(m)
    use = s8.astype(bool)
    ex = np.where(use, 1, mex).astype(np.uint8)
    pv = np.where(use, val, mval).astype(np.uint8)
    return line + [ex, pv]


def fused(m):
    return layout_cost(m, [2, _fused(m), 22])


def fused_conf(m):
    has, val, s8, _ = features_lz.lz48(m)
    return layout_cost(m, [2, _fused(m) + [s8], 22])


def v3_curve(m):
    f = features_combo.features_line.line_curve(m) + features_combo.features_match.match48(m)
    return cost(m, f, 24, alpha=1 / 4, at=2)


# ---- mixing two complete models (dev only) ----
import realcoder  # noqa: E402
import realconfigs  # noqa: E402


def _probs_v3_v4(m):
    x = m.ravel()
    nb = neighbour_bits(m, 24)
    f = [np.asarray(p, dtype=np.uint8).ravel() for p in features_combo.line_match48(m)]
    pa = realcoder.probabilities(nb[:2] + f + nb[2:], x, 24)
    pb = realcoder.probabilities(realconfigs.v4_bits(m), x, 24)
    return x, pa, pb


def ctw_v3_seq(m):
    x, pa, _ = _probs_v3_v4(m)
    pa = np.clip(pa, 1 / 65536, 1 - 1 / 65536)
    return float(-np.sum(np.where(x == 1, np.log2(pa), np.log2(1 - pa))))


def mix_v3_v4(m):
    x, pa, pb = _probs_v3_v4(m)
    has, val, s8, s64 = features_lz.lz48(m)
    sel = (s8.ravel().astype(np.int64) + s64.ravel()).astype(np.int64)  # 0, 1, 2
    return realcoder.mix_bits(x.astype(np.int64), pa, pb, sel, 3, 0.02)


def mix_v3_v4_1set(m):
    x, pa, pb = _probs_v3_v4(m)
    return realcoder.mix_bits(x.astype(np.int64), pa, pb, np.zeros(x.size, np.int64), 1, 0.02)
