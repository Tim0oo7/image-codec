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
