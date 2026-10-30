"""Tests for the feature harness: the causality check must catch leaks."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from harness import check_causal, check_planes, cost  # noqa: E402


def west(m):
    out = np.zeros_like(m)
    out[:, 1:] = m[:, :-1]
    return [out]


def leaky_current(m):
    return [m.copy()]


def leaky_row_sum(m):
    # whole-row information: depends on pixels to the right of (y, x)
    return [np.repeat((m.sum(axis=1, keepdims=True) > 0).astype(np.uint8), m.shape[1], axis=1)]


def sample():
    rng = np.random.default_rng(3)
    m = (rng.random((20, 25)) < 0.4).astype(np.uint8)
    m[4:12, 6:18] = 1
    return m


def test_causal_feature_passes():
    check_causal(west, sample())


@pytest.mark.parametrize("fn", [leaky_current, leaky_row_sum])
def test_leaks_are_caught(fn):
    with pytest.raises(AssertionError):
        check_causal(fn, sample(), trials=40)


def test_cost_matches_neighbours_only_when_feature_is_redundant():
    m = sample()
    check_planes(west(m), m)
    # the west neighbour is already the first neighbour, so a duplicate bit costs a little but
    # must stay close to the plain model
    plain = cost(m, [], 24)
    dup = cost(m, west(m), 24)
    assert dup >= plain - 1e-6
    assert dup < plain * 1.1
