"""Round trip of the real coder, and its size against the ideal CTW code length."""

import sys
from pathlib import Path

import numpy as np
import pytest

numba = pytest.importorskip("numba")
sys.path.insert(0, str(Path(__file__).parent))
import realcoder  # noqa: E402
from harness import neighbour_bits  # noqa: E402
from idealcost import ctw_bits  # noqa: E402


def sample():
    rng = np.random.default_rng(5)
    m = (rng.random((40, 50)) < 0.15).astype(np.uint8)
    m[8:30, 10:42] = 1
    m[15:20, 20:30] = 0
    return m


def test_round_trip_and_size():
    m = sample()
    bits = neighbour_bits(m, 12)
    data = realcoder.encode(bits, m.ravel())
    assert np.array_equal(realcoder.decode(bits, m.size, data), m.ravel())
    ideal = ctw_bits(m.ravel(), bits, 12, realcoder.ALPHA) / 8
    assert ideal <= len(data) <= ideal + 4
