"""The reference codec against the research code: same bytes compiled or not, the same context
features as the research predictors, and close to realconfigs.real_mix in size."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

numba = pytest.importorskip("numba")
sys.path.insert(0, str(Path(__file__).parent))
import features_combo  # noqa: E402
import features_lz  # noqa: E402
import fixedcodec  # noqa: E402
import realconfigs  # noqa: E402
from harness import load  # noqa: E402

from image_codec import alpha_binary  # noqa: E402

CORPUS = Path(__file__).resolve().parents[2] / "corpus"


@pytest.fixture(scope="module")
def codec():
    return fixedcodec.Codec()


def synthetic():
    rng = np.random.default_rng(7)
    m = (rng.random((60, 70)) < 0.1).astype(np.uint8)
    yy, xx = np.mgrid[0:60, 0:70]
    m |= ((xx - 30) ** 2 / 400 + (yy - 28) ** 2 / 200 < 1).astype(np.uint8)
    m[5:20, 40:66] = np.tile(np.eye(5, dtype=np.uint8), (3, 6))[:, :26]
    return m


def corpus_masks(n, max_pixels):
    path = CORPUS / "split_dev_quick.json"
    if not path.exists():
        pytest.skip("no local corpus")
    items = [it for it in json.load(open(path)) if it["w"] * it["h"] <= max_pixels]
    masks = []
    for it in items[:: max(1, len(items) // n)][:n]:
        try:
            masks.append(load(it)[0])
        except OSError:
            pass
    return masks


def test_compiled_equals_pure(codec):
    m = synthetic()
    data = codec.encode_mask(m)
    assert data == alpha_binary.encode_mask(m.tolist())
    assert np.array_equal(codec.decode_mask(data, m.shape[1], m.shape[0]), m)


def check_features(m):
    """The traced context features equal the research predictors (whose causality is checked
    by harness.check_causal). The match table is made large, so that collisions are unlikely."""
    big = fixedcodec.Codec(MATCH_MIN=24, MATCH_MAX=24)
    data, trace = big.encode_mask(m, trace=True)
    planes = features_combo.line_match48(m) + features_lz.lz48(m)
    for k, plane in enumerate(planes):
        assert np.array_equal(trace[k], np.asarray(plane).ravel()), k
    assert np.array_equal(big.decode_mask(data, m.shape[1], m.shape[0]), m)


def test_features_synthetic():
    check_features(synthetic())


def test_corpus_roundtrip_and_features():
    for m in corpus_masks(12, 120_000):
        check_features(m)


def test_corpus_pure_python(codec):
    for m in corpus_masks(3, 3000):
        data = alpha_binary.encode_mask(m.tolist())
        assert data == codec.encode_mask(m)
        assert alpha_binary.decode_mask(data, m.shape[1], m.shape[0]) == m.tolist()


def test_size_against_real_mix(codec):
    """Within 1 % of the floating-point coder, both with realconfigs' header bits."""
    masks = corpus_masks(40, 300_000)
    ours = sum(fixedcodec.file_bits(m, codec.encode_mask(m)) for m in masks)
    ref = sum(realconfigs.real_mix(m) for m in masks)
    assert ours < 1.01 * ref
