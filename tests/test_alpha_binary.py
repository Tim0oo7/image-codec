import math
import random

import pytest

from image_codec import alpha_binary as ab


def roundtrip_mask(mask):
    data = ab.encode_mask(mask)
    assert ab.decode_mask(data, len(mask[0]), len(mask)) == [list(row) for row in mask]
    return data


def rect(w, h, x0, y0, x1, y1):
    return [[int(x0 <= x < x1 and y0 <= y < y1) for x in range(w)] for y in range(h)]


def disc(w, h, cx, cy, r):
    return [[int((x - cx) ** 2 + (y - cy) ** 2 < r * r) for x in range(w)] for y in range(h)]


def noise(w, h, p, seed):
    rng = random.Random(seed)
    return [[int(rng.random() < p) for _ in range(w)] for _ in range(h)]


def polygon(w, h):
    """A triangle with edges of several slopes, so the line predictor has work to do."""
    return [
        [int(y > 0.4 * x + 3 and y < 70 - 1.7 * x and 2 * y > 75 - x) for x in range(w)]
        for y in range(h)
    ]


def sprites(w, h):
    """The same 8x8 sprite repeated, so the match and copy predictors have work to do."""
    sprite = [
        [int((x - 3.5) ** 2 + (y - 3.5) ** 2 < 12 and (x, y) != (2, 3)) for x in range(8)]
        for y in range(8)
    ]
    return [
        [sprite[y % 8][x % 8] if (x // 8 + y // 8) % 3 else 0 for x in range(w)] for y in range(h)
    ]


MASKS = {
    "1x1 zero": [[0]],
    "1x1 one": [[1]],
    "row": [[1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0]],
    "column": [[v] for v in (0, 1, 1, 0, 1, 0, 0, 0, 1)],
    "empty": [[0] * 23 for _ in range(17)],
    "full": [[1] * 23 for _ in range(17)],
    "rectangle": rect(30, 20, 4, 3, 25, 15),
    "disc": disc(40, 40, 19.5, 21, 15),
    "polygon": polygon(48, 48),
    "checkerboard": [[(x + y) % 2 for x in range(16)] for y in range(16)],
    "stripes": [[(x // 3) % 2 for x in range(30)] for y in range(10)],
    "noise": noise(24, 24, 0.5, 1),
    "sparse noise": noise(40, 30, 0.03, 2),
    "sprites": sprites(48, 40),
}


@pytest.mark.parametrize("name", MASKS)
def test_mask_roundtrip(name):
    roundtrip_mask(MASKS[name])


def test_compression():
    """Simple shapes cost a few bytes; 20 copies of a sprite cost much less than 20 sprites."""
    assert len(roundtrip_mask(MASKS["empty"])) <= 4
    assert len(roundtrip_mask(MASKS["rectangle"])) <= 12
    assert len(roundtrip_mask(MASKS["disc"])) <= 40
    assert len(roundtrip_mask(MASKS["noise"])) <= 24 * 24 / 8 + 12
    one = len(roundtrip_mask(sprites(16, 8)))
    assert len(roundtrip_mask(MASKS["sprites"])) < 6 * one


def test_mask_values():
    with pytest.raises(ValueError):
        ab.encode_mask([[0, 2]])


@pytest.mark.parametrize(
    "lo, hi, bits",
    [
        (0, 255, "1"),
        (0, 7, "00 00000111"),
        (9, 255, "010 00001001"),
        (3, 200, "011 00000011 11001000"),
        (255, 0, "011 11111111 00000000"),
    ],
)
def test_values_code(lo, hi, bits):
    writer = ab.BitWriter()
    ab.write_values(writer, lo, hi)
    assert "".join(map(str, writer.bits)) == bits.replace(" ", "")
    assert ab.read_values(ab.BitReader(writer.to_bytes())) == (lo, hi)


@pytest.mark.parametrize("lo, hi", [(0, 255), (0, 128), (17, 255), (17, 128)])
def test_stream_roundtrip(lo, hi):
    mask = MASKS["disc"]
    alpha = [[hi if v else lo for v in row] for row in mask]
    data = ab.encode(alpha)
    assert ab.decode(data) == alpha
    assert data[-1] != 0  # trailing zero bits are dropped


@pytest.mark.parametrize("value", [0, 1, 254, 255])
def test_constant_alpha(value):
    alpha = [[value] * 5 for _ in range(3)]
    assert ab.decode(ab.encode(alpha)) == alpha


def test_stream_size():
    """Dimensions (10 bits for 40x40) + 1 bit for (0, 255) + the payload."""
    alpha = [[255 * v for v in row] for row in MASKS["disc"]]
    payload = ab.encode_mask(MASKS["disc"])
    assert len(ab.encode(alpha)) <= math.ceil((10 + 1 + 8 * len(payload)) / 8)


def test_too_many_values():
    with pytest.raises(ValueError):
        ab.encode([[0, 1, 2]])


def test_neighbours():
    """The first neighbours, nearest first (docs/format-spec.md, figure)."""
    assert ab.NEIGHBOURS[:12] == [
        (-1, 0),
        (0, -1),
        (-1, -1),
        (-1, 1),
        (-2, 0),
        (0, -2),
        (-2, -1),
        (-2, 1),
        (-1, -2),
        (-1, 2),
        (-2, -2),
        (-2, 2),
    ]
    assert len(set(ab.NEIGHBOURS)) == 48
    assert max(max(-dy, abs(dx)) for dy, dx in ab.NEIGHBOURS) == ab.PAD


def test_tables_are_platform_independent():
    """Every table entry is far from a rounding boundary, so any accurate log2/pow gives it."""
    for i in range(4097):
        v = 65536 * math.log2(1 + i / 4096)
        assert abs(v - math.floor(v) - 0.5) > 1e-6
    for i in range(-4096, 4097):
        v = 65536 / (1 + 2 ** (-i / 256))
        assert abs(v - math.floor(v) - 0.5) > 1e-6
    assert ab.LOG[0] == 0 and ab.LOG[4096] == 65536
    for i in range(4097):
        assert ab.SQUASH[4096 + i] + ab.SQUASH[4096 - i] == 65536


def test_log2():
    for p in [1, 2, 3, 1000, 1 << 20, 123456789, (1 << 31) - 1, 1 << 31, (1 << 32) - 1]:
        assert abs(ab._log2(p, ab.LOG) / 65536 - math.log2(p / 2**32)) < 3e-5
    rng = random.Random(3)
    for _ in range(2000):
        p = rng.randint(1, (1 << 32) - 1)
        assert abs(ab._log2(p, ab.LOG) / 65536 - math.log2(p / 2**32)) < 3e-5


@pytest.mark.parametrize(
    "deltas, expect",
    [
        ([], (0, 0)),
        ([0, 1, 0, 1, 0, 1], (0, 1)),  # steep line, alternating
        ([2, 2, 3, 2, 2, 3, 2, 3, 2, 2], (3, 1)),  # shallow line, run complete
        ([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0], (0, 0)),  # ambiguous run length
        ([0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], (1, 1)),  # longest run reached
        ([0, 0, 1, 0, 0, 0, 1, 0], (0, 1)),  # shorter than every complete run
        ([0, 0, 1, 0, 0, 0, 1], (0, 1)),  # the isolated step was just taken
        ([0, 0, 0, 0], (0, 0)),  # vertical edge
        ([0, 0, 0, 5], (5, 0)),  # corner
    ],
)
def test_next_step(deltas, expect):
    """The worked examples of docs/alpha-binary-line-predictor.md and a few more."""
    assert ab.next_step(deltas, len(deltas)) == expect
