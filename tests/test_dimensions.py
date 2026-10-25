import random

import pytest

from image_codec.bitio import BitReader, BitWriter
from image_codec.dimensions import (
    MAX_SIDE,
    dimension_bits,
    encodings,
    read_dimensions,
    write_dimensions,
)


def roundtrip(width, height):
    writer = BitWriter()
    write_dimensions(writer, width, height)
    reader = BitReader(writer.to_bytes())
    assert read_dimensions(reader) == (width, height)
    assert reader.pos == len(writer)


def test_all_small_sizes():
    for width in range(1, 300):
        for height in range(1, 300):
            roundtrip(width, height)


def test_every_encoding_decodes():
    for width in range(1, 70):
        for height in range(1, 70):
            for fields in encodings(width, height):
                writer = BitWriter()
                for value, nbits in fields:
                    writer.write(value, nbits)
                assert read_dimensions(BitReader(writer.to_bytes())) == (width, height)


SIZES = [
    (1920, 1080),
    (1280, 720),
    (3840, 2160),
    (2560, 1440),
    (4032, 3024),
    (6000, 4000),
    (2048, 1365),
    (1366, 768),
    (854, 480),
    (1000, 563),
    (2880, 1800),
    (1080, 1350),
    (1080, 2400),
    (1440, 3200),
    (2480, 3508),
    (595, 842),
    (5760, 2880),
    (1170, 2532),
    (137, 291),
    (728, 90),
    (16384, 1),
    (16385, 16385),
    (40000, 5000),
    (MAX_SIDE, MAX_SIDE),
    (MAX_SIDE, 1),
]


@pytest.mark.parametrize("width, height", SIZES)
def test_sizes(width, height):
    roundtrip(width, height)
    roundtrip(height, width)


def test_random_sizes():
    rng = random.Random(1)
    for _ in range(3000):
        long = rng.randint(1, 1 << rng.randint(1, 32))
        roundtrip(long, rng.randint(1, long))
        roundtrip(rng.randint(1, long), long)


@pytest.mark.parametrize(
    "width, height, bits",
    [
        (1, 1, 4),
        (16, 16, 8),
        (32, 32, 9),
        (44, 44, 10),
        (256, 256, 12),
        (1920, 1080, 16),
        (1080, 1920, 16),
        (3840, 2160, 17),
        (2880, 1800, 17),
        (1080, 2400, 16),
        (4032, 3024, 19),
        (1024, 768, 17),
        (1080, 1350, 18),
        (1366, 768, 20),
        (6000, 4000, 21),
        (2048, 1365, 19),
        (2480, 3508, 20),
        (137, 291, 23),
        (1170, 2532, 29),
    ],
)
def test_bits(width, height, bits):
    assert dimension_bits(width, height) == bits


@pytest.mark.parametrize(
    "width, height, fields",
    [
        (32, 32, "0101 0 1111"),
        (1920, 1080, "1011 0 010 0 1 110111"),
        (2048, 1365, "1011 0 001 0 1111111111"),
        (137, 291, "1001 10 1 00100010 10001000"),
    ],
)
def test_spec_examples(width, height, fields):
    writer = BitWriter()
    write_dimensions(writer, width, height)
    assert "".join(map(str, writer.bits)) == fields.replace(" ", "")


@pytest.mark.parametrize("width, height", [(0, 1), (1, 0), (MAX_SIDE + 1, 1)])
def test_out_of_range(width, height):
    with pytest.raises(ValueError):
        write_dimensions(BitWriter(), width, height)
