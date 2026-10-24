"""Bit-level reading and writing, most significant bit first."""


class BitWriter:
    """Collects bits and packs them into bytes, MSB first, zero-padded at the end."""

    def __init__(self):
        self.bits = []

    def __len__(self):
        return len(self.bits)

    def write(self, value, nbits):
        """Append value as an nbits-bit unsigned integer, most significant bit first."""
        if nbits < 0 or value < 0 or value >> nbits:
            raise ValueError(f"{value} does not fit in {nbits} bits")
        self.bits.extend((value >> i) & 1 for i in range(nbits - 1, -1, -1))

    def to_bytes(self):
        out = bytearray((len(self.bits) + 7) // 8)
        for i, bit in enumerate(self.bits):
            out[i >> 3] |= bit << (7 - (i & 7))
        return bytes(out)


class BitReader:
    """Reads bits from bytes, MSB first."""

    def __init__(self, data):
        self.data = data
        self.pos = 0

    def read(self, nbits):
        """Next nbits as an unsigned integer (0 when nbits is 0)."""
        if self.pos + nbits > 8 * len(self.data):
            raise EOFError("read past the end of the data")
        value = 0
        for _ in range(nbits):
            bit = (self.data[self.pos >> 3] >> (7 - (self.pos & 7))) & 1
            value = (value << 1) | bit
            self.pos += 1
        return value
