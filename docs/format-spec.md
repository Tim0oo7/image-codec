# Format specification (draft)

Nothing here is final.

## Container / header

### Image dimensions

Width W and height H, with 1 ≤ W, H ≤ 2³². Design notes and the comparison with other formats
are in [dimensions.md](dimensions.md).

The field codes the long side L = max(W, H) and a shape, from which the short side S = min(W, H)
follows. Fields are written most significant bit first, in this order:

| # | Field | Bits | Present when |
|---|---|---|---|
| 1 | n, the number of binary digits of L − 1 | 4. If they read 15: 5 more bits e, and n = 15 + e | always |
| 2 | kind | 1 or 2, table below | n ≥ 1 |
| 3 | ratio index | 3 | kind is ratio |
| 4 | portrait | 1 | kind is ratio or free |
| 5 | exact | 1 | kind is ratio and the ratio has an exact bit |
| 6 | value of L | below | n ≥ 1 |
| 7 | short side | truncated binary over L − 1 values | kind is free |

n = 0 means a 1×1 image, and nothing follows. n > 32 is invalid. An image is *small* when
n ≤ 8, which is the same as L ≤ 256.

**Kind**

| Code | Small (n ≤ 8) | Large (n ≥ 9) |
|---|---|---|
| `0` | square | ratio |
| `10` | free | free |
| `11` | ratio | square |

**Ratios**

| Index | Ratio | p | q | Exact bit |
|---|---|---|---|---|
| 0 | 4:3 | 4 | 3 | yes |
| 1 | 3:2 | 3 | 2 | no |
| 2 | 16:9 | 16 | 9 | yes |
| 3 | 16:10 | 16 | 10 | yes |
| 4 | 5:4 | 5 | 4 | yes |
| 5 | 2:1 | 2 | 1 | no |
| 6 | A4 | 99 | 70 | no |
| 7 | 20:9 | 20 | 9 | yes |

**Value of L**

- Exact bit present and 1: L = p·k. With k_min = ⌈(2ⁿ⁻¹ + 1) / p⌉ and
  count = ⌊2ⁿ / p⌋ − k_min + 1, k = k_min + t, where t is coded in truncated binary over count
  values. count = 0 is invalid.
- Otherwise: the n − 1 binary digits of L − 1 after its leading 1, so L = 2ⁿ⁻¹ + 1 + digits.

**Short side**

| Kind | S |
|---|---|
| square | L |
| ratio, exact | q·k |
| ratio, otherwise | ⌊(2·L·q + p) / (2·p)⌋, which is L·q/p rounded with halves up |
| free | 1 + t, with t in truncated binary over L − 1 values |

portrait = 0 gives W = L, H = S. portrait = 1 gives W = S, H = L.

**Truncated binary** of t in [0, m): with k = ⌈log₂ m⌉ and u = 2ᵏ − m, t < u is written in
k − 1 bits and t ≥ u as t + u in k bits. m = 1 takes no bits.

**Encoder.** Every encoding that decodes to (W, H) is valid. The reference encoder
(`src/image_codec/dimensions.py`) writes the shortest, and on ties the first of: the ratios in
index order, then free.

**Examples**

| W×H | Fields | Bits |
|---|---|---|
| 32×32 | `0101` `0` `1111` | 9 |
| 1920×1080 | `1011` `0` `010` `0` `1` `110111` | 16 |
| 2048×1365 | `1011` `0` `001` `0` `1111111111` | 19 |
| 137×291 | `1001` `10` `1` `00100010` `10001000` | 23 |

## Color transforms

## Prediction

## Entropy coding

## Alpha channel

- Constant alpha: header flag + value, no payload.
- Binary alpha: below.
- Whether alpha is coded before or after the colour channels, and how the alpha mode is
  signalled, is still open.

### Binary alpha

Reference implementation: `src/image_codec/alpha_binary.py`. The design and its measurements
are in [alpha-binary.md](alpha-binary.md). All arithmetic below is on integers; no intermediate
value reaches 2⁶³, `>>` rounds down (also for negative numbers), and `clamp(v, a, b)` is
min(max(v, a), b).

**Stream.** A binary alpha stream is, without any alignment in between:

1. the image dimensions W × H (above);
2. the two alpha values lo and hi: `1` = (0, 255), `00` + hi = (0, hi), `010` + lo = (lo, 255),
   `011` + lo + hi otherwise, each value in 8 bits;
3. the payload: the arithmetic-coded mask, read 8 bits at a time. Reading past the end of the
   stream gives 0 bits, so an encoder may drop trailing zero bits.

The mask m has m = 1 where alpha = hi and 0 where alpha = lo. Its pixels are decoded in raster
order, index i = y·W + x. Every context bit depends only on pixels decoded before, and pixels
outside the image read as 0.

**Decoding a pixel.** For every pixel:

1. compute the context bits (neighbours, line, match and copy predictors);
2. compute P(1) of model A (30 context bits) and model B (34 bits), two CTWs;
3. mix them into one 16-bit probability p;
4. decode the bit with p;
5. update both CTWs, the mixer, the match table and the copy predictor.

**Small masks** (W·H ≤ 1024) use a single CTW over the 24 neighbours nb₀ … nb₂₃ (D = 24,
model A's table) and no mixer: p = clamp((P + 2¹⁵) >> 16, 1, 65535). The match table and the
copy predictor are kept up to date as below, but their bits are not used. Larger masks use the
full model. The size rule and why it is at 1,024 pixels: [alpha-binary.md](alpha-binary.md) §3.4.

#### Context bits

**Neighbours.** The 48 nearest decoded neighbours, numbered in this order (· is the current
pixel):

```
            44 40 34 41 45
         35 30 24 22 25 31 36
      37 28 18 14 12 15 19 29 38
   46 32 20 10  6  4  7 11 21 33 47
   42 26 16  8  2  0  3  9 17 27 43
   39 23 13  5  1  ·
```

That is: by distance dx² + dy², then by |dx|, then the left one first.
nb_k is the value of neighbour k (0 outside the image).

**Line predictor** (bits here, past, near, certain). Defined in
[alpha-binary-line-predictor.md](alpha-binary-line-predictor.md): transitions, links (nearest
transition of the same kind in the row above, the left one on a tie), the track history of up to
24 steps, and next_step. A decoder computes the transitions, links and predictions of row y − 1
once, when row y starts. Additions to that text:

- The bits are 0 at x = 0, in row 0, when the run start s has no link, and when the predicted
  end E is negative.
- For the row-end transition, E = W and certain = 0 (so near = 1 at x = W − 1).

**Match predictor** (bits exists, value). The match key is
K = 2⁴⁸ + Σₖ nb_k · 2⁴⁷⁻ᵏ (the 48 neighbours after a leading 1). A table of 2^mbits slots holds
(key, position) pairs, all 0 at the start. The slot of K is

```
h    = ((K mod 2²⁴) · 0x9E3779B1 + (K >> 24) · 0x85EBCA77) mod 2³²
h    = ((h xor (h >> 16)) · 0x2C1B3C6D) mod 2³²
slot = h >> (32 − mbits)
```

exists = (key in the slot = K), and then value = the mask at the slot's position, else 0. After
the pixel is decoded, the slot gets (K, i), whatever it held before.

**Copy predictor** (bits copy, copied, streak8, streak64). State: an offset d (0 = none, at the
start) and a streak count.

```
if exists and d = 0:  d = i − position from the match slot, streak = 0
if d > 0:  copy = 1, copied = m[i − d], streak8 = (streak ≥ 8), streak64 = (streak ≥ 64)
else:      all four bits 0
after the pixel:  if d > 0:  if bit = copied: streak += 1,  else: streak = 0, d = 0
```

**Context order.** Model A uses 30 bits and model B 34, in this order:

| Model | Context bits |
|---|---|
| A | nb₀ nb₁ · here past near certain · exists value · nb₂ … nb₂₃ |
| B | nb₀ nb₁ · here past near certain · exists value · nb₂ … nb₁₃ · copy copied streak8 streak64 · nb₁₄ … nb₂₃ |

#### Tables

Two tables, defined by formulas. Every entry is at least 10⁻⁶ away from a rounding boundary, so
any log2 and pow accurate to 10⁻¹² give the same table (the tests check this).

```
LOG[i]           = round(2¹⁶ · log2(1 + i / 4096)),                 i = 0 … 4096
SQUASH[i + 4096] = clamp(round(2¹⁶ / (1 + 2^(−i / 256))), 1, 65535),  i = −4096 … 4096
```

```
log2fix(p):  for 1 ≤ p < 2³², 2¹⁶ · log2(p / 2³²)
    e = ⌊log2 p⌋                      (0 … 31)
    v = p << (31 − e)                 (2³¹ ≤ v < 2³²)
    j = (v >> 19) − 4096,  f = v mod 2¹⁹
    return (e − 32) · 2¹⁶ + LOG[j] + ((LOG[j + 1] − LOG[j]) · f >> 19)

squash(t):   for t in 2⁻¹⁶ bit, P(1) · 2¹⁶ of the log-odds t
    return SQUASH[clamp((t + 128) >> 8, −4096, 4096) + 4096]

stretch(p):  for a CTW probability p (P(1) · 2³²)
    return clamp(log2fix(p) − log2fix(2³² − p), −20 · 2¹⁶, 20 · 2¹⁶)
```

#### The two CTW models

Both models work the same way, with their own node table and context length D (30 for A, 34
for B). The table has 2^tbits slots of (key, n0, n1, lb), all 0 at the start. With N = W·H and
bitlen(N) the number of binary digits of N:

```
tbits = clamp(bitlen(N) + 5, 12, 24)
mbits = clamp(bitlen(N) + 3, 8, 22)        (the match table)
```

A node needs 35 bits of key, two 16-bit counts and a 22-bit lb, so the largest tables take about
2 × 2²⁴ × 12 bytes. Capping tbits at 20 instead of 24 would cost 0.08 % on the test corpus
(up to 6 % on dense text), see [alpha-binary.md](alpha-binary.md) §4.1d.

**Path.** Node d (d = 0 … D) is the context's first d bits c₀ … c_{d−1}. Its key is those bits
after a leading 1, and its slot comes from a hash along the path:

```
h = 0, key = 1
for d = 0 … D:
    slot_d = h >> (32 − tbits)
    if the key in slot_d ≠ key:  slot_d := (key, 0, 0, 0)      (a collision resets the slot)
    if d < D:
        key = 2 · key + c_d
        h = (((h xor (h >> 15)) + c_d + 1) · 0x2C1B3C6D) mod 2³²
```

**Probability.** A node holds counts n0, n1 and lb = log2 β in 2⁻¹⁶ bit, where
β = Pe(node) / (Pw(child 0) · Pw(child 1)) as in CTW, so squash(lb) is the node's weight
β / (1 + β). From the deepest node up:

```
for d = D … 0:
    pe_d = ((4 · n1 + 1) << 32) / (4 · (n0 + n1) + 2)         (integer division; α = 1/4)
    pw_d = pe_d                                  if d = D
    pw_d = pw_{d+1} + ((pe_d − pw_{d+1}) · squash(lb) >> 16)    otherwise
P(1) = pw_0  (· 2⁻³²)
```

**Update** with the decoded bit b, for every node d on the path:

```
if d < D:
    lb = clamp(lb + L(pe_d) − L(pw_{d+1}), −24 · 2¹⁶, 24 · 2¹⁶)
    where L(q) = log2fix(q) if b = 1, log2fix(2³² − q) if b = 0
n_b = n_b + 1
if n0 + n1 = 65535:  n0 = (n0 + 1) >> 1,  n1 = (n1 + 1) >> 1
```

If two nodes of one path share a slot, the later one has reset it and both steps act on the same
slot, in the order above.

#### Mixer and coder

Two weights, w0 = 65536 and w1 = 0 at the start (in 2⁻¹⁶):

```
sA = stretch(P_A),  sB = stretch(P_B)
p  = squash((w0 · sA + w1 · sB) >> 16)                     (P(1) · 2¹⁶, 1 … 65535)
after the pixel:  err = b · 2¹⁶ − p
                  w0 = clamp(w0 + (err · sA >> 23), −2²⁰, 2²⁰)
                  w1 = clamp(w1 + (err · sB >> 23), −2²⁰, 2²⁰)
```

The binary arithmetic decoder keeps an interval [x1, x2] and a code value v, 32 bits each.
At the start x1 = 0, x2 = 2³² − 1, and v is the first 4 payload bytes. For each pixel:

```
xmid = x1 + ((x2 − x1) · p >> 16)
b = 1 if v ≤ xmid else 0
if b = 1:  x2 = xmid   else:  x1 = xmid + 1
while x1 and x2 have the same top byte:
    x1 = (x1 << 8) mod 2³²,  x2 = ((x2 << 8) mod 2³²) + 255,  v = ((v << 8) mod 2³²) + next byte
```

**Encoder.** It runs the same model and, for each pixel, narrows [x1, x2] in the same way,
writing the top byte of x2 at each shift. At the end it writes any 4-byte value in [x1, x2]; the
reference encoder takes the one with the most trailing zero bits and then drops all trailing
zero bits of the stream.

## Metadata
