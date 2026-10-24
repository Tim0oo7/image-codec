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
- Binary alpha: see [alpha-binary.md](alpha-binary.md).
- Proposed: alpha is coded after the YCoCg channels, so its model can use the decoded colour
  (including a colour-key mode with no alpha payload). See alpha-binary.md §6.

## Metadata
