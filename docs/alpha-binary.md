# Binary alpha: the prediction model

Status: research v3.2 (2026-10-30). Nothing here is final. Every number can be reproduced with the
scripts in [research/alpha_binary](../research/alpha_binary/README.md). v2 of this document
(format survey, colour-key ideas, first measurements) is in the git history.

This document is only about **prediction**: the probability p that the next pixel is opaque,
given everything decoded so far. The cost of a pixel is −log₂ p of what it really is, and an
arithmetic coder reaches that within a few bytes per image, so the coder itself is out of scope.

## Summary

- **What's measured:** 2,130 binary masks from four kinds of real assets. These are pixel art,
  icon-font glyphs rendered without anti-aliasing, app icons and GIFs from this machine, and
  vector shapes. The vector shapes come from the QOI benchmark suite's icons, cutouts and plants,
  thresholded at 50 % coverage.
- **Where the bits go:** 96 % of pixels are flat and cost almost nothing. Nearly half of all
  bits are spent on *staircase steps*, the pixels where an edge moves sideways.
- **The proposed predictor** adds small, side-information-free models to the context of a CTW
  model, and mixes two such models:
  - a **line predictor** that follows each edge upward and predicts its next step from the
    digital-straight-line pattern;
  - a **match predictor** that copies what followed the same 48-pixel neighbourhood last time;
  - a second CTW that also sees an **LZ-style copy predictor**, and an online two-weight mixer
    that learns per image how much to trust it.
- **Real files, untouched test set:** a working coder (sequential CTW in a bounded hash table,
  16-bit probabilities, arithmetic coding, the dimensions field, the alpha header) was run on
  a test split that no design decision was made on (§4.0). In total it is **2.13× smaller than
  JPEG XL, 2.6× smaller than WebP, 4.2× smaller than PNG, 6.0× smaller than AVIF and 1.37×
  smaller than an idealised JBIG2 without headers**. Per mask, it is smaller than JPEG XL, PNG
  and AVIF on all 1,421, and smaller than WebP on 1,398 (WebP's wins are animation sprite strips).
- **Compared with v2:** 15.8 % smaller as real files on the test set (same headers).
- **What's stored:** the dimensions field, then an alpha-values code that is a single bit for
  the usual pair (0, 255), then the pixels. No mode flags, tables, offsets, colour keys or
  encoder choices. The decoder infers everything from pixels it already has.
- **Checked end to end:** a pixel-by-pixel decoder simulation reproduces the measured cost
  exactly, so the predictor is causal (decodable) and the numbers aren't an artefact.

## 1. Corpus

Binary alpha means exactly two alpha values. Downloaded sets are git-ignored (`corpus/`) and the
file lists of private images never leave the machine.

| Category | Source | Masks | Notes |
|---|---|---|---|
| `pixelart` | Kenney CC0 packs: 1-Bit, Pixel Platformer, Tiny Dungeon, Micro Roguelike, Pixel Shmup, Pixel UI | 369 | Sprites, tiles and tile sheets |
| `glyphs` | Material Icons, Font Awesome solid + regular, rendered without anti-aliasing at 24, 64 and 192 px | 400 of 1,960 | Clean vector cutouts. Colour carries no information |
| `system` | `C:\Windows`, `Program Files` | 400 of 1,417 | Application icons, GIFs |
| `user` | `C:\Users\<user>` (app data, browser caches) | 378 of 715 | Real-world web and app images |
| `icons_thr` | QOI suite `icon_64`, `icon_512`, alpha ≥ 128 | 380 | Derived (see below) |
| `cutouts_thr` | QOI suite `pngimg` (product cutouts), alpha ≥ 128 | 141 | Derived, up to 2.1 Mpx |
| `plants_thr` | QOI suite `textures_plants`, alpha ≥ 128 | 54 | Derived, alpha-tested foliage |
| `misc` | the 9 truly binary QOI images | 8 | |

How common binary alpha is: of the 21,919 distinct images with transparency on this machine, 2,221
(10 %) are binary. The QOI benchmark suite has almost none: its icons, cutouts and textures use
anti-aliased alpha. That's why the `*_thr` categories exist. Thresholding coverage-based alpha at
50 % gives, for straight edges, about the same pixels as rasterising the same vector shape
without anti-aliasing. They are labelled *derived* everywhere.

Images per category are capped at 400, chosen by hash. "Category-balanced" averages the
per-category ratios, so the many small icons don't hide the large shapes or the other way round.

## 2. Where the bits go

The per-pixel cost of an adaptive 16-neighbour model, split by the pixel's 3×3 geometry (analysis
only, `where_bits.py`):

| Geometry | Share of pixels | Share of bits | Bits per pixel |
|---|---|---|---|
| Flat (all 8 neighbours equal) | 96.5 % | 12 % | 0.002 |
| Straight horizontal / vertical edge | 1.4 % | 13 % | 0.14 / 0.22 |
| Diagonal step | 0.6 % | 11 % | 0.36 |
| Staircase corner (the edge moves) | 0.7 % | **46 %** | 1.29 |
| 1-px lines and spikes | 0.2 % | 11 % | 1.08 |
| Isolated pixels | 0.01 % | 2 % | 3.99 |

So the question that decides the size is: **where does the edge step next?** That's what the
line predictor is for. In pixel art the bits are spread more evenly (lines 27 %, corners 29 %,
diagonals 16 %), which is where repetition (the match predictor) helps.

## 3. The predictor

Pixels are coded in raster order. Pixels outside the image read as 0. The probability comes from
CTW (context-tree weighting) over a context of 30 bits, in this order:

```
 2 nearest neighbours          W, N
 4 line-predictor bits         here, past, near, certain
 2 match-predictor bits        exists, value
22 further neighbours          nearest first (Euclidean distance, ties: left, then up)
```

CTW weighs every context depth from 0 to 30 at once, so small images use short contexts and large
images use long ones, with nothing to tune. Every node uses the Dirichlet estimator with α = 1/4:
p(1) = (n₁ + ¼) / (n₀ + n₁ + ½).

### 3.1 Line predictor (edges)

Every row has a list of **transitions**: position 0 with the row's first value, every x where the
value changes, and the row end W. Each transition is linked to the nearest transition of the same
kind (0→1, 1→0, or end) in the row above, which chains transitions into **edge tracks**.

While decoding pixel (y, x):

1. The current run in row y started at s, the last transition before x.
2. s is linked to its counterpart in row y − 1. The *next* transition in row y − 1 is where that
   run ended one row up. Its track gives the steps Δ (horizontal moves per row) of the last 24
   rows.
3. A straight digital edge only ever steps by two neighbouring amounts, a and a + 1. One of them
   is isolated, and the runs of the other are L or L + 1 long. So the next step is predicted as:

| Situation | Prediction | Certain |
|---|---|---|
| The isolated step was just taken | the other step | yes |
| Current run of the frequent step is shorter than any complete run | same step | yes |
| Current run is as long as the longest complete run | the other step | yes (if runs differ by ≤ 1) |
| Only one step value seen, or not a straight line (curve, corner) | same step | no |

4. E = the edge's position in row y − 1 plus the predicted step. The four bits are:
   **here** = (x = E), **past** = (x > E), **near** = (|x − E| = 1), and **certain**.

The tracker keeps no state of its own: it only reads rows that are already decoded.

### 3.2 Match predictor (repetition)

The 48 nearest causal neighbours form a key. The decoder keeps a table *key → value of the pixel
that followed this key most recently*. The two bits are **exists** (the key was seen before) and
**value**. This copies repeated tiles, sprites, glyph parts and dither patterns without stored
offsets, like LZ77 but per pixel. In a real decoder the table is a fixed-size hash table; the
measurement uses exact keys, and collisions in a reasonably sized table cost very little.

### 3.3 Mixing in a copy model

A second CTW uses a longer context: the same 30 bits with 4 more inserted after the 14th
neighbour. They come from an **LZ-style copy predictor**:

- it keeps a copy offset d (in raster positions) and a streak, the number of pixels in a row that
  `x[i − d]` predicted correctly;
- when a prediction fails, d is replaced by the offset to the most recent pixel with the same
  48-neighbour context;
- the bits are [offset set], [x[i − d]], [streak ≥ 8] and [streak ≥ 64].

Both CTWs give a probability, and the coded probability is

```
p = squash(w0 · st(p_a) + w1 · st(p_b)),   st(p) = ln(p / (1 − p)),   squash = st⁻¹
after the pixel:  w_k += 0.02 · (x − p) · st(p_k),     starting from w0 = 1, w1 = 0
```

So each image starts with the plain model and moves weight to the copy model only if the copy
model is actually better there. As a fixed context bit, the copy predictor won on sprite strips
but lost elsewhere and didn't generalise (§4.3). Mixed, it keeps the wins and loses at most
0.6 % on any category. Cost: a second CTW pass.

### 3.4 What is stored

| Field | Size |
|---|---|
| The two alpha values (lo, hi) | `1` = (0, max) · `00` + hi · `010` + lo (hi = max) · `011` + lo + hi |
| Arithmetic-coded pixels | — |

Of the 4,564 binary-alpha images found, 96.7 % use (0, 255), 98.4 % have lo = 0 and 98.2 % have
hi = 255, so the values cost about 1.3 bits on average instead of 16. That saves about 2 bytes
per image: 8 % on a set of small pixel-art tiles.

The exact line-predictor definition, with tie-breaking and worked examples, is in
[alpha-binary-line-predictor.md](alpha-binary-line-predictor.md).

Nothing else. In particular, these items from v2 are **dropped**:

| v2 element | Why it's gone |
|---|---|
| Colour-key mode and colour context bits | Colour is a separate decision for when the YCoCg coder exists. The predictor has to be strong on its own: the glyph set has no colour information at all |
| Encoder-chosen context order, AT pixels, transpose, repeat offsets, dither period, upscale factor | Side information and spec complexity. CTW + match cover most of what they were for |
| Mixer + SSE stage | One CTW over a well-ordered context does the job and is easier to specify |
| Raw fallback mode | CTW's worst-case overhead on random data is a few bytes |
| A trained starting table in the spec | Measured: only 1.2 % on top of CTW (§4.3). Not worth a 4,096-entry table in the spec |

## 4. Measurements

### 4.0 Method: development and test split

After the review of v3, the corpus was split by **source object** (`split.py`): all renderings
of a glyph, a whole Kenney pack, the same icon at 64 and 512 px, and one folder of app/cache
images always land on the same side. Every design decision after v3 was made on the dev split.
The test split (1,421 masks: cutouts 70, glyphs 300, icons 194, pixel art 260 from two packs,
plants 21, system 300, user 271, misc 5) was only used for the final numbers in §4.1c. The v3
design itself was chosen before the split, on data that overlaps both halves, so its test
numbers may be slightly optimistic.

Exact ideal code lengths (`harness.py`, from per-context counts). Bytes, totals per category, and
the ratio to the v2 plan (CTW over 24 neighbours, α = 1/16).

### 4.1 The predictor, full set (2,130 masks)

| Category | Masks | v2 plan | + α = 1/4 | + line | + match | **+ line + match** |
|---|---|---|---|---|---|---|
| cutouts_thr | 141 | 90,738 | 0.995 | 0.909 | 0.962 | **0.894** |
| glyphs | 400 | 29,200 | 0.989 | 0.936 | 1.005 | **0.916** |
| icons_thr | 380 | 38,978 | 0.998 | 0.813 | 0.925 | **0.771** |
| misc | 8 | 9,750 | 0.992 | 0.980 | 0.990 | **0.984** |
| pixelart | 369 | 69,616 | 0.978 | 0.979 | 0.933 | **0.935** |
| plants_thr | 54 | 173,200 | 0.993 | 0.957 | 0.981 | **0.957** |
| system | 400 | 15,395 | 0.984 | 1.010 | 0.972 | **0.994** |
| user | 378 | 58,676 | 0.995 | 0.973 | 0.875 | **0.870** |
| **All** | 2,130 | 485,553 | 0.991 | 0.943 | 0.954 | **0.916** |
| Category-balanced | | | 0.990 | 0.945 | 0.955 | **0.915** |

The two predictors are complementary: the line predictor works on shapes (icons, cutouts, glyphs),
and the match predictor on repetition (pixel art, app and web images).

### 4.1b Against existing formats

The same masks coded on their own as 1-bit images. Our sizes are ideal costs plus a 4-byte
allowance per image for the alpha header. The codecs' sizes include their own container, which
matters for the smallest icons.

| Category | Masks | New predictor | JPEG XL e9 | WebP | PNG | JXL / new | New smaller than JXL |
|---|---|---|---|---|---|---|---|
| cutouts_thr | 141 | 81,701 | 178,419 | 262,970 | 436,702 | 2.18 | 141/141 |
| glyphs | 400 | 28,351 | 59,661 | 83,782 | 141,032 | 2.10 | 400/400 |
| icons_thr | 380 | 31,567 | 89,354 | 132,886 | 263,007 | 2.83 | 380/380 |
| misc | 8 | 9,626 | 13,903 | 21,882 | 31,007 | 1.44 | 8/8 |
| pixelart | 369 | 66,568 | 98,278 | 97,782 | 124,171 | 1.48 | 369/369 |
| plants_thr | 54 | 165,932 | 251,590 | 433,314 | 607,259 | 1.52 | 54/54 |
| system | 400 | 16,904 | 36,241 | 42,658 | 68,976 | 2.14 | 400/400 |
| user | 378 | 52,552 | 104,015 | 88,062 | 129,098 | 1.98 | 378/378 |
| **All** | 2,130 | **453,201** | 831,461 | 1,163,336 | 1,801,252 | **1.83** | **2,130/2,130** |

Pixel art is the closest race: there WebP's LZ77 is as good as JPEG XL, because tile sheets repeat.

### 4.1c Real files on the test split

`realcoder.py` codes for real: sequential CTW in a direct-mapped table of 2²⁴ nodes (a collision
resets the slot), the probability rounded to 16 bits, and a 32-bit binary arithmetic coder that
ends with the fewest bytes that still decode. Files include the dimensions field
(`image_codec.dimensions`) and the alpha-values code, and the round trip is checked (CTW
streams in full; mixed streams with the two models' probabilities passed in). Model arithmetic
is still floating point. The table is for v3.2 (mixed). On the same files, v3 without the copy
model totals 219,988 bytes, 3.4 % more.

| Category | Masks | Real file (bytes) | JPEG XL | WebP | PNG | AVIF | JBIG2, ideal, no header |
|---|---|---|---|---|---|---|---|
| cutouts_thr | 70 | 50,928 | 2.07× (70/70) | 3.05× (70/70) | 4.89× (70/70) | 7.89× (70/70) | 1.24× (70/70) |
| glyphs | 300 | 22,477 | 2.12× (300/300) | 3.00× (300/300) | 5.02× (300/300) | 5.80× (300/300) | 1.31× (299/300) |
| icons_thr | 194 | 14,778 | 2.95× (194/194) | 4.34× (194/194) | 8.60× (194/194) | 12.78× (194/194) | 1.56× (194/194) |
| misc | 5 | 8,007 | 1.43× (5/5) | 2.25× (5/5) | 3.19× (5/5) | 2.43× (5/5) | 1.10× (5/5) |
| pixelart | 260 | 5,737 | 2.59× (260/260) | 3.20× (260/260) | 5.28× (260/260) | 14.56× (260/260) | 1.24× (259/260) |
| plants_thr | 21 | 47,649 | 1.60× (21/21) | 2.79× (21/21) | 3.84× (21/21) | 3.06× (21/21) | 1.15× (21/21) |
| system | 300 | 14,627 | 2.09× (300/300) | 2.47× (300/300) | 3.95× (300/300) | 10.59× (300/300) | 1.24× (277/300) |
| user | 271 | 48,352 | 2.54× (271/271) | 1.43× (248/271) | 1.99× (271/271) | 3.24× (271/271) | 1.79× (265/271) |
| **All** | 1,421 | 212,555 | 2.13× (1,421/1,421) | 2.64× (1,398/1,421) | 4.15× (1,421/1,421) | 6.03× (1,421/1,421) | 1.37× (1,390/1,421) |

Each cell is the other format's total divided by ours, with the number of masks where ours is
smaller. JBIG2 is the generic-region template 0 with an ideal adaptive estimator and **no
header at all**, so it is flattered. Its 31 wins are tiny icons where our dimensions and alpha
header outweigh the difference (model against model it wins on 5 masks, by 5 bytes in total).

Real against ideal: the real files are 3.6 % above the ideal cost in total. That covers headers,
termination, 16-bit probabilities and table collisions, and it rises to 10–18 % on sets of tiny
icons, where the fixed header matters most. A 2²² table instead of 2²⁴ costs up to 36 % on the
most complex mask (text), so the node table needs care in the spec (§6).

### 4.2 Synthetic shapes: how close to the theory

Bits per boundary pixel (pairs of differing 4-neighbours), CTW over 24 neighbours:

| Mask | v2 plan | + edge step = last step | + line predictor |
|---|---|---|---|
| Random polygons, 1024² | 0.86 | 0.62 | **0.41** |
| Rotated ellipse, 1024² | 0.99 | 0.74 | **0.60** |
| Organic blobs, 1024² | 1.16 | 1.17 | 1.18 |

A digital straight segment of length n is one of about n³/π² possibilities (Mignosi's count of
Sturmian factors), so long straight edges need only ~0.2 bits per boundary pixel. The line
predictor halves the cost on polygons and gets about halfway to that bound. Irregular blobs gain
nothing, as expected: their edges carry real information.

### 4.3 Things that were measured and rejected

| Idea | Result | Verdict |
|---|---|---|
| Bigger neighbourhood (32 or 48 neighbours instead of 24) | −1 to −2 %, mostly on large shapes | The match predictor gets the long-range benefit more cheaply |
| Trained starting table (prior) for CTW, 12 or 16 bits | −1.2 % held out (−0.8 % with 16 bits) | Not worth a table in the spec |
| Same, for a plain 16-pixel model without CTW | −12 % | Shows CTW already solves most of the learning cost on small images |
| Edge step from more history, averaged (k = 4, 8 rows) | worse than the last step | The run-length (Sturmian) rule replaces it |
| Match key of 80 or 120 neighbours | worse than 48 on small images | 48 kept |
| Feature bits first in the context (before W, N) | 0.4–0.8 % worse | after 2 neighbours |
| **Dev split from here on** | | |
| Left-right mirror (axis from a symmetric row above) | −3 % glyphs, ±0 elsewhere, −0.4 % balanced | Not worth 2 bits |
| Top-bottom mirror, row repetition | ±0 | Dropped |
| LZ-style copy with a streak counter (keeps an offset while it predicts right) | Sprite strip 928 → 236 bytes, text 3,889 → 1,637. Dev −1.5 % (ideal), **test +0.7 % (real)** | Rejected as a context bit (doesn't generalise, adds table pressure). **Adopted through mixing** (§3.3): dev −1.8 %, test −3.4 % real |
| Copy fused into the match bits | +2 % | Rejected |
| Curve rule (straight stretch only, run-length trend) | Polygons 0.40 → 0.35 bits per boundary pixel, but +0.4 % on dev | Not adopted |
| Scan order: Hilbert, coarse-to-fine (same 16-pixel window plus decoded-or-not bits) | Hilbert 1.35× larger, coarse-to-fine 2.9× larger than raster | Raster |
| α = 1/32, 1/16, 1/8, 1/2 | 1/4 is best, with and without the new bits (−1 % on the v2 model too) | α = 1/4 |

### 4.4 Verification

`simulate_decoder.py` decodes masks pixel by pixel. At each pixel it zeroes everything not yet
decoded, recomputes all feature bits from that partial image, and runs a plain sequential CTW.
On glyph, pixel-art, app-icon and cache images the total matches the count-based number to
10⁻⁴ bits. `harness.check_causal` also tests every feature by replacing all future pixels
(random, all 0, all 1, inverted, from random positions and row starts) and requiring identical
feature values up to that point.

## 5. Ideas from other sources, for binary alpha

| Idea | Verdict for a 1-bit mask |
|---|---|
| Context modelling with a 2D neighbourhood + arithmetic coding | The core of this design |
| "Predict the predictor" (choose a model per region) | CTW does this per context, implicitly and without side information |
| Chain codes / polygon outlines | Their strength (an edge's direction memory) is now inside the line predictor, while raster order keeps blobs, dither and 1-px detail cheap |
| Quadtree, blocks (fixed or variable) | No gain: flat areas are already ~0.002 bits per pixel. Point-cloud research shows the decomposition matters less than the context model |
| Hilbert or zigzag scan order | Same total in theory (chain rule). Raster order gives the most decoded neighbours above and to the left, and the edge tracker relies on whole rows |
| Anchor points + diffusion, graphs, wavelets, GSP transforms | Made for many-valued pixels. A 1-bit mask has no residuals to shrink |
| MED / LOCO-I / Paeth, YCoCg, palette sorting, BWT/MTF | For the colour channels, not the mask |
| Colour correlation | Real, but a separate decision once colour coding exists (v2, §6 in git history) |
| BPG (HEVC) | Alpha is a second, monochrome HEVC stream, optionally premultiplied ([spec](https://bellard.org/bpg/bpg_spec.txt)). Nothing binary-specific; it should land near AVIF, which is the same idea with AV1. Not measured: no encoder can be run here |
| FUIF (FLIF, then JPEG XL modular) | Alpha is another channel, coded with MANIAC context trees that can use other channels' values as properties ([FUIF](https://github.com/cloudinary/fuif)). Its descendant, JPEG XL modular, is measured above. The idea worth keeping is cross-channel context, i.e. the colour correlation |
| Scan order (rows, Hilbert, coarse-to-fine) | Measured (§4.3). Rows win because each pixel sees a full half-plane of decoded neighbours |

## 6. Next steps

0. **More models through the mixer.** Mixing (§3.3) is how future predictors should enter:
   symmetry, the curve rule and larger windows all helped some images and hurt others as
   context bits. The sprite strips are the last place where WebP still wins (23 masks).

1. **Fixed-point, bounded CTW.** Integer counts, a fixed-size node table, and a specified hash for
   the match table. Measure the loss against the ideal numbers here.
2. **Better lines on curves.** Ellipses still cost 0.6 bits per boundary pixel, and the bound is
   much lower. Try a second-order rule (the run lengths of the run lengths) or a curvature step.
3. **Corners and shape starts.** Track births and deaths (where a shape begins or ends in a row)
   are not predicted yet.
4. **Speed.** Everything here is Python research code. The decoder's work per pixel is one CTW
   update of depth 30, a hash lookup, and a few comparisons.

## Appendix A. How existing formats store binary alpha

| Format | Representation of binary alpha | Coding | Colour under transparent pixels |
|---|---|---|---|
| GIF | One palette index means "transparent" ([GIF89a](https://www.w3.org/Graphics/GIF/spec-gif89a.txt)) | LZW over palette indices | That palette entry: a colour key |
| PNG palette + tRNS | Alpha value per palette entry ([PNG](https://www.w3.org/TR/png-3/)) | Deflate over byte-filtered rows | Per entry |
| PNG grey/RGB + tRNS | "A single pixel value distinguishing the fully transparent pixels from the fully opaque pixels": a colour key | Deflate | Must equal the key |
| PNG RGBA / LA | Full 8- or 16-bit alpha channel, no 1-bit alpha type | Deflate | Stored |
| WebP lossy + ALPH | Alpha plane, raw or VP8L lossless (in the green channel), optional spatial filter and level reduction ([RFC 9649](https://www.rfc-editor.org/rfc/rfc9649.txt)) | Prefix codes, LZ77 with 2D distance codes, colour cache. A palette of ≤ 2 colours bundles 8 pixels per symbol | Lossy colour |
| WebP lossless | Alpha inside ARGB | As above | Format keeps it, but `cwebp` only does so with `-exact` (default off) ([cwebp](https://developers.google.com/speed/webp/docs/cwebp)) |
| JPEG XL | Extra channel with its own bit depth, 1-bit allowed | Modular mode: MA tree over local properties chooses predictor + context, "ANS ... with static probabilities", optional LZ77 and patches (JBIG2-like pattern matching) ([JXL paper](https://arxiv.org/html/2506.05987)). MA-tree properties include 4 values per *previous channel* (\|v\|, v, \|v − pred\|, v − pred), up to 11 channels ([libjxl source](https://raw.githubusercontent.com/libjxl/libjxl/main/lib/jxl/modular/encoding/context_predict.h), [encoder options](https://raw.githubusercontent.com/libjxl/libjxl/main/lib/include/jxl/encode.h)) | `keep_invisible` defaults to on for lossless |
| AVIF | Auxiliary monochrome AV1 image ([avifenc](https://man.archlinux.org/man/extra/libavif/avifenc.1.en)) | Lossless AV1. Palette mode (2–8 colours, index map coded with neighbour contexts in wavefront order, [SVT-AV1 docs](https://toscode.gitee.com/mirrors/SVT-AV1/blob/master/Docs/Appendix-Palette-Prediction.md)) and intra block copy exist for screen content | Stored (premultiplication optional) |
| FLIF | Alpha plane, coded **first** (order A, Y, Co, Cg) ([FLIF spec](https://flif.info/spec.html)) | MANIAC: adaptive binary arithmetic coding of binarised integers, with context trees signalled up front whose branches only become active after enough visits. Alpha is a context property for Y, Co and Cg | `alpha_zero` skips invisible colour (lossy for invisible pixels) |
| MPEG-4 Visual (video objects) | "Binary alpha plane" in 16×16 binary alpha blocks: all transparent / all opaque / coded | Context-based arithmetic encoding (CAE): 10-pixel intra context, 9-pixel inter context (5 from the motion-compensated reference), **fixed** probability table ([patent text](https://patents.google.com/patent/GB2329785A/en), [CAE architecture paper](https://ir.lib.nycu.edu.tw/handle/11536/23821)) | n/a |
| GPU textures (BC1, ETC2 RGB8A1) | "Punch-through": a reserved colour code in a block means transparent ([D3D docs](https://learn.microsoft.com/en-ie/Windows/Win32/direct3d9/textures-with-alpha-channels), [ETC2](https://developer.mozilla.org/docs/Web/API/WEBGL_compressed_texture_etc)) | Fixed-rate blocks | Transparent black |

Observations:

- **Colour keys are the classic binary-alpha encoding** (GIF, PNG tRNS, GPU punch-through).
  They are exactly lossless when every transparent pixel has the key colour and no opaque pixel
  does. They were the most common case on this machine (v2).
- **Only FLIF and JPEG XL model alpha jointly with colour.** FLIF codes alpha first and uses it
  as context for colour. JPEG XL codes colour first and lets the alpha MA tree split on
  previous-channel values. Neither has a dedicated binary model or a colour-key mode.
- **No format uses an adaptive *binary* model with large contexts** (CTW, mixing, free trees)
  for alpha. The closest are FLIF's MANIAC (adaptive, but it binarises integers) and JBIG2 in PDF.

## Appendix B. Binary image coding: the literature

### B.1 Pixel-context coders (the core)

| Work | Idea | Result |
|---|---|---|
| JBIG (T.82, 1993), JBIG2 generic (2000) | 10- and 16-pixel templates with movable "AT" pixels, adaptive binary arithmetic coders (QM, MQ) | The reference baselines ([JBIG2](https://en.wikipedia.org/wiki/Jbig_2)) |
| Free tree coding (Martins & Forchhammer 1998) | Encoder searches a context tree over many candidate pixels and signals it | Best general-purpose bilevel coder of its time ([DTU](https://orbit.dtu.dk/en/publications/bi-level-image-compression-with-tree-coding/)) |
| Context-tree modelling of map layers (Kopylov & Fränti 2003–05) | Large-neighbourhood context trees, inter-layer contexts | ~50 % better than JBIG on maps ([report](https://cs.uef.fi/pub/Reports/A-2003-3.pdf)) |
| Context weighting for bilevel images (~2006) | Weights several context models instead of selecting one | No manual model choice, largest gains on halftones ([PubMed](https://pubmed.ncbi.nlm.nih.gov/17076388/)) |
| Digital straight line contexts (Aghito & Forchhammer 2004/06) | Straightness of the coded boundary as context | Better than JBIG, JBIG2 and MPEG-4 CAE on shapes ([PubMed](https://pubmed.ncbi.nlm.nih.gov/16900670/)) |
| Per-image context selection (Ramalho & Peixoto 2020) | Choose the best context pixels per image, signal them in 16 bits | Beats fixed templates, more so on point-cloud slices ([SBrT](https://biblioteca.sbrt.org.br/articles/2280)) |
| Sparse binary masks (Mohideen, Peter & Weickert 2020) | Systematic comparison incl. PAQ variants | DjVu best of the standard codecs. Logistic mixing of 12 nested contexts (BPAQ-2D-L) best overall. A second SSE stage *hurt* on binary data ([arXiv 2010.13634](https://arxiv.org/pdf/2010.13634)) |

### B.2 Contour coders (chain codes, vertices)

- Chain codes (3OT, AF8) beat JBIG on irregular shapes: about 29 % on average in one study,
  and in over 85 % of cases in another ([SPIE 2005](https://opticalengineering.spiedigitallibrary.org/journals/optical-engineering/volume-44/issue-09/097004/Compressing-bilevel-images-by-means-of-a-three-bit-chain/10.1117/1.2052793.full),
  [Pattern Recognition 2007](https://www.academia.edu/3127136/Efficiency_of_chain_codes_to_represent_binary_objects)).
- Context-adaptive extended chain coding (InterDigital, 2026) reaches 139,045 bytes on V-PCC
  occupancy maps, against JBIG1 at 153,454 and FLIF at 150,109 (about 9 % better than JBIG1).
  It is worse on a sequence with many fragmented blobs ([arXiv 2603.03073](https://arxiv.org/abs/2603.03073)).
- Vertex-based shape coders, which are usually lossy, report up to 27 % fewer bits than MPEG-4
  intra CAE ([paper](https://www.datalearner.com/academic/journal-papers/0923-5965/volumes-and-issues/61/paper-detail/102913)).

**Interpretation.** Contour coders win against *weak* pixel models (JBIG's 10-pixel template),
by roughly the same margin that CTW wins over a JBIG-like template (10–34 %, v2). Nobody has
compared them with CTW- or mixing-class pixel models. What contour coders exploit is long-range
direction memory along the boundary, which a small raster template cannot see. That's a
reason to feed an **edge-geometry model** into the mixer, not to switch to contours, since
contours fail on fragmented and dithered masks.

### B.3 Hierarchical / occupancy coding

Point-cloud geometry is binary occupancy, coded as octrees (the 3D quadtree) or as bilevel
slices. Both become competitive only when each occupancy bit is coded with rich neighbour
contexts: slice coding with JBIG-like contexts beats JBIG and octree coders, and learned
octree contexts save up to 30 % over MPEG G-PCC ([UnB summary](https://repositorio.unb.br/bitstream/10482/42616/1/2021_EvaristoRamalhoLucchezidaSilva.pdf),
[arXiv 2104.09859](https://arxiv.org/pdf/2104.09859)). The lesson for quadtrees: the
decomposition doesn't matter much, the context model does.

### B.4 Learned coders

No learned lossless coder for binary masks or alpha was found that beats classic context
modelling on images. A 2024 dictionary method is preliminary and loses to JBIG2 on several
images ([arXiv 2406.03087](https://arxiv.org/html/2406.03087v1)). Learned models need training
data and a large decoder, so they are out of scope.

### B.5 Probability estimation and mixing

- CTW (1995) is still the principled answer to the template-size problem. Context Tree
  Switching (2012) and Partition Tree Weighting (2013) extend it while keeping its guarantees
  ([CTS](https://ar5iv.arxiv.org/html/1111.3182), [PTW](https://ar5iv.arxiv.org/html/1211.0587)).
- Logistic mixing (PAQ, cmix) and Gated Linear Networks are the practical state of the art for
  combining different models ([GLN](https://arxiv.org/pdf/1712.01897)).
- VVC's CABAC keeps two probability estimates per context with different adaptation rates and
  averages them ([MPU papers](https://research.mpu.edu.mo/en/publications/using-four-hypothesis-probability-estimators-for-cabac-in-versati-2/)):
  multi-rate estimation is standard practice.

