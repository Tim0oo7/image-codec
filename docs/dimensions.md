# Image dimensions: design notes

Status: decided 2026-10-24. The normative definition is in
[format-spec.md](format-spec.md#image-dimensions), the implementation in
`src/image_codec/dimensions.py`.

## Goal

A header field for width and height that is smaller than JPEG XL's `SizeHeader` on the images we
expect, without betting on round numbers. `SizeHeader` costs 9 to 34 bits: a flag for sizes that
are multiples of 8 up to 256, the height in 9, 13, 18 or 30 bits, a 3-bit aspect ratio with
width = ⌊height × ratio⌋, and otherwise the width ([libjxl headers.cc](https://raw.githubusercontent.com/libjxl/libjxl/main/lib/jxl/headers.cc)).

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Which side | Long side L first, short side S derived from the shape | The shape can depend on the size. Rounding from the long side reproduces both resizing conventions (long side fixed, or short side fixed and long side rounded). A free short side is bounded by L. Coding the short side first saves nothing on average: by the chain rule, the bits come back as a "which of two long sides" bit |
| Length of L | 4-bit digit count n of L − 1, then the digits after the leading 1 | Each doubling costs exactly one bit, with no jumps. Fixed classes (8, 12, 16 or 32 bits behind a 2-bit type) are cheaper only inside the range they bet on (2049–4352: 14 bits instead of 15–16), cost 4 bits extra just above each boundary (256 → 257), and give a quarter of the code space to sizes above 69888 |
| Short side of a ratio | L·q/p, rounded | Covers resized images such as 2048×1365 and 1366×768, where JPEG XL's height-first truncation fails |
| Kind | Square, free or ratio, in a size-dependent order | Small images are mostly square icons. Large images are assumed to have a standard ratio more than two thirds of the time, with about 20 % free |
| Orientation | 1 bit for non-square shapes | JPEG XL only has landscape ratios, so a portrait 1080×1920 costs it 34 bits |
| Ratios | 4:3, 3:2, 16:9, 16:10, 5:4, 2:1, A4 (99:70), 20:9 | Cameras and phones, screens, Instagram 4:5, panoramas and 18:9 phones, scans and PDF renders (A4 is exactly 210 × 297 mm), Android phones. 21:9 is left out: no ultrawide screen is exactly 21:9 (2560×1080 is 64:27, 3440×1440 is 43:18) |
| Exact bit | For 4:3, 16:9, 16:10, 5:4 and 20:9 | Native resolutions are exact multiples (1920×1080 = 120 × 16:9), so coding k = L/p saves log₂ p bits for a 1-bit flag. That pays off when more than 1/log₂ p of the images with that ratio are exact: 25 % for 16:9, 50 % for 4:3, 63 % for 3:2 (left out). A4 renders are never exact. Resized images pay 1 bit; free sizes are unaffected |
| Free short side | Truncated binary over L − 1 values | Within 0.09 bit on average of log₂(L − 1), the minimum for a short side without a pattern |

## Rejected

- **Height first, width = ⌊height × ratio⌋** (JPEG XL): no portrait ratios, and it fails on
  resized images.
- **Ratio × scale factor k only:** works for exact multiples only.
- **Trailing zero bits** ("L is a multiple of 4ˢ", 2 bits): wins 2–4 bits on round sizes, costs
  2 bits on every other size.
- **Huffman tables** for the length and for shape plus orientation, built from assumed
  frequencies: about 1.2 bits per image better under those assumptions, but the frequencies are
  guesses. Fields 1 and 2 can be replaced by tables later if a measured list of dimensions shows
  the gain. Arithmetic coding of the header would gain less than 0.1 bit over Huffman, and
  tools would need the entropy decoder just to read the dimensions.
- **Tables of exact device or icon sizes:** device sizes date quickly, and an icon table only pays
  off if most icons have standard sizes (the binary-alpha corpus in
  [alpha-binary.md](alpha-binary.md) has a median of 44×44).
- **Deriving dimensions from the file length:** breaks streaming and embedding.

## Cost compared with JPEG XL

Bits for the dimensions alone.

| W×H | JPEG XL | This format |
|---|---|---|
| 1×1 | 15 | **4** |
| 16×16 | 9 | **8** |
| 44×44 | 15 | **10** |
| 256×256 | **9** | 12 |
| 768×512 | **15** | 18 |
| 1920×1080 | 19 | **16** |
| 1080×1920 | 34 | **16** |
| 3840×2160 | 19 | **17** |
| 4032×3024 | 19 | 19 |
| 6000×4000 | **19** | 21 |
| 2048×1365 | 34 | **19** |
| 1366×768 | 34 | **20** |
| 2880×1800 | 34 | **17** |
| 1080×1350 | 34 | **18** |
| 1080×2400 | 34 | **16** |
| 2480×3508 | 34 | **20** |
| 137×291 | 26 | **23** |
| 1170×2532 | 34 | **29** |
| 300×250 | **15** | 24 |

JPEG XL wins on landscape camera photos (6000×4000, Kodak's 768×512), on square icons that are
multiples of 8, and on 300×250 through its 12:10 ratio. Those are the cases its fixed fields
and multiple-of-8 flag are built for.

## How other formats store dimensions

| Format | Representation | Bits for 1920×1080 |
|---|---|---|
| PNG, BMP, QOI | 2 × 32 bits | 64 |
| GIF, JPEG | 2 × 16 bits (JPEG may leave the height 0 and define it later in a DNL marker) | 32 |
| WebP lossless | 2 × 14 bits, up to 16384 | 28 |
| FLIF | varints of width − 1 and height − 1 | 32 |
| HEVC (HEIC) | Exp-Golomb in the sequence parameter set | 42 |
| AV1 (AVIF) | 4-bit lengths, then the values. AVIF repeats them as 2 × 32 bits in its `ispe` box | 30 |
| JPEG XL | `SizeHeader`, see above | 19 |
