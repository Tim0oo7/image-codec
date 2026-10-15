# Binary alpha: research and plan

Status: research v2 (2026-10-15). Nothing here is final. The measurements can be reproduced
with the scripts in [research/alpha_binary](../research/alpha_binary/README.md).

## Summary

- **Binary alpha is common.** Of the 12,214 distinct images with transparency found on this
  machine, 15 % (1,782) use exactly two alpha values. Half of those are GIFs.
- **In practice it is usually redundant with colour.** 94 % of the binary-alpha images have one
  single colour under every transparent pixel, and 70 % are an exact *colour key*: alpha is a
  function of colour, so it costs zero bits once colour is known.
- **A pure context model already beats the existing formats on the mask alone.** CTW over the
  24 nearest neighbours (exact ideal costs) codes the 1,527 measured masks in 133 KB in total,
  against 241 KB for JPEG XL and 229 KB for WebP. It is smaller than JPEG XL on every single
  image, and smaller than all four baselines on 98 % of them. On synthetic shapes and dithering
  it is 1.3–4.9× smaller than JPEG XL.
- **It loses on repeated content**: text, sprite sheets and tiled textures, where LZ77, patches
  and intra block copy win. A match model is therefore mandatory. Detecting nearest-neighbour
  upscaling makes upscaled pixel art another 2.5× smaller.
- **Alpha and colour should be coded jointly.** The proposal is to code colour (YCoCg) first and
  alpha last, with a colour-key mode and colour-derived context bits. On the 1,527 images this
  costs 55 KB for all the alpha, against the 108 KB (WebP) and 156 KB (JPEG XL) that those codecs
  add when they code the full RGBA image losslessly.
- **The largest remaining headroom is on edges.** On synthetic polygon and ellipse masks CTW
  spends 0.86–1.0 bits per boundary pixel, while the theory of digital straight segments allows
  about 0.2 on long straight edges. An edge model is the next thing to measure.

## 1. Scope and decisions

| Decision | Choice |
|---|---|
| Constant alpha | Header only (mode + value), zero payload |
| Binary alpha | Dedicated coding path, optimised purely for size |
| "Lossless" | Includes the colour under alpha = 0. Encoder tricks that rewrite invisible colour (WebP's default, FLIF's `alpha_zero`, `oxipng --alpha`, `zopflipng --lossy_transparent`) are therefore **not** allowed |
| Colour space | YCoCg for the colour channels (§6 covers how alpha interacts with it) |

## 2. Theory

1. **Size = model.** Every decodable code implicitly defines a probability distribution
   (Kraft inequality), and an arithmetic coder gets within a few bytes of −log₂ P(data). Every
   "representation" (quadtree, RLE, chain code, polygons) is an implicit model, so the design
   problem is building the best sequential predictor.
2. **Flat regions are nearly free.** An adaptive estimator in a context that only ever sees one
   value spends about ½·log₂ n + 1 bits on n pixels (Krichevsky–Trofimov), so 10⁶ transparent
   pixels cost about 11 bits. The information is on the boundary.
3. **Edges cost what their predictability says.** A simple edge crossing a scanline moves left,
   stays or moves right (≤ log₂ 3 ≈ 1.58 bits per crossing). Contexts that see the edge's slope
   and curvature cost far less. The figure of merit is **bits per boundary pixel**.
4. **Learning costs bits (regret).** Every context used pays about ½·log₂ n bits to learn its
   probability. Bigger templates predict better but have more contexts to learn. Measured below:
   on random noise a static 16-pixel "oracle" model appears to reach 0.38 bpp, while every adaptive
   model correctly costs 1.0 bpp. That gap is pure learning cost.
5. **Order is free in theory.** By the chain rule, H(A, C) = H(A) + H(C | A) = H(C) + H(A | C)
   for alpha A and colour C, so coding alpha or colour first gives the same ideal total. In practice
   the order decides which conditional model you have to build, and one of them is much easier (§6).
6. **Adaptive costs can be computed exactly without a coder.** The KT/Dirichlet estimator, and CTW
   built on it, give a sequence probability that depends only on the final counts per context. So
   the cost of coding an image adaptively, pixel by pixel, follows exactly from counts, and that's
   what the measurements in §5 use. Mixing models (PAQ-style) need a sequential simulation instead.

Two engineering consequences:

- **Probability precision.** Clamping probabilities at 12 bits costs at least
  −log₂(1 − 2⁻¹²) ≈ 3.5·10⁻⁴ bits on every pixel, about 365 bytes on a 4K mask. At 16 bits the
  floor is about 23 bytes. Use ≥ 16-bit probabilities, and aim for 24.
- **Prefix (Huffman) codes have a 1-bit-per-symbol floor**, which is why PNG and WebP need runs,
  LZ77 or pixel bundling to make binary planes cheap. An adaptive binary arithmetic coder has no
  such floor.

## 3. How existing formats store binary alpha

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
  does. That's the most common real-world case (§5.3).
- **Only FLIF and JPEG XL model alpha jointly with colour.** FLIF codes alpha first and uses it
  as context for colour. JPEG XL codes colour first and lets the alpha MA tree split on
  previous-channel values. Neither has a dedicated binary model or a colour-key mode.
- **No format uses an adaptive *binary* model with large contexts** (CTW, mixing, free trees)
  for alpha. The closest are FLIF's MANIAC (adaptive, but it binarises integers) and JBIG2 in PDF.

## 4. Binary image coding: the literature

### 4.1 Pixel-context coders (the core)

| Work | Idea | Result |
|---|---|---|
| JBIG (T.82, 1993), JBIG2 generic (2000) | 10- and 16-pixel templates with movable "AT" pixels, adaptive binary arithmetic coders (QM, MQ) | The reference baselines ([JBIG2](https://en.wikipedia.org/wiki/Jbig_2)) |
| Free tree coding (Martins & Forchhammer 1998) | Encoder searches a context tree over many candidate pixels and signals it | Best general-purpose bilevel coder of its time ([DTU](https://orbit.dtu.dk/en/publications/bi-level-image-compression-with-tree-coding/)) |
| Context-tree modelling of map layers (Kopylov & Fränti 2003–05) | Large-neighbourhood context trees, inter-layer contexts | ~50 % better than JBIG on maps ([report](https://cs.uef.fi/pub/Reports/A-2003-3.pdf)) |
| Context weighting for bilevel images (~2006) | Weights several context models instead of selecting one | No manual model choice, largest gains on halftones ([PubMed](https://pubmed.ncbi.nlm.nih.gov/17076388/)) |
| Digital straight line contexts (Aghito & Forchhammer 2004/06) | Straightness of the coded boundary as context | Better than JBIG, JBIG2 and MPEG-4 CAE on shapes ([PubMed](https://pubmed.ncbi.nlm.nih.gov/16900670/)) |
| Per-image context selection (Ramalho & Peixoto 2020) | Choose the best context pixels per image, signal them in 16 bits | Beats fixed templates, more so on point-cloud slices ([SBrT](https://biblioteca.sbrt.org.br/articles/2280)) |
| Sparse binary masks (Mohideen, Peter & Weickert 2020) | Systematic comparison incl. PAQ variants | DjVu best of the standard codecs. Logistic mixing of 12 nested contexts (BPAQ-2D-L) best overall. A second SSE stage *hurt* on binary data ([arXiv 2010.13634](https://arxiv.org/pdf/2010.13634)) |

### 4.2 Contour coders (chain codes, vertices)

- Chain codes (3OT, AF8) beat JBIG on irregular shapes: about 29 % on average in one study,
  and in over 85 % of cases in another ([SPIE 2005](https://opticalengineering.spiedigitallibrary.org/journals/optical-engineering/volume-44/issue-09/097004/Compressing-bilevel-images-by-means-of-a-three-bit-chain/10.1117/1.2052793.full),
  [Pattern Recognition 2007](https://www.academia.edu/3127136/Efficiency_of_chain_codes_to_represent_binary_objects)).
- Context-adaptive extended chain coding (InterDigital, 2026) reaches 139,045 bytes on V-PCC
  occupancy maps, against JBIG1 at 153,454 and FLIF at 150,109 (about 9 % better than JBIG1).
  It is worse on a sequence with many fragmented blobs ([arXiv 2603.03073](https://arxiv.org/abs/2603.03073)).
- Vertex-based shape coders, which are usually lossy, report up to 27 % fewer bits than MPEG-4
  intra CAE ([paper](https://www.datalearner.com/academic/journal-papers/0923-5965/volumes-and-issues/61/paper-detail/102913)).

**Interpretation.** Contour coders win against *weak* pixel models (JBIG's 10-pixel template),
by roughly the same margin that CTW wins over a JBIG-like template in §5 (10–34 %). Nobody has
compared them with CTW- or mixing-class pixel models. What contour coders exploit is long-range
direction memory along the boundary, which a small raster template cannot see. That's a
reason to feed an **edge-geometry model** into the mixer, not to switch to contours, since
contours fail on fragmented and dithered masks.

### 4.3 Hierarchical / occupancy coding

Point-cloud geometry is binary occupancy, coded as octrees (the 3D quadtree) or as bilevel
slices. Both become competitive only when each occupancy bit is coded with rich neighbour
contexts: slice coding with JBIG-like contexts beats JBIG and octree coders, and learned
octree contexts save up to 30 % over MPEG G-PCC ([UnB summary](https://repositorio.unb.br/bitstream/10482/42616/1/2021_EvaristoRamalhoLucchezidaSilva.pdf),
[arXiv 2104.09859](https://arxiv.org/pdf/2104.09859)). The lesson for quadtrees: the
decomposition doesn't matter much, the context model does.

### 4.4 Learned coders

No learned lossless coder for binary masks or alpha was found that beats classic context
modelling on images. A 2024 dictionary method is preliminary and loses to JBIG2 on several
images ([arXiv 2406.03087](https://arxiv.org/html/2406.03087v1)). Learned models need training
data and a large decoder, so they are out of scope.

### 4.5 Probability estimation and mixing

- CTW (1995) is still the principled answer to the template-size problem. Context Tree
  Switching (2012) and Partition Tree Weighting (2013) extend it while keeping its guarantees
  ([CTS](https://ar5iv.arxiv.org/html/1111.3182), [PTW](https://ar5iv.arxiv.org/html/1211.0587)).
- Logistic mixing (PAQ, cmix) and Gated Linear Networks are the practical state of the art for
  combining different models ([GLN](https://arxiv.org/pdf/1712.01897)).
- VVC's CABAC keeps two probability estimates per context with different adaptation rates and
  averages them ([MPU papers](https://research.mpu.edu.mo/en/publications/using-four-hypothesis-probability-estimators-for-cabac-in-versati-2/)):
  multi-rate estimation is standard practice.

## 5. Measurements

### 5.1 Method

- **Ideal costs** use `idealcost.py`: exact adaptive code lengths of a fixed template with a KT
  estimator (KT-10 ≈ JBIG, KT-16 ≈ JBIG2 generic), and of **CTW over the 24 nearest causal
  neighbours** (ordered by distance, Dirichlet α = 1/16). Our numbers include a 4-byte header
  allowance. Real coders add a few bytes on top.
- **Baselines** code the mask on its own as a greyscale image: JPEG XL lossless 1-bit (effort 9,
  and 10 up to 1.1 Mpx; the smaller of the two per image, since effort 10 was often *worse* on
  small images), WebP lossless (method 6, exact), PNG 1-bit (optimised), AVIF lossless (speed 4).
  All through Pillow / `imagecodecs`.
- **Joint coding** is measured as RGBA size − RGB size for JPEG XL (effort 9) and WebP (exact):
  what alpha adds when the codec can also use colour.
- **Container overhead** is large for small images (AVIF especially). The payload table
  subtracts each codec's size for an all-opaque mask of the same dimensions.

### 5.2 Synthetic masks

Bytes. CTW-24 includes the 4-byte header allowance. JXL = the better of effort 9 and 10.

| Mask | Size | Boundary px | CTW-24 | KT-10 (≈ JBIG) | KT-16 (≈ JBIG2) | JPEG XL | WebP | PNG | AVIF | JXL / CTW |
|---|---|---|---|---|---|---|---|---|---|---|
| Rotated ellipse | 1024² | 2,740 | **343** | 422 | 408 | 764 | 1,430 | 3,017 | 7,666 | 2.23 |
| Random polygons | 1024² | 8,186 | **880** | 1,193 | 1,135 | 1,566 | 2,986 | 6,509 | 2,096 | 1.78 |
| Organic blobs | 1024² | 23,959 | **3,475** | 3,874 | 4,017 | 4,413 | 7,062 | 9,792 | 5,770 | 1.27 |
| Floyd–Steinberg dither | 512² | 186,474 | **8,241** | 12,462 | 9,578 | 14,456 | 15,854 | 21,795 | 17,472 | 1.75 |
| Bayer (ordered) dither | 512² | 215,400 | **1,113** | 10,981 | 6,188 | 5,450 | 1,742 | 2,134 | 6,372 | 4.89 |
| Text (repeated glyphs) | 1024×512 | 64,824 | 4,969 | 8,346 | 6,591 | **1,430** | 8,130 | 11,117 | 4,280 | 0.29 |
| Sprite sheet (32² tiles) | 512² | 30,400 | 2,219 | 4,140 | 3,466 | 3,063 | 2,372 | 2,641 | **1,618** | 1.38 |
| Pixel art, 8× upscaled | 512² | 6,728 | 426 | 438 | 456 | 549 | **394** | 740 | 908 | 1.29 |
| Random noise | 256² | 65,322 | **8,197** | 8,526 | 8,818 | 8,229 | 8,258 | 8,516 | 8,552 | 1.00 |

What this shows:

- **CTW beats JPEG XL by 1.3–2.2× on smooth shapes, blobs and error-diffusion dither**, and by
  4.9× on ordered (Bayer) dither, where the period fits inside the 24-pixel context.
- **CTW solves the template-size problem.** A bigger fixed template is not automatically better:
  on a 512² disc, KT-16 costs 242 bytes and KT-24 costs 263 (learning cost), while CTW-24 costs
  206. CTW-24 is smaller than both fixed templates on every mask in the table.
- **Repeated content beats pure context models.** On the text mask, which reuses the same glyphs
  everywhere, JPEG XL (LZ77/patches) is 3.5× smaller than CTW. On the sprite sheet, AVIF (likely
  palette + intra block copy) and WebP (LZ77) win. A match model is mandatory.
- **Upscaled pixel art needs a transform.** The 8× nearest-neighbour upscale costs 426 bytes with
  CTW. Detecting the factor and coding the 64×64 original costs about 170 bytes, against 394 for
  WebP, the best baseline.
- **Straight edges have large headroom.** CTW spends 0.86 bits per boundary pixel on the polygon
  mask (1.0 on the ellipse), because a 24-pixel context sees only about 3 pixels of edge. But a
  digital straight segment of length n is one of about n³/π² possibilities: Mignosi's count of
  Sturmian factors, 1 + Σᵢ (n − i + 1)·φ(i) ([proof note](https://www.di.ens.fr/reports/1992/liens-92-21.A4.pdf)).
  It therefore carries only log₂(n³/π²) ≈ 17 bits for n = 100, about 0.17 bits per pixel. An
  edge model that follows the segment (input E in §7.2) could cut edge costs by up to ~5× on
  shape masks.
- **Noise** costs 1.0 bpp everywhere, as it must. The raw fallback covers it.

### 5.3 Real binary-alpha images

**Corpus.** `scan_alpha.py` walked `C:\Windows`, `C:\Program Files`, `C:\Program Files (x86)`
and the per-user installed programs folder: 68,410 PNG/GIF files. Personal folders (documents,
pictures, downloads, browser caches) were not scanned, and no image leaves the machine. After removing duplicate pixel data, 12,214 images had transparency:

| Alpha | Images | Share |
|---|---|---|
| Constant | 1,020 | 8 % |
| **Binary** | **1,782** | **15 %** |
| General (> 2 values) | 9,412 | 77 % |

Of the binary-alpha images, 888 are GIFs and 894 PNGs. 1,697 use 0/255, 21 use 254/255
(near-opaque noise), and the rest use other pairs. Under the transparent pixels, 94 % have one
single colour (most often #818181, white, black, #C0C0C0, #FF00FF) and 70 % are an exact colour
key. Most are application UI icons (MATLAB, Granta EduPack, Office). The 1,527 with both sides
≥ 16 px were measured: median 44×44 px, and 420 at least 128×128.

**Totals in bytes.** "Best of both" picks, per image, CTW with or without the colour bits
(1 bit to signal) and uses key mode where it applies. All our columns include the 4-byte header.

| Set | Images | CTW-24, alpha only | CTW-24 + colour bits | Best of both + key mode | JPEG XL (mask) | WebP (mask) | PNG | AVIF | JPEG XL RGBA − RGB | WebP RGBA − RGB |
|---|---|---|---|---|---|---|---|---|---|---|
| All | 1,527 | 133,484 | 61,319 | **55,095** | 241,075 | 228,864 | 355,878 | 715,693 | 156,034 | 107,650 |
| ≥ 128×128 | 420 | 95,076 | 54,292 | **48,640** | 158,567 | 129,832 | 204,122 | 349,920 | 128,667 | 100,448 |
| < 128×128 | 1,107 | 38,408 | 7,027 | **6,455** | 82,508 | 99,032 | 151,756 | 365,773 | 27,367 | 7,202 |
| No colour key | 454 | 62,834 | 56,233 | **50,803** | 108,686 | 77,150 | 115,643 | 275,240 | 106,760 | 101,760 |

Payload only (container overhead subtracted, see §5.1): JPEG XL e9 192,989, WebP 177,764,
PNG 208,475, AVIF 267,113, against **127,376** for alpha-only CTW-24 without a header.

What this shows:

1. **The mask alone:** CTW-24 is smaller than JPEG XL on all 1,527 images, and smaller than
   every baseline on 1,493 (98 %). In total it's 1.8× smaller than JPEG XL and 1.7× smaller than
   WebP (1.5× and 1.4× on payload only). AVIF is far behind on these mostly small images.
2. **All 34 exceptions are WebP wins on repeated content.** Animation sprite sheets (4800×96
   frame strips: WebP 148 bytes vs CTW 1,235), and a 254/255 noise texture that repeats
   horizontally every 256 pixels (96 % of pixels match at that offset): WebP 10.2 KB vs CTW
   20.1 KB. LZ77 catches both. A match model, or a context pixel at the encoder-detected repeat
   offset, would too.
3. **Real masks are easier than the synthetic ones.** On images ≥ 128×128 with more than 200
   boundary pixels, CTW spends a median 0.11 bits per boundary pixel (axis-aligned UI shapes),
   with a 90th percentile of 1.46.
4. **Colour halves the cost.** Colour-aware contexts take the total from 133 KB to 61 KB.
   Choosing the order per image plus key mode gives 55 KB, against 108 KB (WebP) and 156 KB
   (JPEG XL) for those codecs' own joint coding.
5. **The joint-coding baselines are noisy per image.** Encoders change their choices when alpha
   is present: WebP's RGBA file was *smaller* than its RGB-only file for 216 images (−18 KB in
   total), and JPEG XL's for 119 (−6 KB). That noise favours the baselines in the totals. Per
   image, our cost ties with or beats WebP's marginal cost on 797 of 1,527 images (52 %). Of the
   730 losses, 435 are images where WebP's RGBA file was no bigger than its RGB file, and 597 are
   within 8 bytes, where our 4-byte header allowance decides. All losses together add up to
   22 KB, against 75 KB of wins.
6. **A fixed CTW bit order can hurt.** Putting the 3 colour bits first made 65 of the 454
   non-key images worse (worst case +26 %, the noise texture). CTW can only skip an uninformative
   first bit by falling back to no context at all. Mixing (or choosing between) both orders
   fixes this: the per-image choice already brings the non-key total from 56,233 to 50,803.

## 6. Alpha and colour

### 6.1 What can be exploited

1. **Exact colour key** (alpha = 1 exactly when colour ≠ key): alpha costs 0 bits given colour.
   This is GIF/PNG-tRNS semantics, detected by the encoder and signalled with one flag + the key.
2. **Partial key** (all transparent pixels share a colour, but some opaque pixels have it too):
   alpha is only uncertain on key-coloured pixels. A context bit "colour == key" makes every
   other pixel free.
3. **Edge alignment**: alpha boundaries coincide with colour boundaries. Context bits such as
   "colour equals the west / north neighbour's colour" carry this.

### 6.2 Which order

- **Colour first, alpha last (proposed).** The alpha coder sees the decoded YCoCg values, so
  1–3 are simple context bits, and the colour-key case is exactly free. The colour coder doesn't
  see alpha, but because invisible colour is kept, alpha would mainly help the colour coder at
  edges, which is the same shared information (chain rule).
- **Alpha first (FLIF).** The colour coder must learn that transparent pixels have the key colour,
  which is cheap but not free, and the colour-key case still pays the full alpha cost.
- The keyed comparison is in YCoCg space without loss of generality, because the colour transform
  is a bijection.

### 6.3 Measured

- **Exact colour key:** 70 % of the measured binary-alpha images (1,073 of 1,527). Alpha costs
  0 bits plus the key in the header.
- **Colour-aware contexts** (colour == key, == west, == north) halve the total alpha cost
  (133 KB → 61 KB). On non-key images they save 10 % in total, with a median per-image ratio
  of 0.56.
- **Per-image order choice + key mode:** 55 KB, against 156 KB that JPEG XL adds for alpha
  inside its own joint RGBA coding, and 108 KB for WebP.

### 6.4 Proposed decision

1. **Code colour (YCoCg) first and alpha last.** The alpha model gets the decoded colour.
2. **Key mode** in the header: one flag + the key triple, no alpha payload.
3. **Colour bits as inputs to the mixer** (an alpha-only CTW and a colour-first CTW mixed),
   not forced into one fixed order (§5.3, point 6).
4. **Don't use alpha as context for colour** (FLIF style) for now. Revisit once the YCoCg coder
   exists, since that's the only way to measure the alpha-first gain.

## 7. Revised design

### 7.1 Pipeline

```
colour (YCoCg) decoded ──► alpha mode?
   constant ─► value from header
   key      ─► alpha = (colour != key) ? hi : lo           (0 payload bits)
   coded    ─► per pixel: model inputs ─► mixer ─► p(1) ─► binary arithmetic decoder
   raw      ─► W·H bits                                     (guarantee: ≤ 1 bpp)
```

### 7.2 Model inputs (in order of measured or expected value)

| # | Input | Evidence |
|---|---|---|
| A | CTW (later CTS) over the nearest causal neighbours, depth ≈ 24 | Smaller than JPEG XL on all 1,527 real masks, and smaller than both fixed templates on every synthetic one (§5) |
| B | Colour-first CTW: the same neighbours, preceded by colour == key, colour == W, colour == N | Halves the real-image total (§5.3). Kept as a *separate* model next to A, because forcing the colour bits first hurt 65 of 454 images |
| C | 2D match model (hash of a large context → last occurrence → predicted pixel) | Required: all 34 real images where CTW lost, and the text/sprite synthetic masks, are repeated content (§5.2, §5.3) |
| D | Small/medium fixed templates (orders 4–16) | Fast learners for small images, standard mixer inputs |
| E | Edge-geometry model (follows the boundary's digital straight segment / curvature) | Up to ~5× headroom on straight edges (digital straight segment bound, §5.2). Contour coders beat weak templates (§4.2) |
| F | Offset contexts at encoder-detected periods (dither, tiled textures) | Bayer dither 4.9× smaller than JPEG XL even without it. The 254/255 texture repeats at 256 px, far outside any template (§5.3) |

Combined by **logistic mixing** with weight sets selected by a small context. One SSE/APM stage
at most (the sparse-mask study found a second one harmful on binary data). Counters:
Dirichlet α = 1/16 was 2 % better than KT (α = ½) in total on real masks, but KT won on half the
images and on error-diffusion dither. So mix both, or adapt the rate per context (two-rate
estimators as in VVC).

### 7.3 Encoder-side search (signalled in a few bytes)

| Parameter | How it's chosen |
|---|---|
| Alpha mode (key / coded / raw) | Exact check for the key. Raw if the coded size > W·H bits |
| Nearest-neighbour upscale factor (x, y) | Exact check. Code the downscaled mask |
| Transpose | Ideal cost on a subsample |
| Context order / AT pixels | Greedy, as in free tree / per-image context selection (§4.1) |
| Repeat offsets (dither period, tiled texture) | Autocorrelation peaks of the mask, e.g. (0, 256) for the noise texture in §5.3 |

### 7.4 Coder

Binary range coder, 32-bit range, probabilities ≥ 16 bits (aim for 24), and **all model
arithmetic in fixed-point integers** so every decoder reproduces every probability bit-exactly.

### 7.5 Bitstream sketch

```
alpha_mode        2 bits   0 none · 1 constant · 2 binary · 3 general
-- if binary:
value_lo, value_hi         bit depth each
binary_mode       2 bits   0 key · 1 coded · 2 raw
key               3 × bit depth    (YCoCg triple, key mode only)
upscale           flag (+ factors)
transpose         1 bit
context_order     preset id or explicit offsets
repeat_offsets    optional list of (dy, dx)
payload           arithmetic-coded (coded mode) / packed bits (raw mode)
```

## 8. Milestones

| M | Deliverable | Status |
|---|---|---|
| M0 | Baselines (JXL, WebP, PNG, AVIF) on a corpus | Done for synthetic + locally found images (§5). A public corpus is still needed |
| M1 | Exact ideal-cost harness (KT, CTW) | Done (`research/alpha_binary`) |
| M2 | Colour-key and partial-key handling | Measured in §5.3. Implement detection |
| M3 | Sequential simulation for models that can't be computed from counts (match model, edge model, mixing). First targets: the 34 repeated-content losses and the straight-edge headroom | Next |
| M4 | Mixer + SSE. Add C, E, F one at a time, keep each only if it lowers total cost | |
| M5 | Encoder search (upscale, transpose, context order, period) + raw fallback | |
| M6 | Fixed-point arithmetic coder + bitstream. Size ≤ ideal + 4 bytes. Round-trip tests | |
| M7 | Speed (compiled port) | Later |

**Target:** smaller than JPEG XL effort 10 on every corpus category, measured as the cost of the
alpha *given* the colour (JXL RGBA − RGB).

## 9. Risks and open questions

1. **Corpus bias.** The real images come from installed applications (mostly UI icons and GIFs).
   A public corpus with large cutouts, sprites, segmentation masks and GIF conversions is needed,
   with licences checked.
2. **Speed.** Every pixel goes through several models and a mixer, for both encoding and decoding.
3. **Memory.** CTW at depth 24 on n pixels can create up to 24·n nodes, so it needs a bounded,
   deterministic table.
4. **Tiles vs one stream.** Every tile restart re-pays the learning cost.
5. **Colour-first order** means the colour coder can't use alpha. That has to be confirmed once
   the YCoCg coder exists.

## Sources

Linked inline above. Main references: [RFC 9649 (WebP)](https://www.rfc-editor.org/rfc/rfc9649.txt),
[PNG 3rd edition](https://www.w3.org/TR/png-3/), [GIF89a](https://www.w3.org/Graphics/GIF/spec-gif89a.txt),
[JPEG XL paper](https://arxiv.org/html/2506.05987), [libjxl](https://github.com/libjxl/libjxl),
[FLIF spec](https://flif.info/spec.html), [CTS](https://ar5iv.arxiv.org/html/1111.3182),
[PTW](https://ar5iv.arxiv.org/html/1211.0587), [GLN](https://arxiv.org/pdf/1712.01897),
[sparse binary masks](https://arxiv.org/pdf/2010.13634), [chain coding 2026](https://arxiv.org/abs/2603.03073).
