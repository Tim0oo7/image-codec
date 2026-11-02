# Binary alpha measurements

Scripts behind the numbers in [docs/alpha-binary.md](../../docs/alpha-binary.md). They are
research tools, not part of the codec.

| File | What it does |
|---|---|
| `idealcost.py` | Exact ideal code length of adaptive context models (KT templates, CTW), computed from counts |
| `harness.py` | Builds the evaluation set, runs feature/cost configurations in parallel, checks causality, reports per category |
| `features_edge.py` | Edge tracks: transitions linked row to row, simple slope extrapolation |
| `features_line.py` | The line predictor: next edge step from the digital-straight-line (Sturmian) run pattern |
| `features_match.py` | The match predictor: value that followed the same 48-pixel neighbourhood last time |
| `features_combo.py`, `estimators.py` | Feature combinations and estimator variants used in the doc |
| `ctw_prior.py`, `prior_configs.py`, `prior_table.py` | Trained starting tables (priors) for CTW and for a plain template |
| `where_bits.py` | Per-pixel cost split by local geometry |
| `simulate_decoder.py` | Pixel-by-pixel decoder simulation that must match the count-based cost exactly |
| `baselines.py` | JPEG XL, WebP, PNG, AVIF and an ideal JBIG2 generic template as harness cost functions |
| `split.py` | Dev/test split by source object (glyph, pack, icon name, folder) |
| `realcoder.py`, `realconfigs.py`, `test_realcoder.py` | Real coder: sequential CTW in a bounded table, 16-bit probabilities, arithmetic coding, round trip; real file sizes incl. headers |
| `simple_codecs.py` | Raw bitmap, RLE (plain and adaptive), position lists, per-row change positions and CCITT Group 4, for comparison |
| `size_classes.py`, `size_classes2.py` | Size-dependent choices: starting tables, context layouts and termination per size bucket |
| `fixedcodec.py`, `test_fixedcodec.py` | The reference codec (`src/image_codec/alpha_binary.py`) compiled with numba: identical bytes, the same features as the research predictors, corpus runs and reports against `realconfigs.real_mix` |
| `features_lz.py`, `features_sym.py`, `scan_order.py` | Rejected experiments: LZ-style copy, mirror/row repetition, scan orders |
| `scan_alpha.py`, `threshold_scan.py`, `render_glyphs.py` | Build the corpus: find binary alpha, derive thresholded masks, render icon fonts |
| `measure.py`, `analyze.py` | The v2 measurements (synthetic masks, colour-aware contexts) |
| `test_*.py` | CTW against a sequential CTW, and the causality checker against known leaks |

## Rebuilding the corpus

Downloads go into `corpus/` (git-ignored), each archive in its own folder:

- QOI benchmark suite: <https://qoiformat.org/benchmark/qoi_benchmark_suite.tar>. Only
  `icon_64`, `icon_512`, `pngimg`, `textures_*` are needed.
- Kenney CC0 packs from <https://kenney.nl/assets>: 1-Bit Pack, Pixel Platformer, Tiny Dungeon,
  Micro Roguelike, Pixel Shmup, Pixel UI Pack.
- Icon fonts: `MaterialIcons-Regular.ttf` (google/material-design-icons) and Font Awesome Free 6
  `fa-solid-900.ttf`, `fa-regular-400.ttf`.

```bash
python research/alpha_binary/render_glyphs.py corpus/glyphs corpus/dl_fonts/MaterialIcons-Regular.ttf corpus/dl_fonts/fa-solid-900.ttf corpus/dl_fonts/fa-regular-400.ttf
python research/alpha_binary/scan_alpha.py corpus/scan_all.json pixelart=corpus/kenney glyphs=corpus/glyphs user=C:/Users/you system=C:/Windows --exclude .
python research/alpha_binary/threshold_scan.py corpus/scan_thr.json icons_thr=corpus/qoi/images/icon_64 cutouts_thr=corpus/qoi/images/pngimg plants_thr=corpus/qoi/images/textures_plants
python research/alpha_binary/harness.py build corpus/manifest.json corpus/scan_all.json corpus/scan_thr.json
```

## Running

Run from this folder, so that `module:function` specs resolve:

```bash
python harness.py run ../../corpus/manifest.json ../../corpus/out.json v2=none@24 new=cost:estimators:lm48_a4
python simulate_decoder.py ../../corpus/manifest_quick.json
python harness.py run ../../corpus/split_test.json ../../corpus/out_test_mix.json mix_real=cost:realconfigs:real_mix
python fixedcodec.py ../../corpus/split_test.json ../../corpus/out_test_fixed.json final max20:TABLE_MAX=20
python fixedcodec.py report ../../corpus/out_test_fixed.json ../../corpus/out_test_mix.json
```

`HARNESS_WORKERS` (default 8) and `HARNESS_PER_CATEGORY` (default 400) are environment variables.
Only paths and statistics are written: no pixel data is copied out of the scanned folders.
