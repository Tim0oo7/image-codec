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
| `baselines.py` | JPEG XL, WebP and PNG as harness cost functions |
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
```

`HARNESS_WORKERS` (default 8) and `HARNESS_PER_CATEGORY` (default 400) are environment variables.
Only paths and statistics are written: no pixel data is copied out of the scanned folders.
