# Binary alpha measurements

Scripts behind the numbers in [docs/alpha-binary.md](../../docs/alpha-binary.md). They are
research tools, not part of the codec.

| File | What it does |
|---|---|
| `idealcost.py` | Exact ideal code length of adaptive context models (KT templates, CTW) computed from counts, so no coder is needed |
| `test_idealcost.py` | Checks the count-based CTW against a pixel-by-pixel CTW |
| `scan_alpha.py` | Walks folders, classifies every PNG/GIF's alpha (none, constant, binary, general), and records colour statistics for binary-alpha images |
| `measure.py` | Codes each mask with JPEG XL, WebP, PNG and AVIF (lossless), and computes the ideal costs, including colour-aware contexts |
| `analyze.py` | Turns `measure.py` output into the tables used in the doc |

```bash
python research/alpha_binary/scan_alpha.py out/scan.json "C:\Program Files" "C:\Windows"
python research/alpha_binary/measure.py out/scan.json out/results.json
python research/alpha_binary/analyze.py out/results.json
```

The `bench` extra (`pip install -e ".[bench]"`) provides the reference codecs. The scanned
images stay on the machine: only file paths and statistics are written to `out/`, which is
git-ignored.
