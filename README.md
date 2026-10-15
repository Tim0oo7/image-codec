# image-codec

An experimental **lossless** image compression format, aiming to be competitive
with lossless AVIF and JPEG XL.

> Status: early research. There is no stable format yet; the bitstream will change.

## Layout

```
src/image_codec/   the codec (encoder/decoder)
tests/             pytest tests
testdata/          small test images committed to the repo
corpus/            larger benchmark images (git-ignored, keep locally)
docs/              format specification and notes
```

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -e ".[dev,bench]"
```

- `dev`: pytest and ruff
- `bench`: [imagecodecs](https://pypi.org/project/imagecodecs/) (PNG, WebP, AVIF,
  JPEG XL, QOI, ...) and matplotlib, for comparing against existing formats

## Development

```bash
pytest            # run tests
ruff check .      # lint
ruff format .     # format
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
