# CAMELYON16 slide-level robustness study

Follow-up to [encoder-or-probe](https://github.com/dinakar0745/encoder-or-probe). That study
found that much of the accuracy a pathology foundation model loses under image degradation
belongs to the linear probe, not the frozen encoder. It was limited to patch classification.

This project asks the same question at slide level, on lymph-node metastasis detection in
CAMELYON16: when the head is a slide-level aggregator over frozen tile embeddings, does a
head trained on clean slides still lose more under degradation than one trained on a mix?

**Status:** the extraction pipeline has run end to end on the first ten slides with all four
encoders. The full extraction, aggregator training and evaluation are still to do. There are
no results.

## Design

| | |
| --- | --- |
| Data | CAMELYON16, 399 slides, from the public AWS bucket; official train and test split |
| Tiles | 224 px at about 10x (level-0 downsample 4), non-overlapping, at least 25% tissue |
| Encoders | Phikon, Phikon-v2, Kaiko ViT-B/16, DINOv2-L (frozen, CLS embedding) |
| Test conditions | clean, blur sigma 2 and 4, JPEG quality 30, 15 and 10, stain alpha 0.3 |
| Training sets | clean; and mixed, where each tile is drawn at random from clean, blur 2 and 4, JPEG 30 and 15, stain 0.3 |

JPEG quality 10 is never seen in training. Every setting is in `config.py`.

Working at 10x keeps the study inside a laptop's disk and GPU. The cost is sensitivity to
micrometastases and isolated tumour cells, which any results will have to be read against.

## Disk use

The dataset is several hundred GB; this project is capped at **50 GB** (`BUDGET_GB` in
`config.py`). A slide is downloaded, cut into tiles and deleted within minutes, so at most
one is on disk at a time. Tile pixels are kept only until every encoder has embedded them.
What stays is the embeddings, tile coordinates and thumbnails.

`budget.py` enforces the cap: every large write is checked first, and the run stops cleanly
before going past it. `python budget.py` shows what is using the space.

Because slides are deleted, everything that needs the slide itself is computed in the one
pass: tile coordinates, tissue fraction, tumour fraction from the lesion annotations, and a
thumbnail. Adding an encoder, a condition or a magnification later means downloading again.

## Running it

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests

python manifest.py           # lists the bucket, writes slides.csv, fetches annotations
python run.py --limit 10     # ten slides, all four encoders
python run.py                # the rest
```

`run.py` downloads and tiles in the background while the GPU embeds, so the GPU does not wait
for downloads. It can be stopped and restarted at any point, and a failed download is retried
for about an hour and a half before the run gives up. After a limited run it prints a
projection of the final size; check that it fits before starting the full run.

Embedding runs in the two environments built for encoder-or-probe, found by default in
`../degradation_study`. To use others, copy `envs.example.json` to `envs.json` and edit it.
`--amp` roughly halves embedding time on the GPU at the cost of slightly different embeddings.

## Files

```
config.py          every setting: paths, budget, tiling, conditions, encoders
budget.py          the 50 GB cap
manifest.py        bucket listing -> slides.csv, plus lesion annotations
tiling.py          tissue mask, tile grid, annotation rasterising
fetch_slide.py     download one slide, tile it, delete it
embed_slides.py    embed cached tiles with one encoder (runs in that encoder's environment)
run.py             drives the above in batches
degradations.py    blur, JPEG and stain shift, unchanged from encoder-or-probe
tests/             run with `python -m pytest tests`
```

Written under `data/` (not in git):

```
data/tiles/<slide>.npz               coords, tissue, tumour (fraction per tile), meta
data/tiles/<slide>.jpg               thumbnail
data/embeddings/<model>/<slide>.npz  one [n_tiles, dim] float16 array per condition
```

## Licence and data terms

Code: MIT. CAMELYON16 is released under CC0. The encoders keep their own licences, as listed
in the encoder-or-probe README; Phikon, Phikon-v2 and the Kaiko encoder are non-commercial.

AI assistance: Claude (Anthropic) was used to help write the code.
