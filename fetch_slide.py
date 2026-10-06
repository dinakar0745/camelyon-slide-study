"""Download one slide, cut it into tissue tiles, then delete the slide.

For each slide this leaves:
    data/tiles/<slide>.npz   tile coordinates, tissue and tumour fractions, metadata (kept)
    data/tiles/<slide>.jpg   thumbnail for heatmap figures (kept)
    data/cache/<slide>.npy   raw tile pixels, uint8 [n, 224, 224, 3] (deleted after embedding)

Everything that needs the slide itself is done in this one pass, because the slide is gone
afterwards. Normally called by run.py; to process a single slide by hand:

    python fetch_slide.py tumor_001
"""
import csv
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

import budget
import config
import tiling

Image.MAX_IMAGE_PIXELS = None  # the low-resolution overview of a slide is still a large image
TILE_BYTES = config.TILE * config.TILE * 3


def read_manifest() -> list[dict]:
    if not config.MANIFEST.exists():
        raise SystemExit(f"{config.MANIFEST.name} not found: run `python manifest.py` first")
    with open(config.MANIFEST, newline="") as f:
        return list(csv.DictReader(f))


def tiles_path(slide: str) -> Path:
    return config.TILES / f"{slide}.npz"


def cache_path(slide: str) -> Path:
    return config.CACHE / f"{slide}.npy"


def download(url: str, dest: Path, size: int, md5: str = "", attempts: int = 8) -> None:
    """Download `url` to `dest`, resuming a partial file, then check its size and checksum."""
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        have = part.stat().st_size if part.exists() else 0
        if have > size:
            part.unlink()
            have = 0
        if have == size:
            break
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"})
            with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab") as f:
                if r.status != 206:  # server ignored the range: start again from the top
                    f.truncate(0)
                while chunk := r.read(1 << 20):
                    f.write(chunk)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code in (403, 404):
                raise
            wait = min(60, 2 ** attempt)
            print(f"  download interrupted ({e}); retry {attempt}/{attempts} in {wait}s")
            time.sleep(wait)
    if part.stat().st_size != size:
        raise IOError(f"{dest.name}: got {part.stat().st_size} bytes, expected {size}")
    if md5:
        h = hashlib.md5()
        with open(part, "rb") as f:
            while chunk := f.read(1 << 22):
                h.update(chunk)
        if h.hexdigest() != md5.lower():
            part.unlink()
            raise IOError(f"{dest.name}: checksum mismatch, partial file removed")
    os.replace(part, dest)


def _level_for(slide, downsample: float) -> int:
    """Coarsest pyramid level that is no coarser than `downsample` (with 1% tolerance)."""
    ok = [i for i, d in enumerate(slide.level_downsamples) if d <= downsample * 1.01]
    return max(ok, key=lambda i: slide.level_downsamples[i])


def overview(slide) -> np.ndarray:
    """The whole slide as RGB at 1/MASK_DOWNSAMPLE scale."""
    w0, h0 = slide.dimensions
    level = _level_for(slide, config.MASK_DOWNSAMPLE)
    if slide.level_downsamples[level] < config.MASK_DOWNSAMPLE / 4:
        raise ValueError(f"no pyramid level near 1/{config.MASK_DOWNSAMPLE}; levels: {slide.level_downsamples}")
    img = slide.read_region((0, 0), level, slide.level_dimensions[level]).convert("RGB")
    target = (max(1, w0 // config.MASK_DOWNSAMPLE), max(1, h0 // config.MASK_DOWNSAMPLE))
    if img.size != target:
        img = img.resize(target, Image.BILINEAR)
    return np.asarray(img)


def tile_slide(slide_file: Path, row: dict) -> int:
    """Cut `slide_file` into tissue tiles and write the three outputs. Returns the tile count."""
    from tiffslide import TiffSlide  # imported here so the rest of the module needs no slide reader

    name = row["slide"]
    config.TILES.mkdir(parents=True, exist_ok=True)
    config.CACHE.mkdir(parents=True, exist_ok=True)
    slide = TiffSlide(str(slide_file))
    try:
        w0, h0 = slide.dimensions
        low = overview(slide)
        tile0 = config.TILE * config.DOWNSAMPLE
        coords, tissue = tiling.grid_fractions(tiling.tissue_mask(low), w0, h0, tile0, config.MASK_DOWNSAMPLE)

        tumour = np.zeros(len(coords), dtype=np.float32)
        ann = config.ANNOTATIONS / f"{name}.xml"
        if ann.exists():
            lesions, exclusions = tiling.parse_annotation(ann.read_text())
            lmask = tiling.lesion_mask(low.shape[:2], lesions, exclusions, config.MASK_DOWNSAMPLE)
            tumour = tiling.grid_fractions(lmask, w0, h0, tile0, config.MASK_DOWNSAMPLE)[1]

        keep = tissue >= config.MIN_TISSUE
        coords, tissue, tumour = coords[keep], tissue[keep], tumour[keep]
        n = len(coords)

        level = _level_for(slide, config.DOWNSAMPLE)
        level_ds = float(slide.level_downsamples[level])
        src = round(tile0 / level_ds)
        if n:
            budget.reserve(n * TILE_BYTES, f"tile cache for {name}")
            tmp = cache_path(name).with_suffix(".tmp")
            out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(n, config.TILE, config.TILE, 3))
            for i, (x, y) in enumerate(coords):
                tile = slide.read_region((int(x), int(y)), level, (src, src)).convert("RGB")
                if src != config.TILE:
                    tile = tile.resize((config.TILE, config.TILE), Image.LANCZOS)
                out[i] = np.asarray(tile)
            out.flush()
            del out
            os.replace(tmp, cache_path(name))

        budget.reserve(4 << 20, f"tile index and thumbnail for {name}")
        thumb = Image.fromarray(low)
        thumb.thumbnail((low.shape[1] // 2, low.shape[0] // 2))
        thumb.save(config.TILES / f"{name}.jpg", quality=85)
        meta = dict(slide=name, split=row["split"], label=row["label"], lesion=row.get("lesion", ""),
                    width0=w0, height0=h0, tile=config.TILE, downsample=config.DOWNSAMPLE,
                    level=level, level_downsample=level_ds, mask_downsample=config.MASK_DOWNSAMPLE,
                    min_tissue=config.MIN_TISSUE, annotated=ann.exists(),
                    mpp_x=slide.properties.get("tiffslide.mpp-x"), n_tiles=n)
        # Written last: its presence means the slide was tiled completely.
        tmp = tiles_path(name).with_suffix(".tmp.npz")
        np.savez(tmp, coords=coords, tissue=tissue.astype(np.float16), tumour=tumour.astype(np.float16),
                 meta=json.dumps(meta))
        os.replace(tmp, tiles_path(name))
        return n
    finally:
        slide.close()


def n_tiles(slide: str) -> int:
    with np.load(tiles_path(slide)) as z:
        return len(z["coords"])


def is_cached(slide: str) -> bool:
    """True if the slide's tiles are ready for embedding (or it has no tissue tiles at all)."""
    return tiles_path(slide).exists() and (cache_path(slide).exists() or n_tiles(slide) == 0)


def process(row: dict) -> int:
    """Download, tile and delete one slide. Safe to call again after an interruption."""
    name = row["slide"]
    if is_cached(name):
        return n_tiles(name)
    size = int(row["bytes"])
    slide_file = config.SLIDES / f"{name}.tif"
    if not slide_file.exists():
        partial = slide_file.with_name(slide_file.name + ".part")
        have = partial.stat().st_size if partial.exists() else 0
        budget.reserve(max(0, size - have), f"slide {name}")
        t0 = time.time()
        download(f"{config.BUCKET_URL}/{urllib.parse.quote(row['key'])}", slide_file, size, row.get("md5", ""))
        print(f"  downloaded {size / budget.GB:.2f} GB in {time.time() - t0:.0f}s")
    t0 = time.time()
    try:
        n = tile_slide(slide_file, row)
    finally:
        for stale in config.CACHE.glob(f"{name}.tmp"):
            stale.unlink()
    slide_file.unlink()
    print(f"  {n} tiles in {time.time() - t0:.0f}s, slide deleted")
    return n


if __name__ == "__main__":
    wanted = set(sys.argv[1:])
    if not wanted:
        raise SystemExit("usage: python fetch_slide.py <slide> [<slide> ...]")
    for r in read_manifest():
        if r["slide"] in wanted:
            print(r["slide"])
            process(r)
