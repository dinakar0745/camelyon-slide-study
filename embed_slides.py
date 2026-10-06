"""Embed the cached tiles of every waiting slide with one frozen encoder.

Runs in the encoder's own environment (see config.ENCODERS). For each slide that has tile
pixels in data/cache/ and no embeddings yet, writes

    data/embeddings/<model>/<slide>.npz

with one float16 array [n_tiles, dim] per condition. Test slides get every condition in
config.TEST_CONDITIONS. Training slides get "clean" and "mixed"; in "mixed" each tile is
degraded with a condition drawn at random from config.MIX_CONDITIONS, and "mixed_cond"
records which (as an index into that list).

    python embed_slides.py --model phikon
"""
import argparse
import json
import os
import time
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

import budget
import config
import degradations


def embedding_path(model: str, slide: str) -> Path:
    return config.EMBEDDINGS / model / f"{slide}.npz"


def conditions_for(split: str) -> list[str]:
    return list(config.TEST_CONDITIONS) if split == "test" else ["clean", "mixed"]


def mixed_assignment(slide: str, n: int) -> np.ndarray:
    """Index into config.MIX_CONDITIONS for each tile. Depends only on the slide name."""
    rng = np.random.default_rng(zlib.crc32(slide.encode()))
    return rng.integers(0, len(config.MIX_CONDITIONS), size=n).astype(np.int8)


class TileDataset:
    """Every tile of one slide under every requested condition, preprocessed for the encoder.

    Item i is tile (i mod n) under the condition at position i of `conditions`, so one pass
    over the dataset yields the conditions back to back. A single DataLoader per slide keeps
    worker start-up, which is slow on Windows, to once per slide.
    """

    def __init__(self, cache_file: str, n: int, conditions: list[str], preprocess):
        assert len(conditions) % n == 0
        self.cache_file, self.n, self.conditions, self.preprocess = cache_file, n, conditions, preprocess
        self._tiles = None  # opened in each worker, not pickled

    def __len__(self):
        return len(self.conditions)

    def __getitem__(self, i):
        if self._tiles is None:
            self._tiles = np.load(self.cache_file, mmap_mode="r")
        img = Image.fromarray(np.array(self._tiles[i % self.n]))
        return self.preprocess(degradations.apply(img, self.conditions[i]))


def per_tile_conditions(condition: str, slide: str, n: int) -> list[str]:
    if condition == "mixed":
        return [config.MIX_CONDITIONS[j] for j in mixed_assignment(slide, n)]
    return [condition] * n


def waiting_slides(model: str) -> list[tuple[str, str]]:
    """(slide, split) for every slide with cached tiles and no embeddings for `model`."""
    out = []
    for cache_file in sorted(config.CACHE.glob("*.npy")):
        slide = cache_file.stem
        tiles = config.TILES / f"{slide}.npz"
        if tiles.exists() and not embedding_path(model, slide).exists():
            with np.load(tiles) as z:
                out.append((slide, json.loads(str(z["meta"]))["split"]))
    return out


def load_model(name: str, device: str):
    """Returns (model, forward, preprocess); forward(batch) gives the CLS embedding."""
    library, hub_id, _ = config.ENCODERS[name]
    if library == "timm":
        import timm
        model = timm.create_model(hub_id, pretrained=True, num_classes=0).eval().to(device)
        cfg = timm.data.resolve_model_data_config(model)
        cfg["crop_pct"] = 1.0  # resize to 224 then centre-crop 224, as on the model card
        return model, model, timm.data.create_transform(**cfg, is_training=False)
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained(hub_id)
    model = AutoModel.from_pretrained(hub_id).eval().to(device)
    return (model,
            lambda batch: model(pixel_values=batch).last_hidden_state[:, 0],
            _HFPreprocess(processor))


class _HFPreprocess:
    """Picklable wrapper so DataLoader workers can use a transformers image processor."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, img):
        return self.processor(images=img, return_tensors="pt")["pixel_values"][0]


def main() -> None:
    import torch
    from torch.utils.data import DataLoader

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(config.ENCODERS), required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true",
                    help="half-precision inference on the GPU: roughly twice as fast, embeddings differ slightly")
    args = ap.parse_args()

    slides = waiting_slides(args.model)
    if not slides:
        print(f"{args.model}: nothing to embed")
        return
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"{args.model}: {len(slides)} slides on {device}" + (" (amp)" if args.amp else ""))
    model, forward, preprocess = load_model(args.model, device)
    use_amp = args.amp and device == "cuda"
    (config.EMBEDDINGS / args.model).mkdir(parents=True, exist_ok=True)

    for slide, split in slides:
        cache_file = config.CACHE / f"{slide}.npy"
        n = len(np.load(cache_file, mmap_mode="r"))
        conditions = conditions_for(split)
        budget.reserve(n * 1024 * 2 * len(conditions), f"{args.model} embeddings for {slide}")
        t0 = time.time()
        flat = [c for condition in conditions for c in per_tile_conditions(condition, slide, n)]
        ds = TileDataset(str(cache_file), n, flat, preprocess)
        feats = []
        with torch.inference_mode(), torch.autocast("cuda" if device == "cuda" else "cpu", enabled=use_amp):
            for batch in DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers):
                feats.append(forward(batch.to(device)).float().cpu().numpy())
        feats = np.concatenate(feats).astype(np.float16)
        arrays = {degradations.tag(c): feats[k * n:(k + 1) * n] for k, c in enumerate(conditions)}
        if "mixed" in conditions:
            arrays["mixed_cond"] = mixed_assignment(slide, n)
        out = embedding_path(args.model, slide)
        tmp = out.with_suffix(".tmp.npz")
        np.savez(tmp, **arrays)
        os.replace(tmp, out)
        dt = time.time() - t0
        print(f"  {slide}: {n} tiles x {len(conditions)} conditions in {dt:.0f}s "
              f"({n * len(conditions) / dt:.0f} img/s)")


if __name__ == "__main__":
    try:
        main()
    except budget.BudgetError as e:
        print(f"stopped: {e}")
        raise SystemExit(3)
