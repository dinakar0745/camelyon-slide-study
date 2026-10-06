"""Runs the whole extraction, a few slides at a time, inside the disk budget.

For each batch: download and tile slides (each slide is deleted as soon as it is tiled)
until the tile cache reaches config.CACHE_GB, embed the batch with every encoder, then
delete the batch's tile pixels. Stop it whenever you like; running it again continues
where it left off.

    python run.py --limit 10     # first ten slides: check speed and size before committing
    python run.py                # everything

Embedding runs in the two environments built for encoder-or-probe. By default they are
looked for in ../degradation_study; to point elsewhere, copy envs.example.json to envs.json.
"""
import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import budget
import config
import embed_slides
import fetch_slide

BUDGET_EXIT = 3  # exit code embed_slides.py uses when the disk budget stops it


def interpreters() -> dict[str, str]:
    sibling = config.ROOT.parent / "degradation_study"
    sub = ("Scripts", "python.exe") if sys.platform == "win32" else ("bin", "python")
    envs = {"main": str(sibling.joinpath(".venv", *sub)), "timm": str(sibling.joinpath(".venv-timm", *sub))}
    override = config.ROOT / "envs.json"
    if override.exists():
        envs.update(json.loads(override.read_text()))
    return envs


def ordered(rows: list[dict]) -> list[dict]:
    """A fixed shuffle, so any prefix of the run is a mix of splits and labels."""
    rows = sorted(rows, key=lambda r: r["slide"])
    random.Random(0).shuffle(rows)
    return rows


def is_done(slide: str, models: list[str]) -> bool:
    if not fetch_slide.tiles_path(slide).exists():
        return False
    return all(embed_slides.embedding_path(m, slide).exists() for m in models) or fetch_slide.n_tiles(slide) == 0


def embed_batch(models: list[str], envs: dict[str, str], extra: list[str]) -> bool:
    """Embed the cached tiles with every encoder. False if the disk budget stopped it."""
    for m in models:
        cmd = [envs[config.ENCODERS[m][2]], str(config.ROOT / "embed_slides.py"), "--model", m, *extra]
        code = subprocess.run(cmd, cwd=config.ROOT).returncode
        if code == BUDGET_EXIT:
            return False
        if code != 0:
            raise SystemExit(f"embedding with {m} failed (exit code {code}); fix the error above and rerun")
    return True


def purge_cache(models: list[str]) -> None:
    """Delete the tile pixels of every slide that all encoders have finished."""
    for f in config.CACHE.glob("*.npy"):
        if is_done(f.stem, models):
            f.unlink()


def summary(rows: list[dict], models: list[str], n_all: int) -> str:
    """Progress, disk use, and a projection of the kept data to all `n_all` slides."""
    done = [r for r in rows if is_done(r["slide"], models)]
    lines = [f"{len(done)} of {len(rows)} slides finished", budget.report()]
    if done and len(done) < n_all:
        emb = budget.folder_bytes(config.EMBEDDINGS) + budget.folder_bytes(config.TILES)
        projected = emb / len(done) * n_all / budget.GB
        verdict = "fits" if projected + config.CACHE_GB < config.BUDGET_GB else "DOES NOT FIT"
        lines.append(f"  projected kept data for all {n_all} slides: {projected:.1f} GB, plus up to "
                     f"{config.CACHE_GB} GB of tile cache: {verdict} in {config.BUDGET_GB} GB")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="only the first N slides of the fixed order")
    ap.add_argument("--models", nargs="+", choices=list(config.ENCODERS), default=list(config.ENCODERS))
    ap.add_argument("--amp", action="store_true", help="passed on to embed_slides.py")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    envs = interpreters()
    for env in {config.ENCODERS[m][2] for m in args.models}:
        if not Path(envs[env]).exists():
            raise SystemExit(f'no Python found for the "{env}" environment at {envs[env]}\n'
                             "copy envs.example.json to envs.json and set the right paths")
    extra = ["--batch-size", str(args.batch_size), "--workers", str(args.workers)] + (["--amp"] * args.amp)

    all_rows = ordered(fetch_slide.read_manifest())
    rows = all_rows[: args.limit]
    pending = [r for r in rows if not is_done(r["slide"], args.models)]
    print(f"{len(rows) - len(pending)} of {len(rows)} slides already finished")

    def flush() -> bool:
        ok = embed_batch(args.models, envs, extra)
        purge_cache(args.models)
        return ok

    try:
        ok = True
        if any(config.CACHE.glob("*.npy")):  # left over from an interrupted run
            ok = flush()
        for i, row in enumerate(pending, 1):
            if not ok:
                break
            print(f"[{i}/{len(pending)}] {row['slide']} ({row['split']}, {row['label']})")
            try:
                fetch_slide.process(row)
            except budget.BudgetError:
                if not any(config.CACHE.glob("*.npy")):
                    raise
                # The tile cache is taking the room: embed and clear it, then try once more.
                ok = flush()
                if not ok:
                    break
                fetch_slide.process(row)
            if budget.folder_bytes(config.CACHE) >= config.CACHE_GB * budget.GB:
                ok = flush()
        if ok:
            ok = flush()
        if not ok:
            print("\nstopped by the disk budget; nothing was written past it")
    except budget.BudgetError as e:
        print(f"\nstopped: {e}")
    except KeyboardInterrupt:
        print("\ninterrupted; run again to continue")
    print()
    print(summary(rows, args.models, len(all_rows)))


if __name__ == "__main__":
    main()
