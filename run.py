"""Runs the whole extraction inside the disk budget.

Two things happen at once. In the background, slides are downloaded and tiled one after
another (each slide is deleted as soon as it is tiled) until the tile cache holds
config.CACHE_GB. In the foreground, whatever is in the cache is embedded with every encoder
and its tile pixels are then deleted, which makes room for more. The GPU therefore does not
wait for downloads. Stop it whenever you like; running it again continues where it left off.

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
import threading
import time
from pathlib import Path

import budget
import config
import embed_slides
import fetch_slide

BUDGET_EXIT = 3  # exit code embed_slides.py uses when the disk budget stops it
POLL_SECONDS = 5
FETCH_RETRIES = 20  # per slide, with waits growing to five minutes: rides out a long network outage
RETRY_STEP_SECONDS = 30


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


def cache_waiting() -> bool:
    """True if some slide is tiled and waiting to be embedded."""
    return any(fetch_slide.tiles_path(f.stem).exists() for f in config.CACHE.glob("*.npy"))


class Fetcher(threading.Thread):
    """Downloads and tiles the pending slides in the background, in order.

    Pauses while the tile cache is full. If the disk budget refuses a slide while tiles are
    still waiting to be embedded, it waits for them to be cleared and tries again; if the
    budget refuses with nothing left to clear, it stops and leaves the reason in `error`.
    """

    def __init__(self, pending: list[dict]):
        super().__init__(daemon=True)
        self.pending = pending
        self.stop = threading.Event()
        self.purges = 0      # incremented by the main thread after each cache purge
        self.error: BaseException | None = None

    def _pause(self, seconds: float) -> bool:
        """Sleep, returning True if asked to stop meanwhile."""
        return self.stop.wait(seconds)

    def _fetch(self, row: dict) -> None:
        failures = 0
        while not self.stop.is_set():
            seen = self.purges
            try:
                fetch_slide.process(row)
                return
            except budget.BudgetError:
                if self.purges == seen and not any(config.CACHE.glob("*.npy")):
                    raise
                while self.purges == seen:  # room will appear when the cache is cleared
                    if self._pause(POLL_SECONDS):
                        return
            except Exception as e:  # network trouble, bad checksum: wait and try the slide again
                failures += 1
                if failures >= FETCH_RETRIES:
                    raise
                wait = min(10, failures) * RETRY_STEP_SECONDS
                print(f"  {row['slide']}: {e}; trying again in {wait:.0f}s ({failures}/{FETCH_RETRIES})")
                if self._pause(wait):
                    return

    def run(self) -> None:
        try:
            for i, row in enumerate(self.pending, 1):
                while budget.folder_bytes(config.CACHE) >= config.CACHE_GB * budget.GB:
                    if self._pause(POLL_SECONDS):
                        return
                if self.stop.is_set():
                    return
                print(f"[{i}/{len(self.pending)}] {row['slide']} ({row['split']}, {row['label']})")
                self._fetch(row)
        except BaseException as e:
            self.error = e


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

    fetcher = Fetcher(pending)
    fetcher.start()
    try:
        while True:
            if cache_waiting():
                ok = embed_batch(args.models, envs, extra)
                purge_cache(args.models)
                fetcher.purges += 1
                if not ok:
                    print("\nstopped by the disk budget while embedding")
                    break
            elif not fetcher.is_alive():
                if not cache_waiting():  # nothing arrived between the two checks
                    break
            else:
                time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        print("\ninterrupted; run again to continue")
    finally:
        fetcher.stop.set()
    if isinstance(fetcher.error, budget.BudgetError):
        print(f"\nstopped: {fetcher.error}")
    print()
    print(summary(rows, args.models, len(all_rows)))
    if fetcher.error is not None and not isinstance(fetcher.error, budget.BudgetError):
        raise SystemExit(f"\ndownloading stopped after repeated failures: {fetcher.error!r}\n"
                         "everything fetched so far was embedded; run again to continue")


if __name__ == "__main__":
    main()
