import json

import numpy as np
import pytest

import budget
import config
import embed_slides
import fetch_slide
import run

MODELS = ["phikon", "kaiko-b16"]
ROWS = [dict(slide=f"normal_{i:03d}", split="train", label="normal", key="k", bytes="10", md5="")
        for i in range(1, 7)]


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    for name, sub in (("SLIDES", "slides"), ("CACHE", "cache"), ("TILES", "tiles"),
                      ("EMBEDDINGS", "embeddings"), ("ANNOTATIONS", "annotations")):
        monkeypatch.setattr(config, name, tmp_path / "data" / sub)
        (tmp_path / "data" / sub).mkdir(parents=True)
    monkeypatch.setattr(config, "DISK_FLOOR_GB", 0)
    return tmp_path


def fake_process(row, n=2):
    """Stands in for download + tiling: same files, 1000 bytes of tile pixels per slide."""
    slide = row["slide"]
    if fetch_slide.is_cached(slide):
        return n
    budget.reserve(1500, f"tiles for {slide}")
    np.savez(fetch_slide.tiles_path(slide), coords=np.zeros((n, 2)), meta=json.dumps(dict(split=row["split"])))
    fetch_slide.cache_path(slide).write_bytes(b"0" * 1000)
    return n


def fake_embed(models, envs, extra):
    """Stands in for embed_slides.py: 50 bytes per encoder per slide, budget checked first."""
    for m in models:
        for f in sorted(config.CACHE.glob("*.npy")):
            out = embed_slides.embedding_path(m, f.stem)
            if out.exists():
                continue
            try:
                budget.reserve(50, f"{m} embeddings for {f.stem}")
            except budget.BudgetError:
                return False
            out.parent.mkdir(exist_ok=True)
            out.write_bytes(b"0" * 50)
    return True


def drive(monkeypatch, argv=()):
    monkeypatch.setattr(fetch_slide, "read_manifest", lambda: list(ROWS))
    monkeypatch.setattr(fetch_slide, "process", fake_process)
    monkeypatch.setattr(run, "embed_batch", fake_embed)
    monkeypatch.setattr(run, "interpreters", lambda: {"main": __file__, "timm": __file__})
    monkeypatch.setattr(run, "POLL_SECONDS", 0.01)
    monkeypatch.setattr("sys.argv", ["run.py", "--models", *MODELS, *argv])
    run.main()


def test_order_is_fixed_and_independent_of_input_order():
    assert run.ordered(ROWS) == run.ordered(ROWS[::-1])
    assert [r["slide"] for r in run.ordered(ROWS)] != [r["slide"] for r in ROWS]


def test_full_run_embeds_everything_and_leaves_no_tile_pixels(project, monkeypatch):
    monkeypatch.setattr(config, "CACHE_GB", 2500 / budget.GB)   # flush after every third slide
    drive(monkeypatch)
    assert all(run.is_done(r["slide"], MODELS) for r in ROWS)
    assert not list(config.CACHE.glob("*")) and not list(config.SLIDES.glob("*"))


def test_limit_takes_a_prefix_and_a_rerun_continues(project, monkeypatch):
    drive(monkeypatch, ["--limit", "2"])
    first = [r["slide"] for r in ROWS if run.is_done(r["slide"], MODELS)]
    assert sorted(first) == sorted(r["slide"] for r in run.ordered(ROWS)[:2])
    drive(monkeypatch)
    assert all(run.is_done(r["slide"], MODELS) for r in ROWS)


def test_full_cache_is_flushed_to_make_room(project, monkeypatch):
    monkeypatch.setattr(config, "CACHE_GB", 1)                   # never flush on cache size
    monkeypatch.setattr(config, "BUDGET_GB", 6000 / budget.GB)   # kept data fits, six caches do not
    drive(monkeypatch)
    assert all(run.is_done(r["slide"], MODELS) for r in ROWS)
    assert budget.used_bytes() <= 6000 and not list(config.CACHE.glob("*.npy"))


def test_embedding_overlaps_with_fetching(project, monkeypatch):
    """While one batch is being embedded, the fetcher keeps tiling the next slides."""
    import time
    waiting_at_start = []

    def slow_embed(models, envs, extra):
        waiting_at_start.append(len(list(config.CACHE.glob("*.npy"))))
        time.sleep(0.3)
        return fake_embed(models, envs, extra)

    monkeypatch.setattr(fetch_slide, "read_manifest", lambda: list(ROWS))
    monkeypatch.setattr(fetch_slide, "process", fake_process)
    monkeypatch.setattr(run, "embed_batch", slow_embed)
    monkeypatch.setattr(run, "interpreters", lambda: {"main": __file__, "timm": __file__})
    monkeypatch.setattr(run, "POLL_SECONDS", 0.01)
    monkeypatch.setattr("sys.argv", ["run.py", "--models", *MODELS])
    run.main()
    assert all(run.is_done(r["slide"], MODELS) for r in ROWS)
    assert len(waiting_at_start) < len(ROWS) and max(waiting_at_start) > 1   # later batches grew meanwhile


def test_cache_limit_pauses_fetching(project, monkeypatch):
    monkeypatch.setattr(config, "CACHE_GB", 2500 / budget.GB)   # room for about two slides of tiles
    peak = []
    real = fake_embed

    def watching_embed(models, envs, extra):
        peak.append(budget.folder_bytes(config.CACHE))
        return real(models, envs, extra)

    monkeypatch.setattr(fetch_slide, "read_manifest", lambda: list(ROWS))
    monkeypatch.setattr(fetch_slide, "process", fake_process)
    monkeypatch.setattr(run, "embed_batch", watching_embed)
    monkeypatch.setattr(run, "interpreters", lambda: {"main": __file__, "timm": __file__})
    monkeypatch.setattr(run, "POLL_SECONDS", 0.01)
    monkeypatch.setattr("sys.argv", ["run.py", "--models", *MODELS])
    run.main()
    assert all(run.is_done(r["slide"], MODELS) for r in ROWS)
    assert max(peak) <= 3000      # never more than the limit plus the slide that crossed it


def test_a_failed_download_is_retried(project, monkeypatch, capsys):
    calls = {"n": 0}

    def flaky(row):
        calls["n"] += 1
        if calls["n"] == 1:
            raise IOError("connection reset")
        return fake_process(row)

    monkeypatch.setattr(fetch_slide, "read_manifest", lambda: list(ROWS))
    monkeypatch.setattr(fetch_slide, "process", flaky)
    monkeypatch.setattr(run, "embed_batch", fake_embed)
    monkeypatch.setattr(run, "interpreters", lambda: {"main": __file__, "timm": __file__})
    monkeypatch.setattr(run, "POLL_SECONDS", 0.01)
    monkeypatch.setattr(run, "RETRY_STEP_SECONDS", 0.001)
    monkeypatch.setattr("sys.argv", ["run.py", "--models", *MODELS])
    run.main()
    assert "trying again" in capsys.readouterr().out
    assert all(run.is_done(r["slide"], MODELS) for r in ROWS)


def test_repeated_download_failure_stops_with_an_error(project, monkeypatch):
    def broken(row):
        raise IOError("no route to host")

    monkeypatch.setattr(fetch_slide, "read_manifest", lambda: list(ROWS))
    monkeypatch.setattr(fetch_slide, "process", broken)
    monkeypatch.setattr(run, "embed_batch", fake_embed)
    monkeypatch.setattr(run, "interpreters", lambda: {"main": __file__, "timm": __file__})
    monkeypatch.setattr(run, "POLL_SECONDS", 0.01)
    monkeypatch.setattr(run, "RETRY_STEP_SECONDS", 0.001)
    monkeypatch.setattr(run, "FETCH_RETRIES", 3)
    monkeypatch.setattr("sys.argv", ["run.py", "--models", *MODELS])
    with pytest.raises(SystemExit, match="repeated failures"):
        run.main()


def test_limited_run_projects_to_the_whole_manifest(project, monkeypatch, capsys):
    drive(monkeypatch, ["--limit", "2"])
    assert f"projected kept data for all {len(ROWS)} slides" in capsys.readouterr().out


def test_budget_stops_the_run_cleanly(project, monkeypatch, capsys):
    monkeypatch.setattr(config, "CACHE_GB", 1)
    monkeypatch.setattr(config, "BUDGET_GB", 2600 / budget.GB)   # kept data for about three slides
    drive(monkeypatch)
    assert "stopped" in capsys.readouterr().out
    assert budget.used_bytes() <= 2600
    done = [r for r in ROWS if run.is_done(r["slide"], MODELS)]
    assert 0 < len(done) < len(ROWS)


def test_leftover_cache_is_embedded_on_restart(project, monkeypatch):
    fake_process(ROWS[0])                                       # tiled, then the run was killed
    drive(monkeypatch, ["--limit", "0"])
    assert run.is_done(ROWS[0]["slide"], MODELS) and not list(config.CACHE.glob("*.npy"))


def test_slide_without_tissue_counts_as_done(project):
    np.savez(fetch_slide.tiles_path("normal_001"), coords=np.zeros((0, 2)), meta="{}")
    assert run.is_done("normal_001", MODELS)
