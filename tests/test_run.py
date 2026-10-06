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
    monkeypatch.setattr(config, "BUDGET_GB", 5000 / budget.GB)   # kept data fits, six caches do not
    drive(monkeypatch)
    assert all(run.is_done(r["slide"], MODELS) for r in ROWS)
    assert budget.used_bytes() <= 5000 and not list(config.CACHE.glob("*.npy"))


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
