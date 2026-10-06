import json

import numpy as np

import config
import embed_slides


def test_conditions_per_split():
    assert embed_slides.conditions_for("train") == ["clean", "mixed"]
    assert embed_slides.conditions_for("test") == config.TEST_CONDITIONS


def test_mixed_assignment_is_repeatable_and_covers_every_condition():
    a = embed_slides.mixed_assignment("tumor_001", 5000)
    assert np.array_equal(a, embed_slides.mixed_assignment("tumor_001", 5000))
    assert not np.array_equal(a, embed_slides.mixed_assignment("tumor_002", 5000))
    counts = np.bincount(a, minlength=len(config.MIX_CONDITIONS))
    assert len(counts) == len(config.MIX_CONDITIONS) and counts.min() > 5000 / len(config.MIX_CONDITIONS) * 0.8


def test_dataset_runs_conditions_back_to_back(tmp_path):
    rng = np.random.default_rng(0)
    tiles = rng.integers(0, 255, size=(3, 224, 224, 3), dtype=np.uint8)
    np.save(tmp_path / "s.npy", tiles)
    flat = ["clean"] * 3 + ["clean", "blur:4", "clean"]      # a clean pass, then a mixed pass
    ds = embed_slides.TileDataset(str(tmp_path / "s.npy"), 3, flat, np.asarray)
    assert len(ds) == 6
    for i in (0, 1, 2, 3, 5):
        assert np.array_equal(ds[i], tiles[i % 3])
    assert ds[4].shape == tiles[1].shape and ds[4].std() < tiles[1].std() / 2   # blurred noise is flatter


def test_per_tile_conditions_follow_the_mixed_assignment():
    got = embed_slides.per_tile_conditions("mixed", "tumor_001", 50)
    want = [config.MIX_CONDITIONS[j] for j in embed_slides.mixed_assignment("tumor_001", 50)]
    assert got == want and embed_slides.per_tile_conditions("jpeg:15", "tumor_001", 2) == ["jpeg:15"] * 2


def test_waiting_slides_skips_embedded_ones(tmp_path, monkeypatch):
    for name, sub in (("CACHE", "cache"), ("TILES", "tiles"), ("EMBEDDINGS", "embeddings")):
        monkeypatch.setattr(config, name, tmp_path / sub)
        (tmp_path / sub).mkdir()
    for slide, split in (("normal_001", "train"), ("test_001", "test")):
        np.save(config.CACHE / f"{slide}.npy", np.zeros((1, 2, 2, 3), dtype=np.uint8))
        np.savez(config.TILES / f"{slide}.npz", coords=np.zeros((1, 2)), meta=json.dumps(dict(split=split)))
    assert embed_slides.waiting_slides("phikon") == [("normal_001", "train"), ("test_001", "test")]
    (config.EMBEDDINGS / "phikon").mkdir()
    np.savez(embed_slides.embedding_path("phikon", "normal_001"), clean=np.zeros(1))
    assert embed_slides.waiting_slides("phikon") == [("test_001", "test")]
    assert len(embed_slides.waiting_slides("phikon-v2")) == 2
