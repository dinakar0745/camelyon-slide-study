import csv

import numpy as np
import pytest

import attention_analysis as aa
import config
import train_heads

DIM = 16


def test_attention_metrics():
    frac = np.array([1.0, 0.8, 0.3, 0.0, 0.0, 0.0])
    att = np.array([0.4, 0.3, 0.1, 0.1, 0.05, 0.05])
    m = aa.attention_metrics(att, frac)
    assert m["tile_auc"] == 1.0 and m["top1_tumour"] == 1
    assert m["attention_on_tumour"] == pytest.approx(0.8) and m["tumour_share"] == pytest.approx(0.5)
    m = aa.attention_metrics(att[::-1], frac)       # attention on the wrong tiles
    assert m["tile_auc"] == 0.0 and m["top1_tumour"] == 0
    assert np.isnan(aa.attention_metrics(att, np.array([0.3, 0, 0, 0, 0, 0]))["tile_auc"])  # no whole-tumour tile


def test_centred_cosine():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(50, DIM)).astype(np.float16)
    mean, std = np.zeros(DIM, np.float32), np.ones(DIM, np.float32)
    assert np.allclose(aa.centred_cosine(a, a, mean, std), 1, atol=1e-3)
    assert np.allclose(aa.centred_cosine(a, -a, mean, std), -1, atol=1e-3)


def test_probe_training_tiles_takes_tumour_from_tumour_slides_and_normal_from_normal_slides():
    bags = [np.full((4, 2), 1.0), np.full((6, 2), 2.0), np.full((5, 2), 3.0)]
    fracs = [np.array([1, 1, 0, 0.2]), np.zeros(6), np.zeros(5)]
    x, y = aa.probe_training_tiles(bags, fracs, ["tumor", "normal", "normal"], np.random.default_rng(0))
    assert y.sum() == 2 and (x[y == 1] == 1).all()
    assert set(np.unique(x[y == 0])) <= {2.0, 3.0} and len(y) == 2 + 11   # all 11 normal tiles, fewer than the cap


# --- End to end on synthetic slides ----------------------------------------------------
# As in test_heads: tumour tiles are shifted along four dimensions and are the first rows of
# each tumour bag; their tumour fraction is 1, two more rows are partial (0.3), the rest 0.

def make(rng, n_pos, noise=0.0, noisy_fraction=1.0):
    n = int(rng.integers(150, 300))
    x = rng.normal(size=(n, DIM))
    x[:n_pos, :4] += 2.5
    noisy = rng.random(n) < noisy_fraction
    x[noisy] += rng.normal(scale=noise, size=(noisy.sum(), DIM))
    frac = np.zeros(n, np.float16)
    frac[:n_pos] = 1.0
    if n_pos:
        frac[n_pos:n_pos + 2] = 0.3
    return x.astype(np.float16), frac


@pytest.fixture
def project(tmp_path, monkeypatch):
    for name, sub in (("EMBEDDINGS", "embeddings"), ("TILES", "tiles"), ("RESULTS", "results"), ("DATA", "data")):
        monkeypatch.setattr(config, name, tmp_path / sub)
        (tmp_path / sub).mkdir()
    monkeypatch.setattr(config, "MANIFEST", tmp_path / "slides.csv")
    monkeypatch.setattr(aa, "HEADS_DIR", tmp_path / "data" / "heads")
    monkeypatch.setattr(config, "ABMIL", dict(config.ABMIL, epochs=12, lr=1e-3))
    (tmp_path / "embeddings" / "toy").mkdir()
    rows = []
    for i in range(60):
        tumour = i % 2 == 0
        n_pos = int(np.random.default_rng(i).integers(3, 30)) if tumour else 0
        clean, frac = make(np.random.default_rng(1000 + i), n_pos)
        mixed, _ = make(np.random.default_rng(1000 + i), n_pos, noise=1.0, noisy_fraction=0.5)
        name = f"tumor_{i:03d}" if tumour else f"normal_{i:03d}"
        np.savez(tmp_path / "embeddings" / "toy" / f"{name}.npz", clean=clean, mixed=mixed)
        np.savez(tmp_path / "tiles" / f"{name}.npz", tumour=frac)
        rows.append(dict(slide=name, split="train", label="tumor" if tumour else "normal", lesion=""))
    for i in range(30):
        lesion = "macro" if i < 8 else "micro" if i < 15 else ""
        n_pos = {"macro": 30, "micro": 3, "": 0}[lesion]
        arrays = {}
        for c in config.TEST_CONDITIONS:
            arrays[c.replace(":", "_")], frac = make(np.random.default_rng(5000 + i), n_pos,
                                                     noise=0.0 if c == "clean" else 1.5)
        name = f"test_{i:03d}"
        np.savez(tmp_path / "embeddings" / "toy" / f"{name}.npz", **arrays)
        np.savez(tmp_path / "tiles" / f"{name}.npz", tumour=frac)
        rows.append(dict(slide=name, split="test", label="tumor" if lesion else "normal", lesion=lesion))
    with open(config.MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["slide", "split", "label", "lesion"])
        w.writeheader()
        w.writerows(rows)
    return tmp_path


def test_analysis_end_to_end(project):
    pytest.importorskip("torch")
    rows = train_heads.read_manifest()
    att, drift, probe, refit = aa.analyse_encoder("toy", rows, [0, 1], "cpu")

    assert len(refit) == 4 and all(r["clean_auc"] > 0.9 for r in refit)
    assert len(att) == 4 * len(config.TEST_CONDITIONS) * 15
    clean = [r for r in att if r["condition"] == "clean"]
    assert np.nanmean([r["tile_auc"] for r in clean]) > 0.85
    assert np.mean([r["top1_tumour"] for r in clean]) > 0.8

    assert len(drift) == (len(config.TEST_CONDITIONS) - 1) * 15
    assert all(r["cos_normal"] < 0.99 for r in drift)
    assert all(r["cos_tumour"] != "" for r in drift)   # every synthetic tumour slide has whole-tumour tiles

    got = {(r["train_set"], r["condition"]): r["tile_auc"] for r in probe}
    assert len(got) == 2 * len(config.TEST_CONDITIONS) and got[("clean", "clean")] > 0.95

    # Heads are cached: a second run loads them and reproduces the same scores.
    again = aa.analyse_encoder("toy", rows, [0, 1], "cpu")[3]
    assert [r["clean_auc"] for r in again] == [r["clean_auc"] for r in refit]

    for name, data in (("attention_toy.csv", att), ("drift_toy.csv", drift), ("tile_probe.csv", probe),
                       ("heads_refit.csv", refit)):
        aa.write_csv(config.RESULTS / name, data)
    md = aa.summary()
    for section in ("Refitted heads", "Attention tile AUC", "Drift", "Tile probe"):
        assert section in md
    assert "| toy | mixed |" in md
