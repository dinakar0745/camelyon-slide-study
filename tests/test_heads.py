import csv

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

import config
import evaluate
import heads
import train_heads

DIM = 16


def test_auc_matches_sklearn_including_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = np.round(rng.normal(size=200) + y, 1)   # rounding creates ties
    assert evaluate.auc(y, s) == pytest.approx(roc_auc_score(y, s))
    assert np.isnan(evaluate.auc(np.zeros(5, dtype=int), s[:5]))


def test_boot_auc_matches_the_plain_auc_on_each_resample():
    rng = np.random.default_rng(3)
    y = np.array([1] * 12 + [0] * 30)
    scores = rng.normal(size=(2, 42)) + y
    idx = evaluate.boot_indices(y, rng)[:50]
    b = evaluate.boot_auc(scores, idx, 12)
    want = [[evaluate.auc(y[i], s[i]) for i in idx] for s in scores]
    assert b.shape == (2, 50) and np.allclose(b, want)


def test_tile_stats_match_numpy():
    rng = np.random.default_rng(1)
    bags = [rng.normal(3, 2, size=(n, 8)).astype(np.float16) for n in (5, 50, 500)]
    mean, std = heads.tile_stats(bags)
    allx = np.concatenate(bags).astype(np.float64)
    assert np.allclose(mean, allx.mean(0), atol=1e-4) and np.allclose(std, allx.std(0), atol=1e-3)


def test_pooling_raw_then_standardising_is_exact():
    rng = np.random.default_rng(2)
    bag = rng.normal(size=(40, 6)).astype(np.float32)
    mean, std = rng.normal(size=6).astype(np.float32), rng.uniform(0.5, 2, 6).astype(np.float32)
    for how, f in (("mean", np.mean), ("max", np.max)):
        assert np.allclose((heads.pool(bag, how) - mean) / std, f((bag - mean) / std, axis=0), atol=1e-5)


def test_split_is_stratified_and_fixed_by_seed():
    y = np.array([1] * 111 + [0] * 159)
    fit, val = train_heads.split(y, 3)
    assert len(val) == 54 and len(set(fit) | set(val)) == 270 and not set(fit) & set(val)
    assert abs(y[val].mean() - y.mean()) < 0.02
    assert np.array_equal(val, train_heads.split(y, 3)[1]) and not np.array_equal(val, train_heads.split(y, 4)[1])


# --- End to end on synthetic bags ------------------------------------------------------
# Normal slides: pure noise. Tumour slides: a few tiles shifted along four dimensions
# (30 for "macro", 3 for "micro"), so mean pooling sees a weak signal and attention a strong
# one. Degraded conditions add noise; "mixed" training bags add it to a random half of tiles.

def make_bag(rng, n_pos, noise=0.0, noisy_fraction=1.0):
    n = rng.integers(150, 300)
    x = rng.normal(size=(n, DIM))
    x[:n_pos, :4] += 2.5
    noisy = rng.random(n) < noisy_fraction
    x[noisy] += rng.normal(scale=noise, size=(noisy.sum(), DIM))
    return x.astype(np.float16)


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EMBEDDINGS", tmp_path / "embeddings")
    monkeypatch.setattr(config, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(config, "MANIFEST", tmp_path / "slides.csv")
    monkeypatch.setattr(config, "ABMIL", dict(config.ABMIL, epochs=12, lr=1e-3))
    (tmp_path / "embeddings" / "toy").mkdir(parents=True)
    (tmp_path / "results").mkdir()
    rng = np.random.default_rng(0)
    rows = []
    for i in range(60):
        tumour = i % 2 == 0
        n_pos = int(rng.integers(3, 30)) if tumour else 0
        seed_rng = np.random.default_rng(1000 + i)
        clean = make_bag(np.random.default_rng(1000 + i), n_pos)
        mixed = make_bag(seed_rng, n_pos, noise=1.0, noisy_fraction=0.5)
        name = f"tumor_{i:03d}" if tumour else f"normal_{i:03d}"
        np.savez(tmp_path / "embeddings" / "toy" / f"{name}.npz", clean=clean, mixed=mixed)
        rows.append(dict(slide=name, split="train", label="tumor" if tumour else "normal", lesion=""))
    for i in range(30):
        lesion = "macro" if i < 8 else "micro" if i < 15 else ""
        n_pos = {"macro": 30, "micro": 3, "": 0}[lesion]
        arrays = {}
        for c in config.TEST_CONDITIONS:
            noise = 0.0 if c == "clean" else 1.0
            arrays[c.replace(":", "_")] = make_bag(np.random.default_rng(5000 + i), n_pos, noise=noise)
        name = f"test_{i:03d}"
        np.savez(tmp_path / "embeddings" / "toy" / f"{name}.npz", **arrays)
        rows.append(dict(slide=name, split="test", label="tumor" if lesion else "normal", lesion=lesion))
    with open(config.MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["slide", "split", "label", "lesion"])
        w.writeheader()
        w.writerows(rows)
    return tmp_path


def test_train_and_evaluate_end_to_end(project):
    pytest.importorskip("torch")   # the attention head needs PyTorch; run in the main environment
    rows = train_heads.read_manifest()
    preds, sel = train_heads.train_encoder("toy", rows, config.HEADS, [0, 1], "cpu")
    assert len(sel) == len(config.HEADS) * 2 * 2
    assert len(preds) == len(config.HEADS) * 2 * 2 * len(config.TEST_CONDITIONS) * 30
    assert all(s["n_fit"] == 48 and s["n_val"] == 12 for s in sel)
    train_heads.write_csv(config.RESULTS / "predictions_toy.csv", train_heads.PRED_COLUMNS, preds)

    by_seed, summary, gaps = evaluate.evaluate(evaluate.load())
    get = {(r["head"], r["train_set"], r["condition"], r["subset"]): r for r in summary}
    # Attention finds a handful of shifted tiles; the mean is diluted by hundreds of normal ones.
    assert get[("abmil", "clean", "clean", "all")]["auc_mean"] > 0.95
    assert get[("abmil", "clean", "clean", "micro")]["auc_mean"] > get[("mean", "clean", "clean", "micro")]["auc_mean"]
    assert get[("max", "clean", "clean", "macro")]["auc_mean"] > 0.95
    for r in summary:
        assert r["ci_low"] - 0.02 <= r["auc_mean"] <= r["ci_high"] + 0.02
    for r in gaps:
        assert r["ci_low"] - 0.02 <= r["gain_mean"] <= r["ci_high"] + 0.02
    assert len(gaps) == len(config.HEADS) * len(config.TEST_CONDITIONS) * len(evaluate.SUBSETS)
    assert len(by_seed) == len(config.HEADS) * 2 * len(config.TEST_CONDITIONS) * len(evaluate.SUBSETS) * 2

    md = evaluate.markdown(summary, gaps)
    assert "## abmil: AUC, all test slides" in md and "| toy | mixed |" in md


def test_finished_encoders_are_skipped(project, monkeypatch, capsys):
    (config.RESULTS / "predictions_toy.csv").write_text("x\n")
    monkeypatch.setattr(config, "ENCODERS", {"toy": None})
    monkeypatch.setattr("sys.argv", ["train_heads.py", "--models", "toy", "--device", "cpu"])
    train_heads.main()
    assert "skipped" in capsys.readouterr().out
