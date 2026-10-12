"""A slide head built from tile-level supervision: score a slide by its top-k tile scores.

The attention head learns from slide labels only. This head instead uses the lesion
annotations: a logistic regression is trained on single tiles (tumour tiles from tumour
slides, tiles sampled from normal slides, as in attention_analysis.py), and a slide's score
is the mean of its k highest tile scores. If this recovers slide AUC where the attention head
fails under degradation, the tumour signal is in the encoder and the attention head is what
loses it.

Same protocol as train_heads.py: for each seed and training set, the tile classifier is fitted
on tiles of the 80% fitting slides only, and k is chosen on the 20% validation slides. The
test set is used only for the final scores. Predictions go to
results/predictions_tileprobe_<model>.csv with head "tileprobe", which evaluate.py reads
alongside the other heads.

    python tile_probe_slides.py
    python evaluate.py
"""
import argparse
import time

import numpy as np
from sklearn.linear_model import LogisticRegression

import attention_analysis as aa
import config
import degradations
import heads
import train_heads

K_GRID = [1, 3, 5, 10, 25, 50]


def topk_scores(tile_scores: list[np.ndarray], k: int) -> np.ndarray:
    """Mean of the k highest tile scores of each slide (all tiles if a slide has fewer)."""
    return np.array([np.sort(s)[-k:].mean() for s in tile_scores])


class TileProbeHead:
    def fit(self, bags_fit, fracs_fit, labels_fit, bags_val, y_val, mean, std, seed: int):
        self.mean, self.std = mean, std
        x, y = aa.probe_training_tiles(bags_fit, fracs_fit, labels_fit, np.random.default_rng(seed))
        self.probe = LogisticRegression(C=aa.PROBE_C, max_iter=3000, class_weight="balanced",
                                        random_state=seed).fit(self._x(x), y)
        val_tiles = self.tile_scores(bags_val)
        best = None
        for k in K_GRID:  # ascending, so ties keep the smaller k
            auc = heads.safe_auc(y_val, topk_scores(val_tiles, k))
            if best is None or auc > best[0]:
                best = (auc, k)
        self.val_auc, self.choice = best
        return self

    def _x(self, x):
        return (np.asarray(x, dtype=np.float32) - self.mean) / self.std

    def tile_scores(self, bags) -> list[np.ndarray]:
        return [self.probe.decision_function(self._x(b)) for b in bags]

    def score(self, bags) -> np.ndarray:
        return topk_scores(self.tile_scores(bags), self.choice)


def run_encoder(model: str, rows: list[dict], seeds: list[int]):
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    names = [r["slide"] for r in train]
    labels = [r["label"] for r in train]
    y = np.array([lab == "tumor" for lab in labels], dtype=int)
    fracs = aa.tumour_fractions(names)

    fitted, selection = [], []
    for train_set in config.TRAIN_SETS:
        bags = train_heads.load_bags(model, names, train_set)
        for seed in seeds:
            t0 = time.time()
            fit, val = train_heads.split(y, seed)
            mean, std = heads.tile_stats([bags[i] for i in fit])
            head = TileProbeHead().fit([bags[i] for i in fit], [fracs[i] for i in fit], [labels[i] for i in fit],
                                       [bags[i] for i in val], y[val], mean, std, seed)
            fitted.append((train_set, seed, head))
            selection.append(dict(model=model, head="tileprobe", train_set=train_set, seed=seed, choice=head.choice,
                                  val_auc=round(head.val_auc, 4), n_fit=len(fit), n_val=len(val)))
            print(f"  {train_set} seed {seed}: top-{head.choice} tiles, val auc {head.val_auc:.3f} "
                  f"({time.time() - t0:.0f}s)")
        del bags

    predictions = []
    test_names = [r["slide"] for r in test]
    for condition in config.TEST_CONDITIONS:
        t0 = time.time()
        bags = train_heads.load_bags(model, test_names, degradations.tag(condition))
        for train_set, seed, head in fitted:
            for r, s in zip(test, head.score(bags)):
                predictions.append(dict(model=model, head="tileprobe", train_set=train_set, seed=seed,
                                        condition=condition, slide=r["slide"], label=int(r["label"] == "tumor"),
                                        lesion=r["lesion"], score=f"{s:.6g}"))
        print(f"  scored test set under {condition} ({time.time() - t0:.0f}s)")
        del bags
    return predictions, selection


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", choices=list(config.ENCODERS), default=list(config.ENCODERS))
    ap.add_argument("--seeds", nargs="+", type=int, default=config.SEEDS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    rows = train_heads.read_manifest()
    for model in args.models:
        out = config.RESULTS / f"predictions_tileprobe_{model}.csv"
        if out.exists() and not args.force:
            print(f"{model}: {out.name} exists, skipped (use --force to redo)")
            continue
        print(model)
        t0 = time.time()
        predictions, selection = run_encoder(model, rows, args.seeds)
        train_heads.write_csv(config.RESULTS / f"selection_tileprobe_{model}.csv", train_heads.SEL_COLUMNS, selection)
        train_heads.write_csv(out, train_heads.PRED_COLUMNS, predictions)
        print(f"{model}: done in {(time.time() - t0) / 60:.1f} min, wrote {out.name}")


if __name__ == "__main__":
    main()
