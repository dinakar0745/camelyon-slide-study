"""Train slide-level heads on clean and on mixed training embeddings, and score the test set
under every condition.

For each encoder, seed and training set ("clean" or "mixed"), every head in config.HEADS is
fitted on 80% of the training slides and tuned on the other 20% (the split depends only on
the seed). The fitted heads then score all 129 test slides under every test condition.

Writes, per encoder:
    results/predictions_<model>.csv   one row per head, training set, seed, condition, slide
    results/selection_<model>.csv     the chosen C or epoch and its validation AUC

    python train_heads.py                     # every encoder
    python train_heads.py --models phikon     # one encoder
Finished encoders are skipped; --force redoes them. Run it in the main encoder-or-probe
environment, which has PyTorch and scikit-learn.
"""
import argparse
import csv
import time

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

import config
import degradations
import heads

PRED_COLUMNS = ["model", "head", "train_set", "seed", "condition", "slide", "label", "lesion", "score"]
SEL_COLUMNS = ["model", "head", "train_set", "seed", "choice", "val_auc", "n_fit", "n_val"]


def read_manifest() -> list[dict]:
    with open(config.MANIFEST, newline="") as f:
        return list(csv.DictReader(f))


def load_bags(model: str, slides: list[str], key: str) -> list[np.ndarray]:
    """One float16 array [n_tiles, dim] per slide; only the requested array is read."""
    out = []
    for s in slides:
        with np.load(config.EMBEDDINGS / model / f"{s}.npz") as z:
            out.append(z[key])
    return out


def split(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=config.VAL_FRACTION, random_state=seed)
    fit, val = next(sss.split(np.zeros(len(labels)), labels))
    return np.sort(fit), np.sort(val)


def pick_device(requested: str | None) -> str:
    if requested:
        return requested
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_encoder(model: str, rows: list[dict], head_names: list[str], seeds: list[int], device: str,
                  verbose: bool = False) -> tuple[list[dict], list[dict]]:
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    names = [r["slide"] for r in train]
    y = np.array([r["label"] == "tumor" for r in train], dtype=int)

    fitted = []      # (head name, training set, seed, head)
    selection = []
    for train_set in config.TRAIN_SETS:
        t0 = time.time()
        bags = load_bags(model, names, train_set)
        pooled = {how: np.stack([heads.pool(b, how) for b in bags]) for how in ("mean", "max")}
        print(f"  {train_set}: loaded {len(bags)} training slides in {time.time() - t0:.0f}s")
        for seed in seeds:
            fit, val = split(y, seed)
            mean, std = heads.tile_stats([bags[i] for i in fit])
            for name in head_names:
                t0 = time.time()
                if name == "abmil":
                    head = heads.ABMILHead(device).fit(
                        [bags[i] for i in fit], y[fit], [bags[i] for i in val], y[val], mean, std, seed,
                        log=print if verbose else None)
                else:
                    head = heads.PooledLR(name).fit(
                        pooled[name][fit], y[fit], pooled[name][val], y[val], mean, std, seed)
                fitted.append((name, train_set, seed, head))
                selection.append(dict(model=model, head=name, train_set=train_set, seed=seed, choice=head.choice,
                                      val_auc=round(head.val_auc, 4), n_fit=len(fit), n_val=len(val)))
                print(f"    seed {seed} {name:5s}: chose {head.choice}, val auc {head.val_auc:.3f} "
                      f"({time.time() - t0:.0f}s)")
        del bags, pooled

    predictions = []
    test_names = [r["slide"] for r in test]
    for condition in config.TEST_CONDITIONS:
        t0 = time.time()
        bags = load_bags(model, test_names, degradations.tag(condition))
        pooled = {how: np.stack([heads.pool(b, how) for b in bags]) for how in ("mean", "max")}
        for name, train_set, seed, head in fitted:
            scores = head.score(bags if name == "abmil" else pooled[name])
            for r, s in zip(test, scores):
                predictions.append(dict(model=model, head=name, train_set=train_set, seed=seed, condition=condition,
                                        slide=r["slide"], label=int(r["label"] == "tumor"), lesion=r["lesion"],
                                        score=f"{s:.6g}"))
        print(f"  scored test set under {condition} ({time.time() - t0:.0f}s)")
        del bags, pooled
    return predictions, selection


def write_csv(path, columns, rows) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", choices=list(config.ENCODERS), default=list(config.ENCODERS))
    ap.add_argument("--heads", nargs="+", choices=config.HEADS, default=config.HEADS)
    ap.add_argument("--seeds", nargs="+", type=int, default=config.SEEDS)
    ap.add_argument("--device", help="cuda or cpu (default: cuda if available)")
    ap.add_argument("--force", action="store_true", help="redo encoders that already have predictions")
    ap.add_argument("--verbose", action="store_true", help="print validation AUC after every epoch")
    args = ap.parse_args()

    config.RESULTS.mkdir(exist_ok=True)
    rows = read_manifest()
    device = pick_device(args.device) if "abmil" in args.heads else "cpu"
    for model in args.models:
        out = config.RESULTS / f"predictions_{model}.csv"
        if out.exists() and not args.force:
            print(f"{model}: {out.name} exists, skipped (use --force to redo)")
            continue
        print(f"{model} on {device}")
        t0 = time.time()
        predictions, selection = train_encoder(model, rows, args.heads, args.seeds, device, args.verbose)
        write_csv(config.RESULTS / f"selection_{model}.csv", SEL_COLUMNS, selection)
        write_csv(out, PRED_COLUMNS, predictions)
        print(f"{model}: done in {(time.time() - t0) / 60:.1f} min, wrote {out.name}")


if __name__ == "__main__":
    main()
