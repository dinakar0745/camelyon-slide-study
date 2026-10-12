"""Slide-level AUC from the saved predictions, per condition, with uncertainty.

Reads results/predictions_*.csv and writes
    results/auc_by_seed.csv   AUC for every encoder, head, training set, condition and seed,
                              on all test slides and on macro- and micro-metastasis slides
                              (each against all normal slides)
    results/summary.csv       mean and sd over seeds with a bootstrap 95% interval for the
                              mean, plus the AUC of the seed-averaged score
    results/gap.csv           mixed minus clean, paired by seed and by bootstrap resample
    results/summary.md        the same as readable tables

    python evaluate.py
"""
import csv
import glob
from collections import defaultdict

import numpy as np
from scipy.stats import rankdata

import config

N_BOOT = 2000
SUBSETS = {"all": None, "macro": "macro", "micro": "micro"}


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUC with ties counted as half; nan if a class is missing."""
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def subset_mask(labels: np.ndarray, lesions: np.ndarray, subset: str) -> np.ndarray:
    if SUBSETS[subset] is None:
        return np.ones(len(labels), dtype=bool)
    return (labels == 0) | (lesions == SUBSETS[subset])


def load() -> dict:
    """{(model, head, train_set, condition): {"slides", "labels", "lesions", "scores": [n_seeds, n]}}"""
    raw = defaultdict(lambda: defaultdict(dict))
    meta = {}
    for path in sorted(glob.glob(str(config.RESULTS / "predictions_*.csv"))):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                key = (r["model"], r["head"], r["train_set"], r["condition"])
                raw[key][int(r["seed"])][r["slide"]] = float(r["score"])
                meta[r["slide"]] = (int(r["label"]), r["lesion"])
    out = {}
    for key, by_seed in raw.items():
        seeds = sorted(by_seed)
        slides = sorted(by_seed[seeds[0]])
        out[key] = dict(
            slides=slides,
            labels=np.array([meta[s][0] for s in slides]),
            lesions=np.array([meta[s][1] for s in slides]),
            seeds=seeds,
            scores=np.array([[by_seed[k][s] for s in slides] for k in seeds]),
        )
    return out


def boot_indices(labels: np.ndarray, rng) -> np.ndarray:
    """N_BOOT stratified resamples of slide indices, shape [N_BOOT, n], positives first."""
    pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    return np.concatenate([rng.choice(pos, (N_BOOT, len(pos))), rng.choice(neg, (N_BOOT, len(neg)))], axis=1)


def boot_auc(scores: np.ndarray, idx: np.ndarray, n_pos: int) -> np.ndarray:
    """AUC of each seed's scores on each resample, shape [n_seeds, N_BOOT].

    `idx` comes from boot_indices, so the first n_pos columns of every resample are the
    positives; that lets every resample be ranked at once.
    """
    n_neg = idx.shape[1] - n_pos
    out = []
    for s in scores:
        r = rankdata(s[idx], axis=1)
        out.append((r[:, :n_pos].sum(1) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    return np.array(out)


def ensemble(scores: np.ndarray) -> np.ndarray:
    """Average over seeds after putting each seed's scores on a common scale."""
    z = (scores - scores.mean(1, keepdims=True)) / (scores.std(1, keepdims=True) + 1e-12)
    return z.mean(0)


def interval(values: np.ndarray) -> tuple[float, float]:
    return round(float(np.percentile(values, 2.5)), 4), round(float(np.percentile(values, 97.5)), 4)


def evaluate(data: dict, seed: int = 0):
    """Every interval is a stratified bootstrap over test slides of the statistic it is
    printed next to: the mean over seeds of the AUC, or of the mixed-minus-clean gain.
    The same resamples are used for every encoder, head, training set and condition."""
    by_seed, summary, gaps = [], [], []
    boots, boot_cache = {}, {}
    for (model, head, train_set, condition), d in sorted(data.items()):
        for subset in SUBSETS:
            m = subset_mask(d["labels"], d["lesions"], subset)
            order = np.argsort(-d["labels"][m], kind="stable")  # positives first, as boot_indices assumes
            y, scores = d["labels"][m][order], d["scores"][:, m][:, order]
            per_seed = [auc(y, s) for s in scores]
            for k, a in zip(d["seeds"], per_seed):
                by_seed.append(dict(model=model, head=head, train_set=train_set, condition=condition,
                                    subset=subset, seed=k, auc=round(a, 4)))
            bkey = (subset, tuple(d["slides"]))
            if bkey not in boot_cache:
                boot_cache[bkey] = boot_indices(y, np.random.default_rng(seed))
            b = boot_auc(scores, boot_cache[bkey], int(y.sum()))
            boots[(model, head, train_set, condition, subset)] = b
            low, high = interval(b.mean(0))
            summary.append(dict(model=model, head=head, train_set=train_set, condition=condition, subset=subset,
                                auc_mean=round(float(np.mean(per_seed)), 4), auc_sd=round(float(np.std(per_seed)), 4),
                                ci_low=low, ci_high=high, auc_ensemble=round(auc(y, ensemble(scores)), 4)))

    for (model, head, train_set, condition), d in sorted(data.items()):
        if train_set != "clean" or (model, head, "mixed", condition) not in data:
            continue
        mx = data[(model, head, "mixed", condition)]
        assert d["slides"] == mx["slides"] and d["seeds"] == mx["seeds"]
        for subset in SUBSETS:
            c_rows = [r for r in by_seed if (r["model"], r["head"], r["train_set"], r["condition"], r["subset"])
                      == (model, head, "clean", condition, subset)]
            m_rows = [r for r in by_seed if (r["model"], r["head"], r["train_set"], r["condition"], r["subset"])
                      == (model, head, "mixed", condition, subset)]
            diff = [b["auc"] - a["auc"] for a, b in zip(c_rows, m_rows)]
            gain_boot = (boots[(model, head, "mixed", condition, subset)]
                         - boots[(model, head, "clean", condition, subset)]).mean(0)
            low, high = interval(gain_boot)
            gaps.append(dict(model=model, head=head, condition=condition, subset=subset,
                             gain_mean=round(float(np.mean(diff)), 4), gain_sd=round(float(np.std(diff)), 4),
                             ci_low=low, ci_high=high))
    return by_seed, summary, gaps


def markdown(summary: list[dict], gaps: list[dict]) -> str:
    conditions = config.TEST_CONDITIONS
    models = sorted({r["model"] for r in summary})
    lines = ["# Slide-level AUC on the CAMELYON16 test set", "",
             "Mean over seeds, in percent, with the standard deviation over seeds. "
             "Gains are mixed minus clean, paired by seed, averaged over seeds; the interval is a "
             "stratified bootstrap over test slides of that average.", ""]
    for head in config.HEADS:
        rows = {(r["model"], r["train_set"], r["condition"]): r for r in summary
                if r["head"] == head and r["subset"] == "all"}
        if not rows:
            continue
        lines += [f"## {head}: AUC, all test slides", "",
                  "| encoder | training | " + " | ".join(conditions) + " |",
                  "|---|---|" + "---|" * len(conditions)]
        for model in models:
            for ts in config.TRAIN_SETS:
                cells = [f"{100 * rows[(model, ts, c)]['auc_mean']:.1f} ± {100 * rows[(model, ts, c)]['auc_sd']:.1f}"
                         if (model, ts, c) in rows else "" for c in conditions]
                lines.append(f"| {model} | {ts} | " + " | ".join(cells) + " |")
        g = {(r["model"], r["subset"], r["condition"]): r for r in gaps if r["head"] == head}
        lines += ["", f"## {head}: gain from mixed training (AUC points, 95% bootstrap interval)", "",
                  "| encoder | slides | " + " | ".join(conditions) + " |",
                  "|---|---|" + "---|" * len(conditions)]
        for model in models:
            for subset in SUBSETS:
                cells = []
                for c in conditions:
                    r = g.get((model, subset, c))
                    cells.append(f"{100 * r['gain_mean']:+.1f} [{100 * r['ci_low']:+.1f}, {100 * r['ci_high']:+.1f}]"
                                 if r else "")
                lines.append(f"| {model} | {subset} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def write_csv(path, rows) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    data = load()
    if not data:
        raise SystemExit("no results/predictions_*.csv found: run train_heads.py first")
    by_seed, summary, gaps = evaluate(data)
    write_csv(config.RESULTS / "auc_by_seed.csv", by_seed)
    write_csv(config.RESULTS / "summary.csv", summary)
    write_csv(config.RESULTS / "gap.csv", gaps)
    md = markdown(summary, gaps)
    (config.RESULTS / "summary.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
