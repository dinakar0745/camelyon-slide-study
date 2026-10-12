"""Where does the attention head look, and how far do tumour tiles move under degradation?

Tests one hypothesis for why mixed training helped Phikon and Kaiko but not Phikon-v2: that
Phikon-v2's degraded tumour tiles end up far from clean tumour tiles, so a head cannot learn
one tumour direction that serves both. Three measurements, per encoder and test condition:

1. Attention on tumour. The attention heads are refitted exactly as in train_heads.py (same
   splits, settings and seeds; cached in data/heads/) and run on every test tumour slide.
   For each slide: tile AUC of the attention weights (tiles at least half tumour against
   tumour-free tiles), whether the most-attended tile touches tumour, and the share of
   attention on tumour tiles next to their share of the slide.
2. Drift. Cosine similarity between each tile's degraded and clean embedding, separately for
   tumour and tumour-free tiles, in the standardised space the heads see.
3. Tile probe. Logistic regression on single tiles, trained on clean or on mixed training
   tiles (tumour: at least half tumour, from tumour slides; normal: from normal slides only,
   since 20 tumour slides are not exhaustively annotated), tested on degraded test tiles.
   If this stays high while the attention head fails, the signal survives in the encoder and
   the slide-level head is what loses it.

Writes results/attention_<model>.csv, drift_<model>.csv, tile_probe.csv, heads_refit.csv
and attention_summary.md. Run in the main encoder-or-probe environment:

    python attention_analysis.py
    python attention_analysis.py --summary-only      # rebuild the tables from the CSVs
"""
import argparse
import csv
import glob
import time
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import config
import degradations
import heads
import train_heads

TUMOUR = 0.5            # a tile counts as tumour if at least this fraction is annotated tumour
PROBE_NORMAL_TILES = 30000
PROBE_C = 1.0
HEADS_DIR = config.DATA / "heads"


def tumour_fractions(slides: list[str]) -> list[np.ndarray]:
    out = []
    for s in slides:
        with np.load(config.TILES / f"{s}.npz") as z:
            out.append(z["tumour"].astype(np.float32))
    return out


def fit_or_load_heads(model: str, train: list[dict], bags: dict, seeds: list[int], device: str) -> dict:
    """{(train_set, seed): ABMILHead}, refitted as in train_heads.py and cached on disk."""
    y = np.array([r["label"] == "tumor" for r in train], dtype=int)
    out = {}
    (HEADS_DIR / model).mkdir(parents=True, exist_ok=True)
    for train_set in config.TRAIN_SETS:
        for seed in seeds:
            path = HEADS_DIR / model / f"abmil_{train_set}_seed{seed}.pt"
            if path.exists():
                out[(train_set, seed)] = heads.ABMILHead.load(path, device)
                continue
            t0 = time.time()
            fit, val = train_heads.split(y, seed)
            b = bags[train_set]
            mean, std = heads.tile_stats([b[i] for i in fit])
            head = heads.ABMILHead(device).fit([b[i] for i in fit], y[fit], [b[i] for i in val], y[val],
                                               mean, std, seed)
            head.save(path)
            out[(train_set, seed)] = head
            print(f"    {train_set} seed {seed}: epoch {head.choice}, val auc {head.val_auc:.3f} "
                  f"({time.time() - t0:.0f}s)")
    return out


def attention_metrics(att: np.ndarray, frac: np.ndarray) -> dict:
    pos, neg, any_t = frac >= TUMOUR, frac == 0, frac > 0
    tile_auc = float(roc_auc_score(np.r_[np.ones(pos.sum()), np.zeros(neg.sum())],
                                   np.r_[att[pos], att[neg]])) if pos.any() and neg.any() else float("nan")
    return dict(tile_auc=tile_auc, top1_tumour=int(any_t[int(np.argmax(att))]),
                attention_on_tumour=float(att[any_t].sum()), tumour_share=float(any_t.mean()))


def centred_cosine(a: np.ndarray, b: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    a = (a.astype(np.float32) - mean) / std
    b = (b.astype(np.float32) - mean) / std
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


def probe_training_tiles(bags: list[np.ndarray], fracs: list[np.ndarray], labels: list[str], rng):
    """Tumour tiles from tumour slides, a random sample of tiles from normal slides."""
    pos = np.concatenate([b[f >= TUMOUR] for b, f, l in zip(bags, fracs, labels) if l == "tumor"])
    normal = [b for b, l in zip(bags, labels) if l == "normal"]
    sizes = np.array([len(b) for b in normal])
    pick = rng.choice(sizes.sum(), size=min(PROBE_NORMAL_TILES, sizes.sum()), replace=False)
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    which = np.searchsorted(starts, pick, side="right") - 1
    neg = np.stack([normal[w][p - starts[w]] for w, p in zip(which, pick)])
    x = np.concatenate([pos, neg]).astype(np.float32)
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return x, y


def analyse_encoder(model: str, rows: list[dict], seeds: list[int], device: str):
    rng = np.random.default_rng(0)
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    train_names = [r["slide"] for r in train]
    t0 = time.time()
    bags = {ts: train_heads.load_bags(model, train_names, ts) for ts in config.TRAIN_SETS}
    train_fracs = tumour_fractions(train_names)
    mean, std = heads.tile_stats(bags["clean"])  # drift is measured in the clean-standardised space
    print(f"  loaded training slides ({time.time() - t0:.0f}s)")

    fitted = fit_or_load_heads(model, train, bags, seeds, device)

    probes = {}
    for ts in config.TRAIN_SETS:
        x, y = probe_training_tiles(bags[ts], train_fracs, [r["label"] for r in train], rng)
        probes[ts] = LogisticRegression(C=PROBE_C, max_iter=3000, class_weight="balanced").fit((x - mean) / std, y)
        print(f"  tile probe on {ts} tiles: {int(y.sum())} tumour, {int((1 - y).sum())} normal")
    del bags

    test_names = [r["slide"] for r in test]
    test_fracs = tumour_fractions(test_names)
    tumour_idx = [i for i, r in enumerate(test) if r["label"] == "tumor"]
    clean = train_heads.load_bags(model, test_names, "clean")

    attention_rows, drift_rows, probe_rows, refit_rows = [], [], [], []
    for condition in config.TEST_CONDITIONS:
        t0 = time.time()
        cond = clean if condition == "clean" else train_heads.load_bags(model, test_names, degradations.tag(condition))

        if condition == "clean":
            y = np.array([r["label"] == "tumor" for r in test], dtype=int)
            for (ts, seed), head in fitted.items():
                refit_rows.append(dict(model=model, train_set=ts, seed=seed,
                                       clean_auc=round(heads.safe_auc(y, head.score(cond)), 4)))

        for (ts, seed), head in fitted.items():
            for i in tumour_idx:
                m = attention_metrics(head.attention(cond[i]), test_fracs[i])
                attention_rows.append(dict(model=model, train_set=ts, seed=seed, condition=condition,
                                           slide=test[i]["slide"], lesion=test[i]["lesion"],
                                           n_tiles=len(test_fracs[i]), n_tumour=int((test_fracs[i] >= TUMOUR).sum()),
                                           **{k: round(v, 5) for k, v in m.items()}))

        if condition != "clean":
            for i in tumour_idx:
                cos = centred_cosine(clean[i], cond[i], mean, std)
                f = test_fracs[i]
                drift_rows.append(dict(model=model, condition=condition, slide=test[i]["slide"],
                                       lesion=test[i]["lesion"],
                                       cos_tumour=round(float(cos[f >= TUMOUR].mean()), 4) if (f >= TUMOUR).any() else "",
                                       cos_normal=round(float(cos[f == 0].mean()), 4)))

        # Scored slide by slide, so the ~175,000 normal-slide tiles never sit in memory as float32.
        pos_bags = [cond[i][test_fracs[i] >= TUMOUR] for i in tumour_idx]
        neg_bags = [cond[i] for i, r in enumerate(test) if r["label"] == "normal"]
        n_pos, n_neg = sum(len(b) for b in pos_bags), sum(len(b) for b in neg_bags)
        yt = np.r_[np.ones(n_pos), np.zeros(n_neg)]
        for ts, probe in probes.items():
            s = np.concatenate([probe.decision_function((b.astype(np.float32) - mean) / std)
                                for b in pos_bags + neg_bags if len(b)])
            probe_rows.append(dict(model=model, train_set=ts, condition=condition,
                                   tile_auc=round(float(roc_auc_score(yt, s)), 4), n_tumour=n_pos, n_normal=n_neg))
        print(f"  {condition}: done ({time.time() - t0:.0f}s)")
    return attention_rows, drift_rows, probe_rows, refit_rows


def read_csvs(pattern: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(str(config.RESULTS / pattern))):
        with open(path, newline="") as f:
            rows += list(csv.DictReader(f))
    return rows


def mean_of(values) -> float:
    v = [float(x) for x in values if x not in ("", "nan") and not np.isnan(float(x))]
    return float(np.mean(v)) if v else float("nan")


def summary() -> str:
    att, drift = read_csvs("attention_*.csv"), read_csvs("drift_*.csv")
    probe, refit = read_csvs("tile_probe.csv"), read_csvs("heads_refit.csv")
    conds = config.TEST_CONDITIONS
    models = sorted({r["model"] for r in att})
    head = "| encoder | training | " + " | ".join(conds) + " |"
    rule = "|---|---|" + "---|" * len(conds)
    out = ["# Attention, drift and tile probes on CAMELYON16 test tumour slides", "",
           f"Tumour tiles are at least {TUMOUR:.0%} annotated tumour. Attention numbers are means over "
           "test tumour slides and seeds.", ""]

    committed = defaultdict(list)
    for r in read_csvs("auc_by_seed.csv"):
        if r["head"] == "abmil" and r["condition"] == "clean" and r["subset"] == "all":
            committed[(r["model"], r["train_set"])].append(float(r["auc"]))
    out += ["## Refitted heads against the committed run (clean test AUC, mean over seeds)", "",
            "| encoder | training | committed | refitted |", "|---|---|---|---|"]
    by = defaultdict(list)
    for r in refit:
        by[(r["model"], r["train_set"])].append(float(r["clean_auc"]))
    for (m, ts), v in sorted(by.items()):
        c = committed.get((m, ts))
        out.append(f"| {m} | {ts} | {100 * np.mean(c):.1f} | {100 * np.mean(v):.1f} |" if c else
                   f"| {m} | {ts} | | {100 * np.mean(v):.1f} |")
    out.append("")

    for metric, title, scale in (("tile_auc", "Attention tile AUC: tumour tiles against tumour-free tiles", 100),
                                 ("top1_tumour", "Most-attended tile touches tumour (% of slides)", 100),
                                 ("enrichment", "Attention on tumour tiles / their share of the slide", 1)):
        out += [f"## {title}", "", head, rule]
        for m in models:
            for ts in config.TRAIN_SETS:
                cells = []
                for c in conds:
                    sel = [r for r in att if r["model"] == m and r["train_set"] == ts and r["condition"] == c]
                    if metric == "enrichment":
                        v = mean_of([float(r["attention_on_tumour"]) / float(r["tumour_share"]) for r in sel
                                     if float(r["tumour_share"]) > 0])
                    else:
                        v = mean_of([r[metric] for r in sel])
                    cells.append(f"{scale * v:.1f}" if scale == 100 else f"{v:.1f}x")
                out.append(f"| {m} | {ts} | " + " | ".join(cells) + " |")
        out.append("")

    out += ["## Drift: cosine between degraded and clean tile embeddings (tumour / tumour-free)", "",
            "| encoder | " + " | ".join(c for c in conds if c != "clean") + " |",
            "|---|" + "---|" * (len(conds) - 1)]
    for m in models:
        cells = []
        for c in conds[1:]:
            sel = [r for r in drift if r["model"] == m and r["condition"] == c]
            cells.append(f"{mean_of(r['cos_tumour'] for r in sel):.2f} / {mean_of(r['cos_normal'] for r in sel):.2f}")
        out.append(f"| {m} | " + " | ".join(cells) + " |")
    out.append("")

    out += ["## Tile probe: tile AUC, tumour against normal-slide tiles", "", head, rule]
    for m in models:
        for ts in config.TRAIN_SETS:
            got = {r["condition"]: float(r["tile_auc"]) for r in probe if r["model"] == m and r["train_set"] == ts}
            out.append(f"| {m} | {ts} | " + " | ".join(f"{100 * got[c]:.1f}" if c in got else "" for c in conds) + " |")
    out.append("")
    return "\n".join(out)


def write_csv(path, rows) -> None:
    if rows:
        train_heads.write_csv(path, list(rows[0]), rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", choices=list(config.ENCODERS), default=list(config.ENCODERS))
    ap.add_argument("--seeds", nargs="+", type=int, default=config.SEEDS)
    ap.add_argument("--device", help="cuda or cpu (default: cuda if available)")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    if not args.summary_only:
        rows = train_heads.read_manifest()
        device = train_heads.pick_device(args.device)
        probe_all, refit_all = read_csvs("tile_probe.csv"), read_csvs("heads_refit.csv")
        for model in args.models:
            out = config.RESULTS / f"attention_{model}.csv"
            if out.exists():
                print(f"{model}: {out.name} exists, skipped")
                continue
            print(f"{model} on {device}")
            t0 = time.time()
            att, drift, probe, refit = analyse_encoder(model, rows, args.seeds, device)
            probe_all = [r for r in probe_all if r["model"] != model] + probe
            refit_all = [r for r in refit_all if r["model"] != model] + refit
            write_csv(config.RESULTS / "tile_probe.csv", probe_all)
            write_csv(config.RESULTS / "heads_refit.csv", refit_all)
            write_csv(config.RESULTS / f"drift_{model}.csv", drift)
            write_csv(out, att)  # last: its presence marks the encoder as done
            print(f"{model}: done in {(time.time() - t0) / 60:.1f} min")
    md = summary()
    (config.RESULTS / "attention_summary.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
