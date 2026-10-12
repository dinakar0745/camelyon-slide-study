"""Three follow-up questions on the saved test predictions. No training; runs in seconds.

1. Sensitivity at 95% specificity, overall and by tumour size (number of tiles touching
   tumour). AUC averages over all thresholds; a screening tool runs at one, so this says how
   small a metastasis each head still catches.
2. Hospital and slide size. CAMELYON16 comes from two centres with different scanners (RUMC, 0.243 um/px;
   UMCU, about 0.226 um/px), and the test micrometastases are mostly UMCU while the normal
   slides are mostly RUMC. For every head and condition: AUC within each centre, and how
   well the score tells the centres apart among normal slides (0.5 means it does not), and
   how strongly it tracks the amount of tissue on normal slides.
3. Which tumour slides each head misses, and whether the attention head and the tile head
   miss the same ones.

Reads results/predictions_*.csv and data/tiles/, writes results/sensitivity.csv,
results/centre.csv, results/failures.csv and results/deep_dive.md.

    python deep_dive.py
"""
import csv
import json
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

import config
import evaluate

SPECIFICITY = 0.95
SIZE_BINS = [(1, 10), (11, 50), (51, 200), (201, 10 ** 9)]
FOCUS_HEADS = ["abmil", "tileprobe"]
FOCUS_CONDITIONS = ["clean", "jpeg:15", "jpeg:10", "blur:4"]


def centre_of(mpp: float) -> str:
    return "RUMC" if mpp > 0.235 else "UMCU"


def slide_info(slides: list[str]) -> dict:
    out = {}
    for s in slides:
        with np.load(config.TILES / f"{s}.npz") as z:
            meta = json.loads(str(z["meta"]))
            frac = z["tumour"].astype(np.float32)
        out[s] = dict(centre=centre_of(float(meta["mpp_x"])), n_tiles=len(frac), n_tumour=int((frac > 0).sum()))
    return out


def size_bin(n: int) -> str:
    for lo, hi in SIZE_BINS:
        if lo <= n <= hi:
            return f"{lo}-{hi}" if hi < 10 ** 9 else f">{lo - 1}"
    return "0"


def detected(scores: np.ndarray, labels: np.ndarray, specificity: float = SPECIFICITY) -> np.ndarray:
    """Which slides score above the threshold that keeps `specificity` of normal slides below it."""
    normal = np.sort(scores[labels == 0])
    allowed = int(np.floor((1 - specificity) * len(normal) + 1e-9))  # normal slides allowed above it
    threshold = normal[len(normal) - allowed - 1]
    return scores > threshold


def analyse(data: dict, info: dict):
    sens_rows, centre_rows, fail = [], [], defaultdict(list)
    for (model, head, train_set, condition), d in sorted(data.items()):
        y, slides = d["labels"], d["slides"]
        size = np.array([size_bin(info[s]["n_tumour"]) if y[i] else "" for i, s in enumerate(slides)])
        centre = np.array([info[s]["centre"] for s in slides])
        hits = np.array([detected(s, y) for s in d["scores"]])      # [seeds, slides]
        groups = [("all", y == 1)] + [(f"tumour tiles {b}", size == b) for b in
                                      sorted({x for x in size if x}, key=lambda b: int(b.split("-")[0].lstrip(">")))]
        groups += [("macro", d["lesions"] == "macro"), ("micro", d["lesions"] == "micro")]
        for name, mask in groups:
            if mask.any():
                sens_rows.append(dict(model=model, head=head, train_set=train_set, condition=condition, group=name,
                                      n_slides=int(mask.sum()), sensitivity=round(float(hits[:, mask].mean()), 4)))
        for c in ("RUMC", "UMCU"):
            m = centre == c
            centre_rows.append(dict(model=model, head=head, train_set=train_set, condition=condition,
                                    measure=f"AUC within {c}",
                                    value=round(float(np.mean([evaluate.auc(y[m], s[m]) for s in d["scores"]])), 4)))
        for c in ("RUMC", "UMCU"):
            m = (centre == c) & ((y == 0) | (d["lesions"] == "micro"))
            centre_rows.append(dict(model=model, head=head, train_set=train_set, condition=condition,
                                    measure=f"micro AUC within {c}",
                                    value=round(float(np.mean([evaluate.auc(y[m], s[m]) for s in d["scores"]])), 4)))
        normals = y == 0
        n_tiles = np.array([info[s]["n_tiles"] for s in slides])
        centre_rows.append(dict(model=model, head=head, train_set=train_set, condition=condition,
                                measure="normal slides: rank correlation of score with tissue tiles",
                                value=round(float(np.mean([spearmanr(s[normals], n_tiles[normals])[0]
                                                           for s in d["scores"]])), 4)))
        is_umcu = (centre[normals] == "UMCU").astype(int)
        centre_rows.append(dict(model=model, head=head, train_set=train_set, condition=condition,
                                measure="normal slides: score separates UMCU from RUMC (AUC)",
                                value=round(float(np.mean([evaluate.auc(is_umcu, s[normals]) for s in d["scores"]])), 4)))
        for i, s in enumerate(slides):
            if y[i]:
                fail[(model, condition, s)].append((head, train_set, float(hits[:, i].mean())))
    failure_rows = []
    for (model, condition, s), entries in sorted(fail.items()):
        row = dict(model=model, condition=condition, slide=s, centre=info[s]["centre"], n_tumour=info[s]["n_tumour"])
        row.update({f"{h}_{ts}": round(rate, 2) for h, ts, rate in entries})
        failure_rows.append(row)
    return sens_rows, centre_rows, failure_rows


def markdown(sens_rows, centre_rows, failure_rows, lesions: dict, info: dict, labels: dict) -> str:
    models = sorted({r["model"] for r in sens_rows})
    conds = config.TEST_CONDITIONS
    S = {(r["model"], r["head"], r["train_set"], r["condition"], r["group"]): r for r in sens_rows}
    C = {(r["model"], r["head"], r["train_set"], r["condition"], r["measure"]): r["value"] for r in centre_rows}
    n_normal = sum(1 for s in labels if labels[s] == 0)
    n_tumour = len(labels) - n_normal

    def by_centre(group):
        slides = [s for s in labels if group(s)]
        return f"{sum(info[s]['centre'] == 'RUMC' for s in slides)} / {sum(info[s]['centre'] == 'UMCU' for s in slides)}"

    out = ["# Deep dive: tumour size, hospital and shared failures", "",
           f"Sensitivity is at {SPECIFICITY:.0%} specificity: the threshold lets at most "
           f"{1 - SPECIFICITY:.0%} of the {n_normal} normal test slides through. Means over seeds.", ""]

    out += [f"## Sensitivity at {SPECIFICITY:.0%} specificity, all {n_tumour} tumour slides", "",
            "| encoder | head | training | " + " | ".join(conds) + " |", "|---|---|---|" + "---|" * len(conds)]
    for m in models:
        for h in FOCUS_HEADS:
            for ts in config.TRAIN_SETS:
                cells = [f"{100 * S[(m, h, ts, c, 'all')]['sensitivity']:.0f}" if (m, h, ts, c, "all") in S else ""
                         for c in conds]
                out.append(f"| {m} | {h} | {ts} | " + " | ".join(cells) + " |")
    out.append("")

    bins = sorted({r["group"] for r in sens_rows if r["group"].startswith("tumour tiles")},
                  key=lambda g: int(g.split()[-1].split("-")[0].lstrip(">")))
    counts = {g: next(r["n_slides"] for r in sens_rows if r["group"] == g) for g in bins}
    for c in FOCUS_CONDITIONS:
        out += [f"## Sensitivity by tumour size under {c}", "",
                "| encoder | head | training | " + " | ".join(f"{g.split()[-1]} tiles (n={counts[g]})" for g in bins) + " |",
                "|---|---|---|" + "---|" * len(bins)]
        for m in models:
            for h in FOCUS_HEADS:
                for ts in config.TRAIN_SETS:
                    cells = [f"{100 * S[(m, h, ts, c, g)]['sensitivity']:.0f}" if (m, h, ts, c, g) in S else ""
                             for g in bins]
                    out.append(f"| {m} | {h} | {ts} | " + " | ".join(cells) + " |")
        out.append("")

    sep = "normal slides: score separates UMCU from RUMC (AUC)"
    size_corr = "normal slides: rank correlation of score with tissue tiles"
    out += ["## Hospital", "",
            f"Test slides (RUMC / UMCU): normal {by_centre(lambda s: labels[s] == 0)}; "
            f"macro {by_centre(lambda s: lesions[s] == 'macro')}; micro {by_centre(lambda s: lesions[s] == 'micro')}. "
            "AUC within each centre "
            "compares tumour and normal slides of that centre only. The last column is how well the score "
            "tells the centres apart among normal slides; 50 means not at all. The last is the Spearman "
            "correlation, among normal slides, between the score and the number of tissue tiles: pooling over "
            "more tiles gives more chances for a high-scoring false positive.", ""]
    for c in FOCUS_CONDITIONS:
        out += [f"### {c}", "", "| encoder | head | training | AUC within RUMC | AUC within UMCU | micro AUC within RUMC "
                "| micro AUC within UMCU | centre from score | score vs tissue (rank corr.) |",
                "|---|---|---|---|---|---|---|---|---|"]
        for m in models:
            for h in FOCUS_HEADS:
                for ts in config.TRAIN_SETS:
                    k = (m, h, ts, c)
                    if k + ("AUC within RUMC",) not in C:
                        continue
                    out.append(f"| {m} | {h} | {ts} | {100 * C[k + ('AUC within RUMC',)]:.1f} | "
                               f"{100 * C[k + ('AUC within UMCU',)]:.1f} | {100 * C[k + ('micro AUC within RUMC',)]:.1f} | "
                               f"{100 * C[k + ('micro AUC within UMCU',)]:.1f} | {100 * C[k + (sep,)]:.1f} | "
                               f"{C[k + (size_corr,)]:+.2f} |")
        out.append("")

    out += ["## Shared failures (mixed training)", "",
            "A slide counts as detected if it clears the 95%-specificity threshold in at least three of "
            "the five seeds.", "",
            "| encoder | condition | caught by both | attention only | tile head only | missed by both | missed by both: micro |",
            "|---|---|---|---|---|---|---|"]
    for m in models:
        for c in FOCUS_CONDITIONS:
            sel = [r for r in failure_rows if r["model"] == m and r["condition"] == c
                   and "abmil_mixed" in r and "tileprobe_mixed" in r]
            if not sel:
                continue
            a = np.array([r["abmil_mixed"] >= 0.6 for r in sel])
            t = np.array([r["tileprobe_mixed"] >= 0.6 for r in sel])
            micro = np.array([lesions[r["slide"]] == "micro" for r in sel])
            out.append(f"| {m} | {c} | {int((a & t).sum())} | {int((a & ~t).sum())} | {int((~a & t).sum())} | "
                       f"{int((~a & ~t).sum())} | {int((~a & ~t & micro).sum())} |")
    out.append("")
    return "\n".join(out)


def write_csv(path, rows) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    data = evaluate.load()
    slides = next(iter(data.values()))["slides"]
    info = slide_info(slides)
    first = next(iter(data.values()))
    lesions = dict(zip(slides, first["lesions"]))
    labels = dict(zip(slides, first["labels"]))
    sens_rows, centre_rows, failure_rows = analyse(data, info)
    write_csv(config.RESULTS / "sensitivity.csv", sens_rows)
    write_csv(config.RESULTS / "centre.csv", centre_rows)
    write_csv(config.RESULTS / "failures.csv", failure_rows)
    md = markdown(sens_rows, centre_rows, failure_rows, lesions, info, labels)
    (config.RESULTS / "deep_dive.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
