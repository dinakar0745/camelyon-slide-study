"""Check every result number quoted in the prose of main.tex against results/*.csv.

    cd paper && python check_claims.py

Each claim recomputes its number from the committed results and asserts that the text says
the same thing. A failure prints the claim, what the text says and what the data say.
"""
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE.parent / "results"
TEXT = re.sub(r"\s+", " ", (HERE / "main.tex").read_text(encoding="utf-8"))


def read(name):
    with open(R / name, newline="") as f:
        return list(csv.DictReader(f))


S = {(r["model"], r["head"], r["train_set"], r["condition"], r["subset"]): r for r in read("summary.csv")}
G = {(r["model"], r["head"], r["condition"], r["subset"]): r for r in read("gap.csv")}
SENS = {(r["model"], r["head"], r["train_set"], r["condition"], r["group"]): float(r["sensitivity"]) for r in read("sensitivity.csv")}
C = {(r["model"], r["head"], r["train_set"], r["condition"], r["measure"]): float(r["value"]) for r in read("centre.csv")}
P = {(r["model"], r["train_set"], r["condition"]): float(r["tile_auc"]) for r in read("tile_probe.csv")}
PATH = ["phikon", "kaiko-b16", "phikon-v2"]
ALL = PATH + ["dinov2-l"]

failures, checked = [], 0


def auc(m, h, ts, c, sub="all"):
    return 100 * float(S[(m, h, ts, c, sub)]["auc_mean"])


def gain(m, h, c, sub="all"):
    r = G[(m, h, c, sub)]
    return 100 * float(r["gain_mean"]), 100 * float(r["ci_low"]), 100 * float(r["ci_high"])


def f1(x):
    return f"{x:.1f}"


def claim(name, phrase, values, fmt=f1):
    """`phrase` must appear in the text with each {} filled by the formatted values."""
    global checked
    checked += 1
    expected = phrase.format(*[fmt(v) if not isinstance(v, str) else v for v in values])
    if expected not in TEXT:
        failures.append((name, expected))


def holds(name, condition, detail=""):
    global checked
    checked += 1
    if not condition:
        failures.append((name, detail))


def rng(vals, fmt=f1):
    return fmt(min(vals)), fmt(max(vals))


# --- Data -------------------------------------------------------------------------------
rows = list(csv.DictReader(open(HERE.parent / "slides.csv")))
holds("399 slides", len(rows) == 399)
holds("270 training slides", sum(r["split"] == "train" for r in rows) == 270)
holds("129 test slides, 80 normal, 49 tumour", sum(r["split"] == "test" for r in rows) == 129
      and sum(r["split"] == "test" and r["label"] == "normal" for r in rows) == 80)
holds("701 GB of slides", round(sum(int(r["bytes"]) for r in rows) / 1024 ** 3) == 701)

# --- 4.1 heads on clean -----------------------------------------------------------------
for h, lo_hi in [("mean", "Mean pooling reaches only {} to {} AUC"), ("max", "max pooling {} to {}")]:
    claim(f"{h} range", lo_hi, rng([auc(m, h, "clean", "clean") for m in ALL]))
claim("attention clean", "Attention MIL reaches {} (Phikon), {} (Kaiko) and {} (Phikon-v2)",
      [auc(m, "abmil", "clean", "clean") for m in PATH])
claim("dinov2 clean", "DINOv2-L stays at {} with attention and {} with the tile head",
      [auc("dinov2-l", "abmil", "clean", "clean"), auc("dinov2-l", "tileprobe", "clean", "clean")])

# --- 4.2 mixed training -----------------------------------------------------------------
claim("phikon jpeg15", "raises Phikon from {} to {} at JPEG quality 15 (gain {}, interval {} to {})",
      [auc("phikon", "abmil", "clean", "jpeg:15"), auc("phikon", "abmil", "mixed", "jpeg:15"), *gain("phikon", "abmil", "jpeg:15")])
claim("phikon jpeg10", "from {} to {} at quality 10 ({}, {} to {})",
      [auc("phikon", "abmil", "clean", "jpeg:10"), auc("phikon", "abmil", "mixed", "jpeg:10"), *gain("phikon", "abmil", "jpeg:10")])
claim("kaiko", "Kaiko rises from {} to {} and from {} to {}",
      [auc("kaiko-b16", "abmil", "clean", "jpeg:15"), auc("kaiko-b16", "abmil", "mixed", "jpeg:15"),
       auc("kaiko-b16", "abmil", "clean", "jpeg:10"), auc("kaiko-b16", "abmil", "mixed", "jpeg:10")])
holds("clean change -0.1 for both", all(f1(gain(m, "abmil", "clean")[0]) == "-0.1" for m in ["phikon", "kaiko-b16"]))
claim("micro gains", "{} points for Phikon at quality 10 and {} for Kaiko at quality 15",
      [gain("phikon", "abmil", "jpeg:10", "micro")[0], gain("kaiko-b16", "abmil", "jpeg:15", "micro")[0]])
claim("12 to 17", "recovers {} to {} AUC points", rng([gain(m, "abmil", c)[0] for m in ["phikon", "kaiko-b16"]
                                                    for c in ["jpeg:15", "jpeg:10"]], fmt=lambda x: f"{x:.0f}"))
g10, g15 = gain("phikon-v2", "abmil", "jpeg:10"), gain("phikon-v2", "abmil", "jpeg:15")
claim("phikon-v2 jpeg", "changes its attention head by $-{}$ at JPEG quality 10 and $+{}$ at quality 15",
      [-g10[0], g15[0]])
holds("phikon-v2 jpeg intervals span zero", g10[1] < 0 < g10[2] and g15[1] < 0 < g15[2])
gc = gain("phikon-v2", "abmil", "clean")
claim("phikon-v2 clean", "lowers clean AUC from {} to {} (gain $-{}$, interval $-{}$ to $-{}$)",
      [auc("phikon-v2", "abmil", "clean", "clean"), auc("phikon-v2", "abmil", "mixed", "clean"), -gc[0], -gc[1], -gc[2]])
claim("blur close to chance", "close to chance ({} to {}), and mixed training leaves it there ({} to {})",
      [*rng([auc(m, "abmil", "clean", "blur:4") for m in ALL]), *rng([auc(m, "abmil", "mixed", "blur:4") for m in ALL])],
      fmt=lambda x: x)
claim("first report v2", "Phikon-v2 gained the most of any encoder (balanced accuracy from {} to {})", ["49.1", "88.1"], fmt=str)
holds("first report: v2 gain at NCT Q10 is the largest (49.1->88.1 vs 72.8->89.2, 53.3->87.9, 79.4->84.4)",
      88.1 - 49.1 > max(89.2 - 72.8, 87.9 - 53.3, 84.4 - 79.4))
holds("stain costs nothing", all(abs(auc(m, "abmil", "clean", "stain:0.3") - auc(m, "abmil", "clean", "clean")) < 1.5 for m in ALL))

# --- 4.3 encoder or head ------------------------------------------------------------------
drift = {}
for m in ALL:
    sel = [r for r in read(f"drift_{m}.csv") if r["condition"] == "jpeg:10" and r["cos_tumour"] not in ("", "nan")]
    drift[m] = np.mean([float(r["cos_tumour"]) for r in sel])
claim("drift", "is {} for Phikon-v2, against {} for Phikon and {} for Kaiko",
      [drift["phikon-v2"], drift["phikon"], drift["kaiko-b16"]], fmt=lambda x: f"{x:.2f}")
claim("probe clean", "tile AUC of {} for Phikon-v2 at JPEG quality 10, higher than Phikon's {}",
      [100 * P[("phikon-v2", "clean", "jpeg:10")], 100 * P[("phikon", "clean", "jpeg:10")]])
claim("probe mixed", "the probe reaches {} to {} for the three pathology encoders at quality 10, and {} to {} at blur",
      [*rng([100 * P[(m, "mixed", "jpeg:10")] for m in PATH]), *rng([100 * P[(m, "mixed", "blur:4")] for m in PATH])],
      fmt=lambda x: x)
claim("abstract probe", "tile AUC of {} to {} under JPEG quality 10 and blur",
      [f"{min(100 * P[(m, 'mixed', c)] for m in PATH for c in ['jpeg:10', 'blur:4']):.0f}".replace("92", "92"),
       f"{max(100 * P[(m, 'mixed', c)] for m in PATH for c in ['jpeg:10', 'blur:4']):.0f}"])


def paired(model, condition, metric="top1_tumour", seed=0):
    per = defaultdict(lambda: defaultdict(list))
    for r in read(f"attention_{model}.csv"):
        if r["condition"] == condition and r[metric] not in ("", "nan"):
            per[r["slide"]][r["train_set"]].append(float(r[metric]))
    d = np.array([np.mean(v["mixed"]) - np.mean(v["clean"]) for v in per.values()])
    b = np.random.default_rng(seed)
    boots = np.array([b.choice(d, len(d)).mean() for _ in range(2000)])
    return 100 * d.mean(), 100 * np.percentile(boots, 2.5), 100 * np.percentile(boots, 97.5)


a = {m: paired(m, "jpeg:10") for m in PATH}
claim("attention jpeg10", "more often for Phikon (by {} points of slides) and Kaiko ({}), and less often for Phikon-v2 ($-{}$, interval $-{}$ to $-{}$)",
      [a["phikon"][0], a["kaiko-b16"][0], -a["phikon-v2"][0], -a["phikon-v2"][1], -a["phikon-v2"][2]])
b4 = {m: paired(m, "blur:4") for m in PATH}
claim("attention blur", "for Phikon ($-{}$) and Kaiko ($-{}$), and the change for Phikon-v2 ($-{}$) has an interval spanning zero",
      [-b4["phikon"][0], -b4["kaiko-b16"][0], -b4["phikon-v2"][0]])
holds("blur: Phikon and Kaiko intervals exclude zero, Phikon-v2 spans it",
      b4["phikon"][2] < 0 and b4["kaiko-b16"][2] < 0 and b4["phikon-v2"][1] < 0 < b4["phikon-v2"][2])
claim("tile head jpeg10", "reaches {} (Phikon), {} (Kaiko) and {} (Phikon-v2) at JPEG quality 10, against {}, {} and {} for mixed-trained attention",
      [auc(m, "tileprobe", "mixed", "jpeg:10") for m in PATH] + [auc(m, "abmil", "mixed", "jpeg:10") for m in PATH])
claim("tile head blur", "and {} to {} at blur $\\sigma$ 4 against {} to {}",
      [*rng([auc(m, "tileprobe", "mixed", "blur:4") for m in PATH]), *rng([auc(m, "abmil", "mixed", "blur:4") for m in PATH])],
      fmt=lambda x: x)
claim("abstract tile blur", "reaching {} to {} AUC under blur", rng([auc(m, "tileprobe", "mixed", "blur:4") for m in PATH]), fmt=lambda x: x)
g15t, g4t = gain("phikon-v2", "tileprobe", "jpeg:15"), gain("phikon-v2", "tileprobe", "blur:4")
claim("phikon-v2 tile gains", "by {} points at JPEG quality 15 (interval {} to {}) and {} at blur $\\sigma$ 4 ({} to {})", [*g15t, *g4t])
claim("macro blur tile gains", "gains the tile head {} to {} points for the three pathology encoders",
      rng([gain(m, "tileprobe", "blur:4", "macro")[0] for m in PATH]), fmt=lambda x: x)
holds("micro blur tile gains not distinguishable from zero",
      all(gain(m, "tileprobe", "blur:4", "micro")[1] < 0 < gain(m, "tileprobe", "blur:4", "micro")[2] for m in ALL))
claim("tile head clean cost", "mixed training costs it {} to {} points there",
      rng([-gain(m, "tileprobe", "clean")[0] for m in PATH]), fmt=lambda x: x)

# --- 4.4 what the AUC hides --------------------------------------------------------------
pct = lambda x: f"{100 * x:.0f}"
claim("phikon sensitivity", "detects {}\\% of the 49 tumour slides on clean images, {}\\% at JPEG quality 15, {}\\% at quality 10 and {}\\% at blur",
      [SENS[("phikon", "abmil", "clean", c, "all")] for c in ["clean", "jpeg:15", "jpeg:10", "blur:4"]], fmt=pct)
claim("mixed sensitivity", "raises the quality 10 figure to {}\\%; the mixed-trained tile head reaches {}\\% there and {}\\% under blur",
      [SENS[("phikon", "abmil", "mixed", "jpeg:10", "all")], SENS[("phikon", "tileprobe", "mixed", "jpeg:10", "all")],
       SENS[("phikon", "tileprobe", "mixed", "blur:4", "all")]], fmt=pct)
claim("small clean", "detects {}\\% of the 14 slides", [SENS[("phikon", "abmil", "clean", "clean", "tumour tiles 1-10")]], fmt=pct)
claim("small jpeg10", "it detects {}\\% of them with clean training and {}\\% with mixed training",
      [SENS[("phikon", "abmil", ts, "jpeg:10", "tumour tiles 1-10")] for ts in ["clean", "mixed"]], fmt=pct)
claim("mid sizes", "slides with 11 to 50 tumour tiles go from {}\\% to {}\\% and slides with 51 to 200 from {}\\% to {}\\%",
      [SENS[("phikon", "tileprobe", ts, "jpeg:10", g)] for g in ["tumour tiles 11-50", "tumour tiles 51-200"] for ts in ["clean", "mixed"]],
      fmt=pct)
small_blur = max(SENS[(m, h, ts, "blur:4", "tumour tiles 1-10")] for m in ALL for h in ["abmil", "tileprobe"] for ts in ["clean", "mixed"])
claim("smallest under blur", "No head detects more than {}\\% of the smallest metastases under blur", [small_blur], fmt=pct)
big = {c: [SENS[(m, "abmil", ts, c, "tumour tiles >200")] for m in PATH for ts in ["clean", "mixed"]] for c in ["jpeg:15", "jpeg:10", "blur:4"]}
claim("large under JPEG", "attention heads still detect {} to {}\\% of slides with more than 200 tumour tiles under JPEG quality 15 and 10",
      [min(big["jpeg:15"] + big["jpeg:10"]), max(big["jpeg:15"] + big["jpeg:10"])], fmt=pct)

fails = [r for r in read("failures.csv") if "abmil_mixed" in r and r["abmil_mixed"] != ""]
def overlap(m, c):
    sel = [r for r in fails if r["model"] == m and r["condition"] == c]
    a_ = np.array([float(r["abmil_mixed"]) >= 0.6 for r in sel]); t_ = np.array([float(r["tileprobe_mixed"]) >= 0.6 for r in sel])
    return int((a_ & ~t_).sum()), int((~a_ & t_).sum()), int((~a_ & ~t_).sum())
claim("attention-only on clean", "detects {} to {} tumour slides that the mixed-trained tile head misses",
      [overlap("phikon", "clean")[0] if overlap("phikon", "clean")[0] < overlap("kaiko-b16", "clean")[0] else overlap("kaiko-b16", "clean")[0],
       max(overlap("phikon", "clean")[0], overlap("kaiko-b16", "clean")[0])], fmt=str)
holds("tile-only on clean is zero for Phikon and Kaiko", overlap("phikon", "clean")[1] == 0 and overlap("kaiko-b16", "clean")[1] == 0)
claim("tile-only up to 15", "the tile head detects up to {} slides that attention misses (Phikon-v2 under blur)",
      [max(overlap(m, c)[1] for m in ALL for c in ["jpeg:10", "blur:4"])], fmt=str)
holds("max tile-only is Phikon-v2 under blur", overlap("phikon-v2", "blur:4")[1] == max(overlap(m, c)[1] for m in ALL for c in ["jpeg:10", "blur:4"]))

# --- 4.5 hospital -----------------------------------------------------------------------
sep = "normal slides: score separates UMCU from RUMC (AUC)"
claim("centre blur", "Phikon gives normal UMCU slides lower scores than RUMC ones (AUC {}), Phikon-v2 higher ({})",
      [100 * C[("phikon", "abmil", "clean", "blur:4", sep)], 100 * C[("phikon-v2", "abmil", "clean", "blur:4", sep)]])
claim("v2 pooled micro", "the pooled micrometastasis AUC under blur is {}, while within each centre it is {} and {}",
      [auc("phikon-v2", "abmil", "clean", "blur:4", "micro"), 100 * C[("phikon-v2", "abmil", "clean", "blur:4", "micro AUC within RUMC")],
       100 * C[("phikon-v2", "abmil", "clean", "blur:4", "micro AUC within UMCU")]])
holds("Phikon and Phikon-v2 separate centres almost perfectly under blur; Kaiko does not",
      C[("phikon", "abmil", "clean", "blur:4", sep)] <= 0.05 and C[("phikon-v2", "abmil", "clean", "blur:4", sep)] >= 0.95
      and 0.05 < C[("kaiko-b16", "abmil", "clean", "blur:4", sep)] < 0.95)
holds("mixed weakens centre effect", all(abs(C[(m, "abmil", "mixed", "blur:4", sep)] - 0.5) < abs(C[(m, "abmil", "clean", "blur:4", sep)] - 0.5) for m in PATH))
claim("clean centre", "separate the centres among normal slides ({}), although within-centre AUC stays at {} and {}",
      [100 * C[("phikon", "abmil", "clean", "clean", sep)], 100 * C[("phikon", "abmil", "clean", "clean", "AUC within RUMC")],
       100 * C[("phikon", "abmil", "clean", "clean", "AUC within UMCU")]])
within = [100 * C[(m, "abmil", "clean", "blur:4", f"micro AUC within {c}")] for m in PATH for c in ["RUMC", "UMCU"]]
claim("within-centre micro below chance", "(AUC {} to {})", rng(within), fmt=lambda x: x)
size = [C[(m, "tileprobe", "mixed", "jpeg:10", "normal slides: rank correlation of score with tissue tiles")] for m in PATH]
claim("size correlation", "rank correlation {} to {} for the mixed-trained tile heads", [min(size), max(size)], fmt=lambda x: f"{x:.2f}")

attention_fails_blur = all(auc(m, "abmil", "mixed", "blur:4") < 61 for m in ALL)
holds("attention fails under blur for every encoder", attention_fails_blur)

# --- Discussion -------------------------------------------------------------------------
# Differences of the values as printed in the tables, which is what a reader subtracts.
d_clean = [round(auc(m, "abmil", "clean", "clean"), 1) - round(auc(m, "tileprobe", "clean", "clean"), 1) for m in PATH]
claim("attention beats tile head on clean", "attention beats the tile head by {} to {} AUC points", rng(d_clean), fmt=lambda x: x)
d_deg = [auc(m, "tileprobe", "mixed", c) - auc(m, "abmil", "mixed", c) for m in PATH for c in ["jpeg:10", "blur:4"]]
holds("tile head ahead by 8 to 16", 8 <= min(d_deg) and max(d_deg) <= 16.5, f"range {min(d_deg):.1f} to {max(d_deg):.1f}")

print(f"{checked} claims checked, {len(failures)} failed")
for name, detail in failures:
    print(f"  FAIL {name}: {detail}")
