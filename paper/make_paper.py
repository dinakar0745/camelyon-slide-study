"""Figures and LaTeX tables for the technical report, built only from results/*.csv.

    cd paper && python make_paper.py && latexmk -pdf main.tex

Writes fig1_slide_auc.pdf, fig2_sensitivity_by_size.pdf and tables/*.tex next to this file.
Every number in the tables comes from the committed CSVs; the per-slide paired bootstrap
for attention localisation (Table 4) is computed here with a fixed seed.
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
TABLES = HERE / "tables"

# Same encoder colours and markers as the first report (colour-blind-checked; markers back up colour).
MODELS = {
    "phikon":    ("Phikon",     "#2a78d6", "o"),
    "phikon-v2": ("Phikon-v2",  "#eb6834", "s"),
    "kaiko-b16": ("Kaiko B/16", "#1baf7a", "^"),
    "dinov2-l":  ("DINOv2-L",   "#eda100", "D"),
}
PATHOLOGY = ["phikon", "phikon-v2", "kaiko-b16"]
CONDS = ["clean", "blur:2", "blur:4", "jpeg:30", "jpeg:15", "jpeg:10", "stain:0.3"]
COND_TEX = {"clean": "Clean", "blur:2": r"Blur $\sigma$ 2", "blur:4": r"Blur $\sigma$ 4", "jpeg:30": "JPEG Q30",
            "jpeg:15": "JPEG Q15", "jpeg:10": "JPEG Q10 (unseen)", "stain:0.3": r"Stain $\alpha$ 0.3"}
COND_TICK = {"clean": "clean", "blur:2": "blur\n2", "blur:4": "blur\n4", "jpeg:30": "JPEG\n30", "jpeg:15": "JPEG\n15",
             "jpeg:10": "JPEG\n10*", "stain:0.3": "stain\n0.3"}
HEAD_TEX = {"mean": "Mean pooling", "max": "Max pooling", "abmil": "Attention MIL", "tileprobe": "Tile head"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"

plt.rcParams.update({
    "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "figure.facecolor": "white", "savefig.facecolor": "white",
})


def read(name):
    with open(RESULTS / name, newline="") as f:
        return list(csv.DictReader(f))


SUMMARY = {(r["model"], r["head"], r["train_set"], r["condition"], r["subset"]): r for r in read("summary.csv")}
GAP = {(r["model"], r["head"], r["condition"], r["subset"]): r for r in read("gap.csv")}
SENS = {(r["model"], r["head"], r["train_set"], r["condition"], r["group"]): r for r in read("sensitivity.csv")}
CENTRE = {(r["model"], r["head"], r["train_set"], r["condition"], r["measure"]): float(r["value"]) for r in read("centre.csv")}
PROBE = {(r["model"], r["train_set"], r["condition"]): float(r["tile_auc"]) for r in read("tile_probe.csv")}


def auc(m, h, ts, c, sub="all"):
    return 100 * float(SUMMARY[(m, h, ts, c, sub)]["auc_mean"])


def sd(m, h, ts, c, sub="all"):
    return 100 * float(SUMMARY[(m, h, ts, c, sub)]["auc_sd"])


def gain(m, h, c, sub="all"):
    r = GAP[(m, h, c, sub)]
    return 100 * float(r["gain_mean"]), 100 * float(r["ci_low"]), 100 * float(r["ci_high"])


def save(fig, name):
    fig.savefig(HERE / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(HERE / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}.pdf")


# --- Figures ---------------------------------------------------------------------------

def fig1():
    """Slide AUC per condition: clean-trained (hollow) and mixed-trained (filled), two heads."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    offsets = np.linspace(-0.27, 0.27, len(MODELS))
    for ax, head, title in [(axes[0], "abmil", "Attention MIL (slide labels only)"),
                            (axes[1], "tileprobe", "Tile head (tile labels from annotations)")]:
        for k, (m, (name, colour, marker)) in enumerate(MODELS.items()):
            for i, c in enumerate(CONDS):
                x = i + offsets[k]
                a, b = auc(m, head, "clean", c), auc(m, head, "mixed", c)
                ax.plot([x, x], [a, b], color=colour, linewidth=1.0, alpha=0.8, zorder=2)
                ax.plot(x, a, marker=marker, markersize=4, markerfacecolor="white", markeredgecolor=colour,
                        markeredgewidth=1.0, linestyle="none", zorder=3)
                ax.plot(x, b, marker=marker, markersize=4, color=colour, markeredgecolor="white",
                        markeredgewidth=0.4, linestyle="none", zorder=4)
        ax.axhline(50, color=MUTED, linewidth=0.6, linestyle=":")
        ax.set_xticks(range(len(CONDS)), [COND_TICK[c] for c in CONDS])
        ax.set_ylim(30, 101)
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(length=2)
        ax.set_title(title, fontsize=8, loc="left", color=INK)
    axes[0].set_ylabel("Slide AUC (%)")
    axes[0].text(6.45, 51, "chance", fontsize=6.5, color=MUTED, va="bottom", ha="right")
    handles = [plt.Line2D([], [], color=c, marker=mk, markersize=4, linestyle="none", label=n)
               for n, c, mk in MODELS.values()]
    handles += [plt.Line2D([], [], color=INK, marker="o", markersize=4, markerfacecolor="white", linestyle="none",
                           label="trained on clean"),
                plt.Line2D([], [], color=INK, marker="o", markersize=4, linestyle="none", label="trained on mixed")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    save(fig, "fig1_slide_auc")


SIZE_GROUPS = ["tumour tiles 1-10", "tumour tiles 11-50", "tumour tiles 51-200", "tumour tiles >200"]


def fig2():
    """Sensitivity at 95% specificity by tumour size; rows = encoders, columns = conditions."""
    conds = [("clean", "Clean"), ("jpeg:10", "JPEG quality 10 (unseen)"), ("blur:4", r"Blur $\sigma$ 4")]
    series = [("abmil", "clean", "attention, clean-trained", MUTED, (0, (3, 2)), "o", "white"),
              ("abmil", "mixed", "attention, mixed-trained", INK, "-", "o", None),
              ("tileprobe", "mixed", "tile head, mixed-trained", INK, (0, (1, 1.2)), "s", "white")]
    counts = {g: int(next(r for k, r in SENS.items() if k[4] == g)["n_slides"]) for g in SIZE_GROUPS}
    fig, axes = plt.subplots(3, 3, figsize=(7.0, 5.6), sharex=True, sharey=True)
    for r, m in enumerate(PATHOLOGY):
        for c, (cond, title) in enumerate(conds):
            ax = axes[r, c]
            for head, ts, label, colour, ls, marker, face in series:
                ys = [100 * float(SENS[(m, head, ts, cond, g)]["sensitivity"]) for g in SIZE_GROUPS]
                ax.plot(range(4), ys, color=colour, linestyle=ls, linewidth=1.4, marker=marker, markersize=4,
                        markerfacecolor=face or colour, markeredgecolor=colour, markeredgewidth=1.0)
            ax.set_ylim(-3, 103)
            ax.yaxis.grid(True, color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            ax.tick_params(length=2)
            if r == 0:
                ax.set_title(title, fontsize=8, loc="left", color=INK)
            if c == 0:
                ax.set_ylabel(f"{MODELS[m][0]}\nSensitivity (%)")
            if r == 2:
                ax.set_xticks(range(4), [f"{g.split()[-1]}\n(n={counts[g]})" for g in SIZE_GROUPS])
                ax.set_xlabel("Tiles touching tumour")
    handles = [plt.Line2D([], [], color=col, linestyle=ls, linewidth=1.4, marker=mk, markersize=4,
                          markerfacecolor=face or col, markeredgecolor=col, label=lab)
               for _, _, lab, col, ls, mk, face in series]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    save(fig, "fig2_sensitivity_by_size")


# --- Tables ----------------------------------------------------------------------------

def write(name, text):
    TABLES.mkdir(exist_ok=True)
    (TABLES / f"{name}.tex").write_text(text, encoding="utf-8")
    print(f"wrote tables/{name}.tex")


def arrow(a, b):
    return f"{a:.1f} \\ra{{}} {b:.1f}"


def table_main():
    cols = ["phikon", "kaiko-b16", "phikon-v2", "dinov2-l"]
    lines = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
             "Condition & " + " & ".join(MODELS[m][0] for m in cols) + r"\\", r"\midrule"]
    for head in ["abmil", "tileprobe"]:
        lines.append(r"\multicolumn{5}{@{}l}{\textit{" + HEAD_TEX[head] + r"}}\\")
        for c in CONDS:
            lines.append(COND_TEX[c] + " & " + " & ".join(arrow(auc(m, head, "clean", c), auc(m, head, "mixed", c))
                                                         for m in cols) + r"\\")
        if head == "abmil":
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("main_results", "\n".join(lines))


def table_gains():
    cols = ["phikon", "kaiko-b16", "phikon-v2", "dinov2-l"]
    lines = [r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
             "Head & Slides & " + " & ".join(MODELS[m][0] for m in cols) + r"\\", r"\midrule"]
    for head in ["abmil", "tileprobe"]:
        for c in ["jpeg:15", "jpeg:10", "blur:4"]:
            for sub, subname in [("all", "all"), ("macro", "macro"), ("micro", "micro")]:
                cells = []
                for m in cols:
                    g, lo, hi = gain(m, head, c, sub)
                    cell = f"{g:+.1f} {{\\scriptsize[{lo:+.1f}, {hi:+.1f}]}}"
                    cells.append(r"\textbf{" + cell + "}" if lo > 0 or hi < 0 else cell)
                lead = f"{HEAD_TEX[head]}, {COND_TEX[c]}" if sub == "all" else ""
                lines.append(f"{lead} & {subname} & " + " & ".join(cells) + r"\\")
            lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    write("gains", "\n".join(lines))


def table_heads_clean():
    cols = ["phikon", "kaiko-b16", "phikon-v2", "dinov2-l"]
    lines = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
             "Head (clean-trained) & " + " & ".join(MODELS[m][0] for m in cols) + r"\\", r"\midrule"]
    for head in ["mean", "max", "abmil", "tileprobe"]:
        lines.append(HEAD_TEX[head] + " & " + " & ".join(
            f"{auc(m, head, 'clean', 'clean'):.1f}{{\\scriptsize$\\pm${sd(m, head, 'clean', 'clean'):.1f}}}"
            for m in cols) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("heads_clean", "\n".join(lines))


def drift_means():
    out = {}
    for m in MODELS:
        rows = read(f"drift_{m}.csv")
        for c in CONDS[1:]:
            sel = [r for r in rows if r["condition"] == c]
            out[(m, c)] = (np.mean([float(r["cos_tumour"]) for r in sel if r["cos_tumour"] not in ("", "nan")]),
                           np.mean([float(r["cos_normal"]) for r in sel]))
    return out


def attention_paired(model, condition, metric, n_boot=2000, seed=0):
    """Mixed minus clean, averaged over seeds per test tumour slide, with a bootstrap over slides."""
    per = defaultdict(lambda: defaultdict(list))
    for r in read(f"attention_{model}.csv"):
        if r["condition"] == condition and r[metric] not in ("", "nan"):
            per[r["slide"]][r["train_set"]].append(float(r[metric]))
    d = np.array([np.mean(v["mixed"]) - np.mean(v["clean"]) for v in per.values() if v["mixed"] and v["clean"]])
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(d, len(d)).mean() for _ in range(n_boot)])
    return 100 * d.mean(), 100 * np.percentile(boots, 2.5), 100 * np.percentile(boots, 97.5), len(d)


def attention_mean(model, condition, train_set, metric):
    vals = [float(r[metric]) for r in read(f"attention_{model}.csv")
            if r["condition"] == condition and r["train_set"] == train_set and r[metric] not in ("", "nan")]
    return 100 * float(np.mean(vals))


def table_encoder_or_head():
    drift = drift_means()
    cols = ["phikon", "kaiko-b16", "phikon-v2", "dinov2-l"]
    lines = [r"\begin{tabular}{@{}l" + "c" * len(cols) + "@{}}", r"\toprule",
             " & " + " & ".join(MODELS[m][0] for m in cols) + r"\\", r"\midrule"]
    for c in ["jpeg:10", "blur:4"]:
        lines.append(r"\multicolumn{5}{@{}l}{\textit{" + COND_TEX[c] + r"}}\\")
        lines.append("Cosine to clean, tumour tiles & " + " & ".join(f"{drift[(m, c)][0]:.2f}" for m in cols) + r"\\")
        lines.append(r"Tile probe, tile AUC (clean \ra{} mixed) & " +
                     " & ".join(arrow(100 * PROBE[(m, 'clean', c)], 100 * PROBE[(m, 'mixed', c)]) for m in cols) + r"\\")
        lines.append(r"Top-attended tile on tumour, \% (clean \ra{} mixed) & " +
                     " & ".join(arrow(attention_mean(m, c, "clean", "top1_tumour"),
                                      attention_mean(m, c, "mixed", "top1_tumour")) for m in cols) + r"\\")
        cells = []
        for m in cols:
            g, lo, hi, _ = attention_paired(m, c, "top1_tumour")
            cell = f"{g:+.1f} {{\\scriptsize[{lo:+.1f}, {hi:+.1f}]}}"
            cells.append(r"\textbf{" + cell + "}" if lo > 0 or hi < 0 else cell)
        lines.append(r"\quad change from mixed training & " + " & ".join(cells) + r"\\")
        lines.append(r"Slide AUC, mixed-trained: attention / tile head & " +
                     " & ".join(f"{auc(m, 'abmil', 'mixed', c):.1f} / {auc(m, 'tileprobe', 'mixed', c):.1f}"
                                for m in cols) + r"\\")
        if c == "jpeg:10":
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("encoder_or_head", "\n".join(lines))


def table_sensitivity():
    rows = [("abmil", "clean", "Attention, clean-trained"), ("abmil", "mixed", "Attention, mixed-trained"),
            ("tileprobe", "clean", "Tile head, clean-trained"), ("tileprobe", "mixed", "Tile head, mixed-trained")]
    conds = ["clean", "jpeg:15", "jpeg:10", "blur:4"]
    lines = [r"\begin{tabular}{@{}ll" + "c" * len(conds) + "@{}}", r"\toprule",
             "Encoder & Head & " + " & ".join(COND_TEX[c] for c in conds) + r"\\", r"\midrule"]
    for m in PATHOLOGY:
        for i, (h, ts, name) in enumerate(rows):
            lead = MODELS[m][0] if i == 0 else ""
            lines.append(f"{lead} & {name} & " + " & ".join(
                f"{100 * float(SENS[(m, h, ts, c, 'all')]['sensitivity']):.0f}" for c in conds) + r"\\")
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    write("sensitivity", "\n".join(lines))


def table_hospital():
    sep = "normal slides: score separates UMCU from RUMC (AUC)"
    cols = ["phikon", "kaiko-b16", "phikon-v2"]
    lines = [r"\begin{tabular}{@{}llccccc@{}}", r"\toprule",
             r"Condition & Encoder, training & AUC RUMC & AUC UMCU & Micro RUMC & Micro UMCU & Centre from score\\",
             r"\midrule"]
    for c in ["clean", "blur:4"]:
        for m in cols:
            for ts in ["clean", "mixed"]:
                k = (m, "abmil", ts, c)
                lead = COND_TEX[c] if (m == cols[0] and ts == "clean") else ""
                lines.append(f"{lead} & {MODELS[m][0]}, {ts} & " + " & ".join(
                    f"{100 * CENTRE[k + (meas,)]:.1f}" for meas in
                    ["AUC within RUMC", "AUC within UMCU", "micro AUC within RUMC", "micro AUC within UMCU", sep]) + r"\\")
        if c == "clean":
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("hospital", "\n".join(lines))


def table_appendix():
    cols = ["phikon", "kaiko-b16", "phikon-v2", "dinov2-l"]
    out = []
    for head in ["mean", "max", "abmil", "tileprobe"]:
        lines = [r"\begin{table}[H]\centering\small",
                 r"\caption{" + HEAD_TEX[head] + r". Slide AUC in \%, all 129 test slides, mean $\pm$ standard "
                 r"deviation over five seeds, for heads trained on clean (C) and mixed (M) embeddings. "
                 r"JPEG Q10 was not in the mixed training set.}",
                 r"\label{tab:app-" + head + "}", r"\setlength{\tabcolsep}{3.5pt}",
                 r"\begin{tabular}{l" + "rr" * len(cols) + "}", r"\toprule",
                 " & " + " & ".join(r"\multicolumn{2}{c}{" + MODELS[m][0] + "}" for m in cols) + r"\\",
                 "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(cols))),
                 "Condition & " + " & ".join("C & M" for _ in cols) + r"\\", r"\midrule"]
        for c in CONDS:
            cells = []
            for m in cols:
                for ts in ["clean", "mixed"]:
                    cells.append(f"{auc(m, head, ts, c):.1f}{{\\scriptsize$\\pm${sd(m, head, ts, c):.1f}}}")
            lines.append(COND_TEX[c].replace(" (unseen)", "") + " & " + " & ".join(cells) + r"\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
        out.append("\n".join(lines))
    write("appendix", "\n".join(out))


def main():
    fig1()
    fig2()
    table_main()
    table_gains()
    table_heads_clean()
    table_encoder_or_head()
    table_sensitivity()
    table_hospital()
    table_appendix()


if __name__ == "__main__":
    main()
