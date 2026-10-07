"""Figure 2 (main text): learning signal vs remaining loss, one dot per training step (raw, unsmoothed, all steps).

Colors, names and legend placement follow Figure 1 (figures/build_teacher_trainability.py).
Inputs (repo-relative): figures/data/{code,math}_signal_collapse_training.csv
Outputs: figures/analysis/signal_collapse_phase.{pdf,png,svg}
"""
import csv, pathlib, sys
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from model_names import display_names
NAMES = display_names()
OUT = ROOT / "figures" / "analysis" / "signal_collapse_phase"
COLORS = ["#0072B2", "#D55E00", "#666666"]          # 与图 1 相同
COLS = [
    dict(csv="figures/data/code_signal_collapse_training.csv", task="Code", student="Qwen3-4B (Non-thinking)", ylim=(0.5, 1000),
         series=[("RL-Code", NAMES["rl4b"]), ("Qwen3-14B", "Qwen3-14B (Non-thinking)")]),
    dict(csv="figures/data/math_signal_collapse_training.csv", task="Math", student="R1-Distill-1.5B", ylim=(0.02, 20),
         series=[("JustRL", "JustRL-1.5B"), ("Skywork", "Skywork-7B"), ("R1-Distill-7B", "R1-Distill-7B")]),
]
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8.5,
                     "axes.labelsize": 9, "xtick.labelsize": 8.4, "ytick.labelsize": 8.4, "axes.linewidth": .65,
                     "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42})
fig = plt.figure(figsize=(6.6, 3.1), facecolor="white")
W, H = .365, .535
for col, spec in enumerate(COLS):
    d = defaultdict(list)
    for r in csv.DictReader(open(ROOT / spec["csv"])):
        d[r["teacher"]].append((int(r["step"]), float(r["remaining_loss_fraction"]), float(r["nu_train"])))
    left = .078 + .490 * col
    ax = fig.add_axes([left, .15, W, H]); handles = []
    for i, (key, label) in enumerate(spec["series"]):
        a = np.array(sorted(d[key])); c = COLORS[i]
        x, y = 100 * a[:, 1], a[:, 2]                  # 原始逐步数据：不平滑、不截断
        ax.scatter(x, y, s=7, color=c, alpha=.55, linewidths=0, zorder=3)
        ax.plot([x[0]], [y[0]], "o", ms=5.5, mfc="white", mec=c, mew=1.2, zorder=4)   # 第 1 步
        handles.append(Line2D([], [], linestyle="none", marker="o", markersize=4.5,
                              markerfacecolor=c, markeredgecolor="none", label=label))
    ax.set_yscale("log"); ax.set_ylim(*spec["ylim"]); ax.set_xlim(150, 0)    # 横轴反向：往右 = loss 降
    ax.set_xticks([0, 50, 100]); ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=6))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_formatter(matplotlib.ticker.LogFormatterMathtext(base=10))
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#e5e5e5", lw=.55); ax.set_axisbelow(True)
    ax.tick_params(length=2.5, width=.6, pad=2)
    ax.set_xlabel(r"Remaining loss $\widehat{\mathcal{L}}_m/\widehat{\mathcal{L}}_1$")
    if col == 0:
        ax.set_ylabel(r"Learning signal $\widehat{\mu}_m$")
    # Match Figure 1 typography and physical header spacing at equal paper width.
    fig.text(left, 1 - .048 / 3.1, f"({chr(97 + col)}) {spec['task']}",
             ha="left", va="top", fontsize=9, weight="bold")
    fig.text(left, 1 - .216 / 3.1, "Student: " + spec["student"],
             ha="left", va="top", fontsize=8.4, color="#444444")
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(left, 1 - .380 / 3.1),
               bbox_transform=fig.transFigure, borderaxespad=0, frameon=False,
               title="OPD teachers:", title_fontsize=8.4, alignment="left",
               fontsize=8.4, handlelength=.8, handleheight=.7, handletextpad=.5,
               labelspacing=.25, borderpad=0)
for ext in ("pdf", "png", "svg"):
    fig.savefig(OUT.with_suffix("." + ext), dpi=300)
print("wrote", OUT)
