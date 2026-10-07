"""Appendix learning-signal figure: premature learning-signal collapse. Layout, colors, names and legend placement follow Figure 1
(figures/build_teacher_trainability.py): columns = tasks, rows = quantities, per-column "OPD teachers:" legends above.

Inputs (repo-relative): figures/data/{code,math}_signal_collapse_training.csv
Outputs: figures/analysis/code_signal_collapse.{pdf,png,svg}
Curves: centered rolling median over WINDOW steps (main line) with raw per-step values drawn faintly behind.
Vertical dotted line: the split step s used in Table 3 (Code 15, Math 30).
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
OUT = ROOT / "figures" / "analysis" / "code_signal_collapse"
WINDOW = 9
COLORS = ["#0072B2", "#D55E00", "#666666"]          # 与图 1 相同：蓝 = 同模型 RL teacher，橙 = 第二个，灰 = 第三个
RAW_ALPHA, GRID, MUTED = 0.22, "#e5e5e5", "#666666"

COLS = [
    dict(csv="figures/data/code_signal_collapse_training.csv", task="Code", student="Qwen3-4B (Non-thinking)", s_ylim=(0.5, 1000), split=15,
         series=[("RL-Code", NAMES["rl4b"]), ("Qwen3-14B", "Qwen3-14B (Non-thinking)")]),
    dict(csv="figures/data/math_signal_collapse_training.csv", task="Math", student="R1-Distill-1.5B", s_ylim=(0.02, 20), split=30,
         series=[("JustRL", "JustRL-1.5B"), ("Skywork", "Skywork-7B"), ("R1-Distill-7B", "R1-Distill-7B")]),
]

def load(path):
    d = defaultdict(list)
    for r in csv.DictReader(open(ROOT / path)):
        d[r["teacher"]].append((int(r["step"]), float(r["remaining_loss_fraction"]), float(r["nu_train"])))
    return {k: np.array(sorted(v)) for k, v in d.items()}

def rolling_median(y, w=WINDOW):
    h = w // 2
    return np.array([np.median(y[max(0, i - h): i + h + 1]) for i in range(len(y))])

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8.5, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "xtick.labelsize": 8.4, "ytick.labelsize": 8.4,
    "axes.linewidth": .65, "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "stix",
})

fig = plt.figure(figsize=(6.6, 4.5), facecolor="white")
W, H = .365, .27
for col, spec in enumerate(COLS):
    data = load(spec["csv"])
    left = .078 + .490 * col
    ax_l = fig.add_axes([left, .505, W, H])
    ax_s = fig.add_axes([left, .16, W, H], sharex=ax_l)
    xmax = max(arr[:, 0].max() for arr in data.values())
    handles = []
    for i, (key, label) in enumerate(spec["series"]):
        arr = data[key]; step, rem, nu = arr[:, 0], 100 * arr[:, 1], arr[:, 2]
        c = COLORS[i]
        ax_l.plot(step, rem, color=c, lw=.6, alpha=RAW_ALPHA)
        ax_s.plot(step, nu, color=c, lw=.6, alpha=RAW_ALPHA)
        ax_l.plot(step, rolling_median(rem), color=c, lw=1.2)
        ax_s.plot(step, rolling_median(nu), color=c, lw=1.2)
        handles.append(Line2D([], [], linestyle="none", marker="o", markersize=4.5, markerfacecolor=c, markeredgecolor="none", label=label))
    ax_s.set_yscale("log")
    for ax in (ax_l, ax_s):   # 表 3 的分割步 s：两块面板各一条竖虚线
        ax.axvline(spec["split"], color=MUTED, lw=.7, ls=(0, (2, 2)), alpha=.9, zorder=1)
        ax.set_xlim(0, xmax)
        ax.set_xticks([0, 50, 100, 150, 200])
        ax.tick_params(length=2.5, width=.6, pad=2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=GRID, lw=.55); ax.set_axisbelow(True)
    ax_l.annotate(f"$s={spec['split']}$", xy=(spec["split"], 1.0), xycoords=("data", "axes fraction"),
                  xytext=(3, -1), textcoords="offset points", ha="left", va="top", fontsize=8.4, color=MUTED)
    ax_l.set_ylim(0, 150); ax_l.set_yticks([0, 50, 100, 150])        # 剩余损失是无量纲百分比：两列共用同一刻度
    ax_l.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=100, decimals=0))
    ax_s.set_ylim(*spec["s_ylim"])                                    # 信号绝对值只在列内比较：各列自己的范围，但都跨 3 个数量级
    # 对数轴只留 decade 主刻度、普通数字标签，去掉小刻度
    ax_s.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=6))
    ax_s.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax_s.yaxis.set_major_formatter(matplotlib.ticker.LogFormatterMathtext(base=10))
    ax_l.tick_params(labelbottom=False)
    ax_s.set_xlabel("OPD step")
    fig.text(left, 1 - .048 / 4.5, f"({chr(97 + col)}) {spec['task']}", ha="left", va="top", fontsize=9, weight="bold")
    fig.text(left, 1 - .216 / 4.5, "Student: " + spec["student"], ha="left", va="top", fontsize=8.4, color="#444444")
    if col == 0:
        ax_l.set_ylabel(r"Remaining loss $\widehat{\mathcal{L}}_m/\widehat{\mathcal{L}}_1$")
        ax_s.set_ylabel(r"Learning signal $\widehat{\mu}_m$")
        for ax in (ax_l, ax_s):
            ax.yaxis.set_label_coords(-0.14, 0.5)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(left, 1 - .380 / 4.5), bbox_transform=fig.transFigure,
               borderaxespad=0, frameon=False, title="OPD teachers:", title_fontsize=8.4, alignment="left",
               fontsize=8.4, handlelength=.8, handleheight=.7, handletextpad=.5, labelspacing=.25, borderpad=0)
for ext in ("pdf", "png", "svg"):
    fig.savefig(OUT.with_suffix("." + ext), dpi=300)
print("wrote", OUT)
