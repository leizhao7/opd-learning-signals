"""Render Figure 1 from audited data; optionally export a standalone preview."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-stem', type=Path, default=ROOT / 'figures/motivation/teacher_trainability_multigroup')
STEM = parser.parse_args().output_stem.resolve()
STEM.parent.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
from model_names import display_names

DATA = ROOT / 'figures/data/opd_teacher_trainability_multigroup.csv'
SUMMARY = ROOT / 'figures/data/opd_teacher_trainability_multigroup_summary.json'
summary = json.loads(SUMMARY.read_text())
rows = list(csv.DictReader(DATA.open()))
names = {'justrl1p5b': 'JustRL-1.5B', 'r1_7b': 'R1-Distill-7B'}
names.update(display_names())
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 8.5, 'axes.labelsize': 9, 'axes.titlesize': 9.5,
    'xtick.labelsize': 8.4, 'ytick.labelsize': 8.4,
    'axes.linewidth': .65, 'pdf.fonttype': 42, 'ps.fonttype': 42,
})
fig = plt.figure(figsize=(6.6, 4.0), facecolor='white')
colors = ['#0072B2', '#D55E00', '#666666']

def rolling_median(y, w=9):
    """Centered rolling median (window w) with shrinking windows at the ends."""
    import statistics
    h = w // 2
    return [statistics.median(y[max(0, i - h): i + h + 1]) for i in range(len(y))]

markers = ['o', 'o', '^']
styles = ['-', '-', '-']
tasks = ['Code', 'Math', 'Math']
students = [names['qwen4b'], names['qwen1p7b'], 'R1-Distill-1.5B']
ylims = [(50, 65), (18, 75), (36, 76)]
yticks = [[50, 55, 60, 65], [20, 30, 40, 50, 60, 70], [40, 50, 60, 70]]
loss_ylims = [(0, .22), (0, .30), (0, .50)]
loss_yticks = [[0, .05, .10, .15, .20], [0, .10, .20, .30],
              [0, .10, .20, .30, .40, .50]]
audit = []
for col, (group_key, group) in enumerate(summary['groups'].items()):
    left = .078 + .315 * col
    acc = fig.add_axes([left, .465, .265, .280])
    loss = fig.add_axes([left, .11, .265, .280], sharex=acc)
    handles = []
    for i, (key, run) in enumerate(group['runs'].items()):
        rr = sorted([r for r in rows if r['group'] == group_key and r['teacher'] == key],
                    key=lambda r: int(r['step']))
        ll = [(int(r['step']), float(r['opd_loss'])) for r in rr if r['opd_loss']]
        vv = [(int(r['step']), float(r['macro_val_accuracy_percent']))
              for r in rr if r['macro_val_accuracy_percent']]
        assert ll[0][0] == 1 and ll[-1][0] == group['max_step']
        assert abs(ll[-1][1] - run['loss_final']) < 1e-12
        assert abs(vv[-1][1] - run['macro_accuracy_final']) < 1e-10
        acc.plot(*zip(*vv), color=colors[i], ls=styles[i], lw=1.1,
                 marker=markers[i], ms=2.4, mew=.5, clip_on=False,
                 markevery=list(range(1, len(vv))) if group.get('shared_initial_evaluation') else None)
        assert all(loss_ylims[col][0] <= y <= loss_ylims[col][1] for _, y in ll)
        # 与图 2 一致：原始逐步损失淡画在后，主线为居中 9 步滚动中位数
        steps_l, vals_l = zip(*ll)
        loss.plot(steps_l, vals_l, color=colors[i], ls=styles[i], lw=.6, alpha=.22)
        loss.plot(steps_l, rolling_median(vals_l), color=colors[i],
                  ls=styles[i], lw=1.05)
        teacher = run.get('teacher_macro_accuracy')
        if teacher is not None:
            acc.axhline(teacher, color=colors[i], ls=(0, (4, 3)),
                        lw=.7, alpha=.55, zorder=1)
            # Put the two close R1 references on opposite sides, and keep
            # the Code reference label below the higher student curve.
            x_fraction, offset, ha, va = .98, (-2, 2), 'right', 'bottom'
            if key == 'justrl1p5b':
                x_fraction, offset, ha, va = .03, (2, -2), 'left', 'top'
            elif key == 'qwen14b':
                offset, va = (-2, -2), 'top'
            acc.annotate(f'Teacher {teacher:.1f}%', xy=(x_fraction, teacher),
                         xycoords=('axes fraction', 'data'), xytext=offset,
                         textcoords='offset points', ha=ha, va=va,
                         color=colors[i], fontsize=8.4, zorder=5,
                         bbox=dict(facecolor='white', edgecolor='none', alpha=.80, pad=.3))
        label = names[key]
        handles.append(Line2D([], [], linestyle='none', marker='o',
                              markersize=4.5, markerfacecolor=colors[i],
                              markeredgecolor='none', label=label))
        audit.append({'group': group_key, 'teacher': key, 'loss_points': len(ll),
                      'accuracy_points': len(vv), 'first_accuracy_step': vv[0][0],
                      'first_accuracy': vv[0][1],
                      'native_first_accuracy_step': run['native_macro_accuracy_first_step'],
                      'shared_initial_evaluation': bool(group.get('shared_initial_evaluation')),
                      'last_step': ll[-1][0],
                      'initial_loss': ll[0][1], 'last_loss': ll[-1][1],
                      'loss_axis_limits': loss_ylims[col],
                      'last_accuracy': vv[-1][1],
                      'teacher_reference': teacher})
    if group.get('shared_initial_evaluation'):
        initial = group['shared_initial_evaluation']['macro_accuracy_percent']
        acc.plot([0], [initial], marker='o', color='#333333', ls='none',
                 ms=3.1, markeredgecolor='white', markeredgewidth=.35,
                 clip_on=False, zorder=7)
    fig.text(left, .988, f'({chr(97 + col)}) {tasks[col]}', ha='left', va='top',
             fontsize=9, weight='bold')
    fig.text(left, .946, 'Student: ' + students[col],
             ha='left', va='top', fontsize=8.4, color='#444444')
    acc.set_ylim(*ylims[col]); acc.set_yticks(yticks[col])
    loss.set_ylim(*loss_ylims[col]); loss.set_yticks(loss_yticks[col])
    loss.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
    for row, ax in enumerate([acc, loss]):
        ax.set_xlim(0, 200)
        ax.set_xticks([0, 50, 100, 150, 200])
        ax.tick_params(length=2.5, width=.6, pad=2)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#e5e5e5', lw=.55)
        ax.set_axisbelow(True)
    acc.tick_params(labelbottom=False)
    if col == 0:
        acc.set_ylabel('Validation accuracy (%)', labelpad=6)
        loss.set_ylabel('OPD loss', labelpad=6)
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(left, .905),
               bbox_transform=fig.transFigure, borderaxespad=0, frameon=False,
               title='OPD teachers:', title_fontsize=8.4, alignment='left',
               fontsize=8.4, handlelength=.8, handleheight=.7, handletextpad=.5,
               labelspacing=.25, borderpad=0)
fig.text(.525, .022, 'OPD step', ha='center', va='bottom', fontsize=9)
for ext in ['pdf', 'svg', 'png']:
    fig.savefig(STEM.with_suffix('.' + ext), dpi=300)
plt.close(fig)
(STEM.with_name(STEM.name + '_audit.json')).write_text(json.dumps({
    'data_sha256': hashlib.sha256(DATA.read_bytes()).hexdigest(),
    'summary_sha256': hashlib.sha256(SUMMARY.read_bytes()).hexdigest(),
    'transformations': ['accuracy shown as percent', 'raw loss shown faintly with centered 9-step rolling medians',
                        'column-specific linear loss axes, each with numeric tick labels',
                        'matched teacher accuracy shown as directly labeled dashed reference lines'],
    'runs': audit}, indent=2) + '\n')
(STEM.with_name(STEM.name + '_caption.txt')).write_text(
    'Student validation accuracy (top) and OPD loss (bottom) over 200 training updates. Panels (a) and (c) use a shared evaluation of the initial student at step 0, shown as a black point. Colors identify the teacher used for OPD; dashed lines show standalone teacher accuracy. Qwen3 models use non-thinking mode. Loss curves are centered 9-step rolling medians, with raw per-step losses shown faintly.\n')
print(STEM.with_suffix('.png'))
print('Verified all seven run endpoints against the audited summary.')
