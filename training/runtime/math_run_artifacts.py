import csv
import json
from pathlib import Path

BENCHES = ('AIME24', 'AIME25', 'AMC23')


def load_rows(root):
    rows = {}
    path = root / 'metrics.jsonl'
    if not path.exists():
        return []
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        step = int(record['step'])
        row = rows.setdefault(step, {'step': step})
        data = record['data']
        if 'critic/advantages/mean_sum_over_k' in data:
            row['loss'] = -float(data['critic/advantages/mean_sum_over_k'])
        if 'actor/pg_loss' in data:
            row['actor_pg_loss'] = float(data['actor/pg_loss'])
        for bench in BENCHES:
            key = f'val-core/{bench}/acc/mean@8'
            if key in data:
                row[bench] = float(data[key])
        if all(bench in row for bench in BENCHES):
            row['math_mean8'] = sum(row[bench] for bench in BENCHES) / 3
    return [rows[step] for step in sorted(rows)]


def render(root):
    root = Path(root)
    rows = load_rows(root)
    if not rows:
        return
    with (root/'curves.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['step','loss','actor_pg_loss',*BENCHES,'math_mean8'])
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    loss = [r for r in rows if 'loss' in r]
    if loss:
        axes[0].plot([r['step'] for r in loss], [r['loss'] for r in loss], color='#2855A1', lw=1.5)
    else:
        axes[0].text(.5, .5, 'Waiting for first training step', ha='center', transform=axes[0].transAxes)
    axes[0].set(title='OPD loss: token mean', xlabel='Training step', ylabel='Loss')
    for key, label, color, width in [
        ('AIME24','AIME24','#388B88',1.2), ('AIME25','AIME25','#DE8F32',1.2),
        ('AMC23','AMC23','#8D65A8',1.2), ('math_mean8','Equal benchmark mean','#24364B',2.2)]:
        values = [r for r in rows if key in r]
        if values:
            axes[1].plot([r['step'] for r in values], [100*r[key] for r in values],
                         label=label, color=color, lw=width, marker='o', ms=3)
    axes[1].set(title='Validation: mean@8, response cap 31,744', xlabel='Training step', ylabel='Accuracy (%)')
    if axes[1].lines:
        axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.18)
        ax.set_xlim(0, 200)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle(root.name, fontsize=12)
    for suffix in ['png','svg']:
        temporary = root/f'.loss_and_eval.tmp.{suffix}'
        fig.savefig(temporary, dpi=170)
        temporary.replace(root/f'loss_and_eval.{suffix}')
    plt.close(fig)


if __name__ == '__main__':
    import sys
    render(sys.argv[1])
