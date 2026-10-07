from pathlib import Path
import json
P=Path(__file__).resolve().parent
data=json.loads((P/'support_rollout_metrics.json').read_text())
lines=[r'''% Generated from support_rollout_metrics.json; values use unsmoothed native losses.
\begin{table}[tbp]
\centering
\normalsize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.18}
\caption{Support and rollout comparisons after 200 updates. Accuracy is
macro-averaged $\mathrm{avg}@8$. Maximum and final loss reduction use
$(1-\widehat{\mathcal L}_{\min}/\widehat{\mathcal L}_1)\times100\%$ and
$(1-\widehat{\mathcal L}_{200}/\widehat{\mathcal L}_1)\times100\%$,
respectively. Each uses its own run's objective. The four top-16, $n=1$
rows are the main-run baselines; the other eight rows are additional runs.}
\label{tab:support-rollout-results}
\begin{tabular}{@{}llcrrrr@{}}
\toprule
Teacher & Support & $n$ & \multicolumn{2}{c}{Accuracy} & \multicolumn{2}{c}{Loss reduction} \\
\cmidrule(lr){4-5}\cmidrule(l){6-7}
 & & & Initial & Final & Maximum & Final \\
\midrule''']
last_group=None;last_teacher=None
names={'rlmath':r'\QwenRLMath{}','qwen4b':'Qwen3-4B','justrl':'JustRL-1.5B','r1_7b':'R1-Distill-7B'}
for r in data:
    if r['group']!=last_group:
        if last_group is not None:lines.append(r'\midrule')
        student='Qwen3-1.7B (Non-thinking)' if r['group']=='qwen' else 'R1-Distill-1.5B'
        lines.append(r'\multicolumn{7}{@{}l}{\textit{Student: '+student+r'}} \\')
        lines.append(r'\addlinespace[2pt]')
    elif r['teacher']!=last_teacher:
        lines.append(r'\addlinespace[3pt]')
    t=names[r['teacher']] if r['teacher']!=last_teacher or r['group']!=last_group else ''
    fields=[t,'Top-16' if r['support']==16 else 'Full',str(r['train_n'])]+[f"{r[k]:.2f}\\%" for k in ['acc_start_pct','acc_final_pct','max_loss_reduction_pct','final_loss_reduction_pct']]
    lines.append(' & '.join(fields)+r' \\')
    last_group=r['group'];last_teacher=r['teacher']
lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
(P/'support_rollout_results.tex').write_text('\n'.join(lines)+'\n')
