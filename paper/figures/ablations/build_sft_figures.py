"""Rebuild appendix figures from the colocated, audited normalized data.

Usage: bundled Python build_sft_figures.py [--output-dir DIR].
Output canvas width is the paper text width (5.5 inches); all ordinary labels
are >=7.5 points without downstream resizing. No benchmark trajectory is
invented for SFT: its figure shows teacher-response cross-entropy only.
"""
import argparse, csv, json, statistics, os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent/'.mplcache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--output-dir',type=Path,default=HERE)
OUT=ap.parse_args().output_dir;OUT.mkdir(parents=True,exist_ok=True)
SFT=list(csv.DictReader((HERE/'sft_curves.csv').open()))
OPD=list(csv.DictReader((HERE/'opd_curves.csv').open()))
EVIDENCE=json.loads((HERE/'sft_evidence.json').read_text())
COLORS={'justrl':'#0072B2','r1_7b':'#666666',
        'qwen_math_rl':'#0072B2','qwen4b_nt':'#D55E00'}
NAMES={'justrl':'JustRL-1.5B','r1_7b':'R1-Distill-7B',
        'qwen_math_rl':'Qwen3-4B-RL-Math','qwen4b_nt':'Qwen3-4B (Non-thinking)'}
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
 'font.size':8,'axes.labelsize':8,'axes.titlesize':8.5,
 'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':7.5,
 'axes.linewidth':.6,'pdf.fonttype':42,'ps.fonttype':42,'mathtext.fontset':'stix'})

def style(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(width=.6,length=2.5,pad=2)
    ax.grid(axis='y',color='#dedede',lw=.45,zorder=0)
    ax.set_axisbelow(True)

def save(fig,stem):
    fig.savefig(OUT/(stem+'.pdf'))
    fig.savefig(OUT/(stem+'.png'),dpi=220)
    plt.close(fig)

# Four single SFT runs; per-teacher datasets make between-teacher CE values
# descriptive only. Holdout points are rows, with repeated questions allowed.
fig,axes=plt.subplots(1,2,figsize=(5.5,2.95))
fig.subplots_adjust(left=.10,right=.97,bottom=.17,top=.65,wspace=.30)
for col,(keys,student) in enumerate([
    (['justrl','r1_7b'],'R1-Distill-1.5B'),
    (['qwen_math_rl','qwen4b_nt'],'Qwen3-1.7B (Non-thinking)')]):
    ax=axes[col];style(ax)
    for key in keys:
        for metric in ['loss','eval_loss']:
            rows=sorted((r for r in SFT if r['teacher_key']==key and r['metric']==metric),
                        key=lambda x:int(x['step']))
            x=[int(r['step']) for r in rows];y=[float(r['value']) for r in rows]
            ax.plot(x,y,color=COLORS[key],lw=1.05 if metric=='loss' else .8,
                    ls='-' if metric=='loss' else '--',
                    marker=None if metric=='loss' else 'o',ms=3,mfc='white',mew=.8)
    ax.set(xlim=(0,300),ylim=(.20,.45),xticks=[0,100,200,300],
           yticks=[.20,.25,.30,.35,.40,.45],xlabel='SFT optimizer update')
    if col==0:ax.set_ylabel('Response cross-entropy')
    ax.text(0,1.38,f'({chr(97+col)}) {student}',transform=ax.transAxes,
            va='bottom',ha='left',fontsize=8.2)
    ax.legend([Line2D([],[],color=COLORS[k],lw=1.2) for k in keys],
              [NAMES[k] for k in keys],loc='lower left',bbox_to_anchor=(0,1.035),
              frameon=False,borderaxespad=0,handlelength=1.3,labelspacing=.35)
fig.legend([Line2D([],[],color='black',lw=1),
            Line2D([],[],color='black',lw=.8,ls='--',marker='o',ms=3,mfc='white')],
           ['Training','Held-out rows'],
           loc='upper center',bbox_to_anchor=(.5,.99),ncol=2,frameon=False,
           handlelength=1.6,columnspacing=1.4)
save(fig,'sft_training_dynamics')

def median(y):
    return [statistics.median(y[max(0,i-4):min(len(y),i+5)]) for i in range(len(y))]

# OPD continuation vs base-initialized OPD: use each run's own evaluation;
# a separate larger open circle denotes the original standalone SFT evaluation.
fig,axes=plt.subplots(2,2,figsize=(5.5,4.25))
fig.subplots_adjust(left=.10,right=.97,bottom=.12,top=.82,hspace=.43,wspace=.30)
for col,(key,student) in enumerate([('r1_7b','R1-Distill-1.5B'),
                                   ('qwen4b_nt','Qwen3-1.7B (Non-thinking)')]):
    color=COLORS[key];acc,loss=axes[0,col],axes[1,col]
    for ax in [acc,loss]:
        style(ax);ax.set_xlim(-3,203);ax.set_xticks([0,50,100,150,200])
    for stage,ls,marker in [('direct','--','o'),('after_sft','-','o')]:
        rows=sorted((r for r in OPD if r['teacher_key']==key and r['stage']==stage),
                    key=lambda r:int(r['step']))
        ar=[r for r in rows if r['accuracy_percent']!='']
        lr=[r for r in rows if r['loss']!='']
        assert len(lr)==200 and int(lr[0]['step'])==1 and int(lr[-1]['step'])==200
        acc.plot([int(r['step']) for r in ar],[float(r['accuracy_percent']) for r in ar],
           color=color,ls=ls,lw=1.05,marker=marker,ms=2.7,
           mfc='white' if stage=='direct' else color,mew=.7,clip_on=False)
        x=[int(r['step']) for r in lr];y=[float(r['loss']) for r in lr]
        loss.plot(x,y,color=color,ls=ls,lw=.55,alpha=.22)
        loss.plot(x,median(y),color=color,ls=ls,lw=1.1)
    original=EVIDENCE[key]['benchmark']['macro_percent']
    acc.plot([0],[original],ls='none',marker='o',ms=5.2,mfc='white',mec=color,mew=1,
             clip_on=False,zorder=8)
    acc.set_ylim((29,46) if key=='r1_7b' else (18,27))
    loss.set_ylim((0,.23) if key=='r1_7b' else (0,.17))
    if col==0:acc.set_ylabel('Math validation accuracy (%)');loss.set_ylabel('OPD loss')
    acc.set_xlabel('OPD update');loss.set_xlabel('OPD update')
    acc.text(0,1.24,f'({chr(97+col)}) {student}',transform=acc.transAxes,
             va='bottom',fontsize=8.2)
    acc.text(0,1.11,'Teacher: '+NAMES[key],transform=acc.transAxes,
             va='bottom',fontsize=7.7,color=color)
fig.legend([
 Line2D([],[],color='black',ls='--',marker='o',mfc='white',ms=3,lw=1),
 Line2D([],[],color='black',ls='-',marker='o',ms=3,lw=1),
 Line2D([],[],color='black',ls='none',marker='o',mfc='white',ms=5.2)],
 ['Direct OPD','After SFT','Standalone SFT'],
 loc='upper center',bbox_to_anchor=(.5,.99),ncol=3,frameon=False,
 handlelength=1.6,columnspacing=1.1)
save(fig,'sft_opd_training_dynamics')
print('Wrote two PDF/PNG figures to',OUT)
