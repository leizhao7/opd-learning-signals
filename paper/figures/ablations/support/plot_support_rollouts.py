"""Render complete, per-run normalized support/rollout ablation curves."""
from pathlib import Path
import argparse, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ap=argparse.ArgumentParser()
ap.add_argument('--data',type=Path,default=Path(__file__).with_name('support_rollout_curves.csv'))
ap.add_argument('--output-dir',type=Path,default=Path(__file__).parent)
args=ap.parse_args()
rows=list(csv.DictReader(args.data.open()))
args.output_dir.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
    'mathtext.fontset':'stix','font.size':8,'axes.titleweight':'normal',
    'axes.titlesize':8.5,'axes.labelsize':8,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
    'legend.fontsize':8,'axes.linewidth':.6,'pdf.fonttype':42,'ps.fonttype':42,
    'savefig.dpi':220})
styles=[(16,1,'Top-16, $n=1$','#4477AA','-'),
        (151936,1,'Full logits, $n=1$','#CC6677','--'),
        (16,8,'Top-16, $n=8$','#228833','-.')]
groups={'qwen':[('rlmath','(a) Qwen3-4B-RL-Math\n'),('qwen4b','(b) Qwen3-4B\n(Non-thinking)')],
        'r1':[('justrl','(a) JustRL-1.5B (self-RL)'),('r1_7b','(b) R1-Distill-7B')]}
for group,teachers in groups.items():
    fig,axs=plt.subplots(2,2,figsize=(5.5,4.45),sharex=True)
    fig.subplots_adjust(left=.13,right=.985,bottom=.11,top=.83,wspace=.28,hspace=.2)
    handles=[]
    for col,(teacher,title) in enumerate(teachers):
        axs[0,col].set_title(title,loc='left',pad=7)
        for support,n,label,color,ls in styles:
            rs=[r for r in rows if r['group']==group and r['teacher']==teacher and int(r['support'])==support and int(r['train_n'])==n]
            acc=[(int(r['step']),float(r['accuracy_pct'])) for r in rs if r['accuracy_pct']]
            loss=[(int(r['step']),float(r['loss'])) for r in rs if r['loss']]
            assert len(acc)==21 and len(loss)==200
            xa,ya=np.array(acc).T; xl,yl=np.array(loss).T
            yl=yl/yl[0]
            smooth=np.array([np.median(yl[max(0,i-4):min(len(yl),i+5)]) for i in range(len(yl))])
            axs[0,col].plot(xa,ya,color=color,ls=ls,lw=1.2,marker='o',markersize=2.2)
            axs[1,col].plot(xl,yl,color=color,alpha=.20,lw=.65)
            axs[1,col].plot(xl,smooth,color=color,ls=ls,lw=1.25)
            axs[1,col].scatter([xl[-1]],[yl[-1]],s=15,color=color,edgecolors='white',linewidths=.45,zorder=5)
            if col==0: handles.append(Line2D([0],[0],color=color,ls=ls,lw=1.4,label=label))
        for row in range(2):
            ax=axs[row,col]
            ax.spines[['top','right']].set_visible(False)
            ax.grid(axis='y',alpha=.16,lw=.5)
            ax.tick_params(length=3,width=.6)
            ax.set_xlim(-2,203);ax.set_xticks([0,50,100,150,200])
        axs[1,col].set_xlabel('OPD step')
        axs[1,col].set_ylim(bottom=0)
    # Matched accuracy and loss limits within each figure, preserving every raw point.
    for row in range(2):
        lo=min(axs[row,c].get_ylim()[0] for c in range(2));hi=max(axs[row,c].get_ylim()[1] for c in range(2))
        for c in range(2):axs[row,c].set_ylim(lo,hi)
    axs[0,0].set_ylabel('Validation accuracy (%)')
    axs[1,0].set_ylabel(r'Remaining loss $\widehat{\mathcal{L}}_m/\widehat{\mathcal{L}}_1$')
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.535,.985),ncol=3,
               frameon=False,handlelength=2.1,columnspacing=1.2)
    for ext in ['pdf','png']:
        fig.savefig(args.output_dir/f'support_rollout_{group}.{ext}')
    plt.close(fig)
