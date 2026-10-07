"""Plot audited IF checkpoints (0--200) and raw signed top-16 objectives (1--200)."""
from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
OUT=Path(__file__).resolve().parent
D=OUT.parent/'data/appendix_if'
scores=list(csv.DictReader((D/'if_composite_curve.csv').open()))
losses=list(csv.DictReader((D/'if_loss_steps.csv').open()))
summary=json.loads((D/'verified_summary.json').read_text())
colors={'ultradata':'#0072B2','r17b':'#D55E00'}
names={'ultradata':'UltraData-IF-1.5B (self-RL)','r17b':'R1-Distill-7B'}
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Times','STIXGeneral','DejaVu Serif'],'mathtext.fontset':'stix','font.size':8,'axes.labelsize':8,'axes.titlesize':8.3,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.6})
fig=plt.figure(figsize=(5.5,4.3))
gs=fig.add_gridspec(2,2,left=.115,right=.98,bottom=.11,top=.79,hspace=.65,wspace=.34,height_ratios=[1.15,1])
a=fig.add_subplot(gs[0,:]);bs=[fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1])]
fig.text(.115,.965,'Student: R1-Distill-1.5B',ha='left',va='top',fontsize=8.3,color='#444444')
handles=[Line2D([],[],color=colors[k],marker='o',markersize=4,linewidth=0,label=names[k]) for k in colors]
leg=fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.101,.945),frameon=False,ncol=2,title='OPD teachers:',title_fontsize=8.3,fontsize=8,handletextpad=.2,columnspacing=1,borderaxespad=0)
leg._legend_box.align='left'
for i,key in enumerate(colors):
 color=colors[key];data=[x for x in scores if x['run']==key]
 a.plot([int(x['step']) for x in data],[float(x['composite_score_pct']) for x in data],color=color,lw=1.35,marker='o',ms=2.5)
 teacher=summary['runs'][key]['score_teacher']
 a.axhline(teacher,color=color,lw=.9,ls=(0,(4,3)),alpha=.8)
 a.text(198,teacher+.85,f'Teacher {teacher:.1f}',ha='right',va='bottom',color=color,fontsize=7.5,bbox={'facecolor':'white','edgecolor':'none','alpha':.8,'pad':.2})
 z=[x for x in losses if x['run']==key];b=bs[i]
 b.plot([int(x['step']) for x in z],[float(x['logged_top16_objective']) for x in z],color=color,lw=.85)
 b.axhline(0,color='#555555',lw=.65,ls=(0,(2,2)),zorder=0)
 b.set_title(f'({chr(98+i)}) '+('UltraData-IF-1.5B' if key=='ultradata' else 'R1-Distill-7B'),loc='left',fontweight='normal',pad=6)
 b.set_xlabel('Training update');b.set_ylabel('Logged top-16 objective')
 if key=='ultradata':b.set_ylim(-.003,.030)
 else:b.set_ylim(-.045,.92)
a.scatter([0],[summary['runs']['ultradata']['score_start']],s=22,color='#333333',zorder=8)
a.set_title('(a) Multi-IF score',loc='left',fontweight='normal',pad=6)
a.set_ylabel('Three-turn composite (%)');a.set_xlabel('Training update');a.set_ylim(21,54)
for ax in [a,*bs]:
 ax.set_xlim(0,200);ax.set_xticks([0,50,100,150,200]);ax.grid(axis='y',alpha=.18,linewidth=.6);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
for suffix in ['pdf','png','svg']:fig.savefig(OUT/f'if_dynamics_200.{suffix}',dpi=220)
plt.close(fig)
