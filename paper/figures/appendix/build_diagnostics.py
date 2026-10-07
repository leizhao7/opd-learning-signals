"""Appendix-only plots from audited per-step/endpoint data; no main assets changed."""
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'figures/data'
OUT = Path(__file__).resolve().parent
OUT.mkdir(exist_ok=True)
BLUE, ORANGE, GRAY, GREEN = '#0072B2', '#D55E00', '#666666', '#009E73'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman','DejaVu Serif'],
 'font.size':8, 'axes.labelsize':8, 'axes.titlesize':8.3,
 'xtick.labelsize':7.5, 'ytick.labelsize':7.5, 'mathtext.fontset':'stix',
 'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.6})
sources = []

def load(name, key='teacher'):
    path = DATA / name
    sources.append({'path':str(path.relative_to(ROOT)), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    out = defaultdict(list)
    for row in csv.DictReader(path.open()):
        out[row[key]].append(row)
    return out

def axis(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(length=2.5,width=.55)
    ax.grid(axis='y',color='#dddddd',lw=.45)
    ax.set_axisbelow(True)

def save(fig, name):
    for ext in ['pdf','png']:
        fig.savefig(OUT / (name+'.'+ext),dpi=220,facecolor='white')
    plt.close(fig)

def med(y):
    return np.array([np.median(y[max(0,i-4):min(len(y),i+5)]) for i in range(len(y))])

occ = load('code_signal_collapse_occupancy.csv')
occ.update(load('math_signal_collapse_occupancy.csv'))
fig, axes = plt.subplots(2,2,figsize=(5.5,3.65))
specs = [('Qwen3-14B','(a) Code / Qwen3-14B',ORANGE),
         ('JustRL','(b) Math / JustRL-1.5B',BLUE),
         ('Skywork','(c) Math / Skywork-7B',ORANGE),
         ('R1-Distill-7B','(d) Math / R1-Distill-7B',GRAY)]
stats = {}
for ax,(key,title,color) in zip(axes.flat,specs):
    rows=sorted(occ[key],key=lambda x:int(x['step']))
    x=np.array([int(r['step']) for r in rows]);y=np.array([float(r['one_plus_chi']) for r in rows])
    assert list(x)==list(range(10,201,10))
    assert all(float(r['g2_crossfit'])>0 for r in rows)
    ax.plot(x,y,'o-',color=color,lw=1,ms=3)
    ax.axhline(0,color='#222222',ls='--',lw=.75)
    ax.axhline(1,color='#888888',ls=':',lw=.65)
    ax.set_title(title,loc='left',pad=6)
    ax.set_xlim(0,205);ax.set_xticks([0,50,100,150,200])
    ax.set_ylim((-.5,2.8) if key=='Qwen3-14B' else (-9,6))
    ax.set_xlabel('OPD step');ax.set_ylabel(r'$1+\widehat{\alpha}$')
    axis(ax)
    stats[key]={'n':len(y),'positive':int((y>0).sum()),'median':float(np.median(y)), 'min':float(y.min()),'max':float(y.max())}
fig.subplots_adjust(left=.095,right=.98,bottom=.105,top=.92,hspace=.62,wspace=.30)
save(fig,'occupancy_dynamics')

training=[load('code_signal_collapse_training.csv'),load('math_signal_collapse_training.csv')]
fig,axes=plt.subplots(1,2,figsize=(5.5,2.65))
specs=[[('RL-Code','Qwen3-4B-RL-Code',BLUE),('Qwen3-14B','Qwen3-14B',ORANGE)],
       [('JustRL','JustRL-1.5B',BLUE),('Skywork','Skywork-7B',ORANGE),('R1-Distill-7B','R1-Distill-7B',GRAY)]]
clip_stats={}
for i,(ax,data,series) in enumerate(zip(axes,training,specs)):
    for key,label,color in series:
        rows=sorted(data[key],key=lambda x:int(x['step']))
        x=np.array([int(r['step']) for r in rows]);y=np.array([float(r['grad_norm_preclip']) for r in rows])
        assert list(x)==list(range(1,201)) and np.all(y>0)
        ax.plot(x,y,color=color,lw=.55,alpha=.2)
        ax.plot(x,med(y),color=color,lw=1.2,label=label)
        clip_stats[key]={'above_clip_threshold':int((y>1).sum()),'total_updates':len(y)}
    ax.axhline(1,color='#555555',ls='--',lw=.7)
    ax.set_yscale('log');ax.set_xlim(0,200);ax.set_xticks([0,50,100,150,200])
    ax.set_xlabel('OPD step');ax.set_ylabel('Pre-clip gradient norm')
    ax.set_title(['(a) Code / Qwen3-4B','(b) Math / R1-Distill-1.5B'][i],loc='left',pad=49)
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.03),fontsize=7.5,frameon=False,ncol=1,borderaxespad=0,handlelength=1.4,labelspacing=.3)
    axis(ax)
fig.subplots_adjust(left=.105,right=.98,bottom=.18,top=.65,wspace=.30)
save(fig,'gradient_norm_dynamics')

cka=load('parameter_representation_cka.csv','pair')
cka.update(load('opd_rlcode_rerun_cka.csv','pair'))
cka.update(load('opd_qwen17_cka.csv','pair'))
teacher_cka=load('rl_teacher_cka.csv','pair')
ifpath=ROOT/'output/if_param_cka_20260925/if_result.json'
sources.append({'path':str(ifpath.relative_to(ROOT)),'sha256':hashlib.sha256(ifpath.read_bytes()).hexdigest()})
ifvalues=json.loads(ifpath.read_text())['if_ultradata_s200']['per_layer']
cka['if_ultradata']=[{'normalized_depth':i/(len(ifvalues)-1),'cka':v} for i,v in enumerate(ifvalues)]
specs=[
 ('(a) OPD / Qwen3-4B',cka,[('qwen4b_rlcode','Qwen3-4B-RL-Code',BLUE),('qwen4b_code14b','Qwen3-14B',ORANGE)]),
 ('(b) OPD / Qwen3-1.7B',cka,[('qwen17_rlmath','Qwen3-4B-RL-Math',BLUE),('qwen17_nt4','Qwen3-4B',ORANGE)]),
 ('(c) OPD / R1-Distill-1.5B',cka,[('r1_justrl','JustRL-1.5B',BLUE),('r1_skywork','Skywork-7B',ORANGE),('r1_r1_7b','R1-Distill-7B',GRAY),('if_ultradata','UltraData-IF-1.5B (IF)',GREEN)]),
 ('(d) Self-RL teachers vs. initial students',teacher_cka,[('rlcode','Qwen3-4B-RL-Code',BLUE),('justrl','JustRL-1.5B',ORANGE),('ultradata_if','UltraData-IF-1.5B',GREEN)])]
fig=plt.figure(figsize=(5.5,5.0))
axes=np.array([[fig.add_axes([left,bottom,.36,.235]) for left in [.11,.605]] for bottom in [.595,.09]])
cka_stats={}
for i,(ax,(title,data,series)) in enumerate(zip(axes.flat,specs)):
    for key,label,color in series:
        rows=[r for r in data[key] if r.get('dataset','ALL')=='ALL' and r.get('stage','OPD')=='OPD']
        rows=sorted(rows,key=lambda x:float(x['normalized_depth']))
        x=np.array([float(r['normalized_depth']) for r in rows]);y=np.array([float(r['cka']) for r in rows])
        if i==3 and key=='justrl':
            color=BLUE
        ax.plot(x,y,color=color,linestyle='--' if i==3 and key=='rlcode' else '-',marker='o',lw=1,ms=2.3,markevery=4,label=label)
        cka_stats[key]={'n_hidden_states':len(y),'min_cka':float(y.min())}
    ax.set_xlim(0,1);ax.set_xticks([0,.25,.5,.75,1]);ax.set_ylim((.96,1.001) if i==3 else (.98,1.0005))
    fig.text([.11,.605][i%2],[.965,.46][i//2],title,ha='left',va='top',fontsize=8)
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.03),frameon=False,fontsize=7.3,borderaxespad=0,handlelength=1.4,labelspacing=.25)
    ax.set_xlabel('Normalized hidden-state depth');ax.set_ylabel('Linear CKA')
    axis(ax)
save(fig,'layerwise_representation')

(OUT/'verification.json').write_text(json.dumps({'sources':sources,'occupancy':stats,'clipping':clip_stats,'cka':cka_stats},indent=2)+'\n')
print(json.dumps({'occupancy':stats,'clipping':clip_stats,'cka':cka_stats},indent=2))
