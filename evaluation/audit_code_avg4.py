import json,hashlib,gzip,base64,collections,time
from pathlib import Path
import pyarrow.parquet as pq
import argparse
parser=argparse.ArgumentParser(description="Audit code validation and reproduce the paper's fixed four-of-eight selection")
parser.add_argument('--validation-dir', type=Path, required=True)
parser.add_argument('--benchmark-dir', type=Path, required=True)
parser.add_argument('--output-dir', type=Path, required=True)
args=parser.parse_args()
args.output_dir.mkdir(parents=True,exist_ok=True)
benchmarks=['humanevalplus','mbppplus','livecodebench']
gtmap=collections.defaultdict(list);datasets=[]
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
for bench,name in zip(benchmarks,['humanevalplus','mbppplus','livecodebench_v6']):
 p=args.benchmark_dir/f'{name}.parquet';table=pq.read_table(p);records=table.to_pylist()
 for i,r in enumerate(records):
  gt=r['reward_model']['ground_truth'];assert isinstance(gt,str)
  gtmap[sha(gt)].append({'benchmark':bench,'dataset_index':i,'prompt':r['prompt'],'ground_truth_sha256':sha(gt)})
 datasets.append({'benchmark':bench,'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'problem_count':len(records),'columns':table.column_names})
print(json.dumps({'type':'datasets','datasets':datasets}),flush=True)
identities=None
for step in range(0,201,10):
 start=time.time();p=args.validation_dir/f'{step}.jsonl';h=hashlib.sha256();groups={};n=0
 with p.open('rb') as f:
  for raw in f:
   h.update(raw);r=json.loads(raw);assert r['step']==step,(step,n,r['step'])
   candidates=gtmap[sha(r['gts'])]
   assert candidates,('unmatched ground truth',step,n)
   if len(candidates)>1:
    candidates=[c for c in candidates if all(msg['content'] in r['input'] for msg in c['prompt'] if msg['role']=='user')]
   assert len(candidates)==1,('ambiguous problem',step,n,len(candidates))
   c=candidates[0];b=c['benchmark'];idx=c['dataset_index'];prompt_hash=sha(r['input']);key=(b,idx)
   if key not in groups:
    assert all(msg['content'] in r['input'] for msg in c['prompt'] if msg['role']=='user'),('prompt mismatch',step,n,key)
    groups[key]={'benchmark':b,'dataset_index':idx,'prompt_sha256':prompt_hash,'ground_truth_sha256':c['ground_truth_sha256'],'rewards':[],'line_indices':[]}
   g=groups[key];assert g['prompt_sha256']==prompt_hash;assert r['score']==r['reward'];assert r['reward'] in [0,1]
   g['rewards'].append(r['reward']);g['line_indices'].append(n);n+=1
 assert all(len(g['rewards'])==8 for g in groups.values()),('sample counts',step,collections.Counter(len(g['rewards']) for g in groups.values()))
 counts=collections.Counter(g['benchmark'] for g in groups.values());assert counts=={d['benchmark']:d['problem_count'] for d in datasets},('problem counts',counts,datasets)
 current={k:g['prompt_sha256'] for k,g in groups.items()}
 if identities is None:identities=current
 else:assert current==identities,('problem identity changed',step)
 selected=[];av4={b:[] for b in benchmarks};av8={b:[] for b in benchmarks}
 for key,g in sorted(groups.items()):
  ranks=[(sha('seed=0\0'+g['benchmark']+'\0'+g['prompt_sha256']+'\0'+str(i)),i) for i in range(8)]
  inds=sorted(i for _,i in sorted(ranks)[:4]);rewards=[g['rewards'][i] for i in inds]
  av4[g['benchmark']].append(sum(rewards)/4);av8[g['benchmark']].append(sum(g['rewards'])/8)
  selected.append({**g,'step':step,'selected_response_indices_zero_based':inds,'selected_rewards':rewards})
 means4={b:sum(v)/len(v) for b,v in av4.items()};means8={b:sum(v)/len(v) for b,v in av8.items()}
 out={'type':'checkpoint','step':step,'source_path':str(p),'source_sha256':h.hexdigest(),'source_bytes':p.stat().st_size,'line_count':n,'problem_counts':dict(counts),'avg4':means4,'native_avg8':means8,'macro_avg4_percent':sum(means4.values())/3*100,'macro_avg8_percent':sum(means8.values())/3*100,'sample_audit_gzip_base64':base64.b64encode(gzip.compress(('\n'.join(json.dumps(v,separators=(',',':')) for v in selected)+'\n').encode())).decode(),'elapsed_seconds':time.time()-start}
 print(json.dumps(out),flush=True)
print(json.dumps({'type':'complete','method':'For each benchmark and prompt SHA256, rank response ordinals 0–7 by SHA256 of UTF-8 seed=0 + NUL + benchmark + NUL + prompt_sha256 + NUL + ordinal; select lowest four hashes, then retain ordinal order. Same problem subset indices at every checkpoint; reward values never enter selection.','seed':0,'selected_n':4,'original_n':8,'response_ordinal':'occurrence order for each benchmark/dataset problem in the original JSONL'}),flush=True)
