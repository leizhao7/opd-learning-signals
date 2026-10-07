"""压测：python test_prime_isolation.py <new|old> <real|stress> <out.json>"""
import collections, json, os, subprocess, sys, time
mode, variant, out = sys.argv[1], sys.argv[2], sys.argv[3]
O = '/workspace/lei/agent1/migrated_from_30023_20260818/opd_theory/OPD'
V = [l for l in open(sys.argv[4])] if len(sys.argv) > 4 else None
import pandas as pd
key2ds = {}
for f in ['humanevalplus', 'mbppplus', 'livecodebench_v6']:
    df = pd.read_parquet(f'{O}/datasets/bench_val/{f}.parquet', columns=['data_source', 'prompt'])
    for ds, pr in zip(df.data_source, df.prompt):
        key2ds[str(list(pr)[0]['content'])[:300]] = str(ds)
rows = [json.loads(l) for l in V]
cache = {}
comps, refs, tasks, recorded = [], [], [], []
for r in rows:
    inp = r['input']
    if inp not in cache:
        cache[inp] = next((ds for k, ds in key2ds.items() if k in inp), None)
    tasks.append(cache[inp]); comps.append(r['output']); refs.append(r['gts']); recorded.append(float(r['score']))
assert None not in tasks, collections.Counter(tasks)
if variant == 'stress':
    lcb = [i for i, t in enumerate(tasks) if 'livecodebench' in t.lower()]
    for j, i in enumerate(lcb[:400]):
        if j % 4 == 0:  # 派生孙进程后死循环：测进程组整组清理与孤儿回收
            comps[i] = "```python\nimport subprocess\nsubprocess.Popen(['sleep','3000'])\nwhile True:\n    pass\n```"
        else:
            comps[i] = "```python\nwhile True:\n    pass\n```"
def zombies():
    return sum(1 for l in subprocess.run(['ps', '-eo', 'stat'], capture_output=True, text=True).stdout.split('\n') if l.startswith('Z'))
def sleepers():
    return sum(1 for l in subprocess.run(['ps', '-eo', 'args'], capture_output=True, text=True).stdout.split('\n') if l.startswith('sleep 3000'))
from verl.workers.reward_manager.prime import run_reward_scoring
import verl.workers.reward_manager.prime as P
from verl.utils.reward_score import default_compute_score
z0, s0 = zombies(), sleepers(); t0 = time.time()
scores = run_reward_scoring(default_compute_score, comps, refs, tasks, extra_info=None, num_processes=64)
wall = time.time() - t0
time.sleep(5)
z1, s1 = zombies(), sleepers()
by = collections.defaultdict(list)
for t, s in zip(tasks, scores): by[t].append(s)
res = dict(mode=mode, variant=variant, module=P.__file__, n=len(scores), wall_s=round(wall, 1),
           zombies_before=z0, zombies_after=z1, sleep3000_before=s0, sleep3000_after=s1,
           mean_by_source={k: round(sum(v) / len(v), 4) for k, v in by.items()},
           recorded_mean_by_source={k: round(sum(recorded[i] for i in range(len(tasks)) if tasks[i] == k) / len(v), 4) for k, v in by.items()},
           exact_match_vs_recorded=sum(1 for a, b in zip(scores, recorded) if a == b),
           mismatch_idx=[i for i, (a, b) in enumerate(zip(scores, recorded)) if a != b][:50])
json.dump(res, open(out, 'w'), indent=1); print(json.dumps(res))
