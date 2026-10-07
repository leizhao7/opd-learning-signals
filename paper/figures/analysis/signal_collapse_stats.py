"""Numbers quoted in Section 3.2 / Table 3 for the learning signal (no plotting dependencies).

Per run, with a task-level split step s (Code: 15, Math: 30):
  before_signal    : median of nu_train over steps 1..s
  post_signal      : median of nu_train over the 30 steps after s (s+1 .. s+30)
  drop             : before_signal / post_signal
  remaining_at_s   : centered 9-step rolling median of remaining_loss_fraction at step s, in percent
  last50_signal    : median of nu_train over the run's last 50 steps
Per task: gap of the successful run's signal over each unsuccessful run, before s (medians over steps 1..s)
  and after s (medians over s+1 .. s+30).
Output: figures/data/signal_collapse_late_level.json
"""
import csv, json, pathlib, statistics as st
from collections import defaultdict
ROOT = pathlib.Path(__file__).resolve().parents[2]
W = 9
SPLIT = {"Code": 15, "Math": 30}
SUCCESS = {"Code": "RL-Code", "Math": "JustRL"}
def rmed(y):
    h = W // 2
    return [st.median(y[max(0, i - h): i + h + 1]) for i in range(len(y))]
out = {"definitions": __doc__.strip(), "split_step": SPLIT, "runs": {}, "gaps": {}}
for task, f in (("Code", "code_signal_collapse_training.csv"), ("Math", "math_signal_collapse_training.csv")):
    d = defaultdict(list)
    for r in csv.DictReader(open(ROOT / "figures/data" / f)):
        d[r["teacher"]].append((int(r["step"]), float(r["nu_train"]), float(r["remaining_loss_fraction"])))
    s = SPLIT[task]; series = {}
    for k, v in d.items():
        v.sort(); nu = [x for _, x, _ in v]; rem = rmed([x for _, _, x in v]); series[k] = nu
        out["runs"][f"{task}/{k}"] = {"split_step": s, "before_signal": st.median(nu[:s]), "post_signal": st.median(nu[s:s + 30]),
                                      "drop": st.median(nu[:s]) / st.median(nu[s:s + 30]), "remaining_at_s_pct": 100 * rem[s - 1],
                                      "last50_signal": st.median(nu[-50:]), "steps": [v[0][0], v[-1][0]]}
    su = series[SUCCESS[task]]
    for k, nu in series.items():
        if k == SUCCESS[task]: continue
        out["gaps"][f"{task}/{SUCCESS[task]} vs {k}"] = {"before_s": st.median(su[:s]) / st.median(nu[:s]),
                                                         "after_s": st.median(su[s:s + 30]) / st.median(nu[s:s + 30])}
p = ROOT / "figures/data/signal_collapse_late_level.json"
p.write_text(json.dumps(out, indent=1)); print("wrote", p)
for k, r in out["runs"].items():
    print(f"{k:22} s={r['split_step']:3d} before={r['before_signal']:7.2f} after={r['post_signal']:6.2f} drop={r['drop']:5.1f}x rem@s={r['remaining_at_s_pct']:5.1f}% last50={r['last50_signal']:.3f}")
for k, g in out["gaps"].items(): print(f"{k:34} before={g['before_s']:4.1f}x after={g['after_s']:5.1f}x")
