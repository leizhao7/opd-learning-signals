#!/usr/bin/env python3
"""离线 Multi-IF 评测：3 轮 x 8 语言，一条 prompt 采 1 个样本（论文 2609.04172 的协议）。

    python run_multiif_eval.py multiif --model <hf-dir> --data eval_3turn.parquet --out results/step_50
    python run_multiif_eval.py rescore --out results/step_50        # 只重新打分，不用 GPU

协议（与论文代码 eval/run_if_eval.py 一致）：T=0.6, top_p=0.95, 每轮最多 16384 token, n=1, seed=0；
后续轮次的对话历史只带上去掉 <think> 的答案；8 语言分数是 8 个语言各自分数的无权平均；
一轮的 overall 是 prompt_strict / inst_strict / prompt_loose / inst_loose 四个子指标的均值。
avg@k 需要用不同 --seed 跑 k 次再对 report 取平均（多轮对话无法在单次里分叉采样）。
"""
from __future__ import annotations
import argparse, json, multiprocessing, os, statistics, sys, traceback
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from multiif_scoring import score_turn, strip_reasoning  # noqa: E402

TURNS = (1, 2, 3)
SUB = ("prompt_strict", "inst_strict", "prompt_loose", "inst_loose")
_PROMPT_RESERVE = 64


def _rates(c):
    p, i = c["prompt_total"], c["inst_total"]
    out = {"prompt_strict": 100.0 * c["prompt_strict_correct"] / p if p else 0.0,
           "prompt_loose": 100.0 * c["prompt_loose_correct"] / p if p else 0.0,
           "inst_strict": 100.0 * c["inst_strict_correct"] / i if i else 0.0,
           "inst_loose": 100.0 * c["inst_loose_correct"] / i if i else 0.0}
    out["overall"] = statistics.fmean(out[m] for m in SUB); out["n"] = int(p)
    return out


def score_multiif(records):
    per_lang = {}
    for rec in records:
        for t, turn in zip(TURNS, rec["turns"]):
            c = score_turn(turn["instruction_id_list"], turn["kwargs"], turn.get("response", ""))
            b = per_lang.setdefault(rec["language"], {}).setdefault(t, {})
            for k, v in c.items(): b[k] = b.get(k, 0) + v
    langs = sorted(per_lang)
    by_lang = {l: {f"turn_{t}": _rates(per_lang[l][t]) for t in TURNS} for l in langs}
    lang_avg = {f"turn_{t}": {m: statistics.fmean(by_lang[l][f"turn_{t}"][m] for l in langs) for m in (*SUB, "overall")} for t in TURNS}
    en = by_lang.get("English")
    capped = {f"turn_{t}": sum(1 for r in records if r["turns"][t - 1].get("finish_reason") == "length") for t in TURNS}
    return {"benchmark": "Multi-IF", "n_conversations": len(records), "languages": langs,
            "headline": {"8lang_turn3": lang_avg["turn_3"]["overall"],
                         "8lang_3turn_mean": statistics.fmean(lang_avg[f"turn_{t}"]["overall"] for t in TURNS),
                         "english_turn3": en["turn_3"]["overall"] if en else None,
                         "english_3turn_mean": statistics.fmean(en[f"turn_{t}"]["overall"] for t in TURNS) if en else None},
            "lang_avg": lang_avg, "by_language": by_lang, "hit_token_cap": capped}


def load_records(path, limit):
    df = pd.read_parquet(path)
    if limit: df = df.head(limit)
    return [{"key": r["key"], "language": r["language"],
             "turns": [{"prompt": r[f"turn_{t}_prompt"], "instruction_id_list": json.loads(r[f"turn_{t}_instruction_id_list"]),
                        "kwargs": json.loads(r[f"turn_{t}_kwargs"])} for t in TURNS]} for _, r in df.iterrows()]


def _prompt_len(tok, history):
    text = tok.apply_chat_template(history, add_generation_prompt=True, tokenize=False)
    return len(tok(text, add_special_tokens=False)["input_ids"])


def _generate_here(records, args):
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization, enforce_eager=True, seed=args.seed)
    params = SamplingParams(n=1, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_out_len, seed=args.seed)
    tok = llm.get_tokenizer()
    hist = [[] for _ in records]
    for t in range(3):
        for h, rec in zip(hist, records): h.append({"role": "user", "content": rec["turns"][t]["prompt"]})
        lens = [_prompt_len(tok, h) for h in hist]
        fits = [i for i, n in enumerate(lens) if n + _PROMPT_RESERVE < args.max_model_len]
        overflow = [i for i in range(len(hist)) if i not in set(fits)]
        for i in overflow:  # 上下文塞不下：按空回答记零分，不截断历史（那会变成另一段对话）
            records[i]["turns"][t]["response"] = ""; records[i]["turns"][t]["finish_reason"] = "prompt_too_long"
            hist[i].append({"role": "assistant", "content": ""})
        outs = llm.chat([hist[i] for i in fits], params, add_generation_prompt=True, use_tqdm=False) if fits else []
        for i, o in zip(fits, outs):
            raw = o.outputs[0].text
            records[i]["turns"][t]["response"] = raw; records[i]["turns"][t]["finish_reason"] = o.outputs[0].finish_reason
            hist[i].append({"role": "assistant", "content": strip_reasoning(raw)})
        capped = sum(1 for r in records if r["turns"][t].get("finish_reason") == "length")
        print(f"[turn {t+1}] generated={len(fits)} capped={capped} overlong={len(overflow)}", flush=True)


def _shard_worker(args, records, shard_path):
    code = 0
    try:
        _generate_here(records, args)
        with open(shard_path, "w") as fh:
            for rec in records: fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except BaseException:
        traceback.print_exc(); code = 1
    sys.stdout.flush(); sys.stderr.flush()
    for ch in multiprocessing.active_children(): ch.terminate()
    os._exit(code)


def _wait_for_free_gpus(min_free_gib=100, timeout_s=1800):
    """上一轮评测的 EngineCore 退出后显存不会立刻归零；开新引擎前等每张可见卡都空出来，否则 vLLM 启动直接报错。"""
    import subprocess, time
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    want = {d.strip() for d in env.split(",")} if env else None
    t0 = time.time()
    while True:
        rows = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total,memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.strip().splitlines()
        busy = []
        for r in rows:
            idx, tot, used = [x.strip() for x in r.split(",")]
            if want is not None and idx not in want: continue
            if (int(tot) - int(used)) / 1024 < min_free_gib: busy.append((idx, used))
        if not busy: return
        if time.time() - t0 > timeout_s:
            print(f"[gpu-wait] still busy after {timeout_s}s: {busy}, proceeding anyway", flush=True); return
        print(f"[gpu-wait] waiting for GPUs to free up: {busy}", flush=True); time.sleep(30)


def generate(records, args):
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    _wait_for_free_gpus()
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env: devices = [d for d in env.split(",") if d.strip()]
    else:
        import torch; devices = [str(i) for i in range(torch.cuda.device_count())]
    n_shards = max(1, len(devices) // args.tp)
    if n_shards == 1:
        _generate_here(records, args); return
    ctx = multiprocessing.get_context("spawn"); tmp = Path(args.out) / "shards"; tmp.mkdir(parents=True, exist_ok=True)
    procs, paths, slices = [], [], []
    for i in range(n_shards):
        part = records[i::n_shards]
        if not part: continue
        path = tmp / f"shard{i}.jsonl"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices[i * args.tp:(i + 1) * args.tp])
        p = ctx.Process(target=_shard_worker, args=(args, part, str(path))); p.start()
        procs.append(p); paths.append(path); slices.append(part)
    if env is None: os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else: os.environ["CUDA_VISIBLE_DEVICES"] = env
    failed = []
    for i, (p, path) in enumerate(zip(procs, paths)):
        p.join()
        if p.exitcode != 0 or not path.exists(): failed.append(f"shard {i} (exit {p.exitcode})")
    if failed: raise SystemExit("generation failed in " + ", ".join(failed))
    for part, path in zip(slices, paths):
        for rec, line in zip(part, path.open()):
            done = json.loads(line)
            for turn, dt in zip(rec["turns"], done["turns"]):
                turn["response"] = dt["response"]; turn["finish_reason"] = dt["finish_reason"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("benchmark", choices=["multiif", "rescore"])
    p.add_argument("--model"); p.add_argument("--data", type=Path); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0); p.add_argument("--tp", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85); p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--max-out-len", type=int, default=16384); p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95); p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.benchmark == "rescore":
        records = [json.loads(l) for l in (args.out / "generations.jsonl").open()]
    else:
        assert args.model and args.data, "--model 与 --data 必填"
        records = load_records(args.data, args.limit or None)
        try:
            generate(records, args)
        except SystemExit as e:
            print(f"[retry] generation failed once ({e}); waiting 120s and retrying", flush=True)
            import time; time.sleep(120)
            records = load_records(args.data, args.limit or None); generate(records, args)
    report = score_multiif(records); report["model"] = args.model
    report["protocol"] = {"temperature": args.temperature, "top_p": args.top_p, "max_out_len": args.max_out_len,
                          "max_model_len": args.max_model_len, "seed": args.seed, "n_per_prompt": 1, "history": "reasoning_stripped"}
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "generations.jsonl").open("w") as fh:
        for rec in records: fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (args.out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    h = report["headline"]
    print(f"\nMulti-IF n={report['n_conversations']}  8lang turn3={h['8lang_turn3']:.2f}  8lang 3turn mean={h['8lang_3turn_mean']:.2f}  English turn3={h['english_turn3']}")
    for t in TURNS:
        row = report["lang_avg"][f"turn_{t}"]; print(f"  turn{t}: " + "  ".join(f"{m}={row[m]:.2f}" for m in (*SUB, "overall")))
    print(f"wrote {args.out}/report.json"); return 0


if __name__ == "__main__":
    sys.exit(main())
