"""verl custom_reward_function：单轮 Multi-IF 指令跟随打分（训练期日志与 val 循环用）。

ground_truth 是 JSON 字符串 {key, prompt, instruction_id_list, kwargs}。
返回 dict：score / acc = prompt 级 strict（0/1），并附带 loose 与 instruction 级比例，
verl 的 process_validation_metrics 会把这些键都聚合成 val-aux 指标。
"""
from __future__ import annotations
import json, sys, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from multiif_scoring import score_turn  # noqa: E402


def reward_func(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    try:
        gt = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
        c = score_turn(list(gt["instruction_id_list"]), list(gt.get("kwargs") or []), solution_str)
        ps = c["prompt_strict_correct"]
        return {
            "score": ps,
            "acc": ps,
            "prompt_strict": ps,
            "prompt_loose": c["prompt_loose_correct"],
            "inst_strict": c["inst_strict_correct"] / max(c["inst_total"], 1.0),
            "inst_loose": c["inst_loose_correct"] / max(c["inst_total"], 1.0),
            "format_score": 1.0 if "</think>" in (solution_str or "") else 0.0,
        }
    except Exception as e:  # 打分器异常不应该把训练拖死，但要留痕
        print(f"[if_reward] error: {e!r}", flush=True)
        traceback.print_exc()
        return {"score": 0.0, "acc": 0.0, "prompt_strict": 0.0, "prompt_loose": 0.0,
                "inst_strict": 0.0, "inst_loose": 0.0, "format_score": 0.0}
