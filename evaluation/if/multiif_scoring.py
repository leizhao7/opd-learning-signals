"""Multi-IF 打分（单轮）：封装官方 facebookresearch/Multi-IF 的 ifeval 检查器。

score_turn() 返回计数而不是比例：prompt 级（整条 prompt 的所有指令都满足才算 1）和
instruction 级（每条指令一个单位）分母不同，按行求平均会把两者混成一个数。
strict / loose 的定义与官方 metrics.py 的 gen_acc_strict / gen_acc_loose 一致。
"""
from __future__ import annotations
import os, sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
# 私有依赖目录（nltk/langdetect/pythainlp/emoji/immutabledict），训练用的 py312 环境里没有
_DEPS = _HERE / "pydeps"
if _DEPS.is_dir() and str(_DEPS) not in sys.path:
    sys.path.append(str(_DEPS))
# nltk punkt_tab 数据随代码目录一起部署
os.environ.setdefault("NLTK_DATA", str(_HERE / "nltk_data"))
import multiif_ifeval as ifeval  # noqa: E402

try:  # langdetect 结果默认不确定，固定种子
    import langdetect
    langdetect.DetectorFactory.seed = 0
except Exception:
    pass


def strip_reasoning(text: str) -> str:
    """R1 系学生把推理写在 <think>...</think> 里，评分只看其后的答案。
    没有 </think>（推理未收尾）时原样返回：该响应通常已被截断，按原文判几乎必失败，
    与论文代码 extract_non_reasoning_content_internal 的处理一致。"""
    if text is None:
        return ""
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text.strip()


def _loose_variants(response: str) -> list[str]:
    r = response.split("\n")
    remove_first = "\n".join(r[1:]).strip()
    remove_last = "\n".join(r[:-1]).strip()
    remove_both = "\n".join(r[1:-1]).strip()
    return [
        response, response.replace("*", ""), remove_first, remove_last, remove_both,
        remove_first.replace("*", ""), remove_last.replace("*", ""), remove_both.replace("*", ""),
    ]


def score_turn(instruction_id_list: list[str], kwargs: list[dict], response: str, strip: bool = True) -> dict:
    """一条响应对一轮的指令列表打分。返回 6 个计数。"""
    if strip:
        response = strip_reasoning(response)
    strict, loose = [], []
    for idx, iid in enumerate(instruction_id_list):
        inst = ifeval.INSTRUCTION_DICT[iid](iid)
        kw = kwargs[idx] if idx < len(kwargs) and kwargs[idx] else {}
        inst.build_description(**kw)
        ok_strict = bool(response) and bool(inst.check_following(response))
        ok_loose = ok_strict or any(v.strip() and inst.check_following(v) for v in _loose_variants(response))
        strict.append(ok_strict); loose.append(ok_loose)
    n = len(instruction_id_list)
    return {
        "prompt_strict_correct": 1.0 if n and all(strict) else 0.0,
        "prompt_loose_correct": 1.0 if n and all(loose) else 0.0,
        "prompt_total": 1.0,
        "inst_strict_correct": float(sum(strict)),
        "inst_loose_correct": float(sum(loose)),
        "inst_total": float(n),
    }
