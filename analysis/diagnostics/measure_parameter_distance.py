#!/usr/bin/env python3
"""Streaming Euclidean diagnostics between two safetensors checkpoints.

The script never instantiates either model. It opens one tensor at a time,
accumulates in float64, and therefore also works when GPUs are occupied.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def weight_map(model_dir: Path) -> dict[str, Path]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        raw = json.loads(index_path.read_text())
        return {name: model_dir / shard for name, shard in raw["weight_map"].items()}
    single = model_dir / "model.safetensors"
    if not single.exists():
        raise FileNotFoundError(f"No safetensors weights under {model_dir}")
    with safe_open(single, framework="pt", device="cpu") as handle:
        return {name: single for name in handle.keys()}


def group_name(name: str) -> str:
    match = re.match(r"model\.layers\.(\d+)\.", name)
    if match:
        return f"layer_{int(match.group(1)):02d}"
    if name.startswith("model.embed_tokens."):
        return "embed_tokens"
    if name.startswith("model.norm."):
        return "final_norm"
    if name.startswith("lm_head."):
        return "lm_head"
    return name.split(".", 1)[0]


def empty_stats() -> dict[str, float | int]:
    return {
        "numel": 0,
        "student_sq": 0.0,
        "teacher_sq": 0.0,
        "delta_sq": 0.0,
        "dot": 0.0,
        "delta_l1": 0.0,
        "max_abs_delta": 0.0,
        "changed_numel": 0,
        "tensor_count": 0,
        "identical_tensor_count": 0,
    }


def add_stats(dst: dict, student: torch.Tensor, teacher: torch.Tensor) -> None:
    student = student.float()
    teacher = teacher.float()
    delta = teacher - student
    dst["numel"] += student.numel()
    dst["student_sq"] += torch.sum(student.double() * student.double()).item()
    dst["teacher_sq"] += torch.sum(teacher.double() * teacher.double()).item()
    dst["delta_sq"] += torch.sum(delta.double() * delta.double()).item()
    dst["dot"] += torch.sum(student.double() * teacher.double()).item()
    dst["delta_l1"] += torch.sum(torch.abs(delta).double()).item()
    dst["max_abs_delta"] = max(dst["max_abs_delta"], torch.max(torch.abs(delta)).item())
    changed = torch.count_nonzero(delta).item()
    dst["changed_numel"] += changed
    dst["tensor_count"] += 1
    dst["identical_tensor_count"] += int(changed == 0)


def merge_stats(dst: dict, src: dict) -> None:
    for key in (
        "numel",
        "student_sq",
        "teacher_sq",
        "delta_sq",
        "dot",
        "delta_l1",
        "changed_numel",
        "tensor_count",
        "identical_tensor_count",
    ):
        dst[key] += src[key]
    dst["max_abs_delta"] = max(dst["max_abs_delta"], src["max_abs_delta"])


def finalize(raw: dict) -> dict:
    n = raw["numel"]
    student_norm = math.sqrt(raw["student_sq"])
    teacher_norm = math.sqrt(raw["teacher_sq"])
    delta_norm = math.sqrt(raw["delta_sq"])
    denom = student_norm * teacher_norm
    result = dict(raw)
    result.update(
        {
            "student_l2": student_norm,
            "teacher_l2": teacher_norm,
            "delta_l2": delta_norm,
            "delta_over_student": delta_norm / student_norm if student_norm else float("nan"),
            "delta_over_teacher": delta_norm / teacher_norm if teacher_norm else float("nan"),
            "symmetric_relative_delta": (
                2.0 * delta_norm / (student_norm + teacher_norm)
                if student_norm + teacher_norm
                else float("nan")
            ),
            "cosine": raw["dot"] / denom if denom else float("nan"),
            "student_rms": student_norm / math.sqrt(n) if n else float("nan"),
            "teacher_rms": teacher_norm / math.sqrt(n) if n else float("nan"),
            "delta_rms": delta_norm / math.sqrt(n) if n else float("nan"),
            "mean_abs_delta": raw["delta_l1"] / n if n else float("nan"),
            "changed_fraction": raw["changed_numel"] / n if n else float("nan"),
        }
    )
    return result


def main() -> None:
    args = parse_args()
    started = time.time()
    student_map = weight_map(args.student)
    teacher_map = weight_map(args.teacher)
    student_keys = set(student_map)
    teacher_keys = set(teacher_map)
    ignored_redundant_teacher_tensors = []
    # Some tied-weight exports redundantly serialize lm_head.weight while the
    # base export stores only model.embed_tokens.weight. The two teacher
    # tensors must be exactly identical before the redundant copy is ignored.
    if teacher_keys - student_keys == {"lm_head.weight"} and {
        "model.embed_tokens.weight",
        "lm_head.weight",
    }.issubset(teacher_keys):
        embed_path = teacher_map["model.embed_tokens.weight"]
        head_path = teacher_map["lm_head.weight"]
        with safe_open(embed_path, framework="pt", device=args.device) as handle:
            teacher_embed = handle.get_tensor("model.embed_tokens.weight")
        with safe_open(head_path, framework="pt", device=args.device) as handle:
            teacher_head = handle.get_tensor("lm_head.weight")
        if torch.equal(teacher_embed, teacher_head):
            teacher_keys.remove("lm_head.weight")
            ignored_redundant_teacher_tensors.append("lm_head.weight")
    if student_keys != teacher_keys:
        report = {
            "only_student": sorted(student_keys - teacher_keys),
            "only_teacher": sorted(teacher_keys - student_keys),
        }
        raise RuntimeError("Checkpoint tensor names differ:\n" + json.dumps(report, indent=2))

    all_raw = empty_stats()
    grouped_raw: dict[str, dict] = defaultdict(empty_stats)
    tensor_rows = []
    open_student_path = None
    open_teacher_path = None
    student_handle = None
    teacher_handle = None

    try:
        for index, name in enumerate(sorted(student_keys), 1):
            student_path = student_map[name]
            teacher_path = teacher_map[name]
            if student_path != open_student_path:
                if student_handle is not None:
                    student_handle.__exit__(None, None, None)
                student_handle = safe_open(student_path, framework="pt", device=args.device)
                student_handle.__enter__()
                open_student_path = student_path
            if teacher_path != open_teacher_path:
                if teacher_handle is not None:
                    teacher_handle.__exit__(None, None, None)
                teacher_handle = safe_open(teacher_path, framework="pt", device=args.device)
                teacher_handle.__enter__()
                open_teacher_path = teacher_path

            student = student_handle.get_tensor(name)
            teacher = teacher_handle.get_tensor(name)
            if student.shape != teacher.shape:
                raise RuntimeError(f"Shape mismatch for {name}: {student.shape} vs {teacher.shape}")
            tensor_raw = empty_stats()
            add_stats(tensor_raw, student, teacher)
            tensor_rows.append({"name": name, "shape": list(student.shape), **finalize(tensor_raw)})
            merge_stats(all_raw, tensor_raw)
            merge_stats(grouped_raw[group_name(name)], tensor_raw)
            print(f"[{index:03d}/{len(student_keys):03d}] {name}", flush=True)
    finally:
        if student_handle is not None:
            student_handle.__exit__(None, None, None)
        if teacher_handle is not None:
            teacher_handle.__exit__(None, None, None)

    report = {
        "student": str(args.student),
        "teacher": str(args.teacher),
        "ignored_redundant_teacher_tensors": ignored_redundant_teacher_tensors,
        "elapsed_seconds": time.time() - started,
        "global": finalize(all_raw),
        "groups": {name: finalize(raw) for name, raw in sorted(grouped_raw.items())},
        "tensors": tensor_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["global"], indent=2), flush=True)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
