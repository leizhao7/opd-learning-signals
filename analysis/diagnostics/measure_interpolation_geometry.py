#!/usr/bin/env python3
"""Measure OPD KL along the straight parameter path from student to teacher."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--anchor-inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--alphas", default="0,0.25,0.5,0.75,0.9,0.95,1")
    parser.add_argument("--prompts-per-source", type=int, default=1000)
    parser.add_argument("--positions-per-prompt", type=int, default=32)
    parser.add_argument("--max-sequence-length", type=int, default=16384)
    parser.add_argument("--logit-chunk", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--attention-implementation", default="sdpa")
    return parser.parse_args()


def base_model(model):
    for name in ("model", "transformer"):
        value = getattr(model, name, None)
        if value is not None:
            return value
    raise AttributeError("Could not locate transformer base model")


def summarize(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "std": float(tensor.std(unbiased=False)),
        "p50": float(torch.quantile(tensor, 0.50)),
        "p90": float(torch.quantile(tensor, 0.90)),
        "p95": float(torch.quantile(tensor, 0.95)),
        "p99": float(torch.quantile(tensor, 0.99)),
        "max": float(tensor.max()),
    }


def prepare_records(args: argparse.Namespace) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for line in args.anchor_inputs.open():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("hit_max_response_length"):
            continue
        if len(record["prompt_ids"]) >= args.max_sequence_length:
            continue
        grouped[str(record["data_source"])].append(record)

    select_rng = random.Random(args.seed)
    position_rng = random.Random(args.seed + 1)
    prepared = []
    for source in sorted(grouped):
        candidates = list(grouped[source])
        select_rng.shuffle(candidates)
        for record in candidates[: args.prompts_per_source]:
            prompt_ids = record["prompt_ids"]
            output_ids = record["output_ids"][
                : args.max_sequence_length - len(prompt_ids)
            ]
            count = min(args.positions_per_prompt, len(output_ids))
            if count == 0:
                continue
            positions = sorted(position_rng.sample(range(len(output_ids)), count))
            prepared.append(
                {
                    "prompt_idx": int(record["prompt_idx"]),
                    "data_source": str(record["data_source"]),
                    "input_ids": prompt_ids + output_ids,
                    "prediction_indices": [
                        len(prompt_ids) - 1 + position for position in positions
                    ],
                }
            )
    return prepared


def aligned_parameters(*models) -> list[tuple[str, tuple[torch.nn.Parameter, ...]]]:
    maps = [dict(model.named_parameters()) for model in models]
    keys = set(maps[0])
    for index, mapping in enumerate(maps[1:], 1):
        if set(mapping) != keys:
            raise RuntimeError(
                f"named parameter mismatch for model {index}: "
                f"missing={sorted(keys - set(mapping))[:5]}, "
                f"extra={sorted(set(mapping) - keys)[:5]}"
            )
    rows = []
    for name in sorted(keys):
        params = tuple(mapping[name] for mapping in maps)
        if any(param.shape != params[0].shape for param in params[1:]):
            raise RuntimeError(f"shape mismatch for {name}")
        rows.append((name, params))
    return rows


def cache_teacher_hidden(teacher, prepared: list[dict]) -> list[torch.Tensor]:
    base = base_model(teacher)
    cached = []
    for index, record in enumerate(prepared, 1):
        input_ids = torch.tensor([record["input_ids"]], dtype=torch.long, device="cuda")
        with torch.inference_mode():
            hidden = base(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                return_dict=True,
            ).last_hidden_state[:, record["prediction_indices"], :]
        cached.append(hidden.cpu())
        if index % 100 == 0 or index == len(prepared):
            print(f"cached teacher hidden {index}/{len(prepared)}", flush=True)
    return cached


def evaluate_alpha(
    work,
    teacher,
    prepared: list[dict],
    teacher_hidden_cache: list[torch.Tensor],
    logit_chunk: int,
) -> dict:
    work_base = base_model(work)
    reverse_kls: list[float] = []
    forward_kls: list[float] = []
    total_variations: list[float] = []
    top1_agreements = 0
    token_count = 0
    source_sums: dict[str, float] = defaultdict(float)
    source_counts: dict[str, int] = defaultdict(int)

    for index, (record, cached_teacher_hidden) in enumerate(
        zip(prepared, teacher_hidden_cache), 1
    ):
        input_ids = torch.tensor([record["input_ids"]], dtype=torch.long, device="cuda")
        with torch.inference_mode():
            work_hidden = work_base(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
                return_dict=True,
            ).last_hidden_state[:, record["prediction_indices"], :]
            teacher_hidden = cached_teacher_hidden.to("cuda")
            positions = work_hidden.shape[1]
            for start in range(0, positions, logit_chunk):
                stop = min(positions, start + logit_chunk)
                work_logits = work.lm_head(work_hidden[:, start:stop, :]).float().squeeze(0)
                teacher_logits = teacher.lm_head(
                    teacher_hidden[:, start:stop, :]
                ).float().squeeze(0)
                log_work = F.log_softmax(work_logits, dim=-1)
                log_teacher = F.log_softmax(teacher_logits, dim=-1)
                prob_work = log_work.exp()
                prob_teacher = log_teacher.exp()
                reverse = torch.sum(prob_work * (log_work - log_teacher), dim=-1)
                forward = torch.sum(prob_teacher * (log_teacher - log_work), dim=-1)
                tv = 0.5 * torch.sum(torch.abs(prob_work - prob_teacher), dim=-1)
                agree = torch.argmax(log_work, dim=-1) == torch.argmax(log_teacher, dim=-1)
                reverse_values = reverse.cpu().tolist()
                reverse_kls.extend(reverse_values)
                forward_kls.extend(forward.cpu().tolist())
                total_variations.extend(tv.cpu().tolist())
                top1_agreements += int(agree.sum())
                token_count += len(reverse_values)
                source_sums[record["data_source"]] += sum(reverse_values)
                source_counts[record["data_source"]] += len(reverse_values)
        if index % 100 == 0 or index == len(prepared):
            print(f"evaluated {index}/{len(prepared)}", flush=True)

    return {
        "num_prefix_positions": token_count,
        "reverse_kl": summarize(reverse_kls),
        "forward_kl": summarize(forward_kls),
        "total_variation": summarize(total_variations),
        "top1_agreement_rate": top1_agreements / token_count,
        "fraction_reverse_kl_gt_0_1": sum(value > 0.1 for value in reverse_kls) / token_count,
        "fraction_reverse_kl_gt_1": sum(value > 1.0 for value in reverse_kls) / token_count,
        "mean_reverse_kl_by_source": {
            source: source_sums[source] / source_counts[source]
            for source in sorted(source_sums)
        },
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    alphas = [float(value) for value in args.alphas.split(",")]
    if alphas != sorted(set(alphas)) or alphas[0] != 0.0 or alphas[-1] != 1.0:
        raise ValueError("alphas must be unique, sorted, and include 0 and 1")
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    prepared = prepare_records(args)
    print(f"prepared {len(prepared)} prompts", flush=True)

    kwargs = dict(
        torch_dtype=dtype,
        attn_implementation=args.attention_implementation,
        trust_remote_code=True,
        device_map={"": "cuda"},
    )
    print("loading base student", flush=True)
    student = AutoModelForCausalLM.from_pretrained(args.student_model, **kwargs).eval()
    print("loading teacher", flush=True)
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher_model, **kwargs).eval()
    print("loading interpolation work model", flush=True)
    work = AutoModelForCausalLM.from_pretrained(args.student_model, **kwargs).eval()
    for model in (student, teacher, work):
        model.config.use_cache = False
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    aligned = aligned_parameters(work, student, teacher)
    delta_squared = 0.0
    for _, (_, student_parameter, teacher_parameter) in aligned:
        delta_squared += float(
            torch.sum((teacher_parameter - student_parameter).float().square())
        )
    print(f"parameter delta_squared={delta_squared:.12g}", flush=True)
    teacher_hidden_cache = cache_teacher_hidden(teacher, prepared)

    started = time.time()
    rows = []
    torch.cuda.reset_peak_memory_stats()
    for alpha in alphas:
        with torch.no_grad():
            for _, (work_parameter, student_parameter, teacher_parameter) in aligned:
                torch.lerp(
                    student_parameter,
                    teacher_parameter,
                    alpha,
                    out=work_parameter,
                )
        print(f"alpha={alpha:g} interpolation complete", flush=True)
        metrics = evaluate_alpha(
            work,
            teacher,
            prepared,
            teacher_hidden_cache,
            args.logit_chunk,
        )
        row = {"alpha": alpha, **metrics}
        rows.append(row)
        print(
            f"alpha={alpha:g} mean_KL={metrics['reverse_kl']['mean']:.9g} "
            f"median={metrics['reverse_kl']['p50']:.9g} "
            f"top1={metrics['top1_agreement_rate']:.6f}",
            flush=True,
        )

    initial_loss = rows[0]["reverse_kl"]["mean"]
    for row in rows:
        alpha = row["alpha"]
        denominator = (1.0 - alpha) ** 2 * initial_loss
        row["quadratic_prediction"] = denominator
        row["loss_over_quadratic_prediction"] = (
            row["reverse_kl"]["mean"] / denominator if denominator > 0 else None
        )

    # Piecewise secants yield a directly observed lower bound on the full
    # gradient PL ratio via Cauchy-Schwarz:
    # ||grad L||^2/(2L) >= (dL/dalpha)^2/(2 L ||Delta||^2).
    secants = []
    for left, right in zip(rows, rows[1:]):
        midpoint = 0.5 * (left["alpha"] + right["alpha"])
        midpoint_loss = 0.5 * (
            left["reverse_kl"]["mean"] + right["reverse_kl"]["mean"]
        )
        slope = (
            right["reverse_kl"]["mean"] - left["reverse_kl"]["mean"]
        ) / (right["alpha"] - left["alpha"])
        secants.append(
            {
                "alpha_midpoint": midpoint,
                "d_loss_d_alpha": slope,
                "directional_pl_lower_bound": (
                    slope * slope / (2.0 * midpoint_loss * delta_squared)
                    if midpoint_loss > 0 and delta_squared > 0
                    else None
                ),
            }
        )

    report = {
        "student_model": args.student_model,
        "teacher_model": args.teacher_model,
        "dtype": args.dtype,
        "alphas": alphas,
        "num_prompts": len(prepared),
        "positions_per_prompt": args.positions_per_prompt,
        "max_sequence_length": args.max_sequence_length,
        "parameter_delta_squared": delta_squared,
        "parameter_delta_l2": math.sqrt(delta_squared),
        "elapsed_seconds_for_alpha_sweep": time.time() - started,
        "peak_gpu_memory_gib_during_alpha_sweep": (
            torch.cuda.max_memory_allocated() / 2**30
        ),
        "rows": rows,
        "secant_directional_pl": secants,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"rows": rows, "secants": secants}, indent=2), flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
