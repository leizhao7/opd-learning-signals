#!/usr/bin/env python3
"""Measure full-vocabulary policy distance on fixed student trajectory prefixes."""

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
    parser.add_argument("--prompts-per-source", type=int, default=8)
    parser.add_argument("--positions-per-prompt", type=int, default=64)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument("--logit-chunk", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attention-implementation", default="sdpa")
    return parser.parse_args()


def quantiles(values: list[float]) -> dict[str, float]:
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


def select_records(args: argparse.Namespace) -> list[dict]:
    records = [json.loads(line) for line in args.anchor_inputs.open() if line.strip()]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record.get("hit_max_response_length"):
            continue
        if len(record["prompt_ids"]) >= args.max_sequence_length:
            continue
        grouped[str(record["data_source"])].append(record)
    rng = random.Random(args.seed)
    selected = []
    for source in sorted(grouped):
        candidates = list(grouped[source])
        rng.shuffle(candidates)
        selected.extend(candidates[: args.prompts_per_source])
    return selected


def base_model(model):
    for name in ("model", "transformer"):
        value = getattr(model, name, None)
        if value is not None:
            return value
    raise AttributeError("Could not locate transformer base model")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    selected = select_records(args)
    print(f"selected {len(selected)} prompts", flush=True)

    load_kwargs = dict(
        torch_dtype=dtype,
        attn_implementation=args.attention_implementation,
        trust_remote_code=True,
        device_map={"": device},
    )
    student = AutoModelForCausalLM.from_pretrained(args.student_model, **load_kwargs)
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher_model, **load_kwargs)
    student.eval()
    teacher.eval()
    student.config.use_cache = False
    teacher.config.use_cache = False
    for model in (student, teacher):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    student_base = base_model(student)
    teacher_base = base_model(teacher)

    rng = random.Random(args.seed + 1)
    token_rows: list[dict] = []
    prompt_rows: list[dict] = []
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    for prompt_number, record in enumerate(selected, 1):
        prompt_ids = record["prompt_ids"]
        max_response = args.max_sequence_length - len(prompt_ids)
        output_ids = record["output_ids"][:max_response]
        if not output_ids:
            continue
        count = min(args.positions_per_prompt, len(output_ids))
        positions = sorted(rng.sample(range(len(output_ids)), count))
        prediction_indices = [len(prompt_ids) - 1 + position for position in positions]
        input_ids = torch.tensor([prompt_ids + output_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            student_all = student_base(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
            student_hidden = student_all[:, prediction_indices, :]
            del student_all
            teacher_all = teacher_base(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
            teacher_hidden = teacher_all[:, prediction_indices, :]
            del teacher_all

            current_rows = []
            for chunk_start in range(0, count, args.logit_chunk):
                chunk_stop = min(count, chunk_start + args.logit_chunk)
                student_logits = student.lm_head(
                    student_hidden[:, chunk_start:chunk_stop, :]
                ).float().squeeze(0)
                teacher_logits = teacher.lm_head(
                    teacher_hidden[:, chunk_start:chunk_stop, :]
                ).float().squeeze(0)
                log_student = F.log_softmax(student_logits, dim=-1)
                log_teacher = F.log_softmax(teacher_logits, dim=-1)
                prob_student = log_student.exp()
                prob_teacher = log_teacher.exp()
                reverse_kl = torch.sum(
                    prob_student * (log_student - log_teacher), dim=-1
                )
                forward_kl = torch.sum(
                    prob_teacher * (log_teacher - log_student), dim=-1
                )
                log_mixture = torch.logaddexp(log_student, log_teacher) - math.log(2.0)
                js = 0.5 * (
                    torch.sum(prob_student * (log_student - log_mixture), dim=-1)
                    + torch.sum(prob_teacher * (log_teacher - log_mixture), dim=-1)
                )
                total_variation = 0.5 * torch.sum(
                    torch.abs(prob_student - prob_teacher), dim=-1
                )
                student_top1 = torch.argmax(log_student, dim=-1)
                teacher_top1 = torch.argmax(log_teacher, dim=-1)
                # Positions are generally non-contiguous, so build exact targets.
                response_chunk = torch.tensor(
                    [output_ids[position] for position in positions[chunk_start:chunk_stop]],
                    dtype=torch.long,
                    device=device,
                )
                actual_logratio = (
                    log_student.gather(1, response_chunk[:, None]).squeeze(1)
                    - log_teacher.gather(1, response_chunk[:, None]).squeeze(1)
                )
                teacher_prob_of_student_top1 = prob_teacher.gather(
                    1, student_top1[:, None]
                ).squeeze(1)
                student_prob_of_teacher_top1 = prob_student.gather(
                    1, teacher_top1[:, None]
                ).squeeze(1)

                for offset in range(chunk_stop - chunk_start):
                    position_offset = chunk_start + offset
                    current_rows.append(
                        {
                            "prompt_idx": int(record["prompt_idx"]),
                            "data_source": str(record["data_source"]),
                            "response_position": int(positions[position_offset]),
                            "reverse_kl_student_teacher": float(reverse_kl[offset]),
                            "forward_kl_teacher_student": float(forward_kl[offset]),
                            "jensen_shannon": float(js[offset]),
                            "total_variation": float(total_variation[offset]),
                            "top1_agree": bool(student_top1[offset] == teacher_top1[offset]),
                            "teacher_prob_of_student_top1": float(
                                teacher_prob_of_student_top1[offset]
                            ),
                            "student_prob_of_teacher_top1": float(
                                student_prob_of_teacher_top1[offset]
                            ),
                            "actual_token_logratio_student_teacher": float(
                                actual_logratio[offset]
                            ),
                        }
                    )
                del (
                    student_logits,
                    teacher_logits,
                    log_student,
                    log_teacher,
                    prob_student,
                    prob_teacher,
                    reverse_kl,
                    forward_kl,
                    log_mixture,
                    js,
                    total_variation,
                    student_top1,
                    teacher_top1,
                    response_chunk,
                    actual_logratio,
                    teacher_prob_of_student_top1,
                    student_prob_of_teacher_top1,
                )
        token_rows.extend(current_rows)
        prompt_rows.append(
            {
                "prompt_idx": int(record["prompt_idx"]),
                "data_source": str(record["data_source"]),
                "prompt_tokens": len(prompt_ids),
                "available_response_tokens": len(record["output_ids"]),
                "scored_response_tokens": len(output_ids),
                "sampled_positions": count,
                "mean_reverse_kl": sum(
                    row["reverse_kl_student_teacher"] for row in current_rows
                ) / len(current_rows),
            }
        )
        elapsed = time.time() - started
        print(
            f"[{prompt_number}/{len(selected)}] idx={record['prompt_idx']} "
            f"source={record['data_source']} seq={input_ids.shape[1]} "
            f"positions={count} elapsed={elapsed:.1f}s",
            flush=True,
        )
        del student_hidden, teacher_hidden, input_ids, attention_mask

    metric_names = (
        "reverse_kl_student_teacher",
        "forward_kl_teacher_student",
        "jensen_shannon",
        "total_variation",
        "teacher_prob_of_student_top1",
        "student_prob_of_teacher_top1",
        "actual_token_logratio_student_teacher",
    )
    summary = {
        name: quantiles([row[name] for row in token_rows]) for name in metric_names
    }
    summary["top1_agreement_rate"] = sum(row["top1_agree"] for row in token_rows) / len(token_rows)
    summary["local_fisher_distance_sqrt_2_mean_reverse_kl"] = math.sqrt(
        2.0 * summary["reverse_kl_student_teacher"]["mean"]
    )
    report = {
        "student_model": args.student_model,
        "teacher_model": args.teacher_model,
        "dtype": args.dtype,
        "seed": args.seed,
        "max_sequence_length": args.max_sequence_length,
        "prompts_per_source": args.prompts_per_source,
        "positions_per_prompt": args.positions_per_prompt,
        "num_prompts": len(prompt_rows),
        "num_sampled_prefix_positions": len(token_rows),
        "elapsed_seconds": time.time() - started,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "summary": summary,
        "prompts": prompt_rows,
        "tokens": token_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
