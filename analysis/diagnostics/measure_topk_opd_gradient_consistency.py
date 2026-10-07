#!/usr/bin/env python3
"""Measure per-trajectory consistency of the practical thunlp/OPD gradient.

This reproduces the code-training recipe:

* top_k_strategy=only_stu
* reward_weight_mode=student_p
* log_prob_top_k=16
* token_reward_direct with token-mean reduction

For frozen rollout log-probabilities S and teacher log-probabilities T on the
student top-k set, the code constructs

    advantage_i = -(S_i - T_i) * softmax(S)_i

and the on-policy PPO gradient is reproduced by the stop-gradient surrogate

    -sum_i advantage_i.detach() * log pi_theta(i).

Only a configurable suffix of student transformer blocks (and optionally the
final norm) is differentiated. This makes the result an explicit parameter
subspace diagnostic rather than an action-support statistic.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM


def load_jsonl(paths: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    for path_string in paths:
        with Path(path_string).open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                source = json.loads(line)
                prompt_idx = int(source["prompt_idx"])
                if prompt_idx in seen:
                    raise ValueError(f"duplicate prompt_idx={prompt_idx}")
                seen.add(prompt_idx)
                records.append(
                    {
                        "prompt_idx": prompt_idx,
                        "data_source": str(source.get("data_source", "")),
                        "prompt_ids": [
                            int(value) for value in source["prompt_ids"]
                        ],
                        "output_ids": [
                            int(value) for value in source["output_ids"]
                        ],
                        "hit_max_response_length": bool(
                            source.get("hit_max_response_length", False)
                        ),
                    }
                )
    return records


def load_metadata(paths: list[str]) -> dict[int, dict[str, Any]]:
    metadata: dict[int, dict[str, Any]] = {}
    for path_string in paths:
        with Path(path_string).open() as stream:
            for line in stream:
                if line.strip():
                    record = json.loads(line)
                    metadata[int(record["prompt_idx"])] = {
                        "student_correct": int(
                            bool(record.get("student_correct", False))
                        )
                    }
    return metadata


def balanced_assignments(
    records: list[dict[str, Any]], world: int
) -> list[list[dict[str, Any]]]:
    assignments: list[list[dict[str, Any]]] = [
        [] for _ in range(world)
    ]
    loads = [0 for _ in range(world)]
    ordered = sorted(
        records,
        key=lambda record: (
            -len(record["prompt_ids"]) - len(record["output_ids"]),
            int(record["prompt_idx"]),
        ),
    )
    for record in ordered:
        rank = min(range(world), key=lambda item: (loads[item], item))
        assignments[rank].append(record)
        loads[rank] += len(record["prompt_ids"]) + len(
            record["output_ids"]
        )
    for shard in assignments:
        shard.sort(
            key=lambda record: (
                len(record["prompt_ids"]) + len(record["output_ids"]),
                int(record["prompt_idx"]),
            )
        )
    return assignments


def get_transformer_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    candidates = [
        getattr(getattr(model, "model", None), "layers", None),
        getattr(
            getattr(getattr(model, "model", None), "model", None),
            "layers",
            None,
        ),
    ]
    for candidate in candidates:
        if isinstance(candidate, torch.nn.ModuleList):
            return candidate
    raise TypeError("could not locate transformer layers")


def get_final_norm(model: torch.nn.Module) -> torch.nn.Module | None:
    candidates = [
        getattr(getattr(model, "model", None), "norm", None),
        getattr(
            getattr(getattr(model, "model", None), "model", None),
            "norm",
            None,
        ),
    ]
    for candidate in candidates:
        if isinstance(candidate, torch.nn.Module):
            return candidate
    return None


def get_base_model(model: torch.nn.Module) -> torch.nn.Module:
    candidate = getattr(model, "model", None)
    if isinstance(candidate, torch.nn.Module):
        return candidate
    raise TypeError("could not locate causal LM base model")


def select_student_parameters(
    model: torch.nn.Module,
    last_n_layers: int,
    include_final_norm: bool,
) -> list[tuple[str, torch.nn.Parameter]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    layers = get_transformer_layers(model)
    if not 1 <= last_n_layers <= len(layers):
        raise ValueError(
            f"last_n_layers must be in [1, {len(layers)}], "
            f"got {last_n_layers}"
        )
    for layer in layers[-last_n_layers:]:
        for parameter in layer.parameters():
            parameter.requires_grad_(True)
    if include_final_norm:
        final_norm = get_final_norm(model)
        if final_norm is None:
            raise TypeError("include_final_norm requested but norm not found")
        for parameter in final_norm.parameters():
            parameter.requires_grad_(True)
    selected = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not selected:
        raise ValueError("no student parameters selected")
    return selected


def squared_norm(tensors: tuple[torch.Tensor, ...]) -> torch.Tensor:
    total = torch.zeros((), dtype=torch.float64, device=tensors[0].device)
    for tensor in tensors:
        total += tensor.detach().double().square().sum()
    return total


def vector_squared_norm(tensors: list[torch.Tensor]) -> float:
    total = 0.0
    for tensor in tensors:
        total += float(tensor.double().square().sum().item())
    return total


def vector_dot(
    left: list[torch.Tensor], right: list[torch.Tensor]
) -> float:
    return sum(
        float(
            x.double().flatten().dot(y.double().flatten()).item()
        )
        for x, y in zip(left, right)
    )


def raw_gradient_moments(
    raw_sum_sqnorm: float,
    sum_grad_sqnorm: float,
    count: int,
    mean_loss: float | None = None,
) -> dict[str, float | None]:
    """Return raw (not per-example normalized) gradient moments.

    The U-statistic removes the positive 1/N diagonal contribution from
    ||mean_i g_i||^2.  It can be negative at finite N when the shared mean
    direction is below the sampling noise floor; callers should report that
    value rather than silently clipping it.
    """
    mean_example_sqnorm = (
        sum_grad_sqnorm / count if count else 0.0
    )
    unbiased_mean_gradient_sqnorm = (
        (raw_sum_sqnorm - sum_grad_sqnorm) / (count * (count - 1))
        if count > 1
        else 0.0
    )
    raw_consistency_unbiased = (
        unbiased_mean_gradient_sqnorm / mean_example_sqnorm
        if mean_example_sqnorm
        else 0.0
    )
    return {
        "mean_example_gradient_sqnorm_M": mean_example_sqnorm,
        "unbiased_mean_gradient_sqnorm": (
            unbiased_mean_gradient_sqnorm
        ),
        "raw_consistency_C_unbiased": raw_consistency_unbiased,
        "biased_mean_gradient_sqnorm": (
            raw_sum_sqnorm / (count * count) if count else 0.0
        ),
        "M_over_mean_loss": (
            mean_example_sqnorm / mean_loss
            if mean_loss is not None and mean_loss > 0.0
            else None
        ),
        "unbiased_mean_gradient_sqnorm_over_mean_loss": (
            unbiased_mean_gradient_sqnorm / mean_loss
            if mean_loss is not None and mean_loss > 0.0
            else None
        ),
    }


def parameter_group(name: str) -> str:
    if ".self_attn." in name:
        suffix = name.split(".self_attn.", 1)[1]
        return "self_attn." + suffix.rsplit(".", 1)[0]
    if ".mlp." in name:
        suffix = name.split(".mlp.", 1)[1]
        return "mlp." + suffix.rsplit(".", 1)[0]
    if "layernorm" in name.lower():
        return name.rsplit(".", 1)[0].split(".")[-1]
    if name.endswith(".norm.weight") or name.endswith(".norm.bias"):
        return "final_norm"
    return name.rsplit(".", 1)[0]


def summary_from_accumulators(
    names: list[str],
    raw_sum: list[torch.Tensor],
    unit_sum: list[torch.Tensor],
    records: list[dict[str, Any]],
    sum_grad_norm: float,
    sum_grad_sqnorm: float,
    sum_kl_token_weighted: float,
    sum_tokens: int,
    proper_raw_sum: list[torch.Tensor] | None = None,
    proper_sum_grad_norm: float = 0.0,
    proper_sum_grad_sqnorm: float = 0.0,
    sum_proper_kl_token_weighted: float = 0.0,
    sum_proper_kl_trajectory_mean: float = 0.0,
) -> dict[str, Any]:
    count = len(records)
    raw_sum_sqnorm = vector_squared_norm(raw_sum)
    unit_sum_sqnorm = vector_squared_norm(unit_sum)
    mean_pairwise_cosine = (
        (unit_sum_sqnorm - count) / (count * (count - 1))
        if count > 1
        else 0.0
    )
    groups: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        groups.setdefault(parameter_group(name), []).append(index)
    group_raw_sum_sqnorm = {
        group: sum(
            float(raw_sum[index].double().square().sum().item())
            for index in indices
        )
        for group, indices in groups.items()
    }
    summary: dict[str, Any] = {
        "n_examples": count,
        "n_tokens": int(sum_tokens),
        "mean_topk_logratio_per_token": (
            sum_kl_token_weighted / sum_tokens if sum_tokens else 0.0
        ),
        "mean_gradient_norm": (
            sum_grad_norm / count if count else 0.0
        ),
        "raw_sum_gradient_norm": math.sqrt(raw_sum_sqnorm),
        "raw_coherence": (
            raw_sum_sqnorm / (count * sum_grad_sqnorm)
            if count and sum_grad_sqnorm
            else 0.0
        ),
        "cancellation_ratio": (
            math.sqrt(raw_sum_sqnorm) / sum_grad_norm
            if sum_grad_norm
            else 0.0
        ),
        "normalized_mean_pairwise_cosine": mean_pairwise_cosine,
        "unit_sum_gradient_norm": math.sqrt(unit_sum_sqnorm),
        "group_raw_sum_gradient_norms": {
            group: math.sqrt(value)
            for group, value in group_raw_sum_sqnorm.items()
        },
    }
    summary["training_surrogate_raw_moments"] = raw_gradient_moments(
        raw_sum_sqnorm,
        sum_grad_sqnorm,
        count,
        None,
    )
    if proper_raw_sum is not None:
        proper_raw_sum_sqnorm = vector_squared_norm(proper_raw_sum)
        proper_mean_loss = (
            sum_proper_kl_trajectory_mean / count if count else 0.0
        )
        cross_dot = vector_dot(raw_sum, proper_raw_sum)
        denominator = math.sqrt(
            raw_sum_sqnorm * proper_raw_sum_sqnorm
        )
        summary["proper_topk"] = {
            "mean_trajectory_token_mean_kl": proper_mean_loss,
            "global_token_mean_kl": (
                sum_proper_kl_token_weighted / sum_tokens
                if sum_tokens
                else 0.0
            ),
            "mean_gradient_norm": (
                proper_sum_grad_norm / count if count else 0.0
            ),
            "raw_sum_gradient_norm": math.sqrt(
                proper_raw_sum_sqnorm
            ),
            "raw_moments": raw_gradient_moments(
                proper_raw_sum_sqnorm,
                proper_sum_grad_sqnorm,
                count,
                proper_mean_loss,
            ),
            "aggregate_direction_cosine_with_training_surrogate": (
                cross_dot / denominator if denominator else 0.0
            ),
        }
    return summary


def save_state(
    path: Path,
    *,
    teacher_label: str,
    teacher_model: str,
    student_model: str,
    student_dtype: str,
    teacher_dtype: str,
    top_k: int,
    last_n_layers: int,
    include_final_norm: bool,
    exclude_hit_max_response_length: bool,
    shard: int,
    world: int,
    names: list[str],
    raw_sum: list[torch.Tensor],
    unit_sum: list[torch.Tensor],
    processed_records: list[dict[str, Any]],
    sum_grad_norm: float,
    sum_grad_sqnorm: float,
    sum_kl_token_weighted: float,
    sum_tokens: int,
    compute_proper_topk_gradient: bool,
    proper_raw_sum: list[torch.Tensor] | None,
    proper_sum_grad_norm: float,
    proper_sum_grad_sqnorm: float,
    sum_proper_kl_token_weighted: float,
    sum_proper_kl_trajectory_mean: float,
    complete: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = summary_from_accumulators(
        names,
        raw_sum,
        unit_sum,
        processed_records,
        sum_grad_norm,
        sum_grad_sqnorm,
        sum_kl_token_weighted,
        sum_tokens,
        proper_raw_sum,
        proper_sum_grad_norm,
        proper_sum_grad_sqnorm,
        sum_proper_kl_token_weighted,
        sum_proper_kl_trajectory_mean,
    )
    state = {
        "schema_version": 2,
        "definition": (
            "Per-trajectory gradient of the practical thunlp/OPD only_stu, "
            "student_p, token_reward_direct loss, projected onto selected "
            "student parameters."
        ),
        "teacher_label": teacher_label,
        "teacher_model": teacher_model,
        "student_model": student_model,
        "student_dtype": student_dtype,
        "teacher_dtype": teacher_dtype,
        "top_k": top_k,
        "compute_proper_topk_gradient": compute_proper_topk_gradient,
        "last_n_layers": last_n_layers,
        "include_final_norm": include_final_norm,
        "exclude_hit_max_response_length": (
            exclude_hit_max_response_length
        ),
        "shard": shard,
        "world": world,
        "selected_parameter_names": names,
        "selected_parameter_shapes": [
            list(tensor.shape) for tensor in raw_sum
        ],
        "raw_sum": [tensor.detach().cpu() for tensor in raw_sum],
        "unit_sum": [tensor.detach().cpu() for tensor in unit_sum],
        "proper_raw_sum": (
            [tensor.detach().cpu() for tensor in proper_raw_sum]
            if proper_raw_sum is not None
            else None
        ),
        "processed_records": processed_records,
        "sum_grad_norm": sum_grad_norm,
        "sum_grad_sqnorm": sum_grad_sqnorm,
        "sum_kl_token_weighted": sum_kl_token_weighted,
        "sum_tokens": sum_tokens,
        "proper_sum_grad_norm": proper_sum_grad_norm,
        "proper_sum_grad_sqnorm": proper_sum_grad_sqnorm,
        "sum_proper_kl_token_weighted": (
            sum_proper_kl_token_weighted
        ),
        "sum_proper_kl_trajectory_mean": (
            sum_proper_kl_trajectory_mean
        ),
        "complete": complete,
        "summary": summary,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)
    sidecar = path.with_suffix(path.suffix + ".json")
    sidecar.write_text(
        json.dumps(
            {
                key: value
                for key, value in state.items()
                if key not in {
                    "raw_sum",
                    "unit_sum",
                    "proper_raw_sum",
                    "processed_records",
                }
            }
            | {
                "processed_prompt_indices": [
                    int(record["prompt_idx"])
                    for record in processed_records
                ],
                "records_tail": processed_records[-10:],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--teacher-label", required=True)
    parser.add_argument(
        "--student-dtype",
        choices=["bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument(
        "--teacher-dtype",
        choices=["bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--anchor-inputs", nargs="+", required=True)
    parser.add_argument("--metadata-inputs", nargs="*", default=[])
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--last-n-layers", type=int, default=1)
    parser.add_argument(
        "--compute-proper-topk-gradient",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also differentiate the non-negative reverse KL obtained by "
            "renormalizing student and teacher on the student's top-k set."
        ),
    )
    parser.add_argument(
        "--include-final-norm", action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--logit-chunk", type=int, default=128)
    parser.add_argument(
        "--student-attention-implementation",
        default="flash_attention_2",
    )
    parser.add_argument(
        "--teacher-attention-implementation",
        default="flash_attention_2",
    )
    parser.add_argument("--prompt-indices", type=int, nargs="*")
    parser.add_argument(
        "--exclude-hit-max-response-length",
        action="store_true",
        help=(
            "Exclude trajectories whose generation terminated at the "
            "configured response cap. Filtering happens after deterministic "
            "shard assignment so an existing non-truncated checkpoint can "
            "be upgraded safely."
        ),
    )
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=20260808,
        help=(
            "Seed for selecting max-prompts globally before length-balanced "
            "sharding. This avoids selecting only the shortest trajectories."
        ),
    )
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.world <= 0 or not 0 <= args.shard < args.world:
        raise ValueError("invalid shard/world")
    if args.top_k <= 1 or args.logit_chunk <= 0:
        raise ValueError("top-k must exceed one and chunks must be positive")
    dtypes = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda:0")
    anchors = load_jsonl(args.anchor_inputs)
    if len(anchors) != 673:
        raise ValueError(f"expected 673 anchors, got {len(anchors)}")
    metadata = load_metadata(args.metadata_inputs)
    if args.prompt_indices:
        requested = set(args.prompt_indices)
        anchors = [
            record
            for record in anchors
            if int(record["prompt_idx"]) in requested
        ]
        missing = requested - {
            int(record["prompt_idx"]) for record in anchors
        }
        if missing:
            raise ValueError(
                f"requested prompts are absent: {sorted(missing)}"
            )
    if args.max_prompts is not None:
        if args.max_prompts <= 0:
            raise ValueError("max-prompts must be positive")
        generator = random.Random(args.sample_seed)
        generator.shuffle(anchors)
        anchors = anchors[: args.max_prompts]
    assigned = balanced_assignments(anchors, args.world)[args.shard]
    if args.exclude_hit_max_response_length:
        assigned = [
            record
            for record in assigned
            if not record["hit_max_response_length"]
        ]

    student_config = AutoConfig.from_pretrained(
        args.student_model, trust_remote_code=True
    )
    teacher_config = AutoConfig.from_pretrained(
        args.teacher_model, trust_remote_code=True
    )
    if int(student_config.vocab_size) != int(teacher_config.vocab_size):
        raise ValueError(
            f"vocab mismatch: student={student_config.vocab_size}, "
            f"teacher={teacher_config.vocab_size}"
        )

    print(
        f"[{args.teacher_label} shard={args.shard}/{args.world}] "
        f"loading student={args.student_model}",
        flush=True,
    )
    student = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        torch_dtype=dtypes[args.student_dtype],
        attn_implementation=args.student_attention_implementation,
        trust_remote_code=True,
        device_map={"": device},
    )
    student.config.use_cache = False
    student.eval()
    selected = select_student_parameters(
        student, args.last_n_layers, args.include_final_norm
    )
    names = [name for name, _ in selected]
    parameters = tuple(parameter for _, parameter in selected)
    selected_count = sum(parameter.numel() for parameter in parameters)
    print(
        f"[{args.teacher_label}] selected_params={selected_count:,} "
        f"({len(parameters)} tensors); loading teacher={args.teacher_model}",
        flush=True,
    )
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=dtypes[args.teacher_dtype],
        attn_implementation=args.teacher_attention_implementation,
        trust_remote_code=True,
        device_map={"": device},
    )
    teacher.config.use_cache = False
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    student_base = get_base_model(student)
    teacher_base = get_base_model(teacher)

    raw_sum = [
        torch.zeros_like(parameter, dtype=torch.float32, device=device)
        for parameter in parameters
    ]
    unit_sum = [
        torch.zeros_like(parameter, dtype=torch.float32, device=device)
        for parameter in parameters
    ]
    proper_raw_sum = (
        [
            torch.zeros_like(
                parameter, dtype=torch.float32, device=device
            )
            for parameter in parameters
        ]
        if args.compute_proper_topk_gradient
        else None
    )
    processed_records: list[dict[str, Any]] = []
    sum_grad_norm = 0.0
    sum_grad_sqnorm = 0.0
    sum_kl_token_weighted = 0.0
    sum_tokens = 0
    proper_sum_grad_norm = 0.0
    proper_sum_grad_sqnorm = 0.0
    sum_proper_kl_token_weighted = 0.0
    sum_proper_kl_trajectory_mean = 0.0
    out_path = Path(args.out)
    if args.resume and out_path.exists():
        state = torch.load(out_path, map_location="cpu")
        expected = {
            "teacher_label": args.teacher_label,
            "teacher_model": args.teacher_model,
            "student_model": args.student_model,
            "student_dtype": args.student_dtype,
            "teacher_dtype": args.teacher_dtype,
            "top_k": args.top_k,
            "compute_proper_topk_gradient": (
                args.compute_proper_topk_gradient
            ),
            "last_n_layers": args.last_n_layers,
            "include_final_norm": args.include_final_norm,
            "exclude_hit_max_response_length": (
                args.exclude_hit_max_response_length
            ),
            "shard": args.shard,
            "world": args.world,
        }
        for key, value in expected.items():
            actual = state.get(key, False)
            if (
                key == "exclude_hit_max_response_length"
                and value
                and not actual
            ):
                excluded_ids = {
                    int(record["prompt_idx"])
                    for record in anchors
                    if record["hit_max_response_length"]
                }
                processed_in_excluded_set = excluded_ids & {
                    int(record["prompt_idx"])
                    for record in state["processed_records"]
                }
                if not processed_in_excluded_set:
                    continue
            if actual != value:
                raise ValueError(
                    f"resume mismatch for {key}: {actual} != {value}"
                )
        if state["selected_parameter_names"] != names:
            raise ValueError("resume parameter names differ")
        for destination, source in zip(raw_sum, state["raw_sum"]):
            destination.copy_(source.to(device))
        for destination, source in zip(unit_sum, state["unit_sum"]):
            destination.copy_(source.to(device))
        if proper_raw_sum is not None:
            state_proper_raw_sum = state.get("proper_raw_sum")
            if state_proper_raw_sum is None:
                raise ValueError(
                    "resume state has no proper top-k gradient buffers"
                )
            for destination, source in zip(
                proper_raw_sum, state_proper_raw_sum
            ):
                destination.copy_(source.to(device))
        processed_records = list(state["processed_records"])
        sum_grad_norm = float(state["sum_grad_norm"])
        sum_grad_sqnorm = float(state["sum_grad_sqnorm"])
        sum_kl_token_weighted = float(
            state["sum_kl_token_weighted"]
        )
        sum_tokens = int(state["sum_tokens"])
        proper_sum_grad_norm = float(
            state.get("proper_sum_grad_norm", 0.0)
        )
        proper_sum_grad_sqnorm = float(
            state.get("proper_sum_grad_sqnorm", 0.0)
        )
        sum_proper_kl_token_weighted = float(
            state.get("sum_proper_kl_token_weighted", 0.0)
        )
        sum_proper_kl_trajectory_mean = float(
            state.get("sum_proper_kl_trajectory_mean", 0.0)
        )
        print(
            f"[{args.teacher_label}] resumed "
            f"{len(processed_records)} records",
            flush=True,
        )
    processed_ids = {
        int(record["prompt_idx"]) for record in processed_records
    }
    remaining = [
        record
        for record in assigned
        if int(record["prompt_idx"]) not in processed_ids
    ]
    print(
        f"[{args.teacher_label}] assigned={len(assigned)} "
        f"remaining={len(remaining)}",
        flush=True,
    )

    started = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    for completed_now, record in enumerate(remaining, 1):
        prompt_ids = record["prompt_ids"]
        output_ids = record["output_ids"]
        if not prompt_ids or not output_ids:
            raise ValueError(
                f"empty prompt/output for {record['prompt_idx']}"
            )
        input_ids = torch.tensor(
            [prompt_ids + output_ids],
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.ones_like(input_ids)
        start = len(prompt_ids) - 1
        stop = start + len(output_ids)

        with torch.no_grad():
            teacher_hidden = teacher_base(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state[:, start:stop, :]
        student_hidden = student_base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state[:, start:stop, :]
        if student_hidden.shape[1] != len(output_ids):
            raise ValueError("student response alignment failure")

        total_surrogate = torch.zeros(
            (), dtype=torch.float32, device=device
        )
        total_topk_logratio = torch.zeros(
            (), dtype=torch.float32, device=device
        )
        total_proper_topk_kl = torch.zeros(
            (), dtype=torch.float32, device=device
        )
        for chunk_start in range(0, len(output_ids), args.logit_chunk):
            chunk_stop = min(
                len(output_ids), chunk_start + args.logit_chunk
            )
            student_logits_native = student.lm_head(
                student_hidden[:, chunk_start:chunk_stop, :]
            )
            # Match the training forward: choose top-k in model dtype. On
            # H200, repeatedly applying FP32 topk to a 151,936-way tensor at
            # 16k sequence length produced a reproducible corrupted int64
            # index after thousands of chunks for the RL checkpoint lane.
            student_values_native, student_ids = torch.topk(
                student_logits_native, args.top_k, dim=-1
            )
            student_logits = student_logits_native.float()
            student_values = student_values_native.float()
            student_log_probs = (
                student_values
                - torch.logsumexp(
                    student_logits, dim=-1, keepdim=True
                )
            )
            with torch.no_grad():
                teacher_logits = teacher.lm_head(
                    teacher_hidden[:, chunk_start:chunk_stop, :]
                ).float()
                if teacher_logits.shape[:2] != student_ids.shape[:2]:
                    raise ValueError(
                        "teacher/student chunk shape mismatch at "
                        f"prompt={record['prompt_idx']} "
                        f"chunk={chunk_start}:{chunk_stop}: "
                        f"teacher={tuple(teacher_logits.shape)}, "
                        f"student_ids={tuple(student_ids.shape)}, "
                        f"teacher_hidden={tuple(teacher_hidden.shape)}, "
                        f"student_hidden={tuple(student_hidden.shape)}"
                    )
                minimum_student_id = int(student_ids.min().item())
                maximum_student_id = int(student_ids.max().item())
                if (
                    minimum_student_id < 0
                    or maximum_student_id >= teacher_logits.shape[-1]
                ):
                    raise ValueError(
                        "student top-k id is outside teacher vocabulary at "
                        f"prompt={record['prompt_idx']} "
                        f"chunk={chunk_start}:{chunk_stop}: "
                        f"min={minimum_student_id}, "
                        f"max={maximum_student_id}, "
                        f"teacher_vocab={teacher_logits.shape[-1]}"
                    )
                teacher_values = torch.gather(
                    teacher_logits, dim=-1, index=student_ids
                )
                teacher_log_probs = (
                    teacher_values
                    - torch.logsumexp(
                        teacher_logits, dim=-1, keepdim=True
                    )
                )
                normalized_student_weights = torch.softmax(
                    student_log_probs.detach(), dim=-1
                )
                logratio = (
                    student_log_probs.detach() - teacher_log_probs
                )
                advantages = (
                    -logratio * normalized_student_weights
                )
                token_topk_logratio = (
                    normalized_student_weights * logratio
                ).sum(dim=-1)
                teacher_topk_log_weights = torch.log_softmax(
                    teacher_values, dim=-1
                )
            student_topk_log_weights = torch.log_softmax(
                student_values, dim=-1
            )
            student_topk_weights = student_topk_log_weights.exp()
            token_proper_topk_kl = (
                student_topk_weights
                * (
                    student_topk_log_weights
                    - teacher_topk_log_weights
                )
            ).sum(dim=-1)
            token_surrogate = -(
                advantages * student_log_probs
            ).sum(dim=-1)
            total_surrogate = (
                total_surrogate + token_surrogate.sum()
            )
            total_topk_logratio = (
                total_topk_logratio
                + token_topk_logratio.sum()
            )
            total_proper_topk_kl = (
                total_proper_topk_kl
                + token_proper_topk_kl.sum()
            )
            del (
                student_logits,
                student_logits_native,
                student_values,
                student_values_native,
                student_ids,
                student_log_probs,
                teacher_logits,
                teacher_values,
                teacher_log_probs,
                normalized_student_weights,
                logratio,
                advantages,
                token_topk_logratio,
                teacher_topk_log_weights,
                student_topk_log_weights,
                student_topk_weights,
                token_proper_topk_kl,
                token_surrogate,
            )
        token_count = len(output_ids)
        loss = total_surrogate / token_count
        proper_loss = total_proper_topk_kl / token_count
        proper_gradients = None
        proper_grad_sqnorm = 0.0
        proper_grad_norm = 0.0
        if args.compute_proper_topk_gradient:
            proper_gradients = torch.autograd.grad(
                proper_loss,
                parameters,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )
            proper_grad_sqnorm = float(
                squared_norm(proper_gradients).item()
            )
            proper_grad_norm = math.sqrt(proper_grad_sqnorm)
            if (
                not math.isfinite(proper_grad_norm)
                or proper_grad_norm <= 0.0
            ):
                raise FloatingPointError(
                    f"invalid proper-KL gradient norm for prompt "
                    f"{record['prompt_idx']}: {proper_grad_norm}"
                )
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )
        grad_sqnorm = float(squared_norm(gradients).item())
        grad_norm = math.sqrt(grad_sqnorm)
        if not math.isfinite(grad_norm) or grad_norm <= 0.0:
            raise FloatingPointError(
                f"invalid gradient norm for prompt "
                f"{record['prompt_idx']}: {grad_norm}"
            )
        inverse_norm = 1.0 / grad_norm
        with torch.no_grad():
            for raw_buffer, unit_buffer, gradient in zip(
                raw_sum, unit_sum, gradients
            ):
                raw_buffer.add_(gradient)
                unit_buffer.add_(
                    gradient, alpha=inverse_norm
                )
            if proper_raw_sum is not None and proper_gradients is not None:
                for proper_buffer, proper_gradient in zip(
                    proper_raw_sum, proper_gradients
                ):
                    proper_buffer.add_(proper_gradient)
        topk_logratio_mean = float(
            (total_topk_logratio / token_count).item()
        )
        sum_grad_norm += grad_norm
        sum_grad_sqnorm += grad_sqnorm
        sum_kl_token_weighted += topk_logratio_mean * token_count
        sum_tokens += token_count
        proper_topk_kl_mean = float(proper_loss.item())
        proper_sum_grad_norm += proper_grad_norm
        proper_sum_grad_sqnorm += proper_grad_sqnorm
        sum_proper_kl_token_weighted += (
            proper_topk_kl_mean * token_count
        )
        sum_proper_kl_trajectory_mean += proper_topk_kl_mean
        processed_records.append(
            {
                "prompt_idx": int(record["prompt_idx"]),
                "data_source": record["data_source"],
                "student_correct": metadata.get(
                    int(record["prompt_idx"]), {}
                ).get("student_correct"),
                "prompt_tokens": len(prompt_ids),
                "response_tokens": token_count,
                "topk_logratio_mean": topk_logratio_mean,
                "gradient_norm": grad_norm,
                "proper_topk_kl_mean": proper_topk_kl_mean,
                "proper_gradient_norm": proper_grad_norm,
            }
        )
        del (
            gradients,
            proper_gradients,
            loss,
            proper_loss,
            total_surrogate,
            total_topk_logratio,
            total_proper_topk_kl,
            student_hidden,
            teacher_hidden,
            input_ids,
            attention_mask,
        )
        if completed_now % 5 == 0 or completed_now == len(remaining):
            elapsed = time.time() - started
            rate = completed_now / elapsed if elapsed else 0.0
            peak_gib = torch.cuda.max_memory_allocated(device) / 2**30
            print(
                f"[{args.teacher_label} shard={args.shard}] "
                f"{completed_now}/{len(remaining)} new, "
                f"total={len(processed_records)}, "
                f"last_idx={record['prompt_idx']}, "
                f"topk_logratio={topk_logratio_mean:.6g}, "
                f"|g|={grad_norm:.6g}, "
                f"rate={rate:.3f} traj/s, peak={peak_gib:.1f} GiB",
                flush=True,
            )
        if (
            args.checkpoint_every > 0
            and len(processed_records) % args.checkpoint_every == 0
        ):
            save_state(
                out_path,
                teacher_label=args.teacher_label,
                teacher_model=args.teacher_model,
                student_model=args.student_model,
                student_dtype=args.student_dtype,
                teacher_dtype=args.teacher_dtype,
                top_k=args.top_k,
                last_n_layers=args.last_n_layers,
                include_final_norm=args.include_final_norm,
                exclude_hit_max_response_length=(
                    args.exclude_hit_max_response_length
                ),
                shard=args.shard,
                world=args.world,
                names=names,
                raw_sum=raw_sum,
                unit_sum=unit_sum,
                processed_records=processed_records,
                sum_grad_norm=sum_grad_norm,
                sum_grad_sqnorm=sum_grad_sqnorm,
                sum_kl_token_weighted=sum_kl_token_weighted,
                sum_tokens=sum_tokens,
                compute_proper_topk_gradient=(
                    args.compute_proper_topk_gradient
                ),
                proper_raw_sum=proper_raw_sum,
                proper_sum_grad_norm=proper_sum_grad_norm,
                proper_sum_grad_sqnorm=proper_sum_grad_sqnorm,
                sum_proper_kl_token_weighted=(
                    sum_proper_kl_token_weighted
                ),
                sum_proper_kl_trajectory_mean=(
                    sum_proper_kl_trajectory_mean
                ),
                complete=False,
            )

    save_state(
        out_path,
        teacher_label=args.teacher_label,
        teacher_model=args.teacher_model,
        student_model=args.student_model,
        student_dtype=args.student_dtype,
        teacher_dtype=args.teacher_dtype,
        top_k=args.top_k,
        last_n_layers=args.last_n_layers,
        include_final_norm=args.include_final_norm,
        exclude_hit_max_response_length=(
            args.exclude_hit_max_response_length
        ),
        shard=args.shard,
        world=args.world,
        names=names,
        raw_sum=raw_sum,
        unit_sum=unit_sum,
        processed_records=processed_records,
        sum_grad_norm=sum_grad_norm,
        sum_grad_sqnorm=sum_grad_sqnorm,
        sum_kl_token_weighted=sum_kl_token_weighted,
        sum_tokens=sum_tokens,
        compute_proper_topk_gradient=(
            args.compute_proper_topk_gradient
        ),
        proper_raw_sum=proper_raw_sum,
        proper_sum_grad_norm=proper_sum_grad_norm,
        proper_sum_grad_sqnorm=proper_sum_grad_sqnorm,
        sum_proper_kl_token_weighted=(
            sum_proper_kl_token_weighted
        ),
        sum_proper_kl_trajectory_mean=(
            sum_proper_kl_trajectory_mean
        ),
        complete=len(processed_records) == len(assigned),
    )
    print(
        json.dumps(
            summary_from_accumulators(
                names,
                raw_sum,
                unit_sum,
                processed_records,
                sum_grad_norm,
                sum_grad_sqnorm,
                sum_kl_token_weighted,
                sum_tokens,
                proper_raw_sum,
                proper_sum_grad_norm,
                proper_sum_grad_sqnorm,
                sum_proper_kl_token_weighted,
                sum_proper_kl_trajectory_mean,
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
