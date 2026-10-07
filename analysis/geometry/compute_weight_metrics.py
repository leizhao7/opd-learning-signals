#!/usr/bin/env python3
"""Compute PRISM-style normalized weight divergence and sparsity for five pairs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open


LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.*)$")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def weight_map(model_dir: Path) -> dict[str, Path]:
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        data = json.loads(index.read_text())["weight_map"]
        return {name: model_dir / file for name, file in data.items()}
    result = {}
    for file in sorted(model_dir.glob("*.safetensors")):
        with safe_open(file, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in result:
                    raise ValueError(f"duplicate parameter {name}")
                result[name] = file
    return result


def classify(name: str) -> tuple[str, int | None, str]:
    match = LAYER_RE.match(name)
    if match:
        layer_index = int(match.group(1))
        suffix = match.group(2)
        if suffix.startswith("self_attn."):
            component = "attn." + suffix.split(".")[1]
        elif suffix.startswith("mlp."):
            component = "mlp." + suffix.split(".")[1]
        elif suffix.startswith("input_layernorm"):
            component = "norm.input"
        elif suffix.startswith("post_attention_layernorm"):
            component = "norm.post_attention"
        else:
            component = "block.other"
        return f"block_{layer_index:02d}", layer_index, component
    if name.startswith("model.embed_tokens"):
        return "embedding", None, "embedding"
    if name.startswith("model.norm"):
        return "final_norm", None, "norm.final"
    if name.startswith("lm_head"):
        return "lm_head", None, "lm_head"
    return "other", None, "other"


def slug_threshold(percent: float) -> str:
    return str(percent).replace(".", "p")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict], keys: tuple[str, ...], thresholds: list[float]) -> list[dict]:
    groups = {}
    for row in rows:
        key = tuple(row[field] for field in keys)
        if key not in groups:
            groups[key] = {field: value for field, value in zip(keys, key)}
            groups[key].update(
                n=0,
                base_sq=0.0,
                final_sq=0.0,
                delta_sq=0.0,
                matrix_count=0,
                matrix_delta_sum=0.0,
            )
            for threshold in thresholds:
                slug = slug_threshold(threshold)
                groups[key][f"tensor_unchanged_numel_{slug}"] = 0
                groups[key][f"element_unchanged_numel_{slug}"] = 0
        group = groups[key]
        for field in ("n", "base_sq", "final_sq", "delta_sq"):
            group[field] += row[field]
        group["matrix_count"] += 1
        group["matrix_delta_sum"] += row["matrix_normalized_l2"]
        for threshold in thresholds:
            slug = slug_threshold(threshold)
            if row["matrix_normalized_l2"] < threshold / 100.0:
                group[f"tensor_unchanged_numel_{slug}"] += row["n"]
            group[f"element_unchanged_numel_{slug}"] += row[
                f"element_unchanged_numel_{slug}"
            ]
    output = []
    for key in sorted(groups, key=lambda values: tuple(str(value) for value in values)):
        row = groups[key]
        row["normalized_l2"] = math.sqrt(row["delta_sq"] / row["base_sq"])
        row["mean_matrix_normalized_l2"] = row["matrix_delta_sum"] / row["matrix_count"]
        for threshold in thresholds:
            slug = slug_threshold(threshold)
            row[f"tensor_weighted_unchanged_fraction_{slug}"] = (
                row[f"tensor_unchanged_numel_{slug}"] / row["n"]
            )
            row[f"elementwise_unchanged_fraction_{slug}"] = (
                row[f"element_unchanged_numel_{slug}"] / row["n"]
            )
        output.append(row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--chunk-elements", type=int, default=4_000_000)
    parser.add_argument("--element-relative-epsilon", type=float, default=1e-8)
    args = parser.parse_args()
    root = args.root.resolve()
    contract = json.loads((root / "experiment_contract.json").read_text())
    thresholds = [float(value) for value in contract["weight_metric"]["threshold_percent"]]
    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    parameter_rows = []

    torch.set_num_threads(12)
    for pair in contract["pairs"]:
        base_dir = Path(pair["student_base"])
        final_dir = Path(pair["final"])
        base_config = json.loads((base_dir / "config.json").read_text())
        final_config = json.loads((final_dir / "config.json").read_text())
        for field in ("model_type", "hidden_size", "num_hidden_layers", "vocab_size"):
            if base_config.get(field) != final_config.get(field):
                raise ValueError(f"{pair['id']}: config mismatch in {field}")
        tied = bool(base_config.get("tie_word_embeddings", False))
        base_map = weight_map(base_dir)
        final_map = weight_map(final_dir)
        base_names = {name for name in base_map if not (tied and name == "lm_head.weight")}
        final_names = {name for name in final_map if not (tied and name == "lm_head.weight")}
        if base_names != final_names:
            raise ValueError(
                f"{pair['id']}: parameter names differ: "
                f"base_only={sorted(base_names-final_names)[:5]} "
                f"final_only={sorted(final_names-base_names)[:5]}"
            )

        with ExitStack() as stack:
            base_handles = {
                path: stack.enter_context(safe_open(path, framework="pt", device="cpu"))
                for path in sorted(set(base_map.values()))
            }
            final_handles = {
                path: stack.enter_context(safe_open(path, framework="pt", device="cpu"))
                for path in sorted(set(final_map.values()))
            }
            for name in sorted(base_names):
                base_slice = base_handles[base_map[name]].get_slice(name)
                final_slice = final_handles[final_map[name]].get_slice(name)
                shape = tuple(int(value) for value in base_slice.get_shape())
                if tuple(final_slice.get_shape()) != shape:
                    raise ValueError(f"{pair['id']}/{name}: shape mismatch")
                width = math.prod(shape[1:]) if len(shape) > 1 else 1
                rows_per_chunk = max(1, args.chunk_elements // width)
                stats = {"n": 0, "base_sq": 0.0, "final_sq": 0.0, "delta_sq": 0.0}
                for threshold in thresholds:
                    stats[f"element_unchanged_numel_{slug_threshold(threshold)}"] = 0
                for start in range(0, shape[0], rows_per_chunk):
                    stop = min(shape[0], start + rows_per_chunk)
                    base = base_slice[start:stop].float()
                    final = final_slice[start:stop].float()
                    delta = final - base
                    stats["n"] += delta.numel()
                    stats["base_sq"] += float(torch.sum(base * base, dtype=torch.float64))
                    stats["final_sq"] += float(torch.sum(final * final, dtype=torch.float64))
                    stats["delta_sq"] += float(torch.sum(delta * delta, dtype=torch.float64))
                    relative_element = delta.abs() / base.abs().clamp_min(
                        args.element_relative_epsilon
                    )
                    for threshold in thresholds:
                        slug = slug_threshold(threshold)
                        stats[f"element_unchanged_numel_{slug}"] += int(
                            (relative_element < threshold / 100.0).sum()
                        )
                    del base, final, delta, relative_element
                layer, layer_index, component = classify(name)
                row = {
                    "pair": pair["id"],
                    "student_base": str(base_dir),
                    "teacher": pair["teacher"],
                    "final": str(final_dir),
                    "parameter": name,
                    "shape": "x".join(map(str, shape)),
                    "layer": layer,
                    "layer_index": layer_index,
                    "component": component,
                    **stats,
                }
                row["base_l2"] = math.sqrt(row["base_sq"])
                row["delta_l2"] = math.sqrt(row["delta_sq"])
                row["matrix_normalized_l2"] = row["delta_l2"] / row["base_l2"]
                parameter_rows.append(row)
        print(f"weight metrics complete pair={pair['id']} matrices={len(base_names)}", flush=True)

    shared = ("pair", "student_base", "teacher", "final")
    global_rows = aggregate(parameter_rows, shared, thresholds)
    layer_rows = aggregate(parameter_rows, shared + ("layer", "layer_index"), thresholds)
    component_rows = aggregate(parameter_rows, shared + ("component",), thresholds)
    global_delta = {row["pair"]: row["delta_sq"] for row in global_rows}
    for rows in (parameter_rows, layer_rows, component_rows):
        for row in rows:
            row["delta_sq_fraction"] = row["delta_sq"] / global_delta[row["pair"]]

    sparsity_rows = []
    for row in global_rows:
        for threshold in thresholds:
            slug = slug_threshold(threshold)
            sparsity_rows.append(
                {
                    "pair": row["pair"],
                    "threshold_percent": threshold,
                    "tensor_weighted_unchanged_fraction": row[
                        f"tensor_weighted_unchanged_fraction_{slug}"
                    ],
                    "elementwise_unchanged_fraction": row[
                        f"elementwise_unchanged_fraction_{slug}"
                    ],
                    "element_relative_epsilon": args.element_relative_epsilon,
                }
            )

    write_csv(out_dir / "weight_parameter_stats.csv", parameter_rows)
    write_csv(out_dir / "weight_layer_stats.csv", layer_rows)
    write_csv(out_dir / "weight_component_stats.csv", component_rows)
    write_csv(out_dir / "weight_global_stats.csv", global_rows)
    write_csv(out_dir / "weight_sparsity.csv", sparsity_rows)
    summary = {
        "schema_version": 1,
        "metric": contract["weight_metric"],
        "element_relative_epsilon": args.element_relative_epsilon,
        "pairs": [
            {
                "pair": row["pair"],
                "n_parameters": row["n"],
                "n_matrices": row["matrix_count"],
                "global_normalized_l2": row["normalized_l2"],
                "mean_matrix_normalized_l2": row["mean_matrix_normalized_l2"],
                "tensor_weighted_unchanged_fraction_1pct": row[
                    "tensor_weighted_unchanged_fraction_1p0"
                ],
                "elementwise_unchanged_fraction_1pct": row[
                    "elementwise_unchanged_fraction_1p0"
                ],
            }
            for row in global_rows
        ],
        "artifacts": {},
    }
    for name in (
        "weight_parameter_stats.csv",
        "weight_layer_stats.csv",
        "weight_component_stats.csv",
        "weight_global_stats.csv",
        "weight_sparsity.csv",
    ):
        summary["artifacts"][name] = sha256(out_dir / name)
    (out_dir / "weight_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
