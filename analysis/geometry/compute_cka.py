#!/usr/bin/env python3
"""Compute PRISM-style centered linear CKA from extracted representations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


REPRESENTATION_PAIRS = {
    "r1_justrl": ("r1_math_base", "r1_math_justrl"),
    "r1_skywork": ("r1_math_base", "r1_math_skywork"),
    "r1_r1_7b": ("r1_math_base", "r1_math_r1_7b"),
    "qwen4b_qwen30b": ("qwen4_math_base", "qwen4_math_qwen30b"),
    "qwen4b_code14b": ("qwen4_code_base", "qwen4_code_code14b"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.to(dtype=torch.float64)
    y = y.to(dtype=torch.float64)
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    k = x @ x.T
    l = y @ y.T
    numerator = torch.sum(k * l)
    denominator = torch.sqrt(torch.sum(k * k) * torch.sum(l * l))
    if denominator <= 0:
        raise ValueError("degenerate CKA denominator")
    value = float((numerator / denominator).item())
    if not -1e-10 <= value <= 1.0 + 1e-9:
        raise ValueError(f"CKA outside range: {value}")
    return min(1.0, max(0.0, value))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = args.root.resolve()
    contract = json.loads((root / "experiment_contract.json").read_text())
    pair_configs = {pair["id"]: pair for pair in contract["pairs"]}
    representation_pairs = contract.get("representation_pairs", REPRESENTATION_PAIRS)
    rep_dir = root / "representations"
    bank_dir = root / "input_banks"
    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    if set(pair_configs) != set(representation_pairs):
        raise ValueError("contract/representation pair mismatch")

    primary_rows = []
    bootstrap_rows = []
    rng = np.random.default_rng(int(contract["cka"]["bootstrap_seed"]))
    for pair_id, (base_label, final_label) in representation_pairs.items():
        base_manifest = json.loads((rep_dir / f"{base_label}.manifest.json").read_text())
        final_manifest = json.loads((rep_dir / f"{final_label}.manifest.json").read_text())
        if base_manifest["bank_sha256"] != final_manifest["bank_sha256"]:
            raise ValueError(f"{pair_id}: input bank mismatch")
        x_array = np.load(rep_dir / f"{base_label}.npy", mmap_mode="r")
        y_array = np.load(rep_dir / f"{final_label}.npy", mmap_mode="r")
        if x_array.shape != y_array.shape:
            raise ValueError(f"{pair_id}: representation shape mismatch")

        bank_key = "qwen4_code" if pair_configs[pair_id]["domain"] == "code" else (
            "r1_math" if pair_id.startswith("r1_") else "qwen4_math"
        )
        bank_key = pair_configs[pair_id].get("bank", bank_key)
        bank = read_jsonl(bank_dir / f"{bank_key}.jsonl")
        if len(bank) != x_array.shape[0]:
            raise ValueError(f"{pair_id}: bank/representation row mismatch")
        dataset_indices = {"ALL": np.arange(len(bank), dtype=np.int64)}
        for dataset in sorted({record["data_source"] for record in bank}):
            dataset_indices[dataset] = np.asarray(
                [i for i, record in enumerate(bank) if record["data_source"] == dataset],
                dtype=np.int64,
            )

        for dataset, indices in dataset_indices.items():
            sample_size = min(100, len(indices))
            bootstrap_indices = [
                rng.choice(indices, size=sample_size, replace=True)
                for _ in range(int(contract["cka"]["bootstrap_replicates"]))
            ]
            for layer in range(x_array.shape[1]):
                x = torch.from_numpy(np.asarray(x_array[:, layer, :])).to(args.device)
                y = torch.from_numpy(np.asarray(y_array[:, layer, :])).to(args.device)
                full_value = linear_cka(x[torch.from_numpy(indices).to(args.device)], y[torch.from_numpy(indices).to(args.device)])
                values = []
                for replicate, chosen in enumerate(bootstrap_indices):
                    chosen_tensor = torch.from_numpy(chosen).to(args.device)
                    value = linear_cka(x[chosen_tensor], y[chosen_tensor])
                    values.append(value)
                    bootstrap_rows.append(
                        {
                            "pair": pair_id,
                            "dataset": dataset,
                            "n_available": len(indices),
                            "sample_size": sample_size,
                            "hidden_state_index": layer,
                            "replicate": replicate,
                            "cka": value,
                        }
                    )
                primary_rows.append(
                    {
                        "pair": pair_id,
                        "domain": pair_configs[pair_id]["domain"],
                        "dataset": dataset,
                        "n_prompts": len(indices),
                        "hidden_state_index": layer,
                        "cka": full_value,
                        "bootstrap_mean": float(np.mean(values)),
                        "bootstrap_std": float(np.std(values, ddof=1)),
                        "bootstrap_min": float(np.min(values)),
                        "bootstrap_max": float(np.max(values)),
                    }
                )
                del x, y
            print(f"CKA complete pair={pair_id} dataset={dataset}", flush=True)

    write_csv(out_dir / "cka_layer_stats.csv", primary_rows)
    write_csv(out_dir / "cka_bootstrap.csv", bootstrap_rows)

    summaries = []
    for pair_id in representation_pairs:
        rows = [row for row in primary_rows if row["pair"] == pair_id and row["dataset"] == "ALL"]
        values = np.asarray([row["cka"] for row in rows])
        summaries.append(
            {
                "pair": pair_id,
                "n_prompts": rows[0]["n_prompts"],
                "n_hidden_states": len(rows),
                "minimum_cka": float(values.min()),
                "minimum_cka_hidden_state_index": int(values.argmin()),
                "median_cka": float(np.median(values)),
                "final_hidden_state_cka": float(values[-1]),
                "maximum_bootstrap_std": max(row["bootstrap_std"] for row in rows),
            }
        )
    output = {
        "schema_version": 1,
        "metric": "centered linear CKA on mean-pooled prompt hidden states",
        "bootstrap": contract["cka"],
        "pairs": summaries,
        "artifacts": {
            "cka_layer_stats.csv": sha256(out_dir / "cka_layer_stats.csv"),
            "cka_bootstrap.csv": sha256(out_dir / "cka_bootstrap.csv"),
        },
    }
    (out_dir / "cka_summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
