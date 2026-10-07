#!/usr/bin/env python3
"""Build exact, tokenizer-audited validation prompt banks for CKA."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer


BANKS = {
    "r1_math": {
        "tokenizer": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "domain": "math",
        "max_prompt_length": 1024,
    },
    "qwen4_math": {
        "tokenizer": "Qwen/Qwen3-4B",
        "domain": "math",
        "max_prompt_length": 1024,
    },
    "qwen4_code": {
        "tokenizer": "Qwen/Qwen3-4B",
        "domain": "code",
        "max_prompt_length": 2048,
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def tokenize(tokenizer, prompt) -> list[int]:
    try:
        return list(
            tokenizer.apply_chat_template(
                prompt,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
    except TypeError:
        return list(
            tokenizer.apply_chat_template(
                prompt,
                tokenize=True,
                add_generation_prompt=True,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    contract = json.loads((root / "experiment_contract.json").read_text())
    out_dir = root / "input_banks"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"schema_version": 1, "banks": {}}
    for key, config in contract.get("banks", BANKS).items():
        tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"], trust_remote_code=True)
        data_paths = [Path(path) for path in contract[f"{config['domain']}_data"]]
        records = []
        raw_prompts = []
        source_counts = {}
        lengths = []
        record_id = 0
        for path in data_paths:
            for row in pq.read_table(path).to_pylist():
                source = str(row.get("data_source") or path.parent.name)
                prompt_ids = tokenize(tokenizer, row["prompt"])
                if len(prompt_ids) > config["max_prompt_length"]:
                    raise ValueError(
                        f"{key} record={record_id} length={len(prompt_ids)} "
                        f"> {config['max_prompt_length']}"
                    )
                records.append(
                    {
                        "record_id": record_id,
                        "data_source": source,
                        "prompt_ids": prompt_ids,
                    }
                )
                raw_prompts.append(row["prompt"])
                lengths.append(len(prompt_ids))
                source_counts[source] = source_counts.get(source, 0) + 1
                record_id += 1
        expected = 143 if config["domain"] == "math" else 673
        if len(records) != expected:
            raise ValueError(f"{key}: expected {expected} prompts, got {len(records)}")
        path = out_dir / f"{key}.jsonl"
        with path.open("w") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        tokenizer_audits = []
        for pair in contract["pairs"]:
            if (
                Path(pair["student_base"]).resolve() != Path(config["tokenizer"]).resolve()
                or pair["domain"] != config["domain"]
            ):
                continue
            final_tokenizer = AutoTokenizer.from_pretrained(pair["final"], trust_remote_code=True)
            if final_tokenizer.get_vocab() != tokenizer.get_vocab():
                raise ValueError(f"{key}/{pair['id']}: tokenizer vocabulary mismatch")
            for record, prompt in zip(records, raw_prompts):
                if tokenize(final_tokenizer, prompt) != record["prompt_ids"]:
                    raise ValueError(
                        f"{key}/{pair['id']}: chat-template token IDs mismatch at "
                        f"record {record['record_id']}"
                    )
            tokenizer_audits.append(
                {
                    "pair": pair["id"],
                    "final_tokenizer": pair["final"],
                    "vocab_equal": True,
                    "all_prompt_ids_equal": True,
                }
            )

        manifest["banks"][key] = {
            **config,
            "path": str(path),
            "sha256": sha256(path),
            "n_prompts": len(records),
            "source_counts": source_counts,
            "min_prompt_length": min(lengths),
            "max_prompt_length_observed": max(lengths),
            "mean_prompt_length": sum(lengths) / len(lengths),
            "tokenizer_size": len(tokenizer),
            "tokenizer_vocab_entries": len(tokenizer.get_vocab()),
            "final_tokenizer_audits": tokenizer_audits,
        }
        print(key, manifest["banks"][key], flush=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
