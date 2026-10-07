#!/usr/bin/env python3
"""Extract batch-size-1 mean-pooled hidden states for one model and input bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def read_bank(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    bank_path = args.bank.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / f"{args.label}.npy"
    partial_path = out_dir / f"{args.label}.partial.npy"
    progress_path = out_dir / f"{args.label}.progress.json"
    manifest_path = out_dir / f"{args.label}.manifest.json"
    if final_path.is_file() and manifest_path.is_file():
        print(f"already complete label={args.label}")
        return

    records = read_bank(bank_path)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    n_hidden_states = int(config.num_hidden_layers) + 1
    hidden_size = int(config.hidden_size)
    shape = (len(records), n_hidden_states, hidden_size)

    completed = 0
    if partial_path.is_file() and progress_path.is_file():
        progress = json.loads(progress_path.read_text())
        if tuple(progress["shape"]) != shape or progress["bank_sha256"] != sha256(bank_path):
            raise ValueError("incompatible representation resume artifact")
        completed = int(progress["completed"])
        array = np.lib.format.open_memmap(partial_path, mode="r+", dtype=np.float32, shape=shape)
    else:
        array = np.lib.format.open_memmap(partial_path, mode="w+", dtype=np.float32, shape=shape)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to("cuda").eval()
    observed_lengths = []
    with torch.inference_mode():
        for index in range(completed, len(records)):
            prompt_ids = records[index]["prompt_ids"]
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
            output = model(
                input_ids=input_ids,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = output.hidden_states
            if len(hidden_states) != n_hidden_states:
                raise RuntimeError(
                    f"hidden-state count mismatch {len(hidden_states)} != {n_hidden_states}"
                )
            pooled = torch.stack(
                [state[0].float().mean(dim=0) for state in hidden_states], dim=0
            )
            array[index] = pooled.cpu().numpy()
            observed_lengths.append(len(prompt_ids))
            del input_ids, output, hidden_states, pooled
            if (index + 1) % 10 == 0 or index + 1 == len(records):
                array.flush()
                progress_path.write_text(
                    json.dumps(
                        {
                            "label": args.label,
                            "shape": list(shape),
                            "completed": index + 1,
                            "bank_sha256": sha256(bank_path),
                        },
                        indent=2,
                    )
                    + "\n"
                )
                print(f"{args.label} {index + 1}/{len(records)}", flush=True)

    del model, array
    torch.cuda.empty_cache()
    os.replace(partial_path, final_path)
    manifest = {
        "schema_version": 1,
        "label": args.label,
        "model": str(model_path),
        "bank": str(bank_path),
        "bank_sha256": sha256(bank_path),
        "representation_path": str(final_path),
        "representation_sha256": sha256(final_path),
        "shape": list(shape),
        "dtype": "float32",
        "model_inference_dtype": "bfloat16",
        "attention_implementation": "flash_attention_2",
        "batch_size": 1,
        "pooling": "arithmetic mean over every prompt token hidden state",
        "hidden_state_indexing": "0=embedding output; final index=post-final-layer normalization",
        "completed": len(records),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    progress_path.unlink(missing_ok=True)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
