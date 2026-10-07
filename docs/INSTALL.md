# Environments

## CPU plotting and packaging checks

Use Python 3.12; the recovered training sources include Python 3.12 syntax. Install `requirements-analysis.txt` and `requirements-test.txt`. Install PyTorch separately to enable `tests/test_full_logits_cpu.py`; this test uses small CPU tensors and does not download models. All figure reconstruction runs without torch.

## GPU training

Training requires a Linux CUDA environment with PyTorch, FlashAttention, Ray and the rollout backend in the selected recipe. The paper runs used H200 GPUs. GPU count, rollout memory settings and sequence limits remain in each recipe; changing them can change resource requirements and numerical behavior.

Install only the snapshot selected by the recipe, for example:

```bash
pip install -e vendor/verl-qwen
```

The recipes load their selected snapshot through `PYTHONPATH`. Installing arbitrary latest versions of PyTorch, transformers and vLLM together is not a verified reproduction environment. The recovered dependency constraints are in each vendor directory. `provenance/runtime_versions.json` records the package metadata that remained accessible at export time: the full-logit runtime overlay contained vLLM 0.11.0 and transformers 4.57.6. The complete old Python environment was not recoverable as an installable lockfile. The manuscript reports PyTorch 2.8.0 and Ray 2.49.0 for the support/rollout additions; these are provenance, not a claim that one stack covers every experiment.

Before a full run, verify imports of the selected snapshot, CUDA/PyTorch compatibility, FlashAttention, the selected vLLM or SGLang backend, and the reward backend on a small job in your own environment. This package has not run a fresh GPU smoke test.

## Data

Training uses veRL parquets with `data_source`, chat-message `prompt`, `ability`, `reward_model` (including `ground_truth` and `style`), and `extra_info`. Supply the original dataset's task-specific reward information, not just text prompts. Benchmark matching, chat templates, thinking mode and response limits must match the paper.

- Code: Eurus-2-RL-Data training prompts; HumanEval+, MBPP+, LiveCodeBench validation.
- Math: DAPO-Math-17K training prompts; AIME24/AIME25/AMC23 validation.
- IF: `data_pipeline/build_multiif_parquets.py` prepares the historical Multi-IF protocol from its original CSV. Its training/evaluation source overlap must be retained as a limitation if reproducing the paper, or removed for a new generalization experiment.

Models and datasets are supplied by the user from their original distributions. This repository does not mirror them.

## SFT

The recovered LLaMA-Factory source is under `vendor/llamafactory`. Install it in a separate compatible environment. Replace all `${...}` placeholders in a `configs/sft_*.json` copy with local paths, including the model, dataset directory, output directory, and a ZeRO-2 configuration. Do not change the recorded thinking template simply to match the OPD launcher's mode: the SFT and OPD prompt protocols differ as documented in the paper.

```bash
torchrun --standalone --nproc_per_node=2 training/sft.py --config /path/to/resolved-sft-config.json
```

## IF evaluation

Install `requirements-if.txt` plus the model inference stack. Supply `NLTK_DATA` containing `punkt_tab` (for example, use NLTK's downloader in your environment). `evaluation/if/run_multiif_eval.py --help` describes input and model options; the metric aggregates languages and turns and is not a simple mean of all raw instruction checks.

## Teacher-response data preparation

`data_pipeline/prepare_dapo.py --root WORKDIR --source DAPO_PARQUET` preserves the original question bodies and verifies the recorded source hash. WORKDIR must contain the historical-style `contract.proposed.json` with a `jobs` list whose entries specify student paths, `model_family` and `sft_config.template`; entries 0 and 2 provide the two tokenizer families. This strict historical helper will reject a different dataset export rather than silently recreate a different experiment.

`data_pipeline/rollout_worker.py --job JOB_JSON --rank 0` runs the recovered acceptance/filtering loop. `configs/rollout_job.example.json` documents the input schema; replace its paths, prompt hash and template with the selected experiment's values. It is an example, not a resolved launch configuration. Cluster-specific GPU reservation checks were removed; run under your own scheduler. The data collection and GPU inference path have not been rerun during packaging.
