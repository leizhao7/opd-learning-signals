# Why On-Policy Distillation Sometimes Fails: Vanishing Learning Signals

Code preparation for the paper's non-anonymous arXiv version. This repository contains the training implementations recovered from the experiment storage, portable recipe launchers, evaluation and geometry utilities, and audited numerical inputs for the paper figures.

Authors: Lei Zhao, Qichao Zhao, Bowen Zuo, and Qishi Zhan.

## Start with the figures (CPU)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-analysis.txt
python -m opd figures
python -m unittest discover -s tests -v
```

Figures are written beneath `paper/figures/`. These commands reconstruct plots from recorded results; they do not train or download language models. See [the experiment map](docs/EXPERIMENTS.md) for coverage and [the reproduction notes](docs/REPRODUCIBILITY.md) for interpretation of the diagnostics.

## Training (GPU)

Prepare a separate GPU environment using [INSTALL.md](docs/INSTALL.md). Supply local model checkpoints and benchmark parquets matching the paper's protocol. Print a command first:

```bash
python -m opd train --recipe qwen_math \
  --student /models/Qwen3-1.7B --teacher /models/Qwen3-4B \
  --train-data /data/dapo-math-17k.parquet \
  --val-data /data/AIME24.parquet /data/AIME25.parquet /data/AMC23.parquet \
  --output /results/qwen-opd --name qwen17-qwen4
```

Add `--execute` to run. Extra Hydra settings are explicit, for example `--override trainer.total_training_steps=200`. Logging defaults to console. The release does not start cluster jobs, allocate GPUs, or upload W&B results by itself. Choose one veRL snapshot at a time; do not install all copies together.

## Unified commands

Run `python -m opd --help` for all entry points, or `python -m opd recipes` to validate and list the 11 recipes. Use `python -m opd figures --group main` to rebuild only Figure 1. Training, SFT, signal analysis, code audits and geometry tools share this entry point; each subcommand keeps its own `--help`. Run commands from the repository root.

The 57 settings shared by every OPD recipe live in `configs/base.json`; individual recipes contain only their differences. Precedence is **base → recipe → command-line overrides**. Every executed launch saves its resolved settings, configuration hashes and backend name in `launch.json`, with separate history files under `launches/`. Historical `scripts/train.py` and `scripts/reproduce_figures.py` commands still work.

See [the organization guide](docs/ORGANIZATION.md) for extension rules and what was verified.

## Layout

| Directory | Contents |
|---|---|
| `opd/` | Unified CLI, recipe loading, launch environment and figure orchestration |
| `configs/` | Shared settings and 11 experiment recipes with source provenance |
| `training/` | OPD entry point, logging helpers, checkpoint directory preparation, SFT entry point |
| `vendor/` | Four recovered veRL variants and the recovered LLaMA-Factory source |
| `evaluation/if/` | Multi-IF scoring and three-turn evaluation; original license retained |
| `data_pipeline/` | Teacher-response generation and IF parquet preparation |
| `analysis/` | Parameter distance, CKA, gradient diagnostics and explicit notation conversion |
| `paper/` | Audited numerical inputs, plot builders and reference tables |
| `tests/` | Recipe validation, rate-factorization checks and optional CPU full-logit equivalence test |
| `provenance/` | Source/output hashes and environment observations |

## Release status

The packaging checks, CPU figure reconstruction and dry-run launchers are tested. Full GPU retraining and all external evaluation dependencies have **not** been rerun from this package. Model weights, raw benchmark responses, training datasets, credentials and cluster orchestration are not bundled. Some convenience recipes are reconstructed from recovered launchers and are labeled as such; they are not presented as byte-identical historical commands.

The diagnostic scripts inherited from earlier experiment stages retain their documented scope. In particular, the suffix-gradient consistency probe is not the paper's all-parameter occupancy estimator. The paper's recorded occupancy values and reanalysis utility are provided; the standalone checkpoint sampling/measurement driver for those values has not been recovered in this package.

Third-party licenses remain with their source trees. The authors' release license still needs to be selected before a public GitHub release. See [THIRD_PARTY.md](THIRD_PARTY.md).
