# Code organization

Run `python -m opd COMMAND` from the repository root. CLI startup and recipe dry-runs need only Python's standard library; training and analysis dependencies are loaded by the selected command.

| Command | Implementation / purpose |
|---|---|
| `train` | `opd/train.py`: resolve configuration, print command, optionally execute |
| `recipes` | Validate all configurations and list their frozen backends |
| `figures --group all` | Rebuild the paper figures; groups: main, signal, if, diagnostics, sft, rollouts |
| `sft` | `training/sft.py`; requires an already resolved SFT configuration and the installed LLaMA-Factory environment |
| `signals` | `analysis/learning_signal.py`; archived cross-fitted diagnostics |
| `code-audit` | `evaluation/audit_code_avg4.py`; fixed four-of-eight selection |
| `input-banks`, `representations`, `cka`, `weights` | Existing geometry pipeline, with its original arguments |

Use `python -m opd COMMAND --help` for arguments (except `recipes`, which takes no arguments). Forwarded commands preserve paths relative to your current directory. Figure builders run in `paper/`, as before. Existing direct script commands remain supported.

## Configuration rules

1. `configs/base.json` contains the 57 assignments identical in all 11 original release recipes.
2. `configs/recipes/NAME.json` selects a frozen backend and adds or replaces settings.
3. Repeated `--override key=value` arguments take final precedence. A leading `+` does not prevent replacement of the same key.

The loader rejects unknown recipes, unsupported backends and duplicate assignments within a configuration. Model/data/output paths supplied to the training CLI are quoted for Hydra; validation data remains a list. Raw `--override` values retain Hydra syntax.

Use recipe-specific overrides for new experiments. If a shared setting needs changing for only one experiment, keep the base unchanged. Retain the original launcher hash and mark derived configurations explicitly. Execution records include the resolved command, settings, backend and configuration hashes; prior launch records survive in `OUTPUT/launches/` when a run is resumed.

## Why the backend snapshots remain separate

The four veRL trees encode distinct historical training and evaluation implementations. A common interface does not establish that their algorithms are interchangeable. They remain frozen under `vendor/`, with their third-party licenses. Refactoring regression tests verify every file against `provenance/frozen_vendor_sha256.json`. No full-logit kernel, reward implementation, optimizer or training-loop code was modified by this organization pass.

## Regression checks

```bash
pip install -r requirements-analysis.txt -r requirements-test.txt
# Install CPU torch separately to run the full-logit gradient check.
python -m unittest discover -s tests -v
python -m opd recipes
python -m opd figures
```

The compact fixture `tests/fixtures/recipes_before_unification.json.gz` retains the pre-refactor recipes. Tests compare every resolved assignment and backend against it, test override precedence and actual Hydra parsing, and retain the numerical full-logit and avg@4 tests. These CPU checks do not certify end-to-end GPU retraining. See `REPRODUCIBILITY.md` for scientific and data-recovery limitations.
