# veRL backends: one base tree plus overlays

The experiments used four veRL 0.7.0.dev trees. They are almost identical, so the repository stores one full tree and the few files that differ:

| Backend | Stored as | Used by |
|---|---|---|
| `verl-original` | `vendor/verl-original/` (full tree) | `code_14b` |
| `verl-code` | base + `overlays/verl-code/` | `code_rl` |
| `verl-qwen` | base + `overlays/verl-qwen/` | `qwen_math`, `sft_opd_qwen` |
| `verl-full` | base + `overlays/verl-full/` | `qwen_full`, `r1_full`, `r1_top16`, `r1_top16_n8`, `qwen_top16_n8`, `if_top16_n8`, `sft_opd_r1` |

`python -m opd train` assembles the selected backend under `vendor/.build/` on first use and checks every file against [`provenance/frozen_vendor_sha256.json`](../../provenance/frozen_vendor_sha256.json). To install or inspect a backend directly:

```bash
python -m opd backend verl-full    # prints the assembled path
pip install -e "$(python -m opd backend verl-full)"
```

## Where the OPD implementation lives

The top-k OPD algorithm is part of the shared base, not an overlay:

- `verl/trainer/ppo/core_algos.py`: the `token_reward_direct` advantage estimator, which uses the per-token distillation reward directly as the advantage, including the `(batch, seq, k)` top-k case.
- `verl/workers/actor/dp_actor.py`: `compute_distillation_reward` and the top-k student/teacher log-probability handling.
- `verl/workers/fsdp_workers.py` and `verl/trainer/ppo/ray_trainer.py`: teacher scoring and the wiring of `log_prob_top_k` through the training loop.
- `verl/workers/config/rollout.py`: the `log_prob_top_k` rollout option.

## What each overlay changes

**`verl-qwen`** changes reporting only.
- `verl/utils/tracking.py` adds the aliases `opd/loss_token_mean` and `math/mean@8`. It also lets the file logger serialize tensors and other non-JSON values instead of failing.

**`verl-full`** adds exact full-vocabulary OPD.
- `verl/workers/full_logits.py` is new. It computes reverse KL over the whole vocabulary, chunked over states. It is active only when `log_prob_top_k` equals the Qwen vocabulary size (151936). It supports only the tested setting: one rollout per prompt, one minibatch, one epoch.
- `verl/workers/actor/dp_actor.py` and `verl/workers/fsdp_workers.py` add early-return hooks into `full_logits` for the reward, log-probability, policy-update and teacher-scoring paths. Top-k runs leave these hooks inactive.
- `verl/trainer/ppo/ray_trainer.py` calls `training/runtime/checkpoint_storage.py` before each save. This refuses to checkpoint when free disk is below `OPD_MIN_FREE_GB` (default 40).
- `verl/utils/tracking.py` makes the same changes as `verl-qwen`. It also moves full-support aggregate statistics into their own `full_support_aggregate/` namespace, and logs the loss as `distill/reverse_kl_full_vocab`.

**`verl-code`** isolates code-reward scoring.
- `verl/workers/reward_manager/prime.py` runs code-reward scoring in an isolated subprocess (`_prime_isolated_runner.py`). It replaces a `ProcessPoolExecutor` that occasionally deadlocked after validation and left zombie processes. Scoring behavior is preserved: same concurrency, FIFO order, 300-second batch deadline and score conversion.
- `test_prime_isolation.py` is the stress test used for that change. Set `OPD_ROOT` to a directory containing `datasets/bench_val/`.

## Changes from the recovered snapshots

- The recovered `verl-code` tree was missing `verl/models/transformers/`, `verl/third_party/` and `verl/utils/megatron/`. `verl/workers/fsdp_workers.py` imports the first two at module load, so the `code_rl` recipe could not start. The assembled `verl-code` now takes these 27 files from the base. They are unmodified veRL files, byte-identical in the other three snapshots.
- `test_prime_isolation.py` previously hard-coded a cluster path. It now reads `OPD_ROOT`.

No algorithm, reward, optimizer or training-loop file changed. `tests/test_unification.py` checks every assembled file against the manifest.
