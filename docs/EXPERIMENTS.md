# Paper-to-code map

| Experiment | Recipe / location | Provenance |
|---|---|---|
| Code, Qwen3-4B → self-RL Code teacher | `code_rl` | Exported 200-step reproduction launcher; veRL evaluator-isolation variant |
| Code, Qwen3-4B → Qwen3-14B | `code_14b` | Reconstructed convenience recipe, four GPUs and native four-response evaluation; historical numerical outputs are separate |
| Math, Qwen3-1.7B → Qwen3-4B or RL-Math | `qwen_math` | Exported two-teacher launcher; select teacher checkpoint explicitly |
| Math, R1-Distill-1.5B → JustRL / Skywork / R1-7B | `r1_top16` | Reconstructed from the recovered full-logit launcher with K=16; historical runtime identity is not asserted |
| R1 full vocabulary | `r1_full` | Exported full-logit launcher; teacher normalized before slicing to aligned student vocabulary |
| Qwen full vocabulary | `qwen_full` | Reconstructed convenience recipe using the full-logit implementation |
| Eight rollouts, Qwen / R1 | `qwen_top16_n8`, `r1_top16_n8` | Exported ablation launchers |
| SFT → OPD | `sft_opd_qwen`, `sft_opd_r1` | Exported continuation launchers; pass the final SFT checkpoint as student |
| Instruction following | `if_top16_n8`, `evaluation/if/` | Exported launcher and evaluation tools; read the IF scope caveat |
| SFT | `training/sft.py`, `configs/sft_*.json`, `vendor/llamafactory/` | Recovered training entry and configs; replace path placeholders before torchrun |
| Parameter changes / CKA | `analysis/geometry/` | Recovered measurement code; bank/model mappings configurable |
| Learning-signal plots | `paper/figures/analysis/` | Uses audited 200-step numerical logs; no stale 140-step curves |

## Expected results

Reference values from the paper's runs after 200 updates. Use them to check whether a rerun behaves like the original. A rerun is not expected to match bit for bit; see [REPRODUCIBILITY.md](REPRODUCIBILITY.md#changes-made-for-portability). Accuracy is Code avg@4 or macro Math avg@8, from the base student to the final checkpoint. Recipes marked *reconstructed* in the table above did not produce these runs directly.

| Recipe | Teacher | Max / final loss reduction | Accuracy |
|---|---|---:|:---:|
| `code_rl` | Qwen3-4B-RL-Code | 99.5% / 99.4% | 52.7 → 60.4 |
| `code_14b` | Qwen3-14B | 40.5% / 25.9% | 52.7 → 53.4 |
| `qwen_math` | Qwen3-4B-RL-Math | 65.1% / 55.4% | 21.1 → 43.9 |
| `qwen_math` | Qwen3-4B | 30.4% / 6.2% | 20.7 → 22.7 |
| `r1_top16` | JustRL-1.5B | 96.3% / 92.9% | 39.9 → 51.9 |
| `r1_top16` | Skywork-7B | 37.7% / 23.9% | 39.9 → 46.1 |
| `r1_top16` | R1-Distill-7B | 34.3% / 13.9% | 39.9 → 42.7 |
| `qwen_full` | Qwen3-4B-RL-Math | 62.7% / 54.9% | 20.5 → 46.4 |
| `qwen_full` | Qwen3-4B | 37.5% / 11.6% | 21.7 → 22.9 |
| `r1_full` | JustRL-1.5B | 96.2% / 91.8% | 39.9 → 50.5 |
| `r1_full` | R1-Distill-7B | 30.8% / −7.0% | 39.9 → 40.3 |
| `qwen_top16_n8` | Qwen3-4B-RL-Math | 60.6% / 51.3% | 22.9 → 45.2 |
| `qwen_top16_n8` | Qwen3-4B | 33.0% / 9.9% | 21.9 → 24.9 |
| `r1_top16_n8` | JustRL-1.5B | 96.2% / 31.1% | 39.9 → 51.9 |
| `r1_top16_n8` | R1-Distill-7B | 35.0% / 18.3% | 40.4 → 38.9 |

Sources: [`teacher_recoverability_metrics.csv`](../paper/figures/data/teacher_recoverability_metrics.csv) for the main runs and [`support_rollout_metrics.json`](../paper/figures/data/appendix_support_rollouts/support_rollout_metrics.json) for the full-vocabulary and eight-rollout runs. The final loss of `r1_top16_n8` with JustRL rises late in training; see [`late_loss_rise_evidence.json`](../paper/figures/data/appendix_support_rollouts/late_loss_rise_evidence.json). For the SFT → OPD and instruction-following recipes, compare against the curves in [`appendix_sft/`](../paper/figures/data/appendix_sft/) and [`appendix_if/`](../paper/figures/data/appendix_if/).

## Important measurement distinctions

- `code_rl` generated eight validation responses per problem. The paper's avg@4 uses a fixed reward-independent four-of-eight selection at every checkpoint, not the native avg@8 log value. The committed plotting CSV already contains the audited avg@4 values.
- Math accuracy is macro-averaged avg@8 across AIME24, AIME25 and AMC23. Accuracy gain and gap recovery use the base student's evaluation as the reference; SFT continuation has its own separately reported starting evaluation.
- The code RL curve is a completed reproduction, not a splice of the old 149-step run. Shared step-zero evaluations and checkpoint resumes are described in the paper.
- The full-logit branch is specialized to the tested on-policy, one-minibatch, one-epoch setup and aligned vocabulary. It is not a generic arbitrary-vocabulary distiller.
- IF training prompts overlap the evaluated Multi-IF conversation source in the historical protocol. Reproducing this protocol does not establish held-out generalization.
- SFT datasets and their exact processing/cache artifacts are not distributed here. The recovered rollout utility and final training configs are included; recreating the exact dataset requires the source datasets and preprocessing protocol in the paper.
