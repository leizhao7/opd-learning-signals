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

## Important measurement distinctions

- `code_rl` generated eight validation responses per problem. The paper's avg@4 uses a fixed reward-independent four-of-eight selection at every checkpoint, not the native avg@8 log value. The committed plotting CSV already contains the audited avg@4 values.
- Math accuracy is macro-averaged avg@8 across AIME24, AIME25 and AMC23. Accuracy gain and gap recovery use the base student's evaluation as the reference; SFT continuation has its own separately reported starting evaluation.
- The code RL curve is a completed reproduction, not a splice of the old 149-step run. Shared step-zero evaluations and checkpoint resumes are described in the paper.
- The full-logit branch is specialized to the tested on-policy, one-minibatch, one-epoch setup and aligned vocabulary. It is not a generic arbitrary-vocabulary distiller.
- IF training prompts overlap the evaluated Multi-IF conversation source in the historical protocol. Reproducing this protocol does not establish held-out generalization.
- SFT datasets and their exact processing/cache artifacts are not distributed here. The recovered rollout utility and final training configs are included; recreating the exact dataset requires the source datasets and preprocessing protocol in the paper.
