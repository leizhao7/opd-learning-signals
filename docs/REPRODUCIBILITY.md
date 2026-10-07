# Reproduction and scientific scope

## Validation done during packaging

- Main Figure 1, learning-signal phase/time-series plots, occupancy/gradient/CKA appendix plots, IF dynamics, SFT/continuation plots, and support/rollout plots regenerate from committed numerical inputs.
- All 11 training recipes render without unresolved substitutions or embedded original cluster paths.
- The full-logit chunked CPU kernel matches a dense reference in scalar value and student hidden-state/weight gradients for multiple chunk sizes and unequal student/teacher vocabulary sizes.
- Code avg@4 is independently recomputed from all 21 checkpoints' archived eight-response reward records. Selected response ordinals match the original reward-independent SHA256 rule, and all macro accuracies match Figure 1's input CSV.
- Python syntax and the release's credential-pattern checks are performed. These are not a guarantee against all possible sensitive content.

## Rate and loss conventions

`analysis/learning_signal.py` maps archived occupancy columns into current manuscript notation. Compute mu from `g2_crossfit / (2 * F_top16)`; do not use the CSV column named `mu_crossfit` as current mu. That column used an earlier notation for gamma. Alpha is `X_dot / g2_crossfit`, gamma is `(g2_crossfit + X_dot) / (2 * F_top16)`.

The training-log proxy uses a squared stochastic, pre-clipping gradient norm divided by twice the logged loss. It includes sampling variance and is not an AdamW update norm. Cross-fitting separates rollout folds on a shared prompt sample and does not automatically remove prompt-sampling uncertainty. Nonpositive cross-fitted g2 estimates must not be silently clipped to positive values. The rate calculation requires positive loss; the IF top-16 surrogate can be signed and should not be interpreted as a nonnegative full KL.

Figure inputs preserve raw loss and use smoothing only for visual trends. Maximum reduction uses the lowest recorded loss, whereas final reduction uses step 200; these are different statistics. Recoverability uses the base student's accuracy, not the post-SFT starting value.

## Measurement utilities

The geometry code was recovered from the experiment storage. `build_input_banks.py` accepts optional `banks` in `experiment_contract.json`; `compute_cka.py` accepts `representation_pairs` and per-pair `bank`, allowing the earlier fixed mapping to be replaced explicitly. Copy `configs/geometry.example.json` into an analysis directory as `experiment_contract.json`, replace model/data paths, build input banks, extract both checkpoints' representations, then run the CKA and weight-metric tools with `--root` pointing to that directory.

The suffix-gradient consistency probe in `analysis/diagnostics` uses the parameter subspace described in its header. It is supplementary source, not a replacement for the all-parameter g/h measurement. The original standalone occupancy measurement driver was not found in the inspected cluster locations; the recorded occupancy CSVs, their formulas and plotting code are available.

## Changes made for portability

No OPD loss or optimizer implementation was intentionally changed. Launch recipes replace storage/model/data paths with CLI inputs; W&B is disabled by default; plotting during training is disabled; cluster watchdogs, GPU reservations, transfers and credential-loading code are excluded. The full-logit checkpoint storage hook now checks free space without requiring a particular cluster directory name. The SFT response worker uses caller-managed GPU scheduling instead of the cluster reservation protocol; filtering logic is preserved. Figure model names no longer depend on reading the paper's LaTeX source. Geometry bank/pair mappings can be supplied through the existing experiment contract.

The recovered source trees are dated snapshots, not verified upstream commits. Hash manifests identify included files. Reconstructed recipes are marked in their JSON files and experiment map. A newly trained checkpoint is not expected to match the archived curves bit-for-bit, particularly across hardware, checkpoint resumes, sampler state and library versions.
