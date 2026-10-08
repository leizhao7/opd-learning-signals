# README figures

These images are copies of the paper's figures, rendered from the committed numerical inputs. They are tracked here because generated images under `paper/figures/` are intentionally ignored by Git.

From the repository root, with `requirements-analysis.txt` installed:

```bash
python -m opd figures --group main
python -m opd figures --group signal
cp paper/figures/motivation/teacher_trainability_multigroup.png docs/assets/teacher-trainability.png
cp paper/figures/analysis/signal_collapse_phase.png docs/assets/learning-signals.png
```

| Image | Builder | Inputs |
|---|---|---|
| `teacher-trainability.png` | [`build_teacher_trainability.py`](../../paper/figures/build_teacher_trainability.py) | [`opd_teacher_trainability_multigroup.csv`](../../paper/figures/data/opd_teacher_trainability_multigroup.csv), audited summary in the same directory |
| `learning-signals.png` | [`plot_signal_collapse_phase.py`](../../paper/figures/analysis/plot_signal_collapse_phase.py) | [`code_signal_collapse_training.csv`](../../paper/figures/data/code_signal_collapse_training.csv), [`math_signal_collapse_training.csv`](../../paper/figures/data/math_signal_collapse_training.csv) |

The README overview is the paper's Figure 2 (`learning-signals.png`), which uses raw per-step values from five runs. The full training curves, Figure 1 (`teacher-trainability.png`), are in a collapsible section and use centered 9-step rolling medians with faint raw losses. Neither image introduces new experiments or measurements. The README's aggregate final loss reductions cover all seven main runs and are computed from [`teacher_recoverability_metrics.csv`](../../paper/figures/data/teacher_recoverability_metrics.csv): self-RL is `rl4b` and `justrl1p5b`; the other five runs form the larger-scale group.
