<div align="center">

<h1>Why On-Policy Distillation Sometimes Fails:<br>Vanishing Learning Signals</h1>

<p>
Lei Zhao<sup>1</sup> · Qichao Zhao<sup>2</sup> · Bowen Zuo<sup>3</sup> · Qishi Zhan<sup>4</sup>
</p>
<p>
<sup>1</sup>University of Pennsylvania &nbsp; <sup>2</sup>Tsinghua University<br>
<sup>3</sup>University of California, Riverside &nbsp; <sup>4</sup>Marquette University
</p>

<p>
<a href="#results">Results</a> &nbsp;•&nbsp;
<a href="#quick-start">Quick start</a> &nbsp;•&nbsp;
<a href="docs/EXPERIMENTS.md">Experiments</a> &nbsp;•&nbsp;
<a href="docs/REPRODUCIBILITY.md">Reproducibility</a> &nbsp;•&nbsp;
<a href="#citation">Citation</a>
</p>

</div>

**Why can distillation stall while the student still differs substantially from its teacher?** We study on-policy distillation (OPD) across code generation and mathematical reasoning, combining training diagnostics with a continuous-time analysis. This repository contains the training recipes, analysis tools, and numerical inputs for the paper's figures.

## Results

<a href="docs/assets/learning-signals.png">
  <img src="docs/assets/learning-signals.png" width="100%" alt="Learning-signal proxy versus remaining loss across five Code and R1 Math runs. Each point is a training update, and open circles mark step 1.">
</a>

*Learning-signal proxy versus remaining OPD loss in code and math.*

| Teacher group | Mean final loss reduction (200 updates) |
|:---|---:|
| Larger-scale teachers | **25.1%** |
| Self-RL teachers | **96.2%** |

Self-RL teachers are obtained by further RL training of the initial student.

<details>
<summary>Reading the plot and result details</summary>

The plot shows five runs across two students. Each dot is one training update; open circles mark the first update. Moving right means less loss remains relative to the first update. Moving down means a smaller gradient-based learning-signal proxy.

The key question is how much loss remains as the signal weakens. Self-RL runs reach low loss, while the larger-scale-teacher runs shown here retain substantial loss.

The table summarizes all seven main runs: five with larger-scale teachers and two with self-RL teachers. It averages each run's `(1 − loss_final / loss_start) × 100%` using raw endpoints after 200 updates. This differs from maximum loss reduction, which uses the lowest recorded loss. Qwen3-4B-RL-Math is a larger-scale teacher for the Qwen3-1.7B student, rather than a self-RL teacher.

These are descriptive results for the runs studied, not a controlled estimate of the effect of teacher size or initialization.

[Full training curves](docs/assets/teacher-trainability.png) · [Evaluation protocol](docs/EXPERIMENTS.md) · [Metric definitions and limitations](docs/REPRODUCIBILITY.md#rate-and-loss-conventions) · [Source data](paper/figures/data/teacher_recoverability_metrics.csv)

</details>

## What we find

- **A recurring plateau.** In the larger-scale-teacher runs studied, loss reduction stalls while substantial teacher–student differences remain. Self-RL teachers support much greater loss reduction and improved validation accuracy.
- **A diagnostic view of learning.** We track gradient strength relative to the remaining loss and separate it from the effect of changing rollout distributions. The idealized flow relates loss decay to both quantities.
- **A local recovery guarantee.** For sufficiently nearby teachers in a shared parameterization, the theory bounds loss by an exponentially decaying term plus a residual under stated regularity conditions over a local time interval.

## Quick start

### Recreate the figures on CPU

```bash
git clone https://github.com/leizhao7/opd-learning-signals.git
cd opd-learning-signals
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-analysis.txt
python -m opd figures
```

Plots are written under `paper/figures/`, using the committed numerical inputs. No model download or GPU is needed. To rebuild the overview plot, use `python -m opd figures --group signal`.

### Run an OPD experiment

Prepare the GPU environment and model/data checkpoints using the [installation guide](docs/INSTALL.md). Preview a launch with:

```bash
python -m opd train --recipe qwen_math \
  --student /models/Qwen3-1.7B --teacher /models/Qwen3-4B \
  --train-data /data/dapo-math-17k.parquet \
  --val-data /data/AIME24.parquet /data/AIME25.parquet /data/AMC23.parquet \
  --output /results/qwen-opd --name qwen17-qwen4
```

Add `--execute` to start training. Use `python -m opd recipes` to list all 11 recipes. Each executed launch saves its resolved configuration and backend in the output directory. Run commands from the repository root.

<details>
<summary>CPU checks and configuration overrides</summary>

```bash
pip install -r requirements-test.txt
# Install CPU PyTorch to include the full-logit value/gradient check.
pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m unittest discover -s tests -v
```

Configuration precedence is **shared settings → recipe → CLI overrides**. For example, `--override trainer.total_training_steps=200` overrides the training horizon. Logging defaults to console. Choose one veRL snapshot at a time as described in the installation guide. The launchers do not allocate cluster resources automatically.

See the [organization guide](docs/ORGANIZATION.md) for all unified commands and extension rules, or the [CPU workflow](.github/workflows/checks.yml) for CI checks.

</details>

## Find your experiment

| Goal | Start here |
|:---|:---|
| Reproduce Code and Math OPD | [11 training recipes](configs/recipes/) · [experiment map](docs/EXPERIMENTS.md) |
| Compare top-16 / full-logit and rollout counts | [Ablation figures](paper/figures/ablations/support/) · [recipe provenance](docs/EXPERIMENTS.md) |
| Run SFT → OPD | [SFT setup](docs/INSTALL.md#sft) · [`sft_opd_qwen`](configs/recipes/sft_opd_qwen.json) · [`sft_opd_r1`](configs/recipes/sft_opd_r1.json) |
| Analyze learning signals and occupancy | [Signal analysis](analysis/learning_signal.py) · [definitions](docs/REPRODUCIBILITY.md#rate-and-loss-conventions) |
| Measure parameter changes and CKA | [Geometry tools](analysis/geometry/) · [example configuration](configs/geometry.example.json) |
| Audit Code avg@4 | [Evaluation audit](evaluation/audit_code_avg4.py) · [protocol](docs/EXPERIMENTS.md#important-measurement-distinctions) |

## Reproduction status

CPU figure reconstruction, numerical checks, and dry-run launchers have been verified. Full GPU retraining and all external evaluation dependencies have **not** been rerun from this package. The experiments use a fixed training horizon and do not establish that the observed plateaus persist under every training configuration.

<details>
<summary>Included artifacts and known gaps</summary>

- Included: training snapshots, portable launchers, evaluation/geometry utilities, audited plotting inputs, and [source hashes](provenance/).
- Models, training datasets, raw benchmark responses, and cluster orchestration are not bundled. Some recipes are reconstructed convenience launchers; the [experiment map](docs/EXPERIMENTS.md) labels their provenance.
- The standalone all-parameter occupancy measurement driver was not recovered. Recorded occupancy values and reanalysis tools are included; the supplementary suffix-gradient probe is not a replacement.
- The historical instruction-following protocol has training/evaluation source overlap; its results do not establish held-out generalization.
- Small parameter changes and high representation similarity are observations. Their role in causing signal collapse remains a hypothesis. The local theorem concerns an idealized flow and nearby teachers with shared parameterization.

Read the [reproduction notes](docs/REPRODUCIBILITY.md) before interpreting or extending these diagnostics.

</details>

## Citation

The manuscript is in preparation. Until a public paper identifier is available:

```bibtex
@unpublished{zhao2026opd,
  title  = {Why On-Policy Distillation Sometimes Fails: Vanishing Learning Signals},
  author = {Zhao, Lei and Zhao, Qichao and Zuo, Bowen and Zhan, Qishi},
  year   = {2026},
  note   = {Manuscript in preparation}
}
```

## Acknowledgments and license

The authors' original code is licensed under the [MIT License](LICENSE).

This work builds on the third-party projects preserved under [`vendor/`](vendor/), including veRL and LLaMA-Factory. Third-party components retain their original licenses and copyright notices; see [third-party notices](THIRD_PARTY.md).

Contact: [Lei Zhao](mailto:leizhao7@upenn.edu).
