"""Rebuild paper figures from audited numerical inputs."""
import argparse
import subprocess
import sys

from .paths import ROOT

GROUPS = {
    'main': ['figures/build_teacher_trainability.py'],
    'signal': ['figures/analysis/plot_signal_collapse_phase.py', 'figures/analysis/plot_signal_collapse.py'],
    'if': ['figures/analysis/plot_if_dynamics.py'],
    'diagnostics': ['figures/appendix/build_diagnostics.py'],
    'sft': ['figures/ablations/build_sft_figures.py'],
    'rollouts': ['figures/ablations/support/plot_support_rollouts.py',
                 'figures/ablations/support/build_support_rollout_table.py'],
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', choices=['all', *GROUPS], default='all')
    args = parser.parse_args(argv)
    groups = GROUPS if args.group == 'all' else {args.group: GROUPS[args.group]}
    for scripts in groups.values():
        for script in scripts:
            subprocess.run([sys.executable, str(ROOT / 'paper' / script)], cwd=ROOT / 'paper', check=True)
