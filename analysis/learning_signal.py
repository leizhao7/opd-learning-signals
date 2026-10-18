"""Interpret the archived diagnostic columns using the manuscript's notation.

In the idealized OPD flow the loss obeys

    dL/dt = -2 * mu * (1 + alpha) * L = -2 * gamma * L,

where
    mu    = ||g||^2 / (2 L)    gradient strength relative to the remaining loss,
    alpha = X_dot / ||g||^2    how the moving rollout distribution reinforces (> 0)
                               or offsets (< 0) the gradient step,
    gamma = mu * (1 + alpha) = (||g||^2 + X_dot) / (2 L).

The archived occupancy CSVs store L as `F_top16`, the cross-fitted ||g||^2 as
`g2_crossfit` and the occupancy term as `X_dot`. Their column `mu_crossfit` uses an
earlier notation for gamma and must not be read as mu.
"""
import argparse
import csv
import json
from pathlib import Path


def crossfit_statistics(loss, g2, cross):
    """Return mu, alpha and gamma for one update.

    Cross-fitted g2 estimates can be nonpositive; they are reported as-is and
    flagged rather than clipped.
    """
    if loss <= 0:
        raise ValueError('The full-KL rate factorization requires positive loss')
    return {
        'mu': g2 / (2 * loss),
        'alpha': cross / g2 if g2 != 0 else None,
        'gamma': (g2 + cross) / (2 * loss),
        'positive_g2': g2 > 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('csv', type=Path, help='occupancy CSV with teacher, step, F_top16, g2_crossfit, X_dot')
    args = parser.parse_args()
    rows = []
    with args.csv.open() as stream:
        for row in csv.DictReader(stream):
            rows.append({'teacher': row['teacher'], 'step': int(row['step']),
                         **crossfit_statistics(float(row['F_top16']), float(row['g2_crossfit']), float(row['X_dot']))})
    print(json.dumps(rows, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
