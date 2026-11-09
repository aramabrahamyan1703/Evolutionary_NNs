"""Phase 3 checks on a folder of runs: has every method levelled off, how many weights sit at the bounds, how long does a run take.

    python scripts/check_runs.py results/raw/check/core_ntrain_sweep

For every (method, n): medians over seeds of the training MSE at half the budget used
and at the end, the relative drop over the second half (small = plateaued),
test MSE, share of weights at the ±bound, and wall time per run.
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from evonn.methods import LABELS


def at_evals(curve, evals):
    """Curve row with the largest checkpoint <= evals."""
    return max((r for r in curve if r["evals"] <= evals), key=lambda r: r["evals"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()

    groups = defaultdict(list)
    for p in sorted(args.folder.glob("*.json")):
        r = json.loads(p.read_text())
        groups[(r["meta"]["method"], r["meta"]["n_train"])].append(r)
    if not groups:
        parser.error(f"no results in {args.folder}")

    med = statistics.median
    print(f"{'method':<9} {'n':>5} {'runs':>4} {'train@½':>8} {'train':>8} {'drop½':>7} "
          f"{'test':>8} {'@bounds':>8} {'time[s]':>8}")
    for (m, n) in sorted(groups, key=lambda k: (list(LABELS).index(k[0]), k[1])):
        runs = groups[(m, n)]
        half = [at_evals(r["curve"], r["budget"]["evals_used"] / 2)["train_mse"] for r in runs]
        end = [r["final"]["train_mse"] for r in runs]
        drop = [(h - e) / e for h, e in zip(half, end)]
        print(f"{LABELS[m]:<9} {n:>5} {len(runs):>4} {med(half):8.4f} {med(end):8.4f} "
              f"{(f'{med(drop):7.1%}' if med(end) > 1e-3 else '      -')} "
              f"{med(r['final']['test_mse'] for r in runs):8.4f} "
              f"{med(r['final']['frac_at_bounds'] for r in runs):8.1%} "
              f"{med(r['budget']['wall_time'] for r in runs):8.1f}")
    print(f"\nnoise floor {runs[0]['final']['noise_floor']:.4f};  drop½ = (train@½ - train) / train, "
          "the relative improvement over the second half of the budget (- if train < 1e-3)")


if __name__ == "__main__":
    main()
