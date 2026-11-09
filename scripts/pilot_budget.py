"""Budget pilot (decision D4): where do the methods stop improving?

    python scripts/pilot_budget.py                         # CMA-ES and L-BFGS-B, 3 seeds, up to 100k (~3 min)
    python scripts/pilot_budget.py --methods cmaes ga lbfgs --seeds 5

Runs each method once per (seed, n) at the largest budget and reads the
smaller budgets from the checkpoint curve: a run cut at B evaluations returns
exactly what a run with budget B would return, because no method uses the
budget to set its parameters.

Prints the median (over seeds) training MSE and clean test MSE at each budget.
"""

import argparse
import copy
import statistics
from pathlib import Path

from evonn.budget import default_checkpoints
from evonn.experiment import load_config, run_cell
from evonn.methods import LABELS

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=ROOT / "configs" / "default.yaml", type=Path)
    parser.add_argument("--methods", nargs="+", default=["cmaes", "lbfgs"])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--budgets", type=int, nargs="+", default=[10_000, 20_000, 50_000, 100_000])
    args = parser.parse_args()

    budgets = sorted(args.budgets)
    off_grid = set(budgets) - set(default_checkpoints(budgets[-1]))
    if off_grid:
        parser.error(f"budgets must be on the 1-2-5 checkpoint grid (e.g. 10000, 20000, 50000): {sorted(off_grid)}")
    cfg = copy.deepcopy(load_config(args.config))
    cfg["budget"]["max_evals"] = budgets[-1]
    sizes = cfg["data"]["n_train"]

    results = {}                                          # (n, method) -> list of {budget: (train, clean)}
    for seed in range(args.seeds):
        for n in sizes:
            for method in args.methods:
                r = run_cell(cfg, method, n, seed)
                curve = {row["evals"]: row for row in r["curve"]}
                results.setdefault((n, method), []).append(
                    {b: (curve[b]["train_mse"], curve[b]["test_mse_clean"]) for b in budgets})
                print(f"seed {seed}  n={n:<5} {LABELS[method]:<9} {r['budget']['wall_time']:5.1f}s", flush=True)

    print(f"\nmedian over {args.seeds} seeds: train MSE (clean test MSE)")
    print(f"{'n':>5} {'method':<9} " + " ".join(f"{b:>17}" for b in budgets))
    for (n, method), runs in sorted(results.items()):
        cells = []
        for b in budgets:
            train = statistics.median(r[b][0] for r in runs)
            clean = statistics.median(r[b][1] for r in runs)
            cells.append(f"{train:.4f} ({clean:.4f})")
        print(f"{n:5d} {LABELS[method]:<9} " + " ".join(f"{c:>17}" for c in cells))


if __name__ == "__main__":
    main()
