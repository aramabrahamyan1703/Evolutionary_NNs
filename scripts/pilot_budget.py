"""Budget pilot (decision D4): where do CMA-ES and L-BFGS-B stop improving?

    python scripts/pilot_budget.py                 # 3 seeds, 4 sizes, up to 100k evals (~3 min)

Runs both smoke-test optimisers once per (seed, n) at the largest budget and
reads the smaller budgets from the checkpoint snapshots: a run truncated at
B evaluations returns exactly what a run with budget B would return, because
neither method uses the budget to set its parameters.

Prints the median (over seeds) training MSE and clean test MSE at each budget.
"""

import argparse
import importlib.util
import statistics
from pathlib import Path

from evonn.budget import CountedObjective
from evonn.data import generate_datasets, TRAIN_SIZES
from evonn.metrics import evaluate_snapshots
from evonn.mlp import MLP

ROOT = Path(__file__).resolve().parents[1]

# reuse the optimisers from the smoke test (moved to ea/ and baselines/ in Phase 2)
_spec = importlib.util.spec_from_file_location("smoke_test", ROOT / "scripts" / "smoke_test.py")
smoke = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(smoke)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--budgets", type=int, nargs="+", default=[10_000, 20_000, 50_000, 100_000])
    args = parser.parse_args()

    budgets = sorted(args.budgets)
    model = MLP()
    results = {}                                          # (n, method) -> list of {budget: (train, clean)}
    for seed in range(args.seeds):
        data = generate_datasets(seed)
        x0 = model.init(seed)
        for n in TRAIN_SIZES:
            X, y = data["train"][n]
            for name in ("CMA-ES", "L-BFGS-B"):
                obj = CountedObjective(model, X, y, max_evals=budgets[-1], grad_cost=2, checkpoints=budgets)
                if name == "CMA-ES":
                    smoke.run_cmaes(model, obj, x0, seed, 0.5, [-5, 5], 0.5)
                else:
                    smoke.run_lbfgs(model, obj, x0, seed, [-5, 5], 0.5, 10)
                rows = evaluate_snapshots(model, obj.snapshots, data, n)
                results.setdefault((n, name), []).append(
                    {r["evals"]: (r["train_mse"], r["test_mse_clean"]) for r in rows})
                print(f"seed {seed}  n={n:<5} {name:<9} {obj.elapsed:5.1f}s", flush=True)

    print(f"\nmedian over {args.seeds} seeds: train MSE (clean test MSE)")
    print(f"{'n':>5} {'method':<9} " + " ".join(f"{b:>17}" for b in budgets))
    for (n, name), runs in sorted(results.items()):
        cells = []
        for b in budgets:
            train = statistics.median(r[b][0] for r in runs)
            clean = statistics.median(r[b][1] for r in runs)
            cells.append(f"{train:.4f} ({clean:.4f})")
        print(f"{n:5d} {name:<9} " + " ".join(f"{c:>17}" for c in cells))


if __name__ == "__main__":
    main()
