"""Smoke test: all methods on one training set, at the same evaluation budget.

    python scripts/smoke_test.py                       # seed 0, n = 100, config defaults
    python scripts/smoke_test.py --n 25 --seed 3 --max-evals 50000 --methods cmaes lbfgs

Runs evonn.experiment.run_cell for each method (same code path as the real
experiment, nothing saved to results/), prints a table and saves a convergence
figure to figures/.
"""

import argparse
import copy
from pathlib import Path

import matplotlib.pyplot as plt

from evonn.experiment import load_config, run_cell
from evonn.methods import LABELS, STYLES

ROOT = Path(__file__).resolve().parents[1]


def plot_convergence(curves, noise_floor, path):
    """Best-so-far train MSE and test MSE against evaluations, one line per method."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, key, title in zip(axes, ["train_mse", "test_mse"], ["(a) Training MSE", "(b) Test MSE"]):
        for method, rows in curves.items():
            ax.plot([r["evals"] for r in rows], [r[key] for r in rows], linewidth=2,
                    markersize=4, label=LABELS[method], **STYLES[method])
        ax.axhline(noise_floor, color="#52514e", linewidth=1, linestyle=":")
        ax.annotate(f"noise floor σ² = {noise_floor:.2f}", (0, noise_floor), xycoords=("axes fraction", "data"),
                    xytext=(4, -4), textcoords="offset points", ha="left", va="top",
                    color="#52514e", fontsize=9)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("function evaluations")
        ax.set_title(title, loc="left")
        ax.grid(True, which="major", color="#e5e4df", linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("MSE (best-so-far network)")
    axes[0].legend(frameon=False)
    plt.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=ROOT / "configs" / "default.yaml", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--max-evals", type=int, default=None, help="override budget.max_evals")
    parser.add_argument("--methods", nargs="+", help="default: experiment.methods in the config")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    cfg = copy.deepcopy(load_config(args.config))
    if args.max_evals:
        cfg["budget"]["max_evals"] = args.max_evals
    methods = args.methods or cfg["experiment"]["methods"]

    print(f"seed={args.seed}  n_train={args.n}  budget={cfg['budget']['max_evals']} evals  "
          f"grad_cost={cfg['budget']['grad_cost']}")
    header = (f"{'method':<9} {'train':>8} {'test':>8} {'clean':>8} {'evals':>7} "
              f"{'calls':>7} {'starts':>6} {'time[s]':>8} {'@bounds':>8}")
    print(header)
    print("-" * len(header))

    curves = {}
    for method in methods:
        r = run_cell(cfg, method, args.n, args.seed)
        f, b = r["final"], r["budget"]
        curves[method] = r["curve"]
        print(f"{LABELS[method]:<9} {f['train_mse']:8.4f} {f['test_mse']:8.4f} {f['test_mse_clean']:8.4f} "
              f"{b['evals_used']:7.0f} {b['n_calls']:7d} {b['starts']:6d} {b['wall_time']:8.2f} "
              f"{f['frac_at_bounds']:8.1%}")

    noise_floor = r["final"]["noise_floor"]
    print(f"\nnoise floor sigma^2 = {noise_floor:.4f}   "
          "(train/test = noisy MSE, clean = MSE against the noise-free f)")

    if not args.no_plot:
        path = ROOT / "figures" / f"smoke_convergence_seed{args.seed}_n{args.n}.png"
        plot_convergence(curves, noise_floor, path)
        print(f"figure: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
