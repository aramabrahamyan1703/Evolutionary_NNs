"""Smoke test: CMA-ES vs L-BFGS-B on one training set, at the same evaluation budget.

    python scripts/smoke_test.py                       # seed 0, n = 100, config defaults
    python scripts/smoke_test.py --n 25 --seed 3 --max-evals 50000

Both methods start from the same initial weights, are bounded to the weight
bounds, and call the same CountedObjective until the budget runs out. When a
method converges early it restarts from new random weights, so neither leaves
budget unused. The returned network is the best training loss seen (D6).

The optimisers are inline here on purpose; Phase 2 moves them into ea/ and
baselines/.
"""

import argparse
from pathlib import Path

import cma
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.optimize import minimize

from evonn.budget import CountedObjective, BudgetExhausted
from evonn.data import generate_datasets
from evonn.metrics import evaluate, evaluate_snapshots
from evonn.mlp import MLP
from evonn.seeding import rng, OPTIMISER

ROOT = Path(__file__).resolve().parents[1]


def run_cmaes(model, obj, x0, seed, sigma0, bounds, init_scale):
    """Bounded CMA-ES with restarts, one obj.loss call per candidate. Returns #generations, #restarts."""
    opt_rng = rng(seed, OPTIMISER)
    generations, restarts = 0, 0
    try:
        while True:
            opts = {"bounds": list(bounds), "verbose": -9,
                    "seed": int(opt_rng.integers(1, 2**31 - 1))}
            es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
            while not es.stop():
                candidates = es.ask()
                es.tell(candidates, [obj.loss(x) for x in candidates])
                generations += 1
            restarts += 1
            x0 = opt_rng.uniform(-init_scale, init_scale, size=model.n_params)
    except BudgetExhausted:
        pass
    return {"generations": generations, "restarts": restarts}


def run_lbfgs(model, obj, x0, seed, bounds, init_scale, maxcor):
    """L-BFGS-B with random restarts until the budget runs out. Returns #restarts."""
    opt_rng = rng(seed, OPTIMISER)
    restarts = 0
    try:
        while True:
            minimize(obj.loss_and_grad, x0, jac=True, method="L-BFGS-B",
                     bounds=[tuple(bounds)] * model.n_params,
                     options={"maxcor": maxcor, "maxiter": 10**9, "maxfun": 10**9})
            restarts += 1
            x0 = opt_rng.uniform(-init_scale, init_scale, size=model.n_params)
    except BudgetExhausted:
        pass
    return {"restarts": restarts}


def plot_convergence(curves, noise_floor, path):
    """Best train MSE and test MSE against evaluations, one line per method."""
    styles = {"CMA-ES": dict(color="#2a78d6", linestyle="-"),
              "L-BFGS-B": dict(color="#eb6834", linestyle="--")}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, key, title in zip(axes, ["train_mse", "test_mse"], ["(a) Training MSE", "(b) Test MSE"]):
        for name, rows in curves.items():
            ax.plot([r["evals"] for r in rows], [r[key] for r in rows],
                    linewidth=2, marker="o", markersize=4, label=name, **styles[name])
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
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    max_evals = args.max_evals or cfg["budget"]["max_evals"]
    grad_cost = cfg["budget"]["grad_cost"]
    bounds = cfg["model"]["weight_bounds"]
    init_scale = cfg["model"]["init_scale"]

    model = MLP(d_in=cfg["data"]["input_dim"], hidden=cfg["model"]["hidden"])
    data = generate_datasets(args.seed, sigma=cfg["data"]["noise_std"])
    X, y = data["train"][args.n]
    x0 = model.init(args.seed, scale=init_scale)          # same start for both methods

    methods = {
        "CMA-ES": lambda obj: run_cmaes(model, obj, x0, args.seed, cfg["methods"]["cmaes"]["sigma0"],
                                        bounds, init_scale),
        "L-BFGS-B": lambda obj: run_lbfgs(model, obj, x0, args.seed, bounds, init_scale,
                                          cfg["methods"]["lbfgs"]["maxcor"]),
    }

    print(f"seed={args.seed}  n_train={args.n}  params={model.n_params}  "
          f"budget={max_evals} evals  grad_cost={grad_cost}")
    header = (f"{'method':<9} {'train':>8} {'test':>8} {'clean':>8} {'evals':>6} "
              f"{'calls':>6} {'restarts':>8} {'time[s]':>8} {'@bounds':>8}")
    print(header)
    print("-" * len(header))

    curves = {}
    for name, run in methods.items():
        obj = CountedObjective(model, X, y, max_evals=max_evals, grad_cost=grad_cost)
        info = run(obj)
        assert np.all(np.abs(obj.best_theta) <= bounds[1] + 1e-12), f"{name} left the weight bounds"
        m = evaluate(model, obj.best_theta, data, args.n, bound=bounds[1])
        curves[name] = evaluate_snapshots(model, obj.snapshots, data, args.n, bound=bounds[1])
        print(f"{name:<9} {m['train_mse']:8.4f} {m['test_mse']:8.4f} {m['test_mse_clean']:8.4f} "
              f"{obj.evals_used:6d} {obj.n_calls:6d} {info['restarts']:8d} {obj.elapsed:8.2f} "
              f"{m['frac_at_bounds']:8.1%}")

    print(f"\nnoise floor sigma^2 = {data['noise_floor']:.4f}   "
          "(train/test = noisy MSE, clean = MSE against the noise-free f)")

    if not args.no_plot:
        path = ROOT / "figures" / f"smoke_convergence_seed{args.seed}_n{args.n}.png"
        plot_convergence(curves, data["noise_floor"], path)
        print(f"figure: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
