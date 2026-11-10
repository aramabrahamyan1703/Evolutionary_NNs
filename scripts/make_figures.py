"""Produce report figures into figures/.

Figures generated:
  fig1_test_train_vs_n.png   main figure: excess test MSE and train MSE vs n (log-log)
  fig2_gap_vs_n.png          generalisation gap (test - train) vs n (log-log)
  fig3_bounds_vs_n.png       fraction of weights at ±5 bounds vs n
  fig4_convergence.png       best-so-far train + test MSE vs evaluations (one seed per method)

Usage:
    python scripts/make_figures.py
    python scripts/make_figures.py --seed 0 --n 100   # convergence at a specific cell
"""

import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]

# ------------------------------------------------------------------ style

METHODS = ["cmaes", "ga", "lbfgs", "sgd"]
LABELS  = {"cmaes": "CMA-ES", "ga": "GA", "lbfgs": "L-BFGS-B", "sgd": "SGD (Adam)"}
COLORS  = {"cmaes": "#2a78d6", "ga": "#2ca02c", "lbfgs": "#eb6834", "sgd": "#9467bd"}
LINES   = {"cmaes": "-", "ga": "--", "lbfgs": "-.", "sgd": ":"}
MARKERS = {"cmaes": "o", "ga": "s", "lbfgs": "^", "sgd": "D"}

NS = [25, 100, 400, 1600]


def style_ax(ax):
    ax.grid(True, which="major", color="#e5e4df", linewidth=0.8)
    ax.grid(True, which="minor", color="#f0efeb", linewidth=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# ------------------------------------------------------------------ fig 1: test + train vs n

def fig_test_train(medians, noise_floor, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)

    for method in METHODS:
        sub = medians[medians["method"] == method].sort_values("n_train")
        ns_m = sub["n_train"].values
        kw = dict(color=COLORS[method], linestyle=LINES[method],
                  marker=MARKERS[method], markersize=6, linewidth=2,
                  label=LABELS[method])

        # excess test MSE (test_mse_clean - noise_floor)
        y_med = sub["test_mse_clean_median"].values
        y_q25 = sub["test_mse_clean_q25"].values
        y_q75 = sub["test_mse_clean_q75"].values
        axes[0].plot(ns_m, y_med, **kw)
        axes[0].fill_between(ns_m, y_q25, y_q75,
                             color=COLORS[method], alpha=0.12)

        # train MSE
        y_med = sub["train_mse_median"].values
        y_q25 = sub["train_mse_q25"].values
        y_q75 = sub["train_mse_q75"].values
        axes[1].plot(ns_m, y_med, **kw)
        axes[1].fill_between(ns_m, y_q25, y_q75,
                             color=COLORS[method], alpha=0.12)

    for ax, title, ylabel in zip(
        axes,
        ["(a) Test MSE (clean) vs training-set size",
         "(b) Training MSE vs training-set size"],
        ["Test MSE against noise-free f", "Training MSE"],
    ):
        ax.axhline(noise_floor, color="#52514e", linewidth=1, linestyle=":",
                   label=f"noise floor σ²={noise_floor:.2f}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Training set size n", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xticks(NS)
        ax.set_xticklabels([str(n) for n in NS])
        style_ax(ax)

    axes[0].legend(frameon=False, fontsize=9)
    plt.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out.relative_to(ROOT)}")


# ------------------------------------------------------------------ fig 2: gap vs n

def fig_gap(medians, out):
    fig, ax = plt.subplots(figsize=(6, 4.5))

    for method in METHODS:
        sub = medians[medians["method"] == method].sort_values("n_train")
        ns_m = sub["n_train"].values
        y_med = sub["gap_median"].values
        y_q25 = sub["gap_q25"].values
        y_q75 = sub["gap_q75"].values
        ax.plot(ns_m, y_med, color=COLORS[method], linestyle=LINES[method],
                marker=MARKERS[method], markersize=6, linewidth=2,
                label=LABELS[method])
        ax.fill_between(ns_m, y_q25, y_q75, color=COLORS[method], alpha=0.12)

    ax.axhline(0, color="#52514e", linewidth=1, linestyle=":")
    ax.set_xscale("log")
    ax.set_xlabel("Training set size n", fontsize=11)
    ax.set_ylabel("Generalisation gap (test − train MSE)", fontsize=11)
    ax.set_title("Generalisation gap vs training-set size", loc="left", fontsize=11)
    ax.set_xticks(NS)
    ax.set_xticklabels([str(n) for n in NS])
    ax.legend(frameon=False, fontsize=9)
    style_ax(ax)
    plt.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out.relative_to(ROOT)}")


# ------------------------------------------------------------------ fig 3: bounds vs n

def fig_bounds(medians, out):
    fig, ax = plt.subplots(figsize=(6, 4.5))

    for method in METHODS:
        sub = medians[medians["method"] == method].sort_values("n_train")
        ns_m = sub["n_train"].values
        y_med = sub["frac_at_bounds_median"].values
        ax.plot(ns_m, y_med * 100, color=COLORS[method], linestyle=LINES[method],
                marker=MARKERS[method], markersize=6, linewidth=2,
                label=LABELS[method])

    ax.set_xscale("log")
    ax.set_xlabel("Training set size n", fontsize=11)
    ax.set_ylabel("Weights at ±5 bounds (%)", fontsize=11)
    ax.set_title("Fraction of weights at ±5 bounds vs n", loc="left", fontsize=11)
    ax.set_xticks(NS)
    ax.set_xticklabels([str(n) for n in NS])
    ax.legend(frameon=False, fontsize=9)
    style_ax(ax)
    plt.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out.relative_to(ROOT)}")


# ------------------------------------------------------------------ fig 4: convergence curves

def fig_convergence(raw_dir, seed, n, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)

    for method in METHODS:
        fname = raw_dir / f"{method}_n{n}_s{seed}.json"
        if not fname.exists():
            print(f"  warning: {fname.name} not found, skipping")
            continue
        d = json.loads(fname.read_text())
        curve = d["curve"]
        if not curve:
            continue
        evals = [r["evals"] for r in curve]
        train = [r["train_mse"] for r in curve]
        test  = [r["test_mse_clean"] for r in curve]
        noise = curve[0]["noise_floor"]

        kw = dict(color=COLORS[method], linestyle=LINES[method], linewidth=2,
                  label=LABELS[method])
        axes[0].plot(evals, train, **kw)
        axes[1].plot(evals, test,  **kw)

    for ax, title, ylabel in zip(
        axes,
        [f"(a) Training MSE  (seed {seed}, n={n})",
         f"(b) Test MSE (clean)  (seed {seed}, n={n})"],
        ["Training MSE (best-so-far)", "Test MSE against noise-free f"],
    ):
        ax.axhline(noise, color="#52514e", linewidth=1, linestyle=":",
                   label=f"noise floor σ²={noise:.2f}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Function evaluations", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, loc="left", fontsize=11)
        style_ax(ax)

    axes[0].legend(frameon=False, fontsize=9)
    plt.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out.relative_to(ROOT)}")


# ------------------------------------------------------------------ main

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=ROOT / "configs" / "core.yaml", type=pathlib.Path)
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for convergence curves (default: 0)")
    parser.add_argument("--n", type=int, default=100,
                        help="n_train for convergence curves (default: 100)")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    exp_name = cfg["experiment"]["name"]
    raw_dir  = ROOT / cfg["experiment"]["output_dir"] / exp_name
    out_dir  = ROOT / "figures"
    out_dir.mkdir(exist_ok=True)
    proc_dir = ROOT / "results" / "processed"
    noise_floor = cfg["data"]["noise_std"] ** 2

    medians_path = proc_dir / "medians.csv"
    if not medians_path.exists():
        print("results/processed/medians.csv not found — run aggregate_results.py first")
        return

    medians = pd.read_csv(medians_path)
    print("generating figures ...")

    fig_test_train(medians, noise_floor,
                   out_dir / "fig1_test_train_vs_n.png")

    fig_gap(medians,
            out_dir / "fig2_gap_vs_n.png")

    fig_bounds(medians,
               out_dir / "fig3_bounds_vs_n.png")

    fig_convergence(raw_dir, args.seed, args.n,
                    out_dir / f"fig4_convergence_seed{args.seed}_n{args.n}.png")

    print("done.")


if __name__ == "__main__":
    main()
