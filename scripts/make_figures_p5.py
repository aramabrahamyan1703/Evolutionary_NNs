"""Phase 5 figures into figures/ (needs the Phase 5 tables in results/processed/).

    python scripts/make_figures_p5.py

  fig5_early_stopping_vs_n.png  clean test MSE vs n: final network (a) vs early-stopped on the
                                selection set (b)                       (p5_selection)
  fig6_sharpness_matched.png    perturbation sharpness (a) and clean test MSE (b) vs n, all
                                methods at the GA's training MSE        (p5_selection/sharpness.csv)
  fig7_chosen_h.png             chosen hidden size H (a) and clean test MSE of the chosen
                                networks (b) vs n                        (p5_arch)
"""

import pathlib

import matplotlib.pyplot as plt
import pandas as pd

from evonn.analysis import PROCESSED
from evonn.methods import LABELS, STYLES

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIGS = ROOT / "figures"
METHODS = ["cmaes", "ga", "lbfgs", "sgd"]
NS = [25, 100, 400, 1600]


def style_ax(ax, log_y=True):
    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xticks(NS)
    ax.set_xticklabels([str(n) for n in NS])
    ax.set_xlabel("Training set size n", fontsize=11)
    ax.grid(True, which="major", color="#e5e4df", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def band(ax, df, value, scale=1.0):
    """Median line with an interquartile band per method, from one row per run."""
    for m in METHODS:
        g = df[df["method"] == m].groupby("n_train")[value]
        med, q25, q75 = g.median() * scale, g.quantile(0.25) * scale, g.quantile(0.75) * scale
        ax.plot(med.index, med.values, **STYLES[m], markersize=6, linewidth=2, label=LABELS[m])
        ax.fill_between(med.index, q25.values, q75.values, color=STYLES[m]["color"], alpha=0.12)


def save(fig, name):
    plt.tight_layout()
    fig.savefig(FIGS / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> figures/{name}")


def fig_early_stopping():
    d = PROCESSED / "p5_selection"
    final, early = pd.read_csv(d / "summary.csv"), pd.read_csv(d / "summary_best_select.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, df, title in zip(axes, [final, early], ["(a) Final network (lowest training MSE)",
                                                     "(b) Early-stopped on the selection set (n points)"]):
        band(ax, df, "test_mse_clean")
        style_ax(ax)
        ax.set_title(title, loc="left", fontsize=11)
    axes[0].set_ylabel("Clean test MSE (floor 0)", fontsize=11)
    axes[0].legend(frameon=False, fontsize=9)
    save(fig, "fig5_early_stopping_vs_n.png")


def fig_sharpness():
    df = pd.read_csv(PROCESSED / "p5_selection" / "sharpness.csv")
    df = df[df["network"] == "matched"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    band(axes[0], df, "perturb_sharpness")
    band(axes[1], df, "test_mse_clean")
    axes[0].set_ylabel("Loss increase at 1% weight perturbation", fontsize=11)
    axes[1].set_ylabel("Clean test MSE (floor 0)", fontsize=11)
    axes[0].set_title("(a) Sharpness at the GA's training MSE", loc="left", fontsize=11)
    axes[1].set_title("(b) Test error of the same networks", loc="left", fontsize=11)
    for ax in axes:
        style_ax(ax)
    axes[0].legend(frameon=False, fontsize=9)
    save(fig, "fig6_sharpness_matched.png")


def fig_chosen_h():
    chosen = pd.read_csv(PROCESSED / "p5_arch" / "chosen.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    band(axes[0], chosen, "hidden")
    band(axes[1], chosen, "test_mse_clean")
    style_ax(axes[0], log_y=False)
    style_ax(axes[1])
    axes[0].set_ylabel("Chosen hidden size H", fontsize=11)
    axes[1].set_ylabel("Clean test MSE (floor 0)", fontsize=11)
    axes[0].set_title("(a) H chosen on the selection set", loc="left", fontsize=11)
    axes[1].set_title("(b) Test error with H and early stopping chosen", loc="left", fontsize=11)
    axes[0].legend(frameon=False, fontsize=9)
    save(fig, "fig7_chosen_h.png")


def main():
    FIGS.mkdir(exist_ok=True)
    print("generating Phase 5 figures ...")
    fig_early_stopping()
    fig_sharpness()
    if (PROCESSED / "p5_arch" / "chosen.csv").exists():
        fig_chosen_h()
    else:
        print("  (p5_arch not summarised yet: run scripts/arch.py first; fig7 skipped)")


if __name__ == "__main__":
    main()
