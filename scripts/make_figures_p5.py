"""Phase 5 figures into figures/ (needs the Phase 5 tables in results/processed/).

    python scripts/make_figures_p5.py

  fig5_early_stopping_vs_n.png  clean test MSE vs n of the networks early-stopped on the
                                selection set (the returned networks are Fig 1a)   (p5_selection)
  fig6_sharpness_matched.png    perturbation sharpness (a) and clean test MSE (b) vs n, all
                                methods at the GA's training MSE        (p5_selection/sharpness.csv)
  fig7_chosen_h.png             how often each hidden size H is chosen, per method and n (p5_arch)
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
    # the returned networks (no early stopping) are Fig 1(a); this figure shows only the early-stopped ones
    early = pd.read_csv(PROCESSED / "p5_selection" / "summary_best_select.csv")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    band(ax, early, "test_mse_clean")
    style_ax(ax)
    ax.set_ylabel("Clean test MSE (floor 0)", fontsize=11)
    ax.set_title("Early-stopped on the selection set (n points)", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
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
    """How often each H is chosen, per method (small multiples): one hue, light = rarely, dark = often."""
    counts = pd.read_csv(PROCESSED / "p5_arch" / "chosen_h.csv").set_index(["method", "n_train"])
    hs = [int(c) for c in counts.columns]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), sharey=True)
    for ax, m in zip(axes, METHODS):
        grid = counts.loc[m].T.to_numpy()                   # rows: H, columns: n
        im = ax.imshow(grid, cmap="Blues", vmin=0, vmax=30, origin="lower", aspect="auto")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                ax.text(j, i, int(grid[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if grid[i, j] > 15 else "#28272a")
        ax.set_xticks(range(len(NS)), [str(n) for n in NS])
        ax.set_yticks(range(len(hs)), [str(h) for h in hs])
        ax.set_xlabel("Training set size n", fontsize=10)
        ax.set_title(LABELS[m], loc="left", fontsize=11)
        for side in ax.spines.values():
            side.set_visible(False)
    axes[0].set_ylabel("Chosen hidden size H", fontsize=10)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01, label="runs (of 30)")
    fig.savefig(FIGS / "fig7_chosen_h.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  -> figures/fig7_chosen_h.png")


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
