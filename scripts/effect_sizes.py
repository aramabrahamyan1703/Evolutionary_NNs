"""Key effect sizes for the report: the GA against the best other method at every n.

    python scripts/effect_sizes.py

For three ways of choosing the network of a run (needs the aggregated tables):
    core          returned network, lowest training MSE        (core_ntrain_sweep/summary.csv)
    early_stop    early-stopped on the selection set (D10)      (p5_selection/summary_best_select.csv)
    choose_h      H and early stopping chosen on the selection set (p5_arch/chosen.csv)
"best other" is the non-GA method with the lowest median clean test MSE at that n. Reports the
difference of medians GA - best with a 95% paired-bootstrap interval (seeds resampled), A12 =
P(GA error > best's error) with its interval, and p_holm from the setting's Wilcoxon table.
Writes results/processed/effect_sizes_ga.csv.
"""

from pathlib import Path

import pandas as pd

from evonn.analysis import PROCESSED, paired, paired_bootstrap
from evonn.methods import LABELS

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = {"core": ("core_ntrain_sweep/summary.csv", "core_ntrain_sweep/wilcoxon.csv"),
            "early_stop": ("p5_selection/summary_best_select.csv", "p5_selection/wilcoxon_best_select.csv"),
            "choose_h": ("p5_arch/chosen.csv", "p5_arch/wilcoxon_chosen.csv")}
METRIC = "test_mse_clean"


def main():
    rows = []
    for setting, (runs_file, wil_file) in SETTINGS.items():
        df, wil = pd.read_csv(PROCESSED / runs_file), pd.read_csv(PROCESSED / wil_file)
        for n in sorted(df["n_train"].unique()):
            med = df[(df["n_train"] == n) & (df["method"] != "ga")].groupby("method")[METRIC].median()
            best = med.idxmin()
            ga, other = paired(df, METRIC, n, "ga", best)
            diff, lo, hi, a12, a_lo, a_hi = paired_bootstrap(ga, other)
            w = wil[(wil["n_train"] == n) & (wil[["method_a", "method_b"]].isin(["ga", best]).all(axis=1))]
            rows.append({"setting": setting, "n_train": n, "best_other": best, "runs": len(ga),
                         "median_ga": float(pd.Series(ga).median()), "median_best": float(med[best]),
                         "diff": diff, "diff_lo": lo, "diff_hi": hi,
                         "a12": a12, "a12_lo": a_lo, "a12_hi": a_hi, "p_holm": float(w["p_holm"].iloc[0])})
    out = pd.DataFrame(rows)
    out.to_csv(PROCESSED / "effect_sizes_ga.csv", index=False)
    print("GA vs best other method (clean test MSE): diff = median GA - median best [95% CI]; "
          "A12 = P(GA worse) [95% CI]")
    for _, r in out.iterrows():
        print(f"  {r.setting:<10} n={r.n_train:4d} vs {LABELS[r.best_other]:<8} diff {r['diff']:+.4f} "
              f"[{r.diff_lo:+.4f}, {r.diff_hi:+.4f}]  A12 {r.a12:.2f} [{r.a12_lo:.2f}, {r.a12_hi:.2f}]  "
              f"p_holm {r.p_holm:.2g}")
    print(f"-> {(PROCESSED / 'effect_sizes_ga.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
