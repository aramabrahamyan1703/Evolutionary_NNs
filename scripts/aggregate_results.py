"""Collect results/raw/<experiment>/*.json into results/processed/*.csv and run statistical tests.

Usage:
    python scripts/aggregate_results.py                        # uses configs/core.yaml
    python scripts/aggregate_results.py --config configs/core.yaml
    python scripts/aggregate_results.py --experiment core_ntrain_sweep

Outputs (all in results/processed/):
    summary.csv          one row per (method, n_train, seed) with all final metrics
    medians.csv          median + IQR per (method, n_train)
    wilcoxon.csv         paired Wilcoxon signed-rank + Holm correction + Cliff's delta
                         for every (EA vs baseline) pair at every n
"""

import argparse
import json
import pathlib
from itertools import combinations

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon

ROOT = pathlib.Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ helpers

def cliffs_delta(x, y):
    """Cliff's delta effect size: P(x > y) - P(x < y)."""
    x, y = np.asarray(x), np.asarray(y)
    greater = np.sum(x[:, None] > y[None, :])
    less = np.sum(x[:, None] < y[None, :])
    return float((greater - less) / (len(x) * len(y)))


def holm_correction(p_values):
    """Holm-Bonferroni correction. Returns adjusted p-values."""
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.zeros(n)
    for rank, idx in enumerate(order):
        adjusted[idx] = min(1.0, p_values[idx] * (n - rank))
    # make monotone
    for i in range(1, n):
        adjusted[order[i]] = max(adjusted[order[i]], adjusted[order[i - 1]])
    return adjusted


# ------------------------------------------------------------------ load

def load_runs(experiment_dir):
    rows = []
    for f in sorted(experiment_dir.glob("*.json")):
        d = json.loads(f.read_text())
        row = {
            "method":    d["meta"]["method"],
            "n_train":   d["meta"]["n_train"],
            "seed":      d["meta"]["seed"],
            "evals_used": d["budget"]["evals_used"],
            "wall_time": d["budget"]["wall_time"],
        }
        row.update(d["final"])
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ aggregate

def make_summary(df):
    return df.sort_values(["method", "n_train", "seed"]).reset_index(drop=True)


def make_medians(df):
    metrics = ["train_mse", "test_mse", "test_mse_clean", "excess_test",
               "gap", "frac_at_bounds", "evals_used", "wall_time"]
    rows = []
    for (method, n), g in df.groupby(["method", "n_train"]):
        row = {"method": method, "n_train": n}
        for m in metrics:
            row[f"{m}_median"] = g[m].median()
            row[f"{m}_q25"]    = g[m].quantile(0.25)
            row[f"{m}_q75"]    = g[m].quantile(0.75)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["n_train", "method"]).reset_index(drop=True)


def make_wilcoxon(df, metric="test_mse_clean"):
    """Paired Wilcoxon signed-rank test for every method pair at every n, with Holm correction."""
    methods = sorted(df["method"].unique())
    ns = sorted(df["n_train"].unique())
    pairs = list(combinations(methods, 2))

    rows = []
    p_values = []
    for n in ns:
        sub = df[df["n_train"] == n]
        for m1, m2 in pairs:
            x = sub[sub["method"] == m1].sort_values("seed")[metric].values
            y = sub[sub["method"] == m2].sort_values("seed")[metric].values
            if len(x) != len(y) or len(x) == 0:
                continue
            try:
                stat, p = wilcoxon(x, y, alternative="two-sided")
            except ValueError:
                stat, p = np.nan, 1.0
            delta = cliffs_delta(x, y)
            med_diff = float(np.median(x) - np.median(y))
            rows.append({
                "n_train": n, "method_a": m1, "method_b": m2,
                "median_a": float(np.median(x)), "median_b": float(np.median(y)),
                "median_diff": med_diff,
                "cliffs_delta": delta,
                "W": stat, "p_raw": p,
            })
            p_values.append(p)

    result = pd.DataFrame(rows)
    if len(result):
        result["p_holm"] = holm_correction(result["p_raw"].values)
        result["significant"] = result["p_holm"] < 0.05
    return result


# ------------------------------------------------------------------ main

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=ROOT / "configs" / "core.yaml", type=pathlib.Path)
    parser.add_argument("--experiment", default=None,
                        help="experiment name (overrides config)")
    parser.add_argument("--metric", default="test_mse_clean",
                        help="metric for Wilcoxon tests (default: test_mse_clean)")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    exp_name = args.experiment or cfg["experiment"]["name"]
    raw_dir = ROOT / cfg["experiment"]["output_dir"] / exp_name
    out_dir = ROOT / "results" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading runs from {raw_dir.relative_to(ROOT)} ...")
    df = load_runs(raw_dir)
    print(f"  {len(df)} runs, {df['method'].nunique()} methods, "
          f"{df['n_train'].nunique()} sizes, {df['seed'].nunique()} seeds")

    summary = make_summary(df)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"  -> results/processed/summary.csv  ({len(summary)} rows)")

    medians = make_medians(df)
    medians.to_csv(out_dir / "medians.csv", index=False)
    print(f"  -> results/processed/medians.csv  ({len(medians)} rows)")

    wilcoxon_df = make_wilcoxon(df, metric=args.metric)
    wilcoxon_df.to_csv(out_dir / "wilcoxon.csv", index=False)
    print(f"  -> results/processed/wilcoxon.csv ({len(wilcoxon_df)} rows)")

    # print a readable summary
    print(f"\nMedian {args.metric} (noise floor = {df['noise_floor'].iloc[0]:.4f}):")
    pivot = medians.pivot(index="method", columns="n_train",
                          values=f"{args.metric}_median")
    print(pivot.to_string(float_format="{:.4f}".format))

    print(f"\nSignificant pairs (p_holm < 0.05):")
    sig = wilcoxon_df[wilcoxon_df["significant"]]
    if len(sig):
        for _, r in sig.iterrows():
            print(f"  n={int(r.n_train):4d}  {r.method_a} vs {r.method_b}: "
                  f"p_holm={r.p_holm:.3f}  delta={r.cliffs_delta:+.2f}  "
                  f"diff={r.median_diff:+.4f}")
    else:
        print("  none")


if __name__ == "__main__":
    main()
