"""Collect results/raw/<experiment>/*.json into tables and run the statistical tests.

    python scripts/aggregate_results.py                                     # configs/core.yaml
    python scripts/aggregate_results.py --config configs/core_wallclock.yaml
    python scripts/aggregate_results.py --config configs/core_ga_long.yaml  # one method: no tests
    python scripts/aggregate_results.py --config configs/p5_selection.yaml --selection best_select

Outputs, in results/processed/<experiment>/ (one folder per experiment, so
experiments never overwrite each other):
    summary.csv    one row per run, with all final metrics
    medians.csv    median + IQR per (method, n_train)
    wilcoxon.csv   paired Wilcoxon signed-rank test (runs matched by seed) for every
                   method pair at every n, Holm correction over the whole table,
                   Cliff's delta as effect size (needs >= 2 methods)
With --selection best_val / best_select the tables describe the early-stopped network
instead of the final one, and the file names get that suffix (summary_best_select.csv, ...).
"""

import argparse
from pathlib import Path

from evonn.analysis import SELECTIONS, load_runs, make_medians, make_wilcoxon, processed_dir
from evonn.experiment import load_config

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=ROOT / "configs" / "core.yaml", type=Path)
    parser.add_argument("--metric", default="test_mse_clean", help="metric for the tests (default: test_mse_clean)")
    parser.add_argument("--selection", default="final", choices=SELECTIONS,
                        help="which network of each run: final (default) or an early-stopping checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)
    name = cfg["experiment"]["name"]
    raw_dir = ROOT / cfg["experiment"]["output_dir"] / name
    out_dir = processed_dir(name)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_runs(raw_dir, args.selection)
    suffix = "" if args.selection == "final" else f"_{args.selection}"
    commits = sorted(df["git_commit"].astype(str).unique())
    print(f"{name} ({args.selection}): {len(df)} runs, {df['method'].nunique()} methods, {df['n_train'].nunique()} sizes, "
          f"{df['seed'].nunique()} seeds; git commit(s): {', '.join(commits)}")

    medians = make_medians(df)
    written = [f"summary{suffix}.csv", f"medians{suffix}.csv"]
    df.to_csv(out_dir / written[0], index=False)
    medians.to_csv(out_dir / written[1], index=False)
    if df["method"].nunique() > 1:
        wil = make_wilcoxon(df, metric=args.metric)
        written.append(f"wilcoxon{suffix}.csv")
        wil.to_csv(out_dir / written[-1], index=False)
    print(f"  -> {out_dir.relative_to(ROOT)}/ {', '.join(written)}")

    print(f"\nmedian {args.metric} (noise floor of the noisy test MSE = {df['noise_floor'].iloc[0]:.4f}):")
    print(medians.pivot(index="method", columns="n_train", values=f"{args.metric}_median")
          .to_string(float_format="{:.4f}".format))
    if len(written) == 3:
        sig = wil[wil["significant"]]
        print(f"\nsignificant pairs (p_holm < 0.05): {len(sig)} of {len(wil)}")
        for _, r in sig.iterrows():
            print(f"  n={int(r.n_train):4d}  {r.method_a} vs {r.method_b}: p_holm={r.p_holm:.2g}  "
                  f"delta={r.cliffs_delta:+.2f}  diff={r.median_diff:+.4f}")


if __name__ == "__main__":
    main()
