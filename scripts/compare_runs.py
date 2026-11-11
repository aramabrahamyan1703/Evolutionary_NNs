"""Compare a secondary experiment with the core sweep, runs matched by seed.

    python scripts/compare_runs.py configs/core_wallclock.yaml   # same conclusions under a time limit?
    python scripts/compare_runs.py configs/core_ga_long.yaml     # GA at 1M evals vs 100k
    python scripts/compare_runs.py configs/core_gradcost1.yaml   # grad_cost 1 instead of 2

Prints and writes to results/processed/<experiment>/:
    compare_methods.csv   every (method, n) of the experiment vs the same method in the
                          core sweep: median train / clean test MSE, paired Wilcoxon on the
                          clean test MSE (Holm over the table), Cliff's delta (> 0: worse)
    compare_pairs.csv     every method pair at every n: significant in the core sweep, and
                          in the experiment? Methods the experiment did not rerun are taken
                          from the core sweep (e.g. grad_cost does not change CMA-ES or the GA)
    rankings              methods by median clean test MSE at every n, core vs experiment
"""

import argparse
from pathlib import Path

from evonn.analysis import (combine_with_base, compare_methods, compare_pairs, load_runs,
                            make_wilcoxon, processed_dir, rankings)
from evonn.experiment import load_config
from evonn.methods import LABELS

ROOT = Path(__file__).resolve().parents[1]


def raw_dir(cfg):
    return ROOT / cfg["experiment"]["output_dir"] / cfg["experiment"]["name"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path, help="config of the secondary experiment")
    parser.add_argument("--base", type=Path, default=ROOT / "configs" / "core.yaml")
    args = parser.parse_args()

    base_cfg, cfg = load_config(args.base), load_config(args.config)
    name = cfg["experiment"]["name"]
    base, other = load_runs(raw_dir(base_cfg)), load_runs(raw_dir(cfg))
    base = base[base["seed"].isin(other["seed"].unique())]        # same seeds in both
    out_dir = processed_dir(name)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_method = compare_methods(base, other)
    per_method.to_csv(out_dir / "compare_methods.csv", index=False)
    print(f"{name} vs {base_cfg['experiment']['name']}  ({other['seed'].nunique()} paired seeds)\n")
    print(f"{'method':<9} {'n':>5} {'train core':>11} {'train exp':>10} {'test core':>10} "
          f"{'test exp':>9} {'delta':>6} {'p_holm':>8}")
    for _, r in per_method.iterrows():
        print(f"{LABELS[r.method]:<9} {int(r.n_train):>5} {r.train_mse_base:11.4f} {r.train_mse_other:10.4f} "
              f"{r.test_mse_clean_base:10.4f} {r.test_mse_clean_other:9.4f} {r.cliffs_delta:+6.2f} {r.p_holm:8.2g}")

    combined, filled = combine_with_base(other, base)
    if combined["method"].nunique() > 1:
        if filled:
            print(f"\nmethods not rerun, taken from the core sweep: {', '.join(LABELS[m] for m in filled)}")
        pairs = compare_pairs(make_wilcoxon(base[base["n_train"].isin(combined["n_train"].unique())]),
                              make_wilcoxon(combined))
        pairs.to_csv(out_dir / "compare_pairs.csv", index=False)
        changed = pairs[~pairs["same_verdict"]]
        print(f"\nmethod pairs with the same verdict as the core sweep: {len(pairs) - len(changed)} of {len(pairs)}")
        for _, r in changed.iterrows():
            print(f"  changed  n={int(r.n_train):4d}  {r.method_a} vs {r.method_b}: core p_holm={r.p_holm_base:.2g} "
                  f"(diff {r.median_diff_base:+.4f}), exp p_holm={r.p_holm_other:.2g} (diff {r.median_diff_other:+.4f})")

        print("\nranking by median clean test MSE (best first):")
        rank_base, rank_other = rankings(base), rankings(combined)
        for n in rank_other:
            core = " < ".join(LABELS[m] for m in rank_base[n])
            exp = " < ".join(LABELS[m] for m in rank_other[n])
            print(f"  n={n:4d}  core: {core}\n          exp:  {exp}{'' if core == exp else '   <- changed'}")
    print(f"\n-> {out_dir.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
