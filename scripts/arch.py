"""Phase 5.3: train every method at every hidden size H, choose (H, checkpoint) on the selection set.

    python scripts/arch.py --jobs 8                  # run what is missing, then summarise
    python scripts/arch.py --summary-only            # only summarise saved runs

Runs go to results/raw/<base experiment>_H<h>/ (the base H reuses the base experiment's
runs). Writes to results/processed/p5_arch/:
    selected_all.csv     early-stopped network of every run at every H
    chosen.csv           per (method, n, seed) the network with the lowest selection MSE
    medians_chosen.csv   medians of the chosen networks; wilcoxon_chosen.csv: method pairs
    compare_vs_fixed.csv chosen H vs the base H (both early-stopped), per method, paired by seed
    chosen_h.csv         how often each H is chosen, per method and n
"""

import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(var, "1")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from evonn.analysis import PROCESSED, compare_methods, make_medians, make_wilcoxon
from evonn.arch import arch_configs, choose_h, load_selected
from evonn.experiment import load_config, run_and_save
from evonn.methods import LABELS

ROOT = Path(__file__).resolve().parents[1]


def run_all(cfgs, jobs):
    cells = [(cfg, m, n, s) for n in sorted(next(iter(cfgs.values()))["data"]["n_train"], reverse=True)
             for h, cfg in sorted(cfgs.items(), reverse=True)
             for s in range(cfg["experiment"]["seeds"]) for m in cfg["experiment"]["methods"]]
    print(f"p5_arch: {len(cfgs)} hidden sizes, {len(cells)} runs (existing ones are skipped)")
    counts = {"done": 0, "skipped": 0}
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(run_and_save, cfg, m, n, s, ROOT / cfg["experiment"]["output_dir"])
                   for cfg, m, n, s in cells]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            counts[fut.result()[1]] += 1
    print(f"done: {counts['done']}, skipped (already there): {counts['skipped']}")


def summarise(cfgs, base_h):
    out = PROCESSED / "p5_arch"
    out.mkdir(parents=True, exist_ok=True)
    df = load_selected(cfgs)
    chosen = choose_h(df)
    fixed = df[df["hidden"] == base_h]
    df.to_csv(out / "selected_all.csv", index=False)
    chosen.to_csv(out / "chosen.csv", index=False)
    med = make_medians(chosen)
    med.to_csv(out / "medians_chosen.csv", index=False)
    wil = make_wilcoxon(chosen)
    wil.to_csv(out / "wilcoxon_chosen.csv", index=False)
    vs_fixed = compare_methods(fixed, chosen)
    vs_fixed.to_csv(out / "compare_vs_fixed.csv", index=False)
    counts = chosen.groupby(["method", "n_train"])["hidden"].value_counts().unstack(fill_value=0)
    counts.to_csv(out / "chosen_h.csv")

    commits = sorted(df["git_commit"].astype(str).unique())
    print(f"{len(df)} runs over H = {sorted(cfgs)}; git commit(s): {', '.join(commits)}")
    print("\nmedian clean test MSE, early stopping + H chosen on the selection set:")
    print(med.pivot(index="method", columns="n_train", values="test_mse_clean_median")
          .rename(index=LABELS).to_string(float_format="{:.4f}".format))
    print(f"\nsame at fixed H = {base_h}:")
    print(fixed.groupby(["method", "n_train"])["test_mse_clean"].median().unstack()
          .rename(index=LABELS).to_string(float_format="{:.4f}".format))
    print("\nmedian chosen H:")
    print(chosen.groupby(["method", "n_train"])["hidden"].median().unstack().rename(index=LABELS).to_string())
    sig = wil[wil["significant"]]
    print(f"\nmethod pairs (chosen networks), p_holm < 0.05: {len(sig)} of {len(wil)}")
    for _, r in sig.iterrows():
        print(f"  n={int(r.n_train):4d}  {r.method_a} vs {r.method_b}: delta={r.cliffs_delta:+.2f} "
              f"p_holm={r.p_holm:.2g}  diff={r.median_diff:+.4f}")
    print(f"\n-> {out.relative_to(ROOT)}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", type=Path, default=ROOT / "configs" / "p5_arch.yaml")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    acfg = load_config(args.arch)
    cfgs = arch_configs(acfg)
    base_h = load_config(ROOT / acfg["base_config"])["model"]["hidden"]
    if not args.summary_only:
        run_all(cfgs, args.jobs)
    summarise(cfgs, base_h)


if __name__ == "__main__":
    main()
