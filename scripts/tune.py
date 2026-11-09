"""Phase 3 tuning (D9): run every method's grid and edge extension, rank by training MSE, report the winners.

    python scripts/tune.py --jobs 8                 # run everything in configs/tuning.yaml, then summarise
    python scripts/tune.py --summary-only           # only summarise runs already saved
    python scripts/tune.py --methods sgd --jobs 8   # one method

Runs are saved under results/raw/tuning/<method>/<setting>/ and skipped if they
exist, so an interrupted run can simply be started again. The summary goes to
results/processed/tuning_summary.csv and the winners to
results/processed/tuning_best.yaml.
"""

import os

# one BLAS thread per process: set before numpy is imported (also in worker processes)
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(var, "1")

import argparse
import csv
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml
from tqdm import tqdm

from evonn.experiment import load_config, run_and_save
from evonn.methods import LABELS
from evonn.tuning import load_results, make_config, method_settings, rank_settings, setting_id

ROOT = Path(__file__).resolve().parents[1]


def run_all(tcfg, base, methods, out_dir, jobs):
    cells = [(make_config(base, m, s), m, n, seed)
             for m in methods for s in method_settings(tcfg, m)
             for n in sorted(tcfg["n_train"], reverse=True) for seed in tcfg["seeds"]]
    print(f"tuning: {len(cells)} runs ({', '.join(methods)}) -> {out_dir / 'tuning'}")
    counts = {"done": 0, "skipped": 0}
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(run_and_save, cfg, m, n, seed, out_dir) for cfg, m, n, seed in cells]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            counts[fut.result()[1]] += 1
    print(f"done: {counts['done']}, skipped (already there): {counts['skipped']}")


def summarise(tcfg, methods, out_dir, processed):
    sizes = sorted(tcfg["n_train"])
    rows, best = [], {}
    for m in methods:
        settings = method_settings(tcfg, m)
        table, stats = {}, {}
        for s in settings:
            sid = setting_id(s)
            runs = load_results(out_dir, m, s)
            table[sid] = {(r["meta"]["seed"], r["meta"]["n_train"]): r["final"]["train_mse"] for r in runs}
            stats[sid] = {n: [r for r in runs if r["meta"]["n_train"] == n] for n in sizes}
        expected = len(tcfg["seeds"]) * len(sizes)
        missing = [sid for sid in table if len(table[sid]) < expected]
        if missing:
            print(f"{LABELS[m]}: incomplete settings {missing}, run without --summary-only first")
            continue
        ranks = rank_settings(table)
        winner = min(ranks, key=ranks.get)
        best[m] = next(s for s in settings if setting_id(s) == winner)

        print(f"\n{LABELS[m]}  (median over {len(tcfg['seeds'])} seeds; rank by training MSE, 1 = best)")
        head = f"  {'setting':<34} {'rank':>5} " + " ".join(
            f"{'train n=' + str(n):>12} {'val n=' + str(n):>11}" for n in sizes) + f" {'time[s]':>8}"
        print(head)
        for sid in sorted(ranks, key=ranks.get):
            row = {"method": m, "setting": sid, "mean_rank": ranks[sid], "winner": sid == winner}
            cells = []
            for n in sizes:
                runs = stats[sid][n]
                row[f"train_n{n}"] = statistics.median(r["final"]["train_mse"] for r in runs)
                row[f"val_n{n}"] = statistics.median(r["final"]["val_mse"] for r in runs)
                row[f"clean_n{n}"] = statistics.median(r["final"]["test_mse_clean"] for r in runs)
                row[f"at_bounds_n{n}"] = statistics.median(r["final"]["frac_at_bounds"] for r in runs)
                cells.append(f"{row[f'train_n{n}']:12.4f} {row[f'val_n{n}']:11.4f}")
            row["time_s"] = statistics.median(r["budget"]["wall_time"] for n in sizes for r in stats[sid][n])
            rows.append(row)
            mark = "*" if sid == winner else " "
            print(f"{mark} {sid:<34} {ranks[sid]:5.2f} " + " ".join(cells) + f" {row['time_s']:8.1f}")

    if rows:
        processed.mkdir(parents=True, exist_ok=True)
        with open(processed / "tuning_summary.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (processed / "tuning_best.yaml").write_text(yaml.safe_dump(best, sort_keys=False))
        print(f"\nwinners -> {(processed / 'tuning_best.yaml').relative_to(ROOT)}, "
              f"table -> {(processed / 'tuning_summary.csv').relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tuning", type=Path, default=ROOT / "configs" / "tuning.yaml")
    parser.add_argument("--methods", nargs="+", help="default: every method in the tuning grids")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    tcfg = load_config(args.tuning)
    base = load_config(ROOT / tcfg["base_config"])
    methods = args.methods or list(tcfg["grids"])
    out_dir = ROOT / base["experiment"]["output_dir"]
    if not args.summary_only:
        run_all(tcfg, base, methods, out_dir, args.jobs)
    summarise(tcfg, methods, out_dir, ROOT / "results" / "processed")


if __name__ == "__main__":
    main()
