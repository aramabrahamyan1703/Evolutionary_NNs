"""Run an experiment: every (method, n_train, seed) cell from a config, one JSON per run.

    python scripts/run_experiment.py                               # everything in configs/core.yaml
    python scripts/run_experiment.py --seeds 2 --jobs 4            # seeds 0-1, 4 processes
    python scripts/run_experiment.py --methods cmaes --n 25 100    # a subset

Results go to <output_dir>/<experiment name>/<method>_n<n>_s<seed>.json.
Existing files are skipped, so an interrupted run can simply be started again
(use --overwrite to redo them).

Wall-clock times from parallel runs are noisy; time the runs used for the
wall-clock comparison with --jobs 1.
"""

import os

# one BLAS thread per process: set before numpy is imported (also in worker processes)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from evonn.experiment import load_config, run_and_save

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "core.yaml")
    parser.add_argument("--methods", nargs="+", help="default: experiment.methods in the config")
    parser.add_argument("--n", nargs="+", type=int, help="training sizes (default: data.n_train)")
    parser.add_argument("--seeds", type=int, help="number of seeds (default: experiment.seeds)")
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1, help="parallel processes")
    parser.add_argument("--out", type=Path, help="default: experiment.output_dir in the config")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    methods = args.methods or cfg["experiment"]["methods"]
    sizes = args.n or cfg["data"]["n_train"]
    n_seeds = args.seeds or cfg["experiment"]["seeds"]
    seeds = range(args.first_seed, args.first_seed + n_seeds)
    out_dir = args.out or ROOT / cfg["experiment"]["output_dir"]
    unknown = set(sizes) - set(cfg["data"]["n_train"])
    if unknown:
        parser.error(f"--n {sorted(unknown)} not in data.n_train {cfg['data']['n_train']}")

    # largest n first: the slowest runs start early, which balances the workers
    cells = [(m, n, s) for n in sorted(sizes, reverse=True) for s in seeds for m in methods]
    print(f"{cfg['experiment']['name']}: {len(methods)} methods x {len(sizes)} sizes x "
          f"{n_seeds} seeds = {len(cells)} runs -> {out_dir}")

    counts = {"done": 0, "skipped": 0}
    if args.jobs == 1:
        for m, n, s in tqdm(cells):
            _, status = run_and_save(cfg, m, n, s, out_dir, args.overwrite)
            counts[status] += 1
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(run_and_save, cfg, m, n, s, out_dir, args.overwrite)
                       for m, n, s in cells]
            for fut in tqdm(as_completed(futures), total=len(futures)):
                _, status = fut.result()
                counts[status] += 1
    print(f"done: {counts['done']}, skipped (already there): {counts['skipped']}")


if __name__ == "__main__":
    main()
