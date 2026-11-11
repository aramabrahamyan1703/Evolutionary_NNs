README.txt — Training Neural Networks with Evolutionary Algorithms
Marta Mateu Lopez · Aram Abrahamyan
Computational Intelligence (CI-MAI), UPC, November 2026

================================================================
REQUIREMENTS
================================================================

Python 3.11 or later. All dependencies are listed in requirements-lock.txt.
Tested on macOS 14 and Ubuntu 22.04.

================================================================
SETUP
================================================================

1. Clone the repository:

    git clone https://github.com/aramabrahamyan1703/Evolutionary_NNs.git
    cd Evolutionary_NNs

2. Create a virtual environment and install dependencies:

    python3 -m venv .venv
    source .venv/bin/activate          # on Windows: .venv\Scripts\activate
    pip install --upgrade pip
    pip install -r requirements-lock.txt
    pip install -e .

3. Verify the installation by running the test suite:

    pytest

    Expected output: 139 passed, 1 skipped (all green).

================================================================
REPRODUCING THE RESULTS
================================================================

All results in the paper come from configs/core.yaml (evaluation budget).
The raw JSON files are gitignored (regenerable); the aggregated CSVs and
figures are committed to results/processed/ and figures/.

--- Core experiment (480 runs, ~17 min with 4 cores) ---

    python scripts/run_experiment.py --config configs/core.yaml --jobs 4

Results are written to results/raw/core_ntrain_sweep/ (one JSON per run).
Existing files are skipped, so an interrupted run can be restarted as-is.

--- Aggregate results and run statistical tests ---

    python scripts/aggregate_results.py

Writes three files to results/processed/:
  summary.csv   — one row per run (480 rows)
  medians.csv   — median + IQR per (method, n_train)
  wilcoxon.csv  — paired Wilcoxon signed-rank + Holm correction + Cliff's delta

--- Generate report figures ---

    python scripts/make_figures.py

Writes four figures to figures/:
  fig1_test_train_vs_n.png     — main result figure (test + train MSE vs n)
  fig2_gap_vs_n.png            — generalisation gap vs n
  fig3_bounds_vs_n.png         — fraction of weights at bounds vs n
  fig4_convergence_seed0_n100.png — convergence curves (seed 0, n=100)

================================================================
SECONDARY ANALYSES
================================================================

--- Wall-clock budget sweep (480 runs, ~90 min, single process) ---

    python scripts/run_experiment.py --config configs/core_wallclock.yaml --jobs 1

Note: must be run with --jobs 1 so wall-clock times are not contaminated
by parallelism. Results go to results/raw/core_ntrain_sweep_wallclock/.

--- GA long-budget run (40 runs, ~15 min with 8 cores) ---

    python scripts/run_experiment.py --config configs/core_ga_long.yaml --jobs 8

1M evaluations, GA only, seeds 0-9. Results go to results/raw/core_ga_long/.

--- Smoke test (single run, writes a convergence figure) ---

    python scripts/smoke_test.py                  # seed 0, n=100
    python scripts/smoke_test.py --n 25 --seed 3

================================================================
REPOSITORY LAYOUT
================================================================

configs/          YAML experiment configs (frozen; do not edit core.yaml)
src/evonn/        installable Python package
  data.py         synthetic data generator
  mlp.py          shared numpy MLP (41 parameters)
  budget.py       evaluation counter and timer
  metrics.py      train/val/test MSE, noise floor, gap
  ea/             CMA-ES (cmaes.py) and GA (ga.py, via deap)
  baselines/      L-BFGS-B (lbfgs.py) and SGD/Adam (sgd.py)
  experiment.py   runs one (method, n_train, seed) cell
scripts/          CLI entry points
tests/            pytest suite (139 tests)
results/raw/      per-run JSON files (gitignored, regenerable)
results/processed/ aggregated CSVs (committed)
figures/          report figures (committed)
docs/             project plan and decisions log
report/           final write-up (managed in Overleaf)

================================================================
RANDOM SEEDS AND REPRODUCIBILITY
================================================================

Every run is fully determined by (config, method, n_train, seed).
- Validation and test sets use data_seed=10000 (fixed for all runs).
- Training sets use default_rng(seed); weight init and optimisers use
  evonn.seeding.rng(seed, stream) to avoid reusing the data stream.
- The git commit hash is stored in every result JSON for traceability.
