README.txt — Training Neural Networks with Evolutionary Algorithms
Marta Mateu Lopez · Aram Abrahamyan
Computational Intelligence (CI-MAI), UPC, November 2026

================================================================
REQUIREMENTS
================================================================

Python 3.11 or later (pyproject.toml). All dependencies are pinned in
requirements-lock.txt. Results were produced with Python 3.14.0 on macOS.

================================================================
SETUP
================================================================

1. Clone the repository:

    git clone https://github.com/aramabrahamyan1703/Evolutionary_NNs.git
    cd Evolutionary_NNs

2. Create a virtual environment and install dependencies:

    python3 -m venv .venv
    source .venv/bin/activate          # on Windows: .venv\Scripts\activate
    # macOS with an iCloud-synced folder: use .venv.nosync and link it, see README.md
    pip install --upgrade pip
    pip install -r requirements-lock.txt
    pip install -e .

3. Verify the installation by running the test suite:

    pytest

    Expected output: 151 passed, 1 skipped (all green).

================================================================
REPRODUCING THE RESULTS
================================================================

All results in the paper come from configs/core.yaml (evaluation budget).
The raw JSON files are gitignored (regenerable); the aggregated CSVs are
committed to results/processed/<experiment>/ (one folder per experiment) and
the figures to figures/.

--- Core experiment (480 runs, ~17 min with 4 cores) ---

    python scripts/run_experiment.py --config configs/core.yaml --jobs 4

Results are written to results/raw/core_ntrain_sweep/ (one JSON per run).
Existing files are skipped, so an interrupted run can be restarted as-is.

--- Aggregate results and run statistical tests ---

    python scripts/aggregate_results.py                  # default: configs/core.yaml

Writes three files to results/processed/core_ntrain_sweep/:
  summary.csv   — one row per run (480 rows)
  medians.csv   — median + IQR per (method, n_train)
  wilcoxon.csv  — paired Wilcoxon signed-rank (runs matched by seed) for every
                  method pair at every n, Holm correction, Cliff's delta

--- Generate report figures (needs the aggregated core sweep) ---

    python scripts/make_figures.py

Writes four figures to figures/:
  fig1_test_train_vs_n.png     — main result figure (test + train MSE vs n)
  fig2_gap_vs_n.png            — generalisation gap vs n
  fig3_bounds_vs_n.png         — fraction of weights at bounds vs n
  fig4_convergence_seed0_n100.png — convergence curves (seed 0, n=100)

================================================================
SECONDARY ANALYSES
================================================================

Each secondary experiment has its own config, raw folder and processed folder.
Run it, aggregate it, then compare it with the core sweep (runs matched by seed;
needs the core sweep's raw results):

    python scripts/run_experiment.py  --config configs/<experiment>.yaml --jobs <j>
    python scripts/aggregate_results.py --config configs/<experiment>.yaml
    python scripts/compare_runs.py configs/<experiment>.yaml

compare_runs.py writes compare_methods.csv (each method vs itself in the core
sweep) and compare_pairs.csv (does each significant method pair stay
significant?) to results/processed/<experiment>/.

  core_wallclock.yaml   wall-clock budget, per-n time limits; 480 runs, ~1 h.
                        Run with --jobs 1 on an otherwise idle machine, so the
                        timings are not distorted by parallel runs.
  core_ga_long.yaml     GA only, 1M evaluations, seeds 0-9; 40 runs, --jobs 8
  core_gradcost1.yaml   grad_cost = 1 instead of 2 (L-BFGS-B and SGD only; CMA-ES
  core_gradcost3.yaml   and the GA never use gradients); 240 runs each, --jobs 8

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
  analysis.py     tables, paired Wilcoxon + Holm, Cliff's delta, comparisons
scripts/          CLI entry points
tests/            pytest suite (151 tests)
results/raw/      per-run JSON files (gitignored, regenerable)
results/processed/ aggregated CSVs, one folder per experiment (committed)
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
