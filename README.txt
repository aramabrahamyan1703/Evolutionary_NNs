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

    Expected output: 172 passed, 1 skipped (all green).

================================================================
REPRODUCING THE RESULTS
================================================================

Run everything from the repository root with the virtual environment active.
Each run writes one JSON file to results/raw/<experiment>/ (not in the
repository: regenerate it with the commands below; ~300 MB in total). Existing
files are skipped, so an interrupted command can simply be started again.
Aggregated tables go to results/processed/<experiment>/, figures to figures/.
Times are for an 11-core laptop (Apple Silicon) with the given --jobs.

Steps 1-4 fix the design (budget, tuned settings, time limits) and only need
to be rerun to check those decisions: their outcome is already frozen in
configs/core.yaml. Steps 5-8 produce every table and figure of the report.

--- 0. Check the installation (seconds) ---

    pytest                                     # 172 passed, 1 skipped
    python scripts/smoke_test.py               # all methods, seed 0, n = 100

--- 1. Budget pilot (decision D4; ~5 min) ---

    python scripts/pilot_budget.py             # CMA-ES and L-BFGS-B, seeds 0-2

Prints training / clean test MSE at 10k, 20k, 50k and 100k evaluations
(docs/PROJECT_PLAN.md, section 4.1; the n = 100 numbers are quoted in the report).

--- 2. Hyperparameter tuning (decision D9; 420 runs, ~15 min) ---

    python scripts/tune.py --jobs 8            # grids in configs/tuning.yaml

Seeds 100-104, n in {100, 400}; winners by mean rank of the training MSE.
-> results/processed/tuning_summary.csv, tuning_best.yaml (copied into
   configs/core.yaml).

--- 3. Checks with the tuned settings (80 runs, ~3 min) ---

    python scripts/run_experiment.py --config configs/core.yaml --first-seed 100 --seeds 5 --jobs 8 --out results/raw/check
    python scripts/check_runs.py results/raw/check/core_ntrain_sweep

Plateau over the second half of the budget and share of weights at +-5
(PROJECT_PLAN 4.2; quoted in the report's Limitations).

--- 4. Time limits for the wall-clock budget (16 runs, ~2 min, serial) ---

    python scripts/run_experiment.py --config configs/core.yaml --first-seed 100 --seeds 1 --jobs 1 --out results/raw/timing
    python scripts/check_runs.py results/raw/timing/core_ntrain_sweep

max_seconds(n) = median over the 4 methods of the time column, rounded up
(4 / 4 / 7 / 12 s, set in configs/core_wallclock.yaml).

--- 5. Core sweep (480 runs, ~13 min) ---

    python scripts/run_experiment.py --config configs/core.yaml --jobs 8
    python scripts/aggregate_results.py        # -> results/processed/core_ntrain_sweep/
    python scripts/make_figures.py             # -> figures/fig1..fig4

aggregate_results.py writes summary.csv (one row per run), medians.csv (median
and IQR per method and n) and wilcoxon.csv (paired Wilcoxon signed-rank test,
runs matched by seed, Holm correction over the 24 tests, Cliff's delta, and
95% paired-bootstrap intervals for the difference of medians and for A12).

--- 6. Robustness checks (each compared with the core sweep, seed by seed) ---

    python scripts/run_experiment.py  --config configs/<experiment>.yaml --jobs <j>
    python scripts/aggregate_results.py --config configs/<experiment>.yaml
    python scripts/compare_runs.py configs/<experiment>.yaml

  core_ga_long.yaml     GA only, 1M evaluations, seeds 0-9; 40 runs, --jobs 8, ~6 min
  core_gradcost1.yaml   grad_cost = 1 instead of 2 (L-BFGS-B and Adam only; CMA-ES
  core_gradcost3.yaml   and the GA never use gradients); 240 runs each, --jobs 8, ~10 min
  core_wallclock.yaml   wall-clock budget, per-n time limits; 480 runs, ~55 min.
                        Run with --jobs 1 on an otherwise idle machine, so the
                        timings are not distorted by parallel runs.

compare_runs.py writes compare_methods.csv (each method vs itself in the core
sweep) and compare_pairs.csv (does each method pair keep its verdict?).

--- 7. Fair model selection, sharpness, hidden size (Phase 5) ---

    python scripts/run_experiment.py  --config configs/p5_selection.yaml --jobs 8   # 480 runs, ~11 min
    python scripts/aggregate_results.py --config configs/p5_selection.yaml           # final networks
    python scripts/aggregate_results.py --config configs/p5_selection.yaml --selection best_select
    python scripts/sharpness.py --jobs 8       # final / early-stopped / matched networks, ~2 min
    python scripts/arch.py --jobs 8            # H in {2,4,6,10,15,20}: 2400 more runs, ~70 min
    python scripts/make_figures_p5.py          # -> figures/fig5..fig7

--- 8. Effect sizes of the GA vs the best other method (seconds) ---

    python scripts/effect_sizes.py             # -> results/processed/effect_sizes_ga.csv

--- Where each element of the report comes from ---

  Figure 1 (data)            notebooks/01_data_check.ipynb (run all cells) -> figures/data_surface.png
  Table 1 (settings)         configs/core.yaml (tuning: step 2)
  Table 2, Figures 2-3       step 5: core_ntrain_sweep/{medians,wilcoxon}.csv, fig1, fig4
  Robustness paragraphs      step 6: <experiment>/compare_methods.csv, compare_pairs.csv
  Table 3, Figure 4          step 7: p5_selection/{medians,wilcoxon}_best_select.csv, fig5
  Table 4, Figure 5          step 7: p5_arch/{medians_chosen,wilcoxon_chosen,chosen_h}.csv, fig7
  Table 5 (effect sizes)     step 8: effect_sizes_ga.csv
  Figure 6, sharpness        step 7: p5_selection/sharpness.csv, sharpness_wilcoxon.csv, fig6
  Pilot and checks           steps 1 and 3 (also in docs/PROJECT_PLAN.md, sections 4.1-4.2)

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
  baselines/      L-BFGS-B (lbfgs.py) and Adam (sgd.py, mini-batch SGD with Adam)
  experiment.py   runs one (method, n_train, seed) cell
  analysis.py     tables, paired Wilcoxon + Holm, Cliff's delta, A12, bootstrap intervals
  sharpness.py    Hessian top eigenvalue, perturbation sharpness (Phase 5)
  arch.py         choosing H and early stopping on the selection set (Phase 5)
scripts/          CLI entry points (one per step above)
notebooks/        data check and Figure 1
tests/            pytest suite (172 tests)
results/raw/      per-run JSON files (gitignored, regenerable)
results/processed/ aggregated CSVs, one folder per experiment (committed)
figures/          report figures (committed)
docs/             project plan and decisions log
report/           LaTeX source of the report (main.tex, references.bib)

================================================================
RANDOM SEEDS AND REPRODUCIBILITY
================================================================

Every run is fully determined by (config, method, n_train, seed).
- Validation and test sets use data_seed=10000 (fixed for all runs).
- Training sets use default_rng(seed); weight init and optimisers use
  evonn.seeding.rng(seed, stream) to avoid reusing the data stream.
- The git commit hash is stored in every result JSON for traceability.
