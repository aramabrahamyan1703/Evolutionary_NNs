# Training Neural Networks with Evolutionary Algorithms

Do evolutionary algorithms (CMA-ES, GA) train MLPs that **generalise** better than
derivative-based methods (L-BFGS, SGD) under the **same computational budget**, or do
they just fit the training data equally well? And does the answer depend on the
**training-set size**?

**Working hypothesis:** the EA doesn't follow the gradient into the nearest sharp
minimum, so any advantage it has should show up with small or noisy training samples
and disappear with large ones.

See [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) for the full design, decisions log
and roadmap, and [`TODO.md`](TODO.md) for current tasks.

## Quickstart

```bash
python3 -m venv .venv.nosync && ln -s .venv.nosync .venv   # .nosync: iCloud won't sync/hide it (macOS)
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-lock.txt   # exact versions, so everyone gets the same results
pip install -e .            # makes `import evonn` work from scripts/, tests/, notebooks/
pytest                      # run the tests
```

Smoke test (all methods on one training set, writes a convergence figure to `figures/`):

```bash
python scripts/smoke_test.py                 # seed 0, n = 100
python scripts/smoke_test.py --n 25 --max-evals 50000
```

Run an experiment (one JSON per run in `results/raw/<experiment>/`; existing runs are skipped):

```bash
python scripts/tune.py --jobs 8                              # Phase 3 tuning (configs/tuning.yaml)
python scripts/run_experiment.py --jobs 4                    # core sweep: everything in configs/core.yaml (frozen)
python scripts/run_experiment.py --seeds 2 --methods cmaes   # a subset
python scripts/check_runs.py results/raw/core_ntrain_sweep   # plateau, weights at bounds, time per run
python scripts/aggregate_results.py                          # JSONs -> results/processed/<experiment>/*.csv + Wilcoxon tests
python scripts/make_figures.py                               # report figures -> figures/
python scripts/compare_runs.py configs/core_gradcost1.yaml   # a secondary experiment vs the core sweep
```

## Repository layout

```
.
├── configs/              # YAML experiment configs (one per experiment)
├── src/evonn/            # the installable package
│   ├── data.py           #   synthetic generators (expose the noise-free f for the Bayes error)
│   ├── mlp.py            #   shared numpy MLP, flat weight vector, manual backprop
│   ├── budget.py         #   evaluation counter + timer (makes the comparison fair)
│   ├── seeding.py        #   separate random streams per run (init, optimiser)
│   ├── metrics.py        #   train/val/test error, noise floor, generalisation gap
│   ├── methods.py        #   registry of training methods (shared interface)
│   ├── experiment.py     #   runs one (method, n_train, seed) cell -> one JSON file
│   ├── ea/               #   cmaes.py (primary), ga.py (DEAP)
│   └── baselines/        #   lbfgs.py (scipy), sgd.py (mini-batch Adam / momentum)
├── scripts/              # CLI entry points: run, aggregate, plot
├── tests/                # pytest (gradient check, budget accounting, ...)
├── notebooks/            # exploration only; anything reusable moves into src/
├── results/
│   ├── raw/              # per-run JSON (gitignored, regenerable)
│   └── processed/        # aggregated CSV tables (committed)
├── figures/              # figures for the report (committed)
├── docs/                 # project plan, literature notes
└── report/               # final write-up
```

## Reproducibility

- Every run is identified by `(config, method, n_train, seed)` and writes one JSON row to `results/raw/`.
- Validation and test sets are generated once from a fixed `data_seed`; training sets are nested subsets.
- `requirements.txt` lists the packages with loose versions (edit this one). After changing it, reinstall and
  refresh the lock file with `pip freeze > requirements-lock.txt`.
- Random streams: data uses `default_rng(seed)`; everything else uses `evonn.seeding.rng(seed, stream)`.
