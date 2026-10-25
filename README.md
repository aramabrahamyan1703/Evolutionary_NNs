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
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .            # makes `import evonn` work from scripts/, tests/, notebooks/
pytest                      # run the tests
```

Run an experiment (once implemented):

```bash
python scripts/run_experiment.py --config configs/default.yaml
python scripts/aggregate_results.py
python scripts/make_figures.py
```

## Repository layout

```
.
├── configs/              # YAML experiment configs (one per experiment)
├── src/evonn/            # the installable package
│   ├── data.py           #   synthetic generators (expose the noise-free f for the Bayes error)
│   ├── mlp.py            #   shared numpy MLP, flat weight vector, manual backprop
│   ├── budget.py         #   evaluation counter + timer (makes the comparison fair)
│   ├── metrics.py        #   train/val/test error, noise floor, generalisation gap
│   ├── experiment.py     #   runs one (method, n_train, seed) cell -> one JSON row
│   ├── ea/               #   cmaes.py (primary), ga.py (DEAP)
│   └── baselines/        #   lbfgs.py (scipy), sgd.py
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
- After installing, freeze exact versions with `pip freeze > requirements-lock.txt`.
