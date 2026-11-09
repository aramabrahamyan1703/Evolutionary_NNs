# TODO

Legend: `[ ]` open · `[~]` in progress · `[x]` done. Phases match `docs/PROJECT_PLAN.md` §5.

## Phase 0: Setup
- [x] Folder structure, README, requirements, project plan, TODO
- [ ] Create venv, `pip install -r requirements-lock.txt`, `pip install -e .` (each team member)
- [x] `pip freeze > requirements-lock.txt`
- [x] First commit and push of the skeleton
- [ ] Read Yao (1999) [25] and the Hansen CMA-ES tutorial, and add notes to `docs/literature/`

## Phase 1: Foundations
- [x] Decisions D1, D2, D7 (generator, noise, seeds) fixed by `data.py`
- [ ] Confirm decisions D3–D6 (budget unit, budget size, H, model selection)
- [x] `seeding.py`: separate random streams for weight init and optimisers (D8) + `tests/test_seeding.py`
- [x] `tests/test_config.py`: config and `data.py` constants must agree
- [ ] Make `experiment.py` pass config values into `generate_datasets` (then the yaml is the only source)
- [x] `data.py`: generator with an exposed noise-free `f`, fixed val/test sets (5000 each), nested training sets
- [x] `tests/test_data.py`: 14 tests covering shapes, nesting, reproducibility, leakage guard, noise floor
- [x] Visual check in `notebooks/01_data_check.ipynb` (surface, noise, floor, nested sets, report figure)
- [x] `mlp.py`: `MLP` dataclass with `n_params`, `pack`/`unpack`, `init`, `forward`, `loss`, `loss_and_grad`, `loss_many`
- [x] `tests/test_mlp.py`: 26 tests incl. finite-difference gradient check (rel. error < 1e-6)
- [x] `budget.py`: `CountedObjective` (counting, `BudgetExhausted`, best-so-far, trace, checkpoint snapshots, time limit)
- [x] `tests/test_budget.py`: 17 tests, the counter never exceeds `max_evals`
- [x] `metrics.py` + `tests/test_metrics.py`: train/val/test MSE, excess, clean test MSE, gap, weights at bounds
- [x] **Smoke test:** `scripts/smoke_test.py` (CMA-ES vs L-BFGS-B, same start, restarts, convergence figure)
- [ ] Decide D4 from the smoke test: CMA-ES never converged in 20k evals (0 restarts); try 50k–100k

## Phase 2: Optimisers
- [ ] `ea/cmaes.py`: bounded CMA-ES, logs best-so-far against evals
- [ ] `baselines/lbfgs.py`: L-BFGS-B with bounds and random restarts that fill the budget
- [ ] `ea/ga.py`: DEAP real-coded GA (tournament, blend crossover, Gaussian mutation, clipping)
- [ ] `baselines/sgd.py` (optional): mini-batch SGD/Adam on the shared MLP
- [ ] Common result dict / JSON schema for all methods
- [ ] `experiment.py` + `scripts/run_experiment.py`: run one cell from the config and write JSON

## Phase 3: Pilot and tuning
- [ ] Convergence curves for each method at n=100 (5 seeds), then fix the budget (D4)
- [ ] Tune the EA hyperparameters and L-BFGS restarts with the same effort, on the validation set
- [ ] Check the number of weights at ±5 bounds
- [ ] Freeze `configs/core.yaml`

## Phase 4: Core experiment
- [ ] Full sweep: 4 sizes × methods × 30 seeds (evaluation budget)
- [ ] Same sweep with the wall-clock budget (or report wall time of the evaluation-budget runs)
- [ ] `grad_cost` sensitivity check (1 vs 2 vs 3)
- [ ] `scripts/aggregate_results.py`: CSV tables, Wilcoxon + Holm, effect sizes
- [ ] `scripts/make_figures.py`: test−σ² vs n, train vs n, convergence curves

## Phase 5: Extensions (choose)
- [ ] E1: evolve the hidden size H with validation fitness. Does H track n?
- [ ] E2: evolve weight decay λ
- [ ] E3: noise sweep / d=5 robustness check
- [ ] E4: sharpness of the minima found

## Phase 6: Report
- [ ] Outline in `report/`
- [ ] Methods and setup section (every decision, with justification)
- [ ] Results and discussion of H1
- [ ] Limitations and future work
- [ ] Final proofread, then tag the release
