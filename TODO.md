# TODO

Legend: `[ ]` open · `[~]` in progress · `[x]` done. Phases match `docs/PROJECT_PLAN.md` §5.

## Phase 0: Setup
- [x] Folder structure, README, requirements, project plan, TODO
- [ ] Create venv, `pip install -r requirements.txt`, `pip install -e .`
- [ ] `pip freeze > requirements-lock.txt`
- [ ] First commit and push of the skeleton
- [ ] Read Yao (1999) [25] and the Hansen CMA-ES tutorial, and add notes to `docs/literature/`

## Phase 1: Foundations
- [ ] Confirm decisions D1–D7 (generator, noise, budget unit, budget size, H, model selection, seeds)
- [ ] `data.py`: generator with an exposed noise-free `f`, fixed val/test sets (5000 each), nested training sets
- [ ] Plot the generator surface plus a sample (sanity check) in `notebooks/01_data.ipynb`
- [ ] `mlp.py`: `n_params`, `unpack`, `forward`, `loss`, `loss_and_grad`
- [ ] `tests/test_mlp.py`: finite-difference gradient check (rel. error < 1e-6)
- [ ] `budget.py`: evaluation counter (with `grad_cost`) and `perf_counter` timer
- [ ] `tests/test_budget.py`: the counter stops exactly at `max_evals`
- [ ] `metrics.py`: train/val/test MSE, noise floor, gap
- [ ] **Smoke test:** CMA-ES vs L-BFGS-B, n=100, 1 seed, print train/test MSE

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
