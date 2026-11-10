# Project Plan: Training Neural Networks with Evolutionary Algorithms

_Last updated: 2026-11-10_

## 1. Where we are

| Item | Status |
|---|---|
| Repo skeleton, requirements, plan, TODO | ✅ done |
| Core decisions (D1–D9 below) | ✅ all confirmed (D3–D6 from measurements, see §4.1) |
| Code (`src/evonn`) | ✅ Phase 1 · ✅ Phase 2: CMA-ES, GA, L-BFGS-B, SGD, experiment runner · ✅ Phase 3: tuning (138 tests) |
| Experiments | ✅ core sweep (480 runs, 30 seeds) · ✅ wall-clock sweep · ✅ GA long run (1M evals) · ✅ aggregate + figures |
| Report | 🟡 in progress (§1–§6 drafted in Overleaf) |

**Next step:** grad_cost sensitivity check + report finalisation (README.txt, proofread, tag release).

---

## 2. Research question and hypothesis

**RQ1.** With the same computational budget, does an EA find MLPs that **generalise** better than a
derivative-based method, or only networks that **fit the training data** equally well?

**RQ2.** Does the answer depend on the training-set size *n*?

**H1 (working hypothesis).** The EA doesn't follow the gradient into the nearest sharp minimum.
Its advantage, if it has one, therefore shows up as a **lower test error at small n** (and/or high
noise). The advantage shrinks to zero at large n, where every reasonable minimiser generalises
about equally well.

**What would falsify H1:** the EA's test error is ≥ the baseline's at every n, or its advantage
grows with n. Another case is when the EA's lower test error comes purely from a **higher
training error** (it under-fits rather than finding flatter minima). To tell these apart we always
report training and test error side by side.

---

## 3. Experimental design (core)

### 3.1 Data
- **Synthetic generator** with a known noise-free function `f`. Targets are `y = f(x) + ε`, `ε ~ N(0, σ²)`.
  - For regression the **noise floor is σ²**, the best achievable test MSE. We report every test error relative to it.
- Input dimension `d = 2` (default, easy to visualise). Possibly `d = 5` as a robustness check.
- **Validation set:** 5000 points, **test set:** 5000 points. Both are generated once from `data_seed` and fixed for every run.
- **Training sets:** for each seed we draw one pool of `max(n_train)` points. The smaller sets are **nested prefixes** of that pool. This reduces variance between sizes.
- **Single factor:** `n_train ∈ {25, 100, 400, 1600}`, which is geometric with ×4 steps. With 41 parameters,
  n = 25 is clearly over-parameterised and n = 1600 is clearly data-rich.

### 3.2 Model
- MLP with 1 hidden layer, `H = 10` tanh units, linear output.
  For d = 2 that gives `d·H + H + H + 1 = 41` parameters.
- **One shared numpy implementation** (`src/evonn/mlp.py`) used by every method. The weights are a flat vector θ and backprop is manual.
  This guarantees the methods optimise exactly the same function and lets us count evaluations honestly.
- **Weights bounded in [−5, 5] for all methods** (the CMA-ES bounds, the GA clipping, the L-BFGS-B bounds), so no method searches a bigger space.
- **Same initial-weight distribution** for all methods (e.g. U(−0.5, 0.5)).

### 3.3 Methods
| Role | Method | Library | Notes |
|---|---|---|---|
| Primary EA | CMA-ES | `cma` | Full covariance is cheap at ~40–120 params |
| Secondary EA | Real-coded GA | `deap` | Tournament selection, blend crossover, Gaussian mutation |
| Primary baseline | L-BFGS-B | `scipy.optimize` | Full-batch, deterministic, bounded |
| Secondary baseline | Mini-batch SGD (Adam by default) | own numpy | A more "typical" deep-learning optimiser; mini-batch noise is a second implicit regulariser |
| Sanity check | `MLPRegressor` | `scikit-learn` | Only to check our numbers are in a plausible range |

### 3.4 Budget (what we equalise and why)
- **Primary:** the **number of error-function evaluations** (full-training-set loss computations).
  - It is hardware- and implementation-independent, it is reproducible, and it is the natural unit for EAs.
  - A loss + gradient call costs `grad_cost` evaluations (decision **D3**). Reverse-mode backprop costs
    1.8–2.3 loss evaluations in our implementation (measured, §4.1), so `grad_cost = 2`, with a sensitivity check at 1 and 3.
- **Mini-batch gradients (SGD):** a gradient on b of the n training points costs `grad_cost · b/n` evaluations, so one
  epoch costs one full gradient. Mini-batch losses are not full training losses, so SGD evaluates the full training loss
  every 5 epochs (cost 1 each, ~9% of its budget) and those evaluations pick the returned network (D6).
- **Secondary:** **wall-clock time** (`time.perf_counter()`), reported for the same runs. This
  shows whether the conclusion survives the EA's per-generation overhead (sampling, covariance updates).
- **Budget-filling:** L-BFGS usually converges long before the budget runs out. Leftover budget goes to
  **random restarts**, and the final model is the one with the best training loss. Otherwise the baseline
  would effectively get a smaller budget.
- **Budget size (D4):** 100 000 evaluations. Results at 10k, 20k and 50k come from the checkpoint snapshots of the
  same runs (a run cut at B evaluations returns exactly what a budget-B run would), so the budget sensitivity is free.
- **Convergence curves:** we log the best-so-far train/val/test error against evaluations, so conclusions don't
  hinge on one arbitrary budget value.
- **Pilot finding:** at n ≤ 100, CMA-ES's test error *rises* with budget (n = 100: clean test MSE 0.039 at 10k → 0.072
  at 100k). An under-converged EA looks like it generalises better because it stops early, without being told to. The main
  result is therefore reported at the converged budget, always next to the budget curve.

### 3.5 Model selection inside a run
- **Core:** the returned network is the one with the **lowest training loss** seen within the budget, the same rule
  for every method. No early stopping on validation. That isolates the question "what kind of minimum does each optimiser find?"
- **Also logged:** each restart's final network (train/val/test) and the best-validation checkpoint, so selection by
  validation can be reported as a secondary analysis without re-running anything.
- The validation set is used only in the extensions (architecture and weight-decay evolution) and for an optional early-stopping ablation.

### 3.6 Measurements per run (one JSON file each)
See the docstring of `src/evonn/experiment.py`: meta (incl. full config and git commit), budget used, final metrics,
final weights, convergence curve, best-validation checkpoint and per-restart results.

### 3.7 Repetitions and statistics
- **30 seeds** per (method × n_train) cell. The seed fixes the training sample and initialisation, and is **shared across methods**, so comparisons are paired.
- For each n: a **Wilcoxon signed-rank** test on test MSE (EA vs baseline), with Holm correction across the 4 sizes, plus an effect size
  (median difference and/or Cliff's δ).
- Main figure: **test MSE − σ² vs n_train** (log-log), one line per method, with median and IQR. Next to it, the same plot for **training MSE**.

---

## 4. Decisions log

| ID | Decision | Proposed | Status | Rationale |
|---|---|---|---|---|
| D1 | Task / generator | Regression, f(x) = sin(πx₁) + sin(πx₂) on [−1,1]² | ✅ | Regression gives an exact noise floor σ² |
| D2 | Noise level | σ = 0.3, noise floor 0.09 (try 0.1 and 0.5 if time allows) | ✅ | "Noisy" is part of the hypothesis |
| D3 | Budget unit | Evaluations, loss+grad = 2 evals; wall-clock as secondary; sensitivity at 1 and 3 | ✅ | Measured cost ratio 1.8–2.3 (§4.1) |
| D4 | Budget size | 100 000 evals; also report 10k/20k/50k from snapshots | ✅ | CMA-ES training MSE changes < 1% from 50k to 100k for n ≥ 100; at 20k it is still 4–8% above its plateau (§4.1) |
| D5 | Hidden size | H = 10 (41 parameters) | ✅ | p/n ranges from 1.6 (n = 25, over-parameterised) to 0.026 (n = 1600); at n = 1600 both methods reach clean test MSE ≈ 0.0026 (3% of σ²), so H = 10 can represent f |
| D6 | Final-model selection | Best training loss within budget, same for all methods; per-restart and best-validation results also logged | ✅ | Isolates the optimiser's effect; logging the rest allows a validation-selection analysis later |
| D7 | Seeds | 30 (run seeds 0–29, `DATA_SEED` = 10000 for val/test) | ✅ | Enough for paired non-parametric tests |
| D8 | Random streams | Data uses `default_rng(seed)`; weight init and optimisers use `evonn.seeding.rng(seed, stream)` | ✅ | Initial weights never reuse the training data's random numbers |
| D9 | Tuning protocol | 8 settings per method, tuning seeds 100–104, n ∈ {100, 400}, full budget; winner = lowest mean rank of final **training** MSE (`configs/tuning.yaml`). Amendment: **one** edge-extension round for every method (a winner at the edge of an ordered parameter gets one step past it); winners re-ranked over both rounds and frozen in `configs/core.yaml` (§4.2) | ✅ | Each method must be as good an optimiser as it can be; tuning on validation error would reward settings that win by under-fitting (hidden early stopping). Validation error is recorded for a secondary table |

Whenever you change a default from the assignment (d = 2–5, H = 5–20, weights in [−5,5], test = 5000), write down the reason here.

### 4.1 Evidence (2026-11-09)

**D3: cost of loss + gradient vs loss** (`MLP`, 41 parameters, mean of 3000 calls):

| n | 25 | 100 | 400 | 1600 |
|---|---|---|---|---|
| ratio | 2.26 | 2.15 | 1.91 | 1.81 |

**D4: budget pilot** (`python scripts/pilot_budget.py`; seeds 0–2, median train MSE with clean test MSE in brackets):

| n | method | 10k | 20k | 50k | 100k |
|---|---|---|---|---|---|
| 25 | CMA-ES | 0.0249 (0.346) | 0.0115 (1.080) | 0.0034 (1.723) | 0.0011 (2.781) |
| 25 | L-BFGS-B | 0.0002 (2.015) | 0.0000 (2.011) | 0.0000 (2.986) | 0.0000 (2.434) |
| 100 | CMA-ES | 0.0660 (0.039) | 0.0577 (0.046) | 0.0536 (0.068) | 0.0532 (0.072) |
| 100 | L-BFGS-B | 0.0494 (0.085) | 0.0475 (0.092) | 0.0448 (0.099) | 0.0440 (0.099) |
| 400 | CMA-ES | 0.0906 (0.0080) | 0.0891 (0.0067) | 0.0850 (0.0117) | 0.0847 (0.0120) |
| 400 | L-BFGS-B | 0.0855 (0.0111) | 0.0845 (0.0125) | 0.0845 (0.0125) | 0.0839 (0.0147) |
| 1600 | CMA-ES | 0.0916 (0.0029) | 0.0869 (0.0018) | 0.0862 (0.0022) | 0.0859 (0.0026) |
| 1600 | L-BFGS-B | 0.0857 (0.0028) | 0.0856 (0.0026) | 0.0854 (0.0026) | 0.0854 (0.0026) |

Three seeds only: this sets the budget, it is not a result. At n = 25 both methods interpolate the training set
(train MSE → 0), so the training curve never flattens there; the budget is set by n ≥ 100.

### 4.2 Tuning and checks (2026-11-10)

**Tuning winners** (D9, `python scripts/tune.py --jobs 8`, 420 runs = 8 settings + edge extension per method;
full table in `results/processed/tuning_summary.csv`), frozen in `configs/core.yaml`.
Settings tried: CMA-ES 9, GA 15, L-BFGS-B 9, SGD 9 (same rule for all; the GA had three parameters at an edge).
Runs are stamped `fcc4242-dirty` (Phase 3 code uncommitted at the time); the commit that adds this code: `8ff680e`.

| Method | Winner | Train MSE n=100 / 400 | Sensitivity |
|---|---|---|---|
| CMA-ES | σ₀ 0.3, popsize default (15) | 0.058 / 0.086 | small: all 9 settings within 0.056–0.060 / 0.084–0.087 |
| GA | mut_sigma 0.03, mut_indpb 0.1, pop 100 | 0.096 / 0.103 | moderate; winner again at the edge (σ 0.03), not extended further (one round only) |
| L-BFGS-B | maxcor 50, ftol 1e-12 | 0.051 / 0.083 | negligible |
| SGD (Adam) | lr 0.01, batch 64 | 0.058 / 0.085 | moderate (lr) |

**Checks with the tuned settings** (`scripts/check_runs.py`; seeds 100–104, all n, 100k evals; medians):

| n | CMA-ES train (drop½) | GA train (drop½) | L-BFGS-B train (drop½) | SGD train (drop½) | weights at ±5: CMA / GA / L-BFGS / SGD |
|---|---|---|---|---|---|
| 25 | 0.0000 | 0.053 (21%) | 0.0000 | 0.0000 | 10 / 2 / 5 / 7 % |
| 100 | 0.058 (7.2%) | 0.096 (9.0%) | 0.051 (0.7%) | 0.058 (3.2%) | 12 / 0 / 17 / 10 % |
| 400 | 0.086 (1.3%) | 0.103 (7.6%) | 0.083 (0.4%) | 0.085 (0.6%) | 7 / 0 / 10 / 10 % |
| 1600 | 0.087 (0.3%) | 0.101 (7.1%) | 0.086 (0.2%) | 0.088 (0.0%) | 5 / 0 / 7 / 2 % |

drop½ = relative improvement of the training MSE over the second half of the budget (50k → 100k).

Findings:
1. **CMA-ES, L-BFGS-B and SGD have levelled off by 100k** (≤ 1.3% at n ≥ 400). At n = 25 all three interpolate the data.
2. **The GA has not levelled off and underfits**: its training MSE stays above the noise floor (0.09) for n ≥ 100 and
   still falls 7–9% over the second half. At 100k evaluations a GA comparison measures "fits less", not
   "generalises better". This is reported as a result (a GA needs more evaluations than CMA-ES in 41 dimensions), and
   a longer-budget GA run is a Phase 4 secondary analysis.
3. **Weight bounds act as a regulariser for some methods only**: 5–17% of the weights of CMA-ES, L-BFGS-B and SGD
   sit at ±5, the GA's (almost) none. Keep this in mind when interpreting test errors.

**Wall-clock limits** (secondary budget, D3): `max_seconds(n)` = median over the 4 methods of the serial (`--jobs 1`)
wall time of a 100k-evaluation run (seed 100), rounded up: n = 25 → 4 s, 100 → 4 s, 400 → 7 s, 1600 → 12 s.
Serial times (s), CMA-ES / GA / L-BFGS-B / SGD: n=25 4.6/2.2/3.9/1.1, n=100 4.9/2.5/4.3/2.2, n=400 6.5/4.2/5.9/7.8,
n=1600 11.8/9.3/10.5/28.2. SGD is the slowest per evaluation at large n (many small Python steps).

---

## 5. Roadmap

### Phase 0: Setup ✅
Repo, venv, requirements, plan, TODO.

### Phase 1: Foundations (shared code)
- Data generator with nested training sets and fixed val/test sets.
- numpy MLP with flat θ, forward, loss, and `loss_and_grad`, plus a **finite-difference gradient check**.
- An evaluation counter and timer wrapper.
- **Smoke test:** CMA-ES vs L-BFGS-B at n = 100 with one seed, printing train and test MSE.

### Phase 2: Optimisers
- CMA-ES (bounded), GA (DEAP), L-BFGS-B with restarts, mini-batch SGD (Adam).
- Each one returns the same result dict and respects the budget exactly.

### Phase 3: Pilot and tuning ✅
✅ Tuned every method with the same protocol (D9, `python scripts/tune.py --jobs 8`), checked plateaus, bounds and
timing (`scripts/check_runs.py`), set the wall-clock limits and froze `configs/core.yaml` (§4.2).

### Phase 4: Core experiment
- 4 sizes × 4 methods × 30 seeds with `configs/core.yaml` (evaluation budget), then the same with
  `configs/core_wallclock.yaml` (per-n time limits, own result folder, run with `--jobs 1`).
- Secondary (confirmed): GA with a 1M-evaluation budget, seeds 0–9, all n, to see whether its test-error difference survives
  once it fits as well as the others (§4.2, finding 2).
- Aggregation, statistical tests, and the main figures.

### Phase 5: Extensions (pick one or two)
- **E1:** Evolve H (number of hidden units) together with the weights, using **validation error** as fitness.
  Does the chosen H grow sensibly with n?
- **E2:** Evolve the weight-decay coefficient λ. Compare with L-BFGS where λ is picked by a validation grid.
- **E3:** Noise sweep (σ as a second factor) and a robustness check at d = 5.
- **E4:** A sharpness analysis of the minima found (Hessian top eigenvalue, or loss increase under random perturbation).

### Phase 6: Write-up
Report: introduction, method, setup (all decisions and why), results, discussion of the hypothesis, limitations.

---

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| The EA doesn't reach a low training loss within the budget, so any "generalisation advantage" is really under-fitting | Always report train and test error together. Check convergence curves. **Happened for the GA (§4.2)**: report it, plus a longer-budget GA run (Phase 4) |
| Unequal tuning effort favours one method | Same tuning protocol and budget for every method (Phase 3) |
| The budget unit is disputed | Report both evaluations and wall-clock, plus the `grad_cost` sensitivity check |
| Weights saturate at ±5 bounds (tanh saturation) | Log how many weights sit at the bounds (`frac_at_bounds`). Measured: 5–17% for CMA-ES / L-BFGS-B / SGD, ~0% for the GA (§4.2) |
| Compute time | ~41 params and ≤1600 points is cheap. Parallelise seeds with `joblib`/multiprocessing |
| Results that aren't reproducible | Seeds everywhere, one JSON row per run, config stored with each result |
