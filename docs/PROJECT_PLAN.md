# Project Plan: Training Neural Networks with Evolutionary Algorithms

_Last updated: 2026-10-25_

## 1. Where we are

| Item | Status |
|---|---|
| Repo skeleton, requirements, plan, TODO | ✅ done |
| Core decisions (D1–D7 below) | 🟡 proposed, need confirming |
| Code (`src/evonn`) | ⬜ stubs only |
| Experiments | ⬜ not started |
| Report | ⬜ not started |

**Next step:** Phase 1, which is building the shared MLP plus a gradient check, then a smoke test comparing CMA-ES and L-BFGS on a single training-set size.

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
| Secondary baseline | Mini-batch SGD/Adam | own numpy | Optional, a more "typical" deep-learning optimiser |
| Sanity check | `MLPRegressor` | `scikit-learn` | Only to check our numbers are in a plausible range |

### 3.4 Budget (what we equalise and why)
- **Primary:** the **number of error-function evaluations** (full-training-set loss computations).
  - It is hardware- and implementation-independent, it is reproducible, and it is the natural unit for EAs.
  - A loss + gradient call costs `grad_cost` evaluations (decision **D3**). Reverse-mode backprop costs
    roughly 2–3 forward passes, so we use `grad_cost = 2` as the default and report a sensitivity check with 1 and 3.
- **Secondary:** **wall-clock time** (`time.perf_counter()`), reported for the same runs. This
  shows whether the conclusion survives the EA's per-generation overhead (sampling, covariance updates).
- **Budget-filling:** L-BFGS usually converges long before the budget runs out. Leftover budget goes to
  **random restarts**, and the final model is the one with the best training loss. Otherwise the baseline
  would effectively get a smaller budget.
- **Convergence curves:** we log the best-so-far train/val/test error against evaluations, so conclusions don't
  hinge on one arbitrary budget value.

### 3.5 Model selection inside a run
- **Core:** the returned network is the one with the **lowest training loss**. No early stopping on validation.
  That isolates the question "what kind of minimum does each optimiser find?"
- The validation set is used only in the extensions (architecture and weight-decay evolution) and for an optional early-stopping ablation.

### 3.6 Measurements per run (one JSON row each)
`method, n_train, seed, train_mse, val_mse, test_mse, noise_floor, evals_used, wall_time, final θ norm, (optional) sharpness proxy`

### 3.7 Repetitions and statistics
- **30 seeds** per (method × n_train) cell. The seed fixes the training sample and initialisation, and is **shared across methods**, so comparisons are paired.
- For each n: a **Wilcoxon signed-rank** test on test MSE (EA vs baseline), with Holm correction across the 4 sizes, plus an effect size
  (median difference and/or Cliff's δ).
- Main figure: **test MSE − σ² vs n_train** (log-log), one line per method, with median and IQR. Next to it, the same plot for **training MSE**.

---

## 4. Decisions log

| ID | Decision | Proposed | Status | Rationale |
|---|---|---|---|---|
| D1 | Task / generator | Regression, smooth nonlinear `f` on [−1,1]² (sum of sines). Alternative: random "teacher" MLP | 🟡 | Regression gives an exact noise floor σ² |
| D2 | Noise level | σ = 0.3 (try 0.1 and 0.5 if time allows) | 🟡 | "Noisy" is part of the hypothesis |
| D3 | Budget unit | Evaluations, with loss+grad = 2 evals, plus wall-clock as a secondary measure | 🟡 | See §3.4 |
| D4 | Budget size | 20 000 evals (to be tuned in the pilot) | 🟡 | Must be large enough that both methods have roughly converged |
| D5 | Hidden size | H = 10 | 🟡 | Middle of the suggested 5–20 range |
| D6 | Final-model selection | Best training loss, no early stopping | 🟡 | Isolates the optimiser's effect |
| D7 | Seeds | 30 | 🟡 | Enough for paired non-parametric tests |

Whenever you change a default from the assignment (d = 2–5, H = 5–20, weights in [−5,5], test = 5000), write down the reason here.

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
- CMA-ES (bounded), GA (DEAP), L-BFGS-B with restarts, and optionally SGD.
- Each one returns the same result dict and respects the budget exactly.

### Phase 3: Pilot and tuning
- A few seeds per method to choose the budget (D4) and sensible EA hyperparameters (σ₀, population size, mutation).
- **Tune every method with the same effort**, ideally on the validation set at one middle n, then freeze the settings.

### Phase 4: Core experiment
- 4 sizes × 2–4 methods × 30 seeds, both budget types.
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
| The EA doesn't reach a low training loss within the budget, so any "generalisation advantage" is really under-fitting | Always report train and test error together. Check convergence curves. Consider a larger budget |
| Unequal tuning effort favours one method | Same tuning protocol and budget for every method (Phase 3) |
| The budget unit is disputed | Report both evaluations and wall-clock, plus the `grad_cost` sensitivity check |
| Weights saturate at ±5 bounds (tanh saturation) | Log how many weights sit at the bounds. Use a small init scale |
| Compute time | ~41 params and ≤1600 points is cheap. Parallelise seeds with `joblib`/multiprocessing |
| Results that aren't reproducible | Seeds everywhere, one JSON row per run, config stored with each result |
