# Project Plan: Training Neural Networks with Evolutionary Algorithms

_Last updated: 2026-11-11_

## 1. Where we are

| Item | Status |
|---|---|
| Repo skeleton, requirements, plan, TODO | ✅ done |
| Core decisions (D1–D11 below) | ✅ all confirmed (D3–D6 from measurements, see §4.1) |
| Code (`src/evonn`) | ✅ Phase 1 · ✅ Phase 2: CMA-ES, GA, L-BFGS-B, SGD, experiment runner · ✅ Phase 3: tuning · ✅ Phase 4: analysis, aggregation, comparisons, figures · ✅ Phase 5: selection sets, sharpness, H selection (169 tests) |
| Experiments | ✅ Phase 4: core, wall-clock, GA long run, `grad_cost` 1 / 3 (§4.3) · ✅ Phase 5: fair early stopping, sharpness, H selection (§4.4) |
| Report | 🟡 in progress (§1–§6 drafted in Overleaf) |

**Next step:** the report (Phase 6), with §4.3 and §4.4 as the results. Report figures to revisit: Fig 2
(linear y: n = 25 flattens the rest) and Fig 1b (log axis down to 1e-13 hides the GA's higher training MSE at n ≥ 100).

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
| D3 | Budget unit | Evaluations, loss+grad = 2 evals; wall-clock as secondary; sensitivity at 1 and 3 | ✅ | Measured cost ratio 1.8–2.3 (§4.1). Checked: `grad_cost` 1 or 3 and the wall-clock budget leave the rankings and the main conclusions unchanged (§4.3) |
| D4 | Budget size | 100 000 evals; also report 10k/20k/50k from snapshots | ✅ | CMA-ES training MSE changes < 1% from 50k to 100k for n ≥ 100; at 20k it is still 4–8% above its plateau (§4.1) |
| D5 | Hidden size | H = 10 (41 parameters) | ✅ | p/n ranges from 1.6 (n = 25, over-parameterised) to 0.026 (n = 1600); at n = 1600 both methods reach clean test MSE ≈ 0.0026 (3% of σ²), so H = 10 can represent f |
| D6 | Final-model selection | Best training loss within budget, same for all methods; per-restart and best-validation results also logged | ✅ | Isolates the optimiser's effect; logging the rest allows a validation-selection analysis later |
| D7 | Seeds | 30 (run seeds 0–29, `DATA_SEED` = 10000 for val/test) | ✅ | Enough for paired non-parametric tests |
| D8 | Random streams | Data uses `default_rng(seed)`; weight init and optimisers use `evonn.seeding.rng(seed, stream)` | ✅ | Initial weights never reuse the training data's random numbers |
| D9 | Tuning protocol | 8 settings per method, tuning seeds 100–104, n ∈ {100, 400}, full budget; winner = lowest mean rank of final **training** MSE (`configs/tuning.yaml`). Amendment: **one** edge-extension round for every method (a winner at the edge of an ordered parameter gets one step past it); winners re-ranked over both rounds and frozen in `configs/core.yaml` (§4.2) | ✅ | Each method must be as good an optimiser as it can be; tuning on validation error would reward settings that win by under-fitting (hidden early stopping). Validation error is recorded for a secondary table |
| D10 | Selection set (Phase 5) | Per run, n points (as many as the training set) from their own random stream (`SELECTION`), nested over n; picks the early-stopping checkpoint (5.1) and H (5.3), the same way for every method | ✅ | A realistic validation set: the 5000-point one is far larger than the training set and remains a measurement only. Selection only chooses among networks fitted on the training set |
| D11 | Sharpness measure (Phase 5.2) | Mean training-loss increase under 200 random perturbations of norm 1% of ‖θ‖ (primary, scale-aware); top Hessian eigenvalue (secondary, with ‖θ‖ and share at bounds next to it); methods compared at the GA's training MSE | ✅ | The Hessian eigenvalue grows with the weight scale, and the non-GA methods have larger weights; comparing at equal training error removes "how far training went" |

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

### 4.3 Phase 4 results (2026-11-11)

All runs at commit `cbc64ec`. Runs stamped `cbc64ec-dirty` differ from it only in `.gitignore` (an entry for an
iCloud duplicate folder, added during the core sweep); no code or config changed. Raw runs: `results/raw/<experiment>/`
(not in git); tables: `results/processed/<experiment>/`; figures: `figures/`.

**Reproduction check.** The rerun of the core sweep matches Marta's first run (commit `b582eb2`) exactly for the GA,
L-BFGS-B and SGD (120/120 runs each). CMA-ES differs in all runs (her environment, most likely a different `cma`/numpy
version; our CMA-ES reproduces itself exactly). Same verdict in 23 of 24 method pairs; the exception is CMA-ES vs
L-BFGS-B at n = 100, borderline in both (p_holm 0.073 here, 0.028 there). The rerun is the record.

**Core sweep** (`core_ntrain_sweep`, 100k evaluations, 30 seeds; medians; clean test MSE is against the noise-free f,
floor 0):

| n | CMA-ES train / test | GA train / test | L-BFGS-B train / test | SGD train / test |
|---|---|---|---|---|
| 25 | 0.0000 / 1.85 | 0.036 / **0.149** | 0.0000 / 2.26 | 0.0000 / 1.69 |
| 100 | 0.054 / 0.067 | 0.082 / **0.030** | 0.045 / 0.092 | 0.052 / 0.070 |
| 400 | 0.079 / **0.011** | 0.095 / 0.014 | 0.078 / 0.014 | 0.080 / **0.011** |
| 1600 | 0.088 / **0.0023** | 0.101 / 0.012 | 0.088 / 0.0027 | 0.089 / **0.0024** |

Paired Wilcoxon on the clean test MSE, Holm over all 24 tests (`wilcoxon.csv`):
- n ≤ 100: the GA beats every other method (Cliff's δ ≈ 0.95–0.98, p_holm < 1e-7). Among the others only L-BFGS-B vs
  SGD at n = 100 differs (SGD better, p_holm = 0.03).
- n = 400: CMA-ES and SGD (equal) beat the GA and L-BFGS-B (p_holm ≤ 0.005).
- n = 1600: the GA is worst (δ = 1 against every method); CMA-ES beats L-BFGS-B; CMA-ES vs SGD not significant.

The GA's training MSE is the highest at every n (above the noise floor for n ≥ 400): **its small-n advantage comes
with fitting less**, and it turns into a disadvantage once n is large enough that fitting more no longer overfits.
The same holds among the other three: L-BFGS-B fits hardest (lowest training MSE at every n ≥ 100) and generalises
worst of them (worse than SGD at n = 100 and 400, than CMA-ES at n = 400 and 1600). At n = 25 all three interpolate (train ≈ 0).

**GA long run** (`core_ga_long`, 1M evaluations, seeds 0–9, paired with the core seeds; `compare_methods.csv`):

| n | GA train 100k → 1M | GA test 100k → 1M | GA 1M vs CMA / L-BFGS-B / SGD at 100k (Cliff's δ on test; < 0: GA better) |
|---|---|---|---|
| 25 | 0.037 → 0.014 | 0.138 → 0.269 (worse, p_holm 0.008) | −0.92 / −0.90 / −0.96 |
| 100 | 0.087 → 0.071 | 0.029 → 0.028 (n.s.) | −0.82 / −0.90 / −0.88 |
| 400 | 0.098 → 0.088 | 0.015 → 0.007 (better, p_holm 0.008) | −0.70 / −0.92 / −0.62 |
| 1600 | 0.099 → 0.090 | 0.012 → 0.003 (better, p_holm 0.008) | +0.62 / +0.26 / +0.50 |

With 10× the budget the GA fits more, its large-n disadvantage mostly disappears, and at n = 25 it starts to overfit.
It still fits less than the other methods at every n (train 0.071 vs 0.043–0.052 at n = 100), so the GA never reaches
"fits as well, generalises better". 10 seeds only: the smallest possible p_holm over 24 pair tests is ≈ 0.047, so the
GA-vs-others column is descriptive (effect sizes), not tested.

**`grad_cost` sensitivity** (`core_gradcost1`, `core_gradcost3`: L-BFGS-B and SGD rerun, CMA-ES and the GA unchanged
by construction; 30 paired seeds):
- L-BFGS-B and SGD change by at most a few percent; no difference vs `grad_cost` 2 is significant, except SGD at
  n = 100 with `grad_cost` 3 (fewer gradient steps → test 0.070 → 0.065, p_holm 0.015).
- Ranking by median test MSE unchanged at every n for `grad_cost` 1; for 3 only CMA-ES and SGD swap at n = 100
  (their difference is not significant in either run). 23 of 24 pair verdicts unchanged in both (the exception is
  the borderline CMA-ES vs L-BFGS-B at n = 100, significant at 1 and 3).

**Wall-clock budget** (`core_ntrain_sweep_wallclock`, per-n time limits 4 / 4 / 7 / 12 s, `--jobs 1`):
- Evaluations reached (median): GA 127k–180k (cheapest per evaluation), CMA-ES 82k–108k, L-BFGS-B 92k–119k,
  SGD 365k at n = 25 down to 42k at n = 1600 (many small Python steps).
- CMA-ES, L-BFGS-B and SGD: no significant change vs the core sweep. The GA gets more evaluations, so it fits a bit
  more: its test MSE drops at n ≥ 100 (e.g. 0.014 → 0.011 at n = 400) and rises at n = 25 (0.149 → 0.170).
- 19 of 24 pair verdicts unchanged and the ranking unchanged except at n = 400, where the GA moves from last (tied
  with L-BFGS-B) to level with CMA-ES and SGD, ahead of L-BFGS-B. Same picture as the GA long run: more GA evaluations → the GA catches up at larger n.

### 4.4 Phase 5 results (2026-11-11)

Runs at commits `ed67e33` (`p5_selection`) and `e0294b4` (`p5_selection_H*`), all clean. Tables in
`results/processed/p5_selection/` and `results/processed/p5_arch/`; figures 5–7.

**5.1 Fair early stopping** (`configs/p5_selection.yaml`: the core sweep plus a selection set of n points (D10)
and 10 checkpoints per decade; final networks identical to the core sweep in 480/480 runs). Median clean test MSE
of the checkpoint with the lowest selection MSE (`*_best_select.csv`, Fig 5):

| n | CMA-ES | GA | L-BFGS-B | SGD |
|---|---|---|---|---|
| 25 | **0.129** | **0.116** | **0.117** | **0.148** |
| 100 | **0.032** | **0.031** | **0.024** | **0.027** |
| 400 | **0.0052** | 0.014 | **0.0057** | 0.0065 |
| 1600 | **0.0012** | 0.012 | **0.0013** | 0.0018 |

(bold in both §4.4 tables: not significantly worse than the best method at that n, paired Wilcoxon, Holm over 24 tests)

- The GA's small-n advantage disappears: no significant difference at n = 25; at n = 100 no method differs from
  the best (L-BFGS-B), but SGD beats the GA (p_holm = 0.011); at n ≥ 400 the GA is worst (δ ≥ 0.95).
- The early-stopping point (median evaluations): L-BFGS-B 100–500, SGD 1.4k–4k, CMA-ES 4k–20k, GA 20k at n = 25 and
  the full 100k at n ≥ 100 (it never starts to overfit within the budget).

**5.2 Sharpness** (D11, `sharpness.csv`, Fig 6). At the GA's training MSE (same seed):
- perturbation sharpness (median ×1e-3): L-BFGS-B 4.2–5.5, CMA-ES 2.6–3.1, GA 1.9–2.4, SGD 2.0–2.3. The GA is flatter
  than L-BFGS-B at every n (p_holm ≤ 4e-5) and than CMA-ES at n = 100 and 1600; GA and SGD are about equal (SGD
  slightly flatter at n = 400, p_holm = 0.04).
- Flatness does not predict generalisation: at n ≥ 100 the sharpest method (L-BFGS-B) has the lowest test MSE of the
  matched networks (e.g. 0.010 vs GA 0.014 at n = 400); only at n = 25 do flatter networks (GA, SGD) also generalise
  better (GA 0.149 vs L-BFGS-B 0.237).
- The final networks differ much more (GA 0.002 vs 0.005–0.044; ‖θ‖ 7–9 vs 13–20), but that is training progress:
  the others keep fitting the noise.

**5.3 H chosen on the selection set** (`configs/p5_arch.yaml`, `scripts/arch.py`; H ∈ {2, 4, 6, 10, 15, 20}, full
budget per H, (H, checkpoint) with the lowest selection MSE; 2880 runs). *Changed from the planned E1* ("evolve H with
validation fitness"): if the EA's fitness is the selection error, the selection set becomes its training data,
which is unfair to methods that fit on the training set; and a grid over ≤ 20 values of H is exhaustive, so an EA over
H could only tie. Instead every method gets the same capacity and early-stopping selection.

| n | CMA-ES test / H | GA test / H | L-BFGS-B test / H | SGD test / H |
|---|---|---|---|---|
| 25 | **0.100** / 10 | **0.102** / 15 | **0.119** / 10 | **0.106** / 10 |
| 100 | 0.024 / 10 | 0.026 / 12.5 | **0.019** / 6 | 0.024 / 6 |
| 400 | **0.0049** / 10 | 0.0115 / 15 | **0.0046** / 8 | 0.0057 / 6 |
| 1600 | **0.0012** / 10 | 0.0087 / 20 | **0.0013** / 10 | 0.0017 / 10 |

(median clean test MSE / median chosen H; `medians_chosen.csv`, `chosen_h.csv`, Fig 7)
- n = 25: no significant differences. n = 100: L-BFGS-B best (beats CMA-ES, GA, SGD; p_holm ≤ 0.031). n ≥ 400: the
  GA is worst (δ ≥ 0.87), CMA-ES and L-BFGS-B best.
- Choosing H improves on fixed H = 10 in 9 of 16 (method, n) cells and is never significantly worse
  (`compare_vs_fixed.csv`); most for the GA at large n (0.012 → 0.0087 at n = 1600). The ranking does not change.
- The chosen H does not track n cleanly: a selection set of n points is noisy (wide IQRs). The GA picks the
  largest networks (median 15–20 at n ≥ 400): extra capacity compensates for its underfitting.
- Optimiser settings were tuned at H = 10 (D9) and reused for every H.

**Phase 5 conclusion (RQ1/H1).** Once every method gets the same realistic validation data for early stopping and
for choosing H, the EAs do not produce networks that generalise better. CMA-ES is on par with the gradient methods
(behind L-BFGS-B only at n = 100 in 5.3). The GA is on par only at n = 25, significantly behind a gradient
method at n = 100 (SGD in 5.1, L-BFGS-B in 5.3) and clearly worse at n ≥ 400; the other methods reach their
early-stopping point with 5–1000× fewer evaluations than the GA. The EAs' small-n advantage in the core sweep (§4.3)
is implicit early stopping (they fit slowly), and the flat-minima mechanism of H1 only shows at n = 25. H1 is not supported as a property of EAs.

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

### Phase 4: Core experiment ✅
Core sweep, wall-clock sweep, GA long run and `grad_cost` 1 / 3 run and analysed (`aggregate_results.py`,
`compare_runs.py`, `make_figures.py`); results in §4.3.

### Phase 5: Extensions ✅
- 5.1 Fair early stopping on a small selection set (D10), 5.2 sharpness at matched training error (E4, D11),
  5.3 H and early stopping chosen on the selection set for every method (E1, redesigned). Results in §4.4.
- Not done: E2 (evolve weight decay λ), E3 (noise sweep / d = 5).

### Phase 6: Write-up
Report: introduction, method, setup (all decisions and why), results, discussion of the hypothesis, limitations.

---

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| The EA doesn't reach a low training loss within the budget, so any "generalisation advantage" is really under-fitting | Always report train and test error together. Check convergence curves. **Happened for the GA (§4.2)**; even at 1M evaluations it fits less than the others (§4.3), so report its test advantage together with its training error |
| Unequal tuning effort favours one method | Same tuning protocol and budget for every method (Phase 3) |
| The budget unit is disputed | Report both evaluations and wall-clock, plus the `grad_cost` sensitivity check |
| Weights saturate at ±5 bounds (tanh saturation) | Log how many weights sit at the bounds (`frac_at_bounds`). Measured: 5–17% for CMA-ES / L-BFGS-B / SGD, ~0% for the GA (§4.2) |
| Compute time | ~41 params and ≤1600 points is cheap. Parallelise seeds with `joblib`/multiprocessing |
| Results that aren't reproducible | Seeds everywhere, one JSON row per run, config stored with each result |
