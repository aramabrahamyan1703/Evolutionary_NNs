# Do Evolutionary Algorithms Generalise Better? A Controlled Comparison with Gradient-Based Methods Under Data Scarcity
> *Alternative title: Beyond Sharp Minima: Can CMA-ES Outgeneralise L-BFGS When Training Data Is Scarce?*

**Authors:** Marta Mateu · Aram Abrahamyan

---

Between 2017 and 2019, OpenAI, Uber AI Labs, and Google Brain demonstrated that
evolutionary algorithms are competitive with backpropagation in control tasks and
architecture search (Salimans et al., 2017; Such et al., 2017; Real et al., 2017).
The theoretical basis for a generalisation advantage exists: Keskar et al. (2017)
showed that optimisers converging to **flat minima** generalise significantly better,
and evolutionary algorithms — operating without gradients over a population — carry a
structural bias towards flatter regions of the loss surface. However, whether this
advantage extends to **supervised regression under data scarcity** has not been
systematically studied. This work addresses that gap directly.

We compare CMA-ES and a real-coded genetic algorithm against L-BFGS across training
set sizes spanning two orders of magnitude (`n ∈ {25, 100, 400, 1600}`), on a
controlled regression task with a known noise floor.

## Report structure

The report follows the 5-section format required by the assignment (§1.2, 10 pages max):

**1. Question and hypothesis**
- Research question: does an EA find networks that generalise better than a gradient-based method under the same budget? Does the answer depend on training-set size?
- H1: the EA does not follow the gradient into the nearest sharp minimum, so any advantage shows up at small n and disappears at large n.
- Falsified by: EA test error ≥ baseline at every n, or advantage grows with n, or lower test error comes purely from higher training error (underfitting, not flat minima).

**2. Design**
- Single factor: `n_train ∈ {25, 100, 400, 1600}` (geometric, ×4 steps).
- Fixed: MLP architecture (1 hidden layer, H=10 tanh units, d=2 inputs → 41 params), weight bounds [−5, 5], same initial distribution, same val/test sets (5000 points each, fixed `data_seed`).
- Budget equalised by number of error-function evaluations (loss+grad = 2 evals); wall-clock reported as secondary.
- Baseline: L-BFGS-B with random restarts to fill the budget. Sanity check: `sklearn.MLPRegressor`.
- 30 seeds per cell, shared across methods (paired comparison).

**3. Results**
- Main figure: test MSE − σ² vs n (log-log), one line per method, median + IQR.
- Effect sizes: Vargha–Delaney A₁₂ and median difference with bootstrap CI (not only p-values).
- Tests: Wilcoxon signed-rank (paired, same seeds) with Holm correction across the 4 sizes.
- Always report train MSE alongside test MSE to distinguish underfitting from flat-minima effects.

**4. Mechanism**
- Support the explanation with a measurement, not only final fitness:
  - Convergence curves: best-so-far train/val/test error vs evaluations.
  - (Optional) Sharpness proxy: loss increase under random weight perturbation at the found minimum.
  - (Optional) Weight norm and fraction of weights at ±5 bounds.

**5. Limitations**
- Synthetic data only (sum-of-sines, d=2); conclusions may not transfer to real datasets.
- Single architecture (H=10); EA advantage may depend on model size relative to n.
- Budget choice (20k evals) may favour one method; sensitivity checked via convergence curves.
- Results are specific to regression; classification landscapes may differ.

## Reproducibility

- Every run is identified by `(config, method, n_train, seed)` and writes one JSON row to `results/raw/`.
- Validation and test sets are generated once from a fixed `data_seed`; training sets are nested subsets.
- After installing, freeze exact versions with `pip freeze > requirements-lock.txt`.
