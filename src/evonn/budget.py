"""Budget accounting.

Wraps the loss function in a counter so every method is charged the same way:
  - one full-training-set loss evaluation = 1 eval
  - one loss + gradient evaluation = `grad_cost` evals (decision D3)
Raises / signals when the budget is exhausted. Also records wall-clock time
with time.perf_counter() for the secondary comparison.
"""
