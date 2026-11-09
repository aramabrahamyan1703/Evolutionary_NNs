"""Budget accounting: one counted objective that every method must call.

Every optimiser calls CountedObjective instead of the MLP directly, so all
methods are charged the same way:

  - loss(theta)            costs 1 evaluation (one full-training-set loss)
  - loss_and_grad(theta)   costs `grad_cost` evaluations (decision D3)
  - loss_many(Theta)       costs P evaluations (one per row)
  - batch_loss_and_grad(theta, idx)  costs grad_cost * len(idx) / n: a mini-batch
                           gradient is charged its share of a full-data gradient.
                           Its loss is a mini-batch loss, so it does not update
                           the best network; mini-batch methods call loss() on the
                           full training set for that. evals_used can then be fractional.

If a call would go over `max_evals` (or the time limit `max_seconds` has
passed), it raises BudgetExhausted *before* evaluating, so the budget is never
exceeded. Optimisers catch the exception and stop.

The objective also keeps the best weights seen in any evaluation (decision D6:
the returned network is the one with the lowest training loss), a trace of
improvements, snapshots of the best weights at checkpoint evaluation counts,
and the best weights of each restart (optimisers call begin_restart() at the
start of every run from new initial weights, including the first). The
snapshots are used later to draw convergence curves of test error without
charging the budget.

Create the objective right before the optimiser starts: the clock starts then.
"""

import time

import numpy as np


class BudgetExhausted(Exception):
    """Raised when the next call would exceed the evaluation or time budget."""


def default_checkpoints(max_evals):
    """1-2-5 spaced evaluation counts up to max_evals, plus max_evals itself."""
    points = []
    scale = 1
    while scale <= max_evals:
        points += [c * scale for c in (1, 2, 5) if c * scale <= max_evals]
        scale *= 10
    if not points or points[-1] != max_evals:
        points.append(max_evals)
    return points


class CountedObjective:
    """Training loss of `model` on (X, y), with evaluation counting and best-so-far tracking."""

    def __init__(self, model, X, y, max_evals, grad_cost=2, max_seconds=None,
                 checkpoints=None, clock=time.perf_counter):
        self.model = model
        self.X = X
        self.y = y
        self.max_evals = max_evals
        self.grad_cost = grad_cost
        self.max_seconds = max_seconds
        self.checkpoints = sorted(checkpoints if checkpoints is not None
                                  else default_checkpoints(max_evals))
        self._clock = clock
        self._t0 = clock()

        self.evals_used = 0
        self.n_calls = 0
        self.best_loss = np.inf
        self.best_theta = None
        self.trace = []        # (evals_used, best_loss, elapsed) at every improvement
        self.snapshots = {}    # checkpoint -> (best_loss, copy of best_theta)
        self.restarts = []     # one dict per restart: start_evals, best_loss, best_theta

    # ------------------------------------------------------------ status

    @property
    def elapsed(self):
        return self._clock() - self._t0

    @property
    def remaining(self):
        return self.max_evals - self.evals_used

    # ------------------------------------------------------------ public calls

    def begin_restart(self):
        """Mark the start of a new run from new initial weights (call before the first one too)."""
        self.restarts.append({"start_evals": self.evals_used, "best_loss": np.inf, "best_theta": None})

    def loss(self, theta):
        self._charge(1)
        value = self.model.loss(theta, self.X, self.y)
        self._record(theta, value)
        return value

    def loss_and_grad(self, theta):
        self._charge(self.grad_cost)
        value, grad = self.model.loss_and_grad(theta, self.X, self.y)
        self._record(theta, value)
        return value, grad

    def loss_many(self, Theta):
        Theta = np.asarray(Theta, dtype=float)
        self._charge(len(Theta))
        values = self.model.loss_many(Theta, self.X, self.y)
        best = int(np.argmin(values))
        self._record(Theta[best], float(values[best]))
        return values

    def batch_loss_and_grad(self, theta, idx):
        """Mini-batch MSE and gradient on the training points idx. Does not update the best network."""
        self._charge(self.grad_cost * len(idx) / len(self.y))
        value, grad = self.model.loss_and_grad(theta, self.X[idx], self.y[idx])
        self._snapshot()
        return value, grad

    # ------------------------------------------------------------ internals

    def _charge(self, cost):
        # small tolerance: fractional mini-batch costs accumulate rounding error
        if self.evals_used + cost > self.max_evals + 1e-9:
            raise BudgetExhausted(
                f"evaluation budget: {self.evals_used} used, call costs {cost}, max {self.max_evals}")
        if self.max_seconds is not None and self.elapsed >= self.max_seconds:
            raise BudgetExhausted(f"time budget: {self.elapsed:.3f}s >= {self.max_seconds}s")
        self._evals_before = self.evals_used
        self.evals_used += cost
        self.n_calls += 1

    def _record(self, theta, value):
        if self.restarts and value < self.restarts[-1]["best_loss"]:
            self.restarts[-1]["best_loss"] = value
            self.restarts[-1]["best_theta"] = np.array(theta, dtype=float, copy=True)
        if value < self.best_loss:
            self.best_loss = value
            self.best_theta = np.array(theta, dtype=float, copy=True)   # scipy reuses its arrays
            self.trace.append((self.evals_used, value, self.elapsed))
        self._snapshot()

    def _snapshot(self):
        # Snapshot every checkpoint passed by this call. With loss_many the
        # snapshot can include up to P - 1 evaluations past the checkpoint.
        if self.best_theta is None:
            return
        for c in self.checkpoints:
            if self._evals_before < c <= self.evals_used:
                self.snapshots[c] = (self.best_loss, self.best_theta.copy())
