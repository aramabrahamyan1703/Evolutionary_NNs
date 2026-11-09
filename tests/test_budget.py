"""Tests for evonn.budget: every call is charged correctly and the budget is never exceeded."""

import numpy as np
import pytest

from evonn.budget import CountedObjective, BudgetExhausted, default_checkpoints
from evonn.mlp import MLP

MODEL = MLP(d_in=2, hidden=10)


def _objective(max_evals=100, grad_cost=2, **kw):
    r = np.random.default_rng(0)
    X = r.uniform(-1, 1, size=(30, 2))
    y = r.normal(size=30)
    return CountedObjective(MODEL, X, y, max_evals=max_evals, grad_cost=grad_cost, **kw)


def _theta(seed):
    return np.random.default_rng(seed).uniform(-1, 1, size=MODEL.n_params)


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


# ---------------------------------------------------------------- charging

def test_loss_costs_one():
    obj = _objective()
    obj.loss(_theta(0))
    assert obj.evals_used == 1


@pytest.mark.parametrize("grad_cost", [1, 2, 3])
def test_loss_and_grad_costs_grad_cost(grad_cost):
    obj = _objective(grad_cost=grad_cost)
    obj.loss_and_grad(_theta(0))
    assert obj.evals_used == grad_cost


def test_loss_many_costs_population_size():
    obj = _objective()
    obj.loss_many(np.stack([_theta(s) for s in range(7)]))
    assert obj.evals_used == 7


# ---------------------------------------------------------------- stopping

def test_stops_exactly_at_max_evals():
    obj = _objective(max_evals=10)
    for _ in range(10):
        obj.loss(_theta(0))
    assert obj.evals_used == 10
    with pytest.raises(BudgetExhausted):
        obj.loss(_theta(0))
    assert obj.evals_used == 10


def test_gradient_call_that_does_not_fit_is_refused():
    obj = _objective(max_evals=5, grad_cost=2)
    obj.loss_and_grad(_theta(0))
    obj.loss_and_grad(_theta(0))
    with pytest.raises(BudgetExhausted):
        obj.loss_and_grad(_theta(0))       # would need 6 > 5
    assert obj.evals_used == 4             # the refused call is not charged
    obj.loss(_theta(0))                    # a 1-eval call still fits
    assert obj.evals_used == 5


def test_population_that_does_not_fit_is_refused():
    obj = _objective(max_evals=10)
    with pytest.raises(BudgetExhausted):
        obj.loss_many(np.stack([_theta(s) for s in range(11)]))
    assert obj.evals_used == 0


def test_time_budget():
    clock = FakeClock()
    obj = _objective(max_evals=1000, max_seconds=1.0, clock=clock)
    obj.loss(_theta(0))
    clock.t = 1.5
    with pytest.raises(BudgetExhausted):
        obj.loss(_theta(0))
    assert obj.evals_used == 1


# ---------------------------------------------------------------- values

def test_values_match_model():
    obj = _objective()
    t = _theta(1)
    assert obj.loss(t) == pytest.approx(MODEL.loss(t, obj.X, obj.y))
    value, grad = obj.loss_and_grad(t)
    ref_value, ref_grad = MODEL.loss_and_grad(t, obj.X, obj.y)
    assert value == pytest.approx(ref_value)
    assert np.allclose(grad, ref_grad)
    Theta = np.stack([_theta(s) for s in range(4)])
    assert np.allclose(obj.loss_many(Theta), MODEL.loss_many(Theta, obj.X, obj.y))


# ---------------------------------------------------------------- best-so-far

def test_best_so_far_tracked():
    obj = _objective()
    thetas = [_theta(s) for s in range(10)]
    losses = [obj.loss(t) for t in thetas]
    k = int(np.argmin(losses))
    assert obj.best_loss == losses[k]
    assert np.array_equal(obj.best_theta, thetas[k])


def test_best_theta_is_a_copy():
    obj = _objective()
    t = _theta(0)
    obj.loss(t)
    t[:] = 99.0                             # caller (e.g. scipy) reuses the array
    assert not np.any(obj.best_theta == 99.0)


def test_loss_many_updates_best():
    obj = _objective()
    Theta = np.stack([_theta(s) for s in range(5)])
    values = obj.loss_many(Theta)
    assert obj.best_loss == values.min()
    assert np.array_equal(obj.best_theta, Theta[np.argmin(values)])


def test_trace_monotone():
    obj = _objective(max_evals=200)
    for s in range(50):
        obj.loss(_theta(s))
        obj.loss_and_grad(_theta(s + 100))
    evals = [e for e, _, _ in obj.trace]
    best = [b for _, b, _ in obj.trace]
    assert evals == sorted(evals)
    assert all(b2 < b1 for b1, b2 in zip(best, best[1:]))


# ---------------------------------------------------------------- snapshots

def test_default_checkpoints():
    assert default_checkpoints(20000) == [1, 2, 5, 10, 20, 50, 100, 200, 500,
                                          1000, 2000, 5000, 10000, 20000]
    assert default_checkpoints(300)[-1] == 300


def test_snapshots_taken_at_checkpoints():
    obj = _objective(max_evals=20, grad_cost=2, checkpoints=[1, 4, 5, 20])
    obj.loss(_theta(0))                     # evals 1   -> checkpoint 1
    obj.loss_and_grad(_theta(1))            # evals 3
    obj.loss_and_grad(_theta(2))            # evals 5   -> checkpoints 4 and 5
    assert set(obj.snapshots) == {1, 4, 5}
    while obj.remaining:
        obj.loss(_theta(3))
    assert set(obj.snapshots) == {1, 4, 5, 20}
    best_at_20, theta_at_20 = obj.snapshots[20]
    assert best_at_20 == obj.best_loss
    assert np.array_equal(theta_at_20, obj.best_theta)


def test_snapshot_is_best_so_far_not_latest():
    obj = _objective(checkpoints=[2])
    good, bad = np.zeros(MODEL.n_params), np.full(MODEL.n_params, 3.0)
    loss_good = obj.loss(good)
    loss_bad = obj.loss(bad)                # evals 2 -> checkpoint, but this one is worse
    assert loss_bad > loss_good
    assert np.array_equal(obj.snapshots[2][1], good)
