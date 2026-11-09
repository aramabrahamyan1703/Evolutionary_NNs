"""Tests that every registered method follows the shared contract (see evonn.methods).

Each method, on a small budget: uses the whole budget, never evaluates weights
outside the bounds, is reproducible for a given seed, improves on the shared
start x0, and records its restarts.
"""

import numpy as np
import pytest

from evonn.budget import CountedObjective
from evonn.data import generate_datasets
from evonn.experiment import ROOT, load_config, method_params
from evonn.methods import METHODS
from evonn.mlp import MLP

CFG = load_config(ROOT / "configs" / "default.yaml")
MODEL = MLP(d_in=2, hidden=10)
DATA = generate_datasets(seed=0)
X, Y = DATA["train"][100]
BUDGET = 2000
BOUND = CFG["model"]["weight_bounds"][1]


class SpyObjective(CountedObjective):
    """CountedObjective that also records every evaluated theta."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.seen = []

    def loss(self, theta):
        value = super().loss(theta)
        self.seen.append(np.array(theta, copy=True))
        return value

    def loss_and_grad(self, theta):
        out = super().loss_and_grad(theta)
        self.seen.append(np.array(theta, copy=True))
        return out

    def batch_loss_and_grad(self, theta, idx):
        out = super().batch_loss_and_grad(theta, idx)
        self.seen.append(np.array(theta, copy=True))
        return out


def _run(method, seed=0, budget=BUDGET):
    obj = SpyObjective(MODEL, X, Y, max_evals=budget, grad_cost=2)
    x0 = MODEL.init(seed)
    info = METHODS[method](obj, MODEL, x0, seed, method_params(CFG, method))
    return obj, x0, info


@pytest.fixture(scope="module", params=sorted(METHODS))
def run0(request):
    method = request.param
    return (method, *_run(method))


def test_uses_whole_budget(run0):
    _, obj, _, _ = run0
    assert obj.max_evals - obj.evals_used < obj.grad_cost     # stops within one call of the budget


def test_stays_within_bounds(run0):
    _, obj, _, _ = run0
    assert np.max(np.abs(np.stack(obj.seen))) <= BOUND


def test_improves_on_start(run0):
    _, obj, x0, _ = run0
    assert obj.best_loss < MODEL.loss(x0, X, Y)


def test_restarts_recorded(run0):
    _, obj, _, info = run0
    assert len(obj.restarts) >= 1
    assert obj.restarts[0]["start_evals"] == 0
    assert info["starts"] == len(obj.restarts)
    assert min(r["best_loss"] for r in obj.restarts) == obj.best_loss


def test_evaluates_shared_start(run0):
    method, obj, x0, _ = run0
    if method == "cmaes":
        pytest.skip("CMA-ES uses x0 as the mean of its first population, not as a candidate")
    assert np.array_equal(obj.seen[0], x0)


@pytest.mark.parametrize("method", sorted(METHODS))
def test_reproducible(method):
    a, _, _ = _run(method, seed=3, budget=600)
    b, _, _ = _run(method, seed=3, budget=600)
    assert np.array_equal(a.best_theta, b.best_theta)
    assert a.evals_used == b.evals_used


@pytest.mark.parametrize("method", sorted(METHODS))
def test_different_seeds_differ(method):
    a, _, _ = _run(method, seed=1, budget=600)
    b, _, _ = _run(method, seed=2, budget=600)
    assert not np.array_equal(a.best_theta, b.best_theta)


# ---------------------------------------------------------------- SGD options

def test_sgd_momentum_variant():
    obj = SpyObjective(MODEL, X, Y, max_evals=BUDGET, grad_cost=2)
    x0 = MODEL.init(0)
    params = {**method_params(CFG, "sgd"), "optimizer": "momentum", "lr": 0.05}
    METHODS["sgd"](obj, MODEL, x0, 0, params)
    assert obj.best_loss < MODEL.loss(x0, X, Y)
    assert np.max(np.abs(np.stack(obj.seen))) <= BOUND


def test_sgd_unknown_optimizer_raises():
    obj = CountedObjective(MODEL, X, Y, max_evals=100)
    with pytest.raises(ValueError):
        METHODS["sgd"](obj, MODEL, MODEL.init(0), 0, {**method_params(CFG, "sgd"), "optimizer": "rmsprop"})


def test_sgd_batch_capped_at_n():
    Xs, Ys = DATA["train"][25]
    obj = SpyObjective(MODEL, Xs, Ys, max_evals=50, grad_cost=2)      # batch_size 32 > n = 25
    METHODS["sgd"](obj, MODEL, MODEL.init(0), 0, method_params(CFG, "sgd"))
    # full batch: every step costs a whole gradient (2) and every loss 1, so the total stays an integer
    assert obj.evals_used == int(obj.evals_used)
    assert obj.max_evals - obj.evals_used < 2
