"""Tests for evonn.metrics."""

import numpy as np
import pytest

from evonn.data import f, generate_datasets, NOISE_FLOOR
from evonn.metrics import mse, evaluate, evaluate_snapshots
from evonn.mlp import MLP

MODEL = MLP(d_in=2, hidden=10)
DATA = generate_datasets(seed=0)


class PerfectModel:
    """Stand-in that predicts the noise-free f exactly."""

    def forward(self, theta, X):
        return f(X)


def test_mse_matches_numpy():
    X, y = DATA["train"][100]
    theta = MODEL.init(seed=0)
    assert mse(MODEL, theta, X, y) == pytest.approx(np.mean((MODEL.forward(theta, X) - y) ** 2))


def test_perfect_predictor_hits_noise_floor():
    m = evaluate(PerfectModel(), np.zeros(41), DATA, n=100)
    assert m["test_mse_clean"] == pytest.approx(0.0, abs=1e-12)
    assert m["test_mse"] == pytest.approx(NOISE_FLOOR, abs=0.005)
    assert m["excess_test"] == pytest.approx(m["test_mse"] - NOISE_FLOOR)


def test_evaluate_keys_and_gap():
    theta = MODEL.init(seed=0)
    m = evaluate(MODEL, theta, DATA, n=25)
    for key in ["train_mse", "val_mse", "test_mse", "excess_test", "test_mse_clean",
                "gap", "noise_floor", "frac_at_bounds", "max_abs_theta", "theta_norm"]:
        assert key in m
    assert m["gap"] == pytest.approx(m["test_mse"] - m["train_mse"])
    assert m["train_mse"] == pytest.approx(MODEL.loss(theta, *DATA["train"][25]))


def test_frac_at_bounds():
    theta = np.zeros(41)
    theta[:4] = 5.0
    theta[4:6] = -5.0
    theta[6] = 4.99999      # CMA-ES-style: close to the bound but not on it -> counts
    theta[7] = 4.9          # clearly inside -> does not count
    m = evaluate(MODEL, theta, DATA, n=25)
    assert m["frac_at_bounds"] == pytest.approx(7 / 41)
    assert m["max_abs_theta"] == 5.0


def test_evaluate_snapshots_sorted():
    snapshots = {200: (0.1, MODEL.init(seed=1)), 100: (0.2, MODEL.init(seed=2))}
    rows = evaluate_snapshots(MODEL, snapshots, DATA, n=100)
    assert [r["evals"] for r in rows] == [100, 200]
