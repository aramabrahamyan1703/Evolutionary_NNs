"""Tests for evonn.sharpness (Phase 5.2, D11)."""

import numpy as np
import pytest

from evonn.data import generate_datasets
from evonn.mlp import MLP
from evonn.sharpness import hessian, matched_checkpoint, measure, perturbation_sharpness, top_eigenvalue


class Quadratic:
    """L(theta) = 1/2 theta^T A theta: Hessian A everywhere (X, y are ignored)."""

    def __init__(self, A):
        self.A = A

    def loss(self, theta, X, y):
        return 0.5 * float(theta @ self.A @ theta)

    def loss_and_grad(self, theta, X, y):
        return self.loss(theta, X, y), self.A @ theta

    def loss_many(self, Theta, X, y):
        return 0.5 * np.einsum("pi,ij,pj->p", Theta, self.A, Theta)


@pytest.fixture
def quad():
    rng = np.random.default_rng(0)
    B = rng.normal(size=(6, 6))
    return Quadratic(B @ B.T)


def test_hessian_of_a_quadratic(quad):
    theta = np.ones(6)
    assert np.allclose(hessian(quad, theta, None, None), quad.A, atol=1e-6)
    assert top_eigenvalue(quad, theta, None, None) == pytest.approx(np.linalg.eigvalsh(quad.A)[-1], rel=1e-6)


def test_perturbation_sharpness_of_a_quadratic(quad):
    # at the minimum theta* = 0 the radius would be 0, so use theta = 1: L(theta + e) - L(theta)
    # = theta^T A e + e^T A e / 2; the linear term averages out, E[e^T A e] = r^2 tr(A) / d
    theta = np.ones(6)
    r = 0.01 * np.linalg.norm(theta)
    expected = 0.5 * r**2 * np.trace(quad.A) / 6
    got = perturbation_sharpness(quad, theta, None, None, n=200_000)
    assert got == pytest.approx(expected, rel=0.05)


def test_hessian_of_the_mlp_matches_second_differences():
    model = MLP()
    X, y = generate_datasets(0)["train"][100]
    theta = model.init(0)
    H = hessian(model, theta, X, y)
    assert np.allclose(H, H.T)
    v = np.random.default_rng(1).normal(size=len(theta))
    v /= np.linalg.norm(v)
    t = 1e-4
    second = (model.loss(theta + t * v, X, y) - 2 * model.loss(theta, X, y) + model.loss(theta - t * v, X, y)) / t**2
    assert v @ H @ v == pytest.approx(second, rel=1e-3)


def test_measure_reports_weight_context():
    model = MLP()
    X, y = generate_datasets(0)["train"][25]
    theta = model.init(0)
    theta[:3] = 5.0
    m = measure(model, theta, X, y, bound=5.0)
    assert set(m) == {"perturb_sharpness", "top_eigenvalue", "theta_norm", "frac_at_bounds"}
    assert m["frac_at_bounds"] == pytest.approx(3 / len(theta))
    assert m["theta_norm"] == pytest.approx(np.linalg.norm(theta))


def test_matched_checkpoint():
    curve = [{"evals": e, "train_mse": t} for e, t in [(1, 1.0), (10, 0.5), (100, 0.2), (1000, 0.1)]]
    assert matched_checkpoint(curve, 0.3)["evals"] == 100      # first one at or below the target
    assert matched_checkpoint(curve, 0.5)["evals"] == 10
    assert matched_checkpoint(curve, 0.05)["evals"] == 1000    # never reached: last checkpoint
