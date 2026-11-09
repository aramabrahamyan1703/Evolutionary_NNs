"""Tests for the shared MLP (evonn.mlp).

Covers: parameter count, pack/unpack, forward, loss, the backprop gradient
(checked against finite differences), loss_many, and init.
"""

import numpy as np
import pytest

from evonn.mlp import MLP
from evonn.seeding import rng, INIT

MODEL = MLP(d_in=2, hidden=10)


def _data(n=50, seed=0, d_in=2):
    r = np.random.default_rng(seed)
    X = r.uniform(-1, 1, size=(n, d_in))
    y = r.normal(size=n)
    return X, y


def _fd_grad(model, theta, X, y, eps=1e-6):
    """Central finite-difference gradient of model.loss."""
    grad = np.zeros_like(theta)
    for k in range(len(theta)):
        e = np.zeros_like(theta)
        e[k] = eps
        grad[k] = (model.loss(theta + e, X, y) - model.loss(theta - e, X, y)) / (2 * eps)
    return grad


def _rel_err(a, b):
    return np.linalg.norm(a - b) / max(np.linalg.norm(a) + np.linalg.norm(b), 1e-12)


# ---------------------------------------------------------------- shapes

def test_n_params_default():
    assert MODEL.n_params == 41


@pytest.mark.parametrize("d_in,hidden", [(1, 1), (2, 5), (5, 20)])
def test_n_params_formula(d_in, hidden):
    assert MLP(d_in, hidden).n_params == d_in * hidden + hidden + hidden + 1


def test_pack_unpack_roundtrip():
    r = np.random.default_rng(1)
    W1, b1, w2, b2 = r.normal(size=(2, 10)), r.normal(size=10), r.normal(size=10), 0.7
    theta = MODEL.pack(W1, b1, w2, b2)
    assert theta.shape == (41,)
    W1b, b1b, w2b, b2b = MODEL.unpack(theta)
    assert np.array_equal(W1, W1b)
    assert np.array_equal(b1, b1b)
    assert np.array_equal(w2, w2b)
    assert b2b == b2


def test_unpack_wrong_length_raises():
    with pytest.raises(ValueError):
        MODEL.unpack(np.zeros(40))


# ---------------------------------------------------------------- forward / loss

def test_forward_shape_and_zero_weights():
    X, _ = _data()
    out = MODEL.forward(np.zeros(41), X)
    assert out.shape == (50,)
    assert np.all(out == 0.0)


def test_forward_hand_computed():
    # d_in = 1, H = 1: y_hat = tanh(x*w1 + b1)*w2 + b2
    m = MLP(d_in=1, hidden=1)
    theta = m.pack([[0.5]], [0.1], [2.0], -0.3)
    x = 0.8
    expected = np.tanh(0.8 * 0.5 + 0.1) * 2.0 - 0.3
    assert m.forward(theta, np.array([[x]]))[0] == pytest.approx(expected)


def test_loss_is_mse():
    X, y = _data()
    theta = MODEL.init(seed=0)
    assert MODEL.loss(theta, X, y) == pytest.approx(np.mean((MODEL.forward(theta, X) - y) ** 2))


def test_loss_and_grad_loss_matches_loss():
    X, y = _data()
    theta = MODEL.init(seed=0)
    loss, _ = MODEL.loss_and_grad(theta, X, y)
    assert loss == pytest.approx(MODEL.loss(theta, X, y))


# ---------------------------------------------------------------- gradient check

@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("d_in,hidden", [(2, 10), (1, 3), (5, 20)])
def test_gradient_matches_finite_differences(seed, d_in, hidden):
    m = MLP(d_in, hidden)
    X, y = _data(seed=seed, d_in=d_in)
    theta = np.random.default_rng(seed).uniform(-1, 1, size=m.n_params)
    _, grad = m.loss_and_grad(theta, X, y)
    assert _rel_err(grad, _fd_grad(m, theta, X, y)) < 1e-6


def test_gradient_with_large_weights():
    # weights near the +-5 bounds: tanh saturates, gradients get small
    X, y = _data(seed=3)
    theta = np.random.default_rng(3).uniform(-5, 5, size=41)
    _, grad = MODEL.loss_and_grad(theta, X, y)
    assert _rel_err(grad, _fd_grad(MODEL, theta, X, y)) < 1e-5


def test_gradient_step_reduces_loss():
    X, y = _data()
    theta = MODEL.init(seed=0)
    loss0, grad = MODEL.loss_and_grad(theta, X, y)
    assert MODEL.loss(theta - 1e-3 * grad, X, y) < loss0


# ---------------------------------------------------------------- loss_many

def test_loss_many_matches_loss():
    X, y = _data()
    Theta = np.random.default_rng(5).uniform(-2, 2, size=(7, 41))
    expected = [MODEL.loss(t, X, y) for t in Theta]
    assert np.allclose(MODEL.loss_many(Theta, X, y), expected)


def test_loss_many_wrong_shape_raises():
    X, y = _data()
    with pytest.raises(ValueError):
        MODEL.loss_many(np.zeros(41), X, y)


# ---------------------------------------------------------------- init

def test_init_reproducible_and_in_range():
    a, b = MODEL.init(seed=4), MODEL.init(seed=4)
    assert np.array_equal(a, b)
    assert a.shape == (41,)
    assert np.all(np.abs(a) <= 0.5)


def test_init_differs_across_seeds():
    assert not np.array_equal(MODEL.init(seed=0), MODEL.init(seed=1))


def test_init_uses_init_stream_not_data_stream():
    theta = MODEL.init(seed=0, scale=0.5)
    assert np.array_equal(theta, rng(0, INIT).uniform(-0.5, 0.5, size=41))
    assert not np.array_equal(theta, np.random.default_rng(0).uniform(-0.5, 0.5, size=41))
