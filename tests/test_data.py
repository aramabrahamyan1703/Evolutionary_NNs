"""Tests for the synthetic data generator (evonn.data).

Covers: f, sample, generate_datasets (nested train sets, fixed val/test,
reproducibility, no leakage, noise floor, custom sigma).
"""

import numpy as np
import pytest

from evonn.data import (
    f,
    sample,
    generate_datasets,
    SIGMA,
    TRAIN_SIZES,
    N_VAL,
    N_TEST,
    X_LOW,
    X_HIGH,
    DATA_SEED,
)


# ---------------------------------------------------------------- f

def test_f_shape_and_range():
    rng = np.random.default_rng(0)
    X = rng.uniform(X_LOW, X_HIGH, size=(50, 2))
    y = f(X)
    assert y.shape == (50,)
    # clean values are sin + sin, so they live in [-2, 2]
    assert np.all(y >= -2.0 - 1e-9)
    assert np.all(y <= 2.0 + 1e-9)


def test_f_known_value():
    # f(0.5, 0.5) = sin(pi/2) + sin(pi/2) = 1 + 1 = 2
    assert f(np.array([[0.5, 0.5]]))[0] == pytest.approx(2.0)


# ---------------------------------------------------------------- sample

def test_sample_shapes_and_range():
    rng = np.random.default_rng(0)
    X, y = sample(200, rng)
    assert X.shape == (200, 2)
    assert y.shape == (200,)
    assert np.all(X >= X_LOW)
    assert np.all(X <= X_HIGH)


def test_sample_reproducible():
    Xa, ya = sample(100, np.random.default_rng(7))
    Xb, yb = sample(100, np.random.default_rng(7))
    assert np.array_equal(Xa, Xb)
    assert np.array_equal(ya, yb)


def test_sample_noise_statistics():
    # On a large sample, the residual y - f(X) should look like N(0, SIGMA^2).
    rng = np.random.default_rng(123)
    X, y = sample(100_000, rng)
    residual = y - f(X)
    assert abs(residual.mean()) < 0.01          # mean close to 0
    assert abs(residual.std() - SIGMA) < 0.01   # std close to SIGMA


# ------------------------------------------------- generate_datasets

def test_dataset_shapes():
    d = generate_datasets(seed=0)
    for n in TRAIN_SIZES:
        X, y = d["train"][n]
        assert X.shape == (n, 2)
        assert y.shape == (n,)
    assert d["val"][0].shape == (N_VAL, 2)
    assert d["test"][0].shape == (N_TEST, 2)


def test_train_sets_are_nested():
    d = generate_datasets(seed=3)
    sizes = sorted(TRAIN_SIZES)
    for small, large in zip(sizes, sizes[1:]):
        Xs, ys = d["train"][small]
        Xl, yl = d["train"][large]
        assert np.array_equal(Xs, Xl[:small])   # X prefix
        assert np.array_equal(ys, yl[:small])   # y prefix


def test_val_test_fixed_across_seeds():
    d0 = generate_datasets(seed=0)
    d1 = generate_datasets(seed=1)
    assert np.array_equal(d0["val"][0], d1["val"][0])
    assert np.array_equal(d0["val"][1], d1["val"][1])
    assert np.array_equal(d0["test"][0], d1["test"][0])
    assert np.array_equal(d0["test"][1], d1["test"][1])


def test_train_differs_across_seeds():
    d0 = generate_datasets(seed=0)
    d1 = generate_datasets(seed=1)
    assert not np.array_equal(d0["train"][25][0], d1["train"][25][0])


def test_generate_datasets_reproducible():
    d_a = generate_datasets(seed=0)
    d_b = generate_datasets(seed=0)
    for n in TRAIN_SIZES:
        assert np.array_equal(d_a["train"][n][0], d_b["train"][n][0])
        assert np.array_equal(d_a["train"][n][1], d_b["train"][n][1])


def test_no_leakage_train_in_val_or_test():
    # No training point should appear in the val or test sets, for seeds 0..29.
    for seed in range(30):
        d = generate_datasets(seed=seed)
        X_train = d["train"][max(TRAIN_SIZES)][0]      # the full pool
        val_set = {tuple(p) for p in d["val"][0]}
        test_set = {tuple(p) for p in d["test"][0]}
        for p in X_train:
            key = tuple(p)
            assert key not in val_set
            assert key not in test_set


def test_noise_floor_matches_test_mse():
    d = generate_datasets(seed=0)
    X_test, y_test = d["test"]
    mse = np.mean((y_test - f(X_test)) ** 2)
    assert mse == pytest.approx(0.09, abs=0.005)       # close to sigma^2
    assert d["noise_floor"] == SIGMA**2


def test_custom_sigma():
    d = generate_datasets(seed=0, sigma=0.5)
    assert d["noise_floor"] == pytest.approx(0.25)


def test_generate_datasets_raises_on_data_seed():
    with pytest.raises(ValueError, match="DATA_SEED"):
        generate_datasets(DATA_SEED)
