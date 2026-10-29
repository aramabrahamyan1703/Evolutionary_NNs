"""Synthetic data generator.

Each data point is y = f(X) + noise, where we choose f and the noise.
Since the noise is random, no model can predict it, so the lowest test error
possible is the noise variance: the noise floor, sigma^2 = 0.09.
We always report test MSE - 0.09. What is left is the model's own error,
for example from fitting the noise.

The problem:
  - Inputs:   X = (x1, x2), each in [-1, 1]
  - Truth:    f(X) = sin(pi*x1) + sin(pi*x2), a value between -2 and 2
  - Noise:    eps ~ N(0, 0.3^2), so y = f(X) + eps

How the data is split:
  - Validation and test sets (5000 points each) are always the same:
    every run and every method is evaluated on the same data.
  - Training sets change with the run seed (0 to 29). Each seed draws
    1600 points and the training set of size n uses the first n of them:

        seed 7:  [p1 p2 ... p25 ... p100 ... p400 ... p1600]
                  n=25 uses p1..p25, n=100 uses p1..p100, and so on.

    So the same seed gives every method the same training data, and the
    only thing that changes between sizes is how many points are used.

Use generate_datasets(seed) to get all the sets for one run.
"""

import numpy as np

SIGMA = 0.3                  # noise standard deviation
NOISE_FLOOR = SIGMA**2       # 0.09, lowest achievable test MSE
X_LOW, X_HIGH = -1.0, 1.0    # range of each input coordinate
D_IN = 2                     # input dimension
TRAIN_SIZES = (25, 100, 400, 1600)  # training sizes compared (the single factor)
N_VAL = N_TEST = 5000        # fixed validation and test sizes
N_RUNS = 30                  # run seeds are 0 .. N_RUNS-1
DATA_SEED = 10_000           # seed of val/test; must be outside the run seeds


def f(X):
    """Noise-free target: f(x1, x2) = sin(pi*x1) + sin(pi*x2).

    X has shape (n, 2); returns an array of shape (n,).
    """
    X = np.asarray(X, dtype=float)
    return np.sin(np.pi * X[:, 0]) + np.sin(np.pi * X[:, 1])


def sample(n, rng, sigma=SIGMA):
    """Draw n points uniformly in [-1, 1]^2 with noisy targets y = f(X) + eps.

    rng is a seeded np.random.Generator, so the same seed gives the same data.
    Returns X of shape (n, 2) and y of shape (n,).
    """
    X = rng.uniform(X_LOW, X_HIGH, size=(n, D_IN))
    y = f(X) + rng.normal(0.0, sigma, size=n)
    return X, y


def generate_datasets(seed, sigma=SIGMA):
    """Return the training, validation and test sets for one run.

    train: dict {n: (X, y)} for n in TRAIN_SIZES. Each set is the first n
           points of one pool of 1600 points drawn with `seed`.
    val, test: (X, y), always the same (they use DATA_SEED).
    noise_floor: sigma**2, the lowest possible test MSE.

    `seed` must not equal DATA_SEED, otherwise the training points would
    coincide with the validation points.
    """
    if seed == DATA_SEED:
        raise ValueError("seed must differ from DATA_SEED to avoid data leakage")

    data_rng = np.random.default_rng(DATA_SEED)
    val = sample(N_VAL, data_rng, sigma)
    test = sample(N_TEST, data_rng, sigma)

    X_pool, y_pool = sample(max(TRAIN_SIZES), np.random.default_rng(seed), sigma)
    train = {n: (X_pool[:n], y_pool[:n]) for n in TRAIN_SIZES}

    return {"train": train, "val": val, "test": test, "noise_floor": sigma**2}
