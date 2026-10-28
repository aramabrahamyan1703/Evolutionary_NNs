"""Synthetic data generator: y = f(X) + noise.

Because we choose f and the noise ourselves, we know the lowest test
error any model can reach: the noise floor, sigma^2 = 0.09. Noise is
random and cannot be predicted, so every test error is reported relative
to it (test MSE - sigma^2): the remaining gap is the model's own error,
for example fitting noise instead of f.

Every generator exposes:
  - f(X): the true function, without noise (ground truth).
  - sample(n, rng): draws n random points X and returns them with their
    noisy targets y = f(X) + noise. rng is a seeded random number
    generator, so results are reproducible.

Conventions (see docs/PROJECT_PLAN.md, section 3.1):
  - Val and test sets (5000 points each) are generated once from a fixed
    data_seed and are identical in every run, so every method is
    evaluated on the same data.
  - Training sets change with the run seed: each seed draws one pool of
    max(n_train) points, and the smaller sets are prefixes of that pool
    (n=25 inside n=100, etc.). Results across sizes then differ only in
    the amount of data, not in which points were drawn. The seed is
    shared across methods, so comparisons are paired.

Problem (decisions D1, D2; sigma may be varied later, extension E3):
  - Inputs:        X in [-1, 1]^2, i.e. two coordinates (x1, x2)
  - True function: f(x1, x2) = sin(pi*x1) + sin(pi*x2), values in [-2, 2]
  - Noise:         y = f(X) + eps, eps ~ N(0, sigma^2), sigma = 0.3
  - Noise floor:   sigma^2 = 0.09, the lowest test MSE on noisy data
"""

import numpy as np

# --- Problem constants (change here if the team changes a decision) ---
SIGMA = 0.3          # standard deviation of the Gaussian noise
NOISE_FLOOR = SIGMA**2  # = 0.09, the best achievable test MSE (our reference line)
X_LOW, X_HIGH = -1.0, 1.0  # each input coordinate lives in [-1, 1]
D_IN = 2             # number of input coordinates (x1, x2)


def f(X):
    """The noise-free target function (the 'ground truth').

    f(x1, x2) = sin(pi * x1) + sin(pi * x2)

    Parameters
    ----------
    X : np.ndarray of shape (n, 2)
        n points, each with two coordinates (x1, x2).

    Returns
    -------
    np.ndarray of shape (n,)
        The clean target value for each point (no noise added).
    """
    X = np.asarray(X, dtype=float)          # make sure we work with a float array
    return np.sin(np.pi * X[:, 0]) + np.sin(np.pi * X[:, 1])
    #            ^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^
    #            wave from x1   +   wave from x2
