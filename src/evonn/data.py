"""Synthetic data generators.

Every generator must expose:
  - f(X):        the noise-free target function (used to compute the noise floor / Bayes error)
  - sample(n, rng): inputs X and noisy targets y = f(X) + noise

Conventions:
  - val and test sets are fixed (data_seed) and large (5000 points).
  - training sets are NESTED: the n=25 set is a subset of the n=100 set, etc.,
    so differences between sizes are not confounded by different samples.
"""
