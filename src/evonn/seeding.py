"""Random streams for one run.

The data already uses np.random.default_rng(seed) (see evonn.data). Every
other use of randomness takes its own stream, so the initial weights never
reuse the random numbers that drew the training points:

    rng(seed, INIT)       initial weights
    rng(seed, OPTIMISER)  CMA-ES / GA sampling, L-BFGS restarts
"""

import numpy as np

INIT = 1
OPTIMISER = 2


def rng(seed, stream):
    """Return an independent np.random.Generator for (run seed, stream)."""
    return np.random.default_rng([seed, stream])
