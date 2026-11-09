"""One-hidden-layer MLP in numpy, shared by ALL methods.

Every method (CMA-ES, GA, L-BFGS, ...) trains this exact network, so they all
optimise the same function and their evaluations can be counted the same way.

The weights live in one flat vector theta, which an EA can treat as a genome
and scipy as an optimisation variable. Layout (length H*(d_in + 2) + 1):

    [ W1 (d_in x H, row-major) | b1 (H) | w2 (H) | b2 (1) ]

For d_in = 2 and H = 10 that is 41 parameters.

The network is  y_hat = tanh(X @ W1 + b1) @ w2 + b2  and the loss is the MSE.
Weight bounds ([-5, 5]) are not enforced here: that is the optimisers' job.
"""

from dataclasses import dataclass

import numpy as np

from evonn.seeding import rng, INIT


@dataclass(frozen=True)
class MLP:
    """Shapes of the network. The weights are passed in as a flat theta."""

    d_in: int = 2
    hidden: int = 10

    @property
    def n_params(self):
        """Number of weights: d_in*H (W1) + H (b1) + H (w2) + 1 (b2)."""
        return self.hidden * (self.d_in + 2) + 1

    def pack(self, W1, b1, w2, b2):
        """Matrices -> flat theta."""
        return np.concatenate([
            np.asarray(W1, dtype=float).reshape(-1),
            np.asarray(b1, dtype=float).reshape(-1),
            np.asarray(w2, dtype=float).reshape(-1),
            np.asarray(b2, dtype=float).reshape(-1),
        ])

    def unpack(self, theta):
        """Flat theta -> (W1 (d_in, H), b1 (H,), w2 (H,), b2 float). Returns views, not copies."""
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.n_params,):
            raise ValueError(f"theta must have shape ({self.n_params},), got {theta.shape}")
        d, H = self.d_in, self.hidden
        i = d * H
        W1 = theta[:i].reshape(d, H)
        b1 = theta[i:i + H]
        w2 = theta[i + H:i + 2 * H]
        b2 = theta[-1]
        return W1, b1, w2, b2

    def init(self, seed, scale=0.5):
        """Initial weights U(-scale, scale), drawn from the INIT stream of `seed`."""
        return rng(seed, INIT).uniform(-scale, scale, size=self.n_params)

    def forward(self, theta, X):
        """Predictions, shape (n,)."""
        W1, b1, w2, b2 = self.unpack(theta)
        return np.tanh(X @ W1 + b1) @ w2 + b2

    def loss(self, theta, X, y):
        """Mean squared error on (X, y)."""
        r = self.forward(theta, X) - y
        return float(np.mean(r ** 2))

    def loss_and_grad(self, theta, X, y):
        """MSE and its gradient w.r.t. theta (manual backprop), same layout as theta."""
        W1, b1, w2, b2 = self.unpack(theta)
        h = np.tanh(X @ W1 + b1)            # (n, H)
        r = h @ w2 + b2 - y                 # (n,)
        g = 2.0 * r / len(y)                # dL/dy_hat
        dz = np.outer(g, w2) * (1.0 - h ** 2)   # (n, H), back through tanh
        grad = self.pack(X.T @ dz, dz.sum(axis=0), h.T @ g, g.sum())
        return float(np.mean(r ** 2)), grad

    def loss_many(self, Theta, X, y):
        """MSE for each row of Theta (P, n_params) at once. Returns shape (P,).

        Same result as calling loss() P times, just vectorised (useful for a
        CMA-ES/GA population).
        """
        Theta = np.asarray(Theta, dtype=float)
        if Theta.ndim != 2 or Theta.shape[1] != self.n_params:
            raise ValueError(f"Theta must have shape (P, {self.n_params}), got {Theta.shape}")
        d, H = self.d_in, self.hidden
        i = d * H
        W1 = Theta[:, :i].reshape(-1, d, H)               # (P, d, H)
        b1 = Theta[:, i:i + H]                            # (P, H)
        w2 = Theta[:, i + H:i + 2 * H]                    # (P, H)
        b2 = Theta[:, -1]                                 # (P,)
        h = np.tanh(np.einsum("nd,pdh->pnh", X, W1) + b1[:, None, :])   # (P, n, H)
        pred = np.einsum("pnh,ph->pn", h, w2) + b2[:, None]             # (P, n)
        return np.mean((pred - y) ** 2, axis=1)
