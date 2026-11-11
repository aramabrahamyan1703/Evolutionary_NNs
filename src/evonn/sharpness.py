"""Phase 5.2: sharpness of the minima found (decision D11).

H1 says an EA avoids the sharp minima a gradient method falls into. Two measures of
how sharp the training loss L is around weights theta:

  perturbation_sharpness  (primary) mean increase of L under random weight perturbations
                          whose norm is a fixed fraction rho of ||theta||. Relative to the
                          weight norm, so larger weights are not automatically "sharper".
  top_eigenvalue          largest eigenvalue of the Hessian of L (41 x 41, central
                          differences of the analytic gradient). Not scale-invariant:
                          always report it with ||theta|| next to it.

Sharpness depends on how far training has gone, so methods are compared at matched
training MSE (matched_checkpoint): the GA fits least, so for each run of another
method we take the first checkpoint whose training MSE is down to the GA's final
training MSE on the same seed.
"""

import numpy as np

RHO = 0.01            # perturbation radius as a fraction of ||theta|| (D11)
N_PERTURB = 200       # random directions per network
FD_STEP = 1e-5        # central-difference step for the Hessian


def hessian(model, theta, X, y, h=FD_STEP):
    """Hessian of the training MSE at theta (symmetrised central differences of the gradient)."""
    theta = np.asarray(theta, dtype=float)
    d = len(theta)
    H = np.empty((d, d))
    for i in range(d):
        e = np.zeros(d)
        e[i] = h
        H[:, i] = (model.loss_and_grad(theta + e, X, y)[1] - model.loss_and_grad(theta - e, X, y)[1]) / (2 * h)
    return (H + H.T) / 2


def top_eigenvalue(model, theta, X, y):
    return float(np.linalg.eigvalsh(hessian(model, theta, X, y))[-1])


def perturbation_sharpness(model, theta, X, y, rho=RHO, n=N_PERTURB, seed=0):
    """Mean of L(theta + eps) - L(theta) over n random eps with ||eps|| = rho * ||theta||."""
    theta = np.asarray(theta, dtype=float)
    u = np.random.default_rng(seed).normal(size=(n, len(theta)))
    u *= rho * np.linalg.norm(theta) / np.linalg.norm(u, axis=1, keepdims=True)
    return float(np.mean(model.loss_many(theta + u, X, y)) - model.loss(theta, X, y))


def matched_checkpoint(curve, target_train_mse):
    """First checkpoint whose training MSE is <= target (the best-so-far training loss only
    falls, so this is where training reaches the target); the last checkpoint if it never does."""
    for row in curve:
        if row["train_mse"] <= target_train_mse:
            return row
    return curve[-1]


def measure(model, theta, X, y, bound):
    """All sharpness numbers for one network, with the weight-size context they need."""
    theta = np.asarray(theta, dtype=float)
    return {"perturb_sharpness": perturbation_sharpness(model, theta, X, y),
            "top_eigenvalue": top_eigenvalue(model, theta, X, y),
            "theta_norm": float(np.linalg.norm(theta)),
            "frac_at_bounds": float(np.mean(np.abs(theta) >= 0.99 * bound))}
