"""Error metrics for a trained network.

Nothing here is charged to the budget: these are measurements, not training.

  - train_mse, val_mse, test_mse: plain MSE on the noisy targets
  - excess_test = test_mse - noise_floor: how far above the best possible error
  - test_mse_clean = mean((pred - f(X_test))^2): error against the noise-free f.
    On average it equals excess_test but varies less, so it is the primary
    measure of the true generalisation error.
  - gap = test_mse - train_mse
  - frac_at_bounds: share of weights with |w| >= 0.99*bound (tanh saturation check).
    Relative, because L-BFGS-B puts weights exactly on the bound while CMA-ES's
    bound handling only approaches it.
  - max_abs_theta: largest |weight|
"""

import numpy as np

from evonn.data import f

WEIGHT_BOUND = 5.0


def mse(model, theta, X, y):
    """Plain mean squared error."""
    return float(np.mean((model.forward(theta, X) - y) ** 2))


def evaluate(model, theta, data, n, bound=WEIGHT_BOUND, at_bound=0.99):
    """All metrics for weights `theta` trained on data["train"][n]. Returns a dict."""
    X_tr, y_tr = data["train"][n]
    X_val, y_val = data["val"]
    X_te, y_te = data["test"]

    pred_te = model.forward(theta, X_te)
    train = mse(model, theta, X_tr, y_tr)
    test = float(np.mean((pred_te - y_te) ** 2))

    out = {
        "train_mse": train,
        "val_mse": mse(model, theta, X_val, y_val),
        "test_mse": test,
        "excess_test": test - data["noise_floor"],
        "test_mse_clean": float(np.mean((pred_te - f(X_te)) ** 2)),
        "gap": test - train,
        "noise_floor": data["noise_floor"],
        "frac_at_bounds": float(np.mean(np.abs(theta) >= at_bound * bound)),
        "max_abs_theta": float(np.max(np.abs(theta))),
        "theta_norm": float(np.linalg.norm(theta)),
    }
    if "select" in data:                       # small validation set for model selection (D10)
        X_s, y_s = data["select"][n]
        out["select_mse"] = mse(model, theta, X_s, y_s)
    return out


def evaluate_snapshots(model, snapshots, data, n, with_theta=False, **kw):
    """evaluate() at every checkpoint of a CountedObjective. Returns a list of dicts sorted by evals
    (with the weights as "theta" if with_theta)."""
    rows = []
    for evals in sorted(snapshots):
        _, theta = snapshots[evals]
        rows.append({"evals": evals, **evaluate(model, theta, data, n, **kw)})
        if with_theta:
            rows[-1]["theta"] = theta.tolist()
    return rows
