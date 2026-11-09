"""Mini-batch SGD baseline (Adam or momentum SGD) on the shared MLP.

Starts from the shared initial weights x0. Each epoch shuffles the training set
(OPTIMISER stream) and takes one step per mini-batch of batch_size points;
weights are clipped to the weight bounds after every step (projected SGD).

Budget: a mini-batch gradient costs grad_cost * batch / n evaluations, so one
epoch costs the same as one full-batch gradient. Every eval_every epochs the
full training loss is evaluated (cost 1); those evaluations pick the returned
network (lowest training loss, D6). No restarts: SGD runs until the budget is
exhausted.
"""

import numpy as np

from evonn.budget import BudgetExhausted
from evonn.seeding import rng, OPTIMISER

BETA1, BETA2, EPS = 0.9, 0.999, 1e-8     # standard Adam constants


def run(obj, model, x0, seed, params):
    """Run until the budget is exhausted.

    params: bounds, lr, batch_size, optimizer ("adam" | "momentum"), momentum, eval_every
    """
    opt_rng = rng(seed, OPTIMISER)
    lo, hi = params["bounds"]
    lr, adam = params["lr"], params["optimizer"] == "adam"
    if params["optimizer"] not in ("adam", "momentum"):
        raise ValueError(f"unknown optimizer {params['optimizer']!r}")
    n = len(obj.y)
    batch = min(params["batch_size"], n)

    theta = np.array(x0, dtype=float)
    m = np.zeros_like(theta)                 # first moment (Adam) / velocity (momentum)
    v = np.zeros_like(theta)                 # second moment (Adam)
    t = 0
    epochs = 0
    obj.begin_restart()
    try:
        obj.loss(theta)
        while True:
            order = opt_rng.permutation(n)
            for start in range(0, n, batch):
                _, grad = obj.batch_loss_and_grad(theta, order[start:start + batch])
                t += 1
                if adam:
                    m = BETA1 * m + (1 - BETA1) * grad
                    v = BETA2 * v + (1 - BETA2) * grad ** 2
                    step = lr * (m / (1 - BETA1 ** t)) / (np.sqrt(v / (1 - BETA2 ** t)) + EPS)
                else:
                    m = params["momentum"] * m - lr * grad
                    step = -m
                theta = np.clip(theta - step, lo, hi)
            epochs += 1
            if epochs % params["eval_every"] == 0:
                obj.loss(theta)
    except BudgetExhausted:
        pass
    return {"starts": len(obj.restarts), "generations": epochs}
