"""L-BFGS-B baseline via scipy.optimize.minimize, bounded to the weight bounds.

Starts from the shared initial weights x0. L-BFGS-B usually converges long
before the budget runs out, so it restarts from new random weights until the
budget is used up (otherwise it would effectively get a smaller budget).
Each call is a loss + gradient, charged grad_cost evaluations.
"""

from scipy.optimize import minimize

from evonn.budget import BudgetExhausted
from evonn.seeding import rng, OPTIMISER


def run(obj, model, x0, seed, params):
    """Run until the budget is exhausted. params: bounds, init_scale, maxcor, ftol (None = scipy default)."""
    opt_rng = rng(seed, OPTIMISER)
    bounds = [tuple(params["bounds"])] * model.n_params
    options = {"maxcor": params["maxcor"], "maxiter": 10**9, "maxfun": 10**9}
    if params.get("ftol") is not None:
        options["ftol"] = params["ftol"]
    try:
        while True:
            obj.begin_restart()
            minimize(obj.loss_and_grad, x0, jac=True, method="L-BFGS-B", bounds=bounds, options=options)
            x0 = opt_rng.uniform(-params["init_scale"], params["init_scale"], size=model.n_params)
    except BudgetExhausted:
        pass
    return {"starts": len(obj.restarts), "generations": None}
