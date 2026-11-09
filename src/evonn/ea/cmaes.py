"""CMA-ES weight optimiser (via the `cma` package), bounded to the weight bounds. Primary EA.

Starts from the shared initial weights x0 (as the initial mean) with step size
sigma0. When CMA-ES stops on its own (converged or stagnated), it restarts
from new random weights, so it always uses the whole budget. One obj.loss call
per candidate.
"""

import cma

from evonn.budget import BudgetExhausted
from evonn.seeding import rng, OPTIMISER


def run(obj, model, x0, seed, params):
    """Run until the budget is exhausted. params: bounds, init_scale, sigma0, popsize (None = default)."""
    opt_rng = rng(seed, OPTIMISER)
    lo, hi = params["bounds"]
    generations = 0
    try:
        while True:
            obj.begin_restart()
            opts = {"bounds": [lo, hi], "verbose": -9,
                    "seed": int(opt_rng.integers(1, 2**31 - 1))}
            if params.get("popsize"):
                opts["popsize"] = params["popsize"]
            es = cma.CMAEvolutionStrategy(x0, params["sigma0"], opts)
            while not es.stop():
                candidates = es.ask()
                es.tell(candidates, [obj.loss(x) for x in candidates])
                generations += 1
            x0 = opt_rng.uniform(-params["init_scale"], params["init_scale"], size=model.n_params)
    except BudgetExhausted:
        pass
    return {"starts": len(obj.restarts), "generations": generations}
