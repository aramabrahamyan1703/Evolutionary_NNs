"""Registry of training methods.

Every method has the same signature:

    run(obj, model, x0, seed, params) -> {"starts": int, "generations": int | None}

  - obj:    CountedObjective; the method only calls obj.loss / obj.loss_and_grad /
            obj.loss_many / obj.batch_loss_and_grad, calls obj.begin_restart()
            before every (re)start, and returns when it raises BudgetExhausted
  - x0:     the shared initial weights model.init(seed), used by every method
  - seed:   run seed; all randomness comes from rng(seed, OPTIMISER)
  - params: the method's config section plus "bounds" and "init_scale"

The trained network is obj.best_theta (decision D6), not a return value.
"""

from evonn.baselines import lbfgs, sgd
from evonn.ea import cmaes, ga

METHODS = {
    "cmaes": cmaes.run,
    "ga": ga.run,
    "lbfgs": lbfgs.run,
    "sgd": sgd.run,
}

LABELS = {"cmaes": "CMA-ES", "ga": "GA", "lbfgs": "L-BFGS-B", "sgd": "SGD"}
