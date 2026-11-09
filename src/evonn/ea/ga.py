"""Real-coded GA weight optimiser (DEAP operators), bounded to the weight bounds. Secondary EA.

Generational GA with elitism:
  - population of pop_size: the shared initial weights x0 plus random
    individuals from the same distribution U(-init_scale, init_scale)
  - tournament selection, blend crossover (cxBlend) with prob cx_prob,
    Gaussian mutation (mutGaussian, per-gene prob mut_indpb) with prob mut_prob
  - children are clipped to the weight bounds
  - the best `elitism` individuals survive unchanged
Children that crossover and mutation leave unchanged keep their fitness and
are not re-evaluated. One obj.loss call per new individual, until the budget
is exhausted.

DEAP's operators draw from Python's global `random` module, so it is seeded
from the OPTIMISER stream at the start of every run (reproducible per seed).
"""

import copy
import random

import numpy as np
from deap import base, creator, tools

from evonn.budget import BudgetExhausted
from evonn.seeding import rng, OPTIMISER

if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)


def _evaluate(obj, individuals):
    """Evaluate every individual without a valid fitness (one obj.loss call each)."""
    for ind in individuals:
        if not ind.fitness.valid:
            ind.fitness.values = (obj.loss(np.asarray(ind)),)


def run(obj, model, x0, seed, params):
    """Run until the budget is exhausted.

    params: bounds, init_scale, pop_size, elitism, tournament_size,
            cx_prob, cx_alpha, mut_prob, mut_sigma, mut_indpb
    """
    opt_rng = rng(seed, OPTIMISER)
    random.seed(int(opt_rng.integers(2**63)))
    lo, hi = params["bounds"]
    P, scale = params["pop_size"], params["init_scale"]

    pop = [creator.Individual(np.asarray(x0, dtype=float).tolist())] + [
        creator.Individual(opt_rng.uniform(-scale, scale, size=model.n_params).tolist())
        for _ in range(P - 1)]
    generations = 0
    obj.begin_restart()
    try:
        _evaluate(obj, pop)
        while True:
            elite = [copy.deepcopy(ind) for ind in tools.selBest(pop, params["elitism"])]
            children = [copy.deepcopy(ind) for ind in
                        tools.selTournament(pop, P - len(elite), params["tournament_size"])]
            for c1, c2 in zip(children[::2], children[1::2]):
                if random.random() < params["cx_prob"]:
                    tools.cxBlend(c1, c2, params["cx_alpha"])
                    del c1.fitness.values, c2.fitness.values
            for c in children:
                if random.random() < params["mut_prob"]:
                    tools.mutGaussian(c, 0.0, params["mut_sigma"], params["mut_indpb"])
                    del c.fitness.values
                c[:] = np.clip(c, lo, hi).tolist()
            pop = elite + children
            _evaluate(obj, pop)
            generations += 1
    except BudgetExhausted:
        pass
    return {"starts": len(obj.restarts), "generations": generations}
