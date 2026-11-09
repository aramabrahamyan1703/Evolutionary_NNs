"""Run one (method, n_train, seed) cell and write one JSON result to results/raw/.

    result = run_cell(cfg, "cmaes", n=100, seed=0)
    path   = run_and_save(cfg, "cmaes", n=100, seed=0, out_dir=Path("results/raw"))

Everything comes from the config (see configs/default.yaml), including the data
settings, so the yaml is the single source of truth for an experiment.

The JSON holds:
  meta      experiment, method, seed, n_train, params, full config, git commit, versions, time
  budget    max_evals, grad_cost, evals_used, n_calls, wall_time, starts, generations
  final     metrics of the returned network (lowest training loss, D6)
  theta     its weights
  curve     metrics of the best-so-far network at every checkpoint (convergence
            curves; results at smaller budgets)
  best_val  the checkpoint with the lowest validation MSE (D6 secondary analysis)
  restarts  metrics of each restart's best network (D6 secondary analysis)
"""

import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import yaml

from evonn.budget import CountedObjective
from evonn.data import generate_datasets
from evonn.methods import METHODS
from evonn.metrics import evaluate, evaluate_snapshots
from evonn.mlp import MLP

ROOT = Path(__file__).resolve().parents[2]
WALL_CLOCK_EVAL_CAP = 10**9     # evaluation cap when the budget is a time limit


def load_config(path):
    return yaml.safe_load(Path(path).read_text())


def build_data(cfg, seed):
    d = cfg["data"]
    return generate_datasets(seed, sigma=d["noise_std"], train_sizes=tuple(d["n_train"]),
                             n_val=d["n_val"], n_test=d["n_test"], data_seed=d["data_seed"])


def method_params(cfg, method):
    """The method's config section plus the shared model settings."""
    return {**(cfg["methods"].get(method) or {}),
            "bounds": list(cfg["model"]["weight_bounds"]),
            "init_scale": cfg["model"]["init_scale"]}


def result_path(cfg, method, n, seed, out_dir):
    return Path(out_dir) / cfg["experiment"]["name"] / f"{method}_n{n}_s{seed}.json"


def _git_commit():
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def _short(metrics):
    keys = ("train_mse", "val_mse", "test_mse", "test_mse_clean")
    return {k: metrics[k] for k in keys}


def run_cell(cfg, method, n, seed):
    """Train `method` on the size-n training set of `seed` and return the result dict."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; available: {sorted(METHODS)}")
    model = MLP(d_in=cfg["data"]["input_dim"], hidden=cfg["model"]["hidden"])
    data = build_data(cfg, seed)
    X, y = data["train"][n]
    x0 = model.init(seed, scale=cfg["model"]["init_scale"])
    params = method_params(cfg, method)
    bound = cfg["model"]["weight_bounds"][1]

    b = cfg["budget"]
    if b["type"] == "evaluations":
        max_evals, max_seconds = b["max_evals"], None
    elif b["type"] == "wall_clock":
        max_evals, max_seconds = WALL_CLOCK_EVAL_CAP, b["max_seconds"]
        if isinstance(max_seconds, dict):           # one limit per training size: {n: seconds}
            max_seconds = max_seconds.get(n, max_seconds.get(str(n)))
        if not max_seconds:
            raise ValueError(f"budget.type = wall_clock needs budget.max_seconds (for n = {n})")
    else:
        raise ValueError(f"unknown budget.type {b['type']!r}")

    obj = CountedObjective(model, X, y, max_evals=max_evals, grad_cost=b["grad_cost"],
                           max_seconds=max_seconds)
    info = METHODS[method](obj, model, x0, seed, params)
    wall_time = obj.elapsed
    # the curve must end at the returned network, also when the run stopped between
    # checkpoints (time limit, or a budget not reachable in steps of grad_cost)
    obj.snapshots.setdefault(obj.evals_used, (obj.best_loss, obj.best_theta.copy()))

    curve = evaluate_snapshots(model, obj.snapshots, data, n, bound=bound)
    best_val = min(curve, key=lambda r: r["val_mse"])
    restarts = [{"start_evals": r["start_evals"],
                 **_short(evaluate(model, r["best_theta"], data, n, bound=bound))}
                for r in obj.restarts if r["best_theta"] is not None]

    return {
        "meta": {
            "experiment": cfg["experiment"]["name"],
            "method": method,
            "seed": seed,
            "n_train": n,
            "params": params,
            "config": cfg,
            "model": {"d_in": model.d_in, "hidden": model.hidden, "n_params": model.n_params},
            "git_commit": _git_commit(),
            "versions": {p: version(p) for p in ("numpy", "scipy", "cma", "deap")},
            "python": platform.python_version(),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "budget": {
            "type": b["type"],
            "max_evals": max_evals,
            "max_seconds": max_seconds,
            "grad_cost": b["grad_cost"],
            "evals_used": obj.evals_used,
            "n_calls": obj.n_calls,
            "wall_time": wall_time,
            **info,
        },
        "final": evaluate(model, obj.best_theta, data, n, bound=bound),
        "theta": obj.best_theta.tolist(),
        "curve": curve,
        "best_val": best_val,
        "restarts": restarts,
    }


def run_and_save(cfg, method, n, seed, out_dir, overwrite=False):
    """run_cell and write the JSON. Skips the run if the file exists (unless overwrite).

    Returns (path, "done" | "skipped"). The file is written to a temporary name
    and renamed, so an interrupted run never leaves a half-written result.
    """
    path = result_path(cfg, method, n, seed, out_dir)
    if path.exists() and not overwrite:
        return path, "skipped"
    result = run_cell(cfg, method, n, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=1))
    os.replace(tmp, path)
    return path, "done"
