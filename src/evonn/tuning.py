"""Hyperparameter tuning (Phase 3, decision D9).

Every method gets the same tuning effort: a small grid of settings plus one
edge-extension round (see configs/tuning.yaml), run on separate tuning seeds, ranked by final training
MSE. One run is an ordinary experiment cell (evonn.experiment.run_and_save)
whose config has the grid point applied, saved under
results/raw/tuning/<method>/<setting id>/.
"""

import copy
import itertools
import json
from pathlib import Path

import numpy as np


def expand_grid(grid):
    """{"a": [1, 2], "b": [3]} -> [{"a": 1, "b": 3}, {"a": 2, "b": 3}] (in a fixed order)."""
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def method_settings(tcfg, method):
    """Round-1 grid plus the edge-extension round (if any), without duplicates, in a fixed order."""
    settings, seen = [], set()
    for grid in (tcfg["grids"][method], (tcfg.get("extension") or {}).get(method)):
        for s in expand_grid(grid) if grid else []:
            if setting_id(s) not in seen:
                seen.add(setting_id(s))
                settings.append(s)
    return settings


def setting_id(setting):
    """Readable, file-system safe name: {"sigma0": 0.3, "popsize": None} -> "sigma0-0.3_popsize-default"."""
    return "_".join(f"{k}-{'default' if v is None else v}" for k, v in setting.items())


def make_config(base_cfg, method, setting):
    """Base config with the method's parameters overridden by `setting`."""
    cfg = copy.deepcopy(base_cfg)
    cfg["methods"][method] = {**(cfg["methods"].get(method) or {}), **setting}
    cfg["experiment"]["name"] = f"tuning/{method}/{setting_id(setting)}"
    return cfg


def load_results(out_dir, method, setting):
    """All saved runs of one setting, as a list of result dicts."""
    folder = Path(out_dir) / "tuning" / method / setting_id(setting)
    return [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))]


def rank_settings(table):
    """Rank settings by training MSE within every (seed, n) cell, then average the ranks.

    table: {setting_id: {(seed, n): train_mse}}, every setting with the same cells.
    Returns {setting_id: mean rank} (1 = best). Ranks within a cell make the
    different training-set sizes comparable (their MSE scales differ).
    """
    ids = list(table)
    cells = sorted(set.intersection(*(set(table[i]) for i in ids)))
    if not cells:
        raise ValueError("settings share no (seed, n) cells")
    ranks = {i: [] for i in ids}
    for cell in cells:
        values = np.array([table[i][cell] for i in ids])
        order = values.argsort(kind="stable").argsort() + 1
        for i, r in zip(ids, order):
            ranks[i].append(r)
    return {i: float(np.mean(r)) for i, r in ranks.items()}
