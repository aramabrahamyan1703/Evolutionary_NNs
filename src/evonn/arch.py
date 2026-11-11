"""Phase 5.3: choose the hidden size H and the early-stopping checkpoint on the selection set.

    acfg = load_config("configs/p5_arch.yaml")
    cfgs = arch_configs(acfg)               # {H: experiment config}; the base H keeps its name
    df   = load_selected(cfgs)              # best_select network of every run, with column "hidden"
    best = choose_h(df)                     # per (method, n, seed): the H with the lowest select_mse
"""

import copy

import pandas as pd

from evonn.analysis import load_runs
from evonn.experiment import ROOT, load_config


def arch_config(base_cfg, hidden):
    """The base experiment with H = hidden; its own experiment name unless H is the base H
    (so the base experiment's runs are reused, not repeated)."""
    cfg = copy.deepcopy(base_cfg)
    if hidden != base_cfg["model"]["hidden"]:
        cfg["model"]["hidden"] = hidden
        cfg["experiment"]["name"] = f"{base_cfg['experiment']['name']}_H{hidden}"
    return cfg


def arch_configs(acfg):
    base = load_config(ROOT / acfg["base_config"])
    return {h: arch_config(base, h) for h in sorted(acfg["hidden"])}


def load_selected(cfgs, selection="best_select"):
    """The selected network of every run of every H, one row each, with a column "hidden"."""
    frames = []
    for h, cfg in sorted(cfgs.items()):
        df = load_runs(ROOT / cfg["experiment"]["output_dir"] / cfg["experiment"]["name"], selection)
        frames.append(df.assign(hidden=h))
    return pd.concat(frames, ignore_index=True)


def choose_h(df):
    """Per (method, n_train, seed) the row with the lowest select_mse (ties: the smaller H)."""
    df = df.sort_values(["method", "n_train", "seed", "hidden"])
    idx = df.groupby(["method", "n_train", "seed"])["select_mse"].idxmin()
    return df.loc[idx].reset_index(drop=True)
