"""Tests for Phase 5.3 (evonn.arch, configs/p5_arch.yaml)."""

import copy

import pandas as pd

from evonn.arch import arch_config, arch_configs, choose_h, load_selected
from evonn.experiment import ROOT, load_config, run_and_save

ACFG = load_config(ROOT / "configs" / "p5_arch.yaml")
P5 = load_config(ROOT / "configs" / "p5_selection.yaml")


def test_arch_configs_change_only_hidden_and_name():
    cfgs = arch_configs(ACFG)
    assert sorted(cfgs) == sorted(ACFG["hidden"]) and P5["model"]["hidden"] in cfgs
    assert cfgs[P5["model"]["hidden"]] == P5                     # base H reuses the base runs
    names = [c["experiment"]["name"] for c in cfgs.values()]
    assert len(set(names)) == len(names)
    for h, cfg in cfgs.items():
        assert cfg["model"]["hidden"] == h
        other = copy.deepcopy(cfg)
        other["model"]["hidden"] = P5["model"]["hidden"]
        other["experiment"]["name"] = P5["experiment"]["name"]
        assert other == P5


def test_choose_h_lowest_selection_mse_ties_to_smaller_h():
    df = pd.DataFrame({"method": "ga", "n_train": 25, "seed": [0, 0, 0, 1, 1],
                       "hidden": [10, 2, 4, 2, 4], "select_mse": [0.3, 0.2, 0.2, 0.5, 0.1]})
    chosen = choose_h(df).set_index("seed")["hidden"]
    assert chosen.to_dict() == {0: 2, 1: 4}


def test_load_selected(tmp_path):
    base = copy.deepcopy(P5)
    base["experiment"].update(name="pytest", output_dir=str(tmp_path))
    base["budget"]["max_evals"] = 300
    cfgs = {h: arch_config(base, h) for h in (2, 10)}
    for cfg in cfgs.values():
        for seed in (0, 1):
            run_and_save(cfg, "lbfgs", 25, seed, tmp_path)
    df = load_selected(cfgs)
    assert sorted(df["hidden"].unique()) == [2, 10] and len(df) == 4
    assert len(choose_h(df)) == 2
