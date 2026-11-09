"""Tests for evonn.tuning and the tuning config."""

import pytest

from evonn.experiment import ROOT, load_config
from evonn.methods import METHODS
from evonn.tuning import expand_grid, make_config, method_settings, rank_settings, setting_id

TUNING = load_config(ROOT / "configs" / "tuning.yaml")
BASE = load_config(ROOT / TUNING["base_config"])


def test_expand_grid():
    assert expand_grid({"a": [1, 2], "b": [3, None]}) == [
        {"a": 1, "b": 3}, {"a": 1, "b": None}, {"a": 2, "b": 3}, {"a": 2, "b": None}]


def test_setting_id_readable_and_unique():
    assert setting_id({"sigma0": 0.3, "popsize": None}) == "sigma0-0.3_popsize-default"
    for grid in TUNING["grids"].values():
        ids = [setting_id(s) for s in expand_grid(grid)]
        assert len(set(ids)) == len(ids)
        assert all("/" not in i for i in ids)


def test_make_config_overrides_only_that_method():
    cfg = make_config(BASE, "cmaes", {"sigma0": 1.0})
    assert cfg["methods"]["cmaes"]["sigma0"] == 1.0
    assert cfg["methods"]["cmaes"]["popsize"] == BASE["methods"]["cmaes"]["popsize"]
    assert cfg["methods"]["ga"] == BASE["methods"]["ga"]
    assert cfg["experiment"]["name"] == "tuning/cmaes/sigma0-1.0"
    assert BASE["methods"]["cmaes"]["sigma0"] == 0.5          # base config untouched


def test_rank_settings():
    table = {"a": {(0, 100): 0.1, (0, 400): 0.3},
             "b": {(0, 100): 0.2, (0, 400): 0.1},
             "c": {(0, 100): 0.3, (0, 400): 0.2}}
    assert rank_settings(table) == {"a": 2.0, "b": 1.5, "c": 2.5}


def test_rank_settings_needs_shared_cells():
    with pytest.raises(ValueError):
        rank_settings({"a": {(0, 100): 0.1}, "b": {(1, 100): 0.1}})


def test_tuning_config_is_fair():
    # same effort: every method gets the same number of settings
    sizes = {m: len(expand_grid(g)) for m, g in TUNING["grids"].items()}
    assert set(sizes.values()) == {8}, sizes
    assert set(TUNING["grids"]) == set(METHODS)
    # tuning seeds never overlap the evaluation seeds
    assert not set(TUNING["seeds"]) & set(range(BASE["experiment"]["seeds"]))
    assert set(TUNING["n_train"]) <= set(BASE["data"]["n_train"])
    # every tuned parameter exists in the base config
    for m, grid in TUNING["grids"].items():
        assert set(grid) <= set(BASE["methods"][m]), (m, set(grid) - set(BASE["methods"][m]))


def test_method_settings_adds_extension_without_duplicates():
    tcfg = {"grids": {"m": {"a": [1, 2], "b": [3]}}, "extension": {"m": {"a": [2, 4], "b": [3]}}}
    assert method_settings(tcfg, "m") == [{"a": 1, "b": 3}, {"a": 2, "b": 3}, {"a": 4, "b": 3}]
    assert method_settings({"grids": tcfg["grids"]}, "m") == expand_grid(tcfg["grids"]["m"])


def test_extension_follows_the_edge_rule():
    for m, ext in (TUNING.get("extension") or {}).items():
        grid = TUNING["grids"][m]
        # same keys in the same order -> settings already run get the same id (and are skipped)
        assert list(ext) == list(grid), m
        for key, values in ext.items():
            old = [v for v in values if v in grid[key]]
            new = [v for v in values if v not in grid[key]]
            assert len(old) == 1, (m, key)                  # the round-1 winner's value
            assert len(new) <= 1, (m, key)                  # at most one step past the edge
            numeric = [v for v in grid[key] if v is not None]
            for v in new:
                assert v < min(numeric) or v > max(numeric), (m, key, v)   # outside the grid, not inside


def test_core_config_is_default_plus_tuning_winners():
    core = load_config(ROOT / "configs" / "core.yaml")
    best = load_config(ROOT / "results" / "processed" / "tuning_best.yaml")
    for m, setting in best.items():
        assert {k: core["methods"][m][k] for k in setting} == setting, m
    # everything else is the base config
    expected = make_config_all(BASE, best, core["experiment"]["name"])
    expected["budget"]["max_seconds"] = core["budget"]["max_seconds"]   # set in Phase 3, one per n
    assert core == expected
    assert set(core["budget"]["max_seconds"]) == set(core["data"]["n_train"])


def make_config_all(base, best, name):
    cfg = base
    for m, setting in best.items():
        cfg = make_config(cfg, m, setting)
    cfg["experiment"]["name"] = name
    return cfg


def test_wallclock_config_differs_only_in_budget_type_and_name():
    core = load_config(ROOT / "configs" / "core.yaml")
    wall = load_config(ROOT / "configs" / "core_wallclock.yaml")
    assert wall["budget"]["type"] == "wall_clock" and core["budget"]["type"] == "evaluations"
    assert wall["experiment"]["name"] != core["experiment"]["name"]     # separate result folders
    for cfg in (core, wall):
        cfg["budget"]["type"] = cfg["experiment"]["name"] = None
    assert wall == core
