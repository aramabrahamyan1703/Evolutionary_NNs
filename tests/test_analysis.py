"""Tests for evonn.analysis and the Phase 4 configs."""

import copy

import numpy as np
import pandas as pd
import pytest

from evonn.analysis import (cliffs_delta, combine_with_base, compare_methods, compare_pairs, holm_correction,
                            load_runs, make_medians, make_wilcoxon, paired, processed_dir, rankings)
from evonn.experiment import ROOT, load_config, run_and_save

CONFIGS = ROOT / "configs"
CORE = load_config(CONFIGS / "core.yaml")


def _df(values, method="a", n=100):
    """Runs of one method at one n: {seed: test_mse_clean}."""
    return pd.DataFrame({"method": method, "n_train": n, "seed": list(values),
                         "test_mse_clean": list(values.values()), "train_mse": list(values.values())})


def test_holm_known_values():
    assert holm_correction([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm_correction([0.5, 0.9]) == pytest.approx([1.0, 1.0])     # capped at 1
    assert len(holm_correction([])) == 0


def test_cliffs_delta():
    assert cliffs_delta([2, 3], [0, 1]) == 1.0
    assert cliffs_delta([0, 1], [2, 3]) == -1.0
    assert cliffs_delta([1, 2], [0, 3]) == 0.0


def test_paired_matches_by_seed_not_by_order():
    df = pd.concat([_df({0: 1.0, 1: 2.0, 2: 3.0}, "a"),
                    _df({2: 30.0, 0: 10.0, 1: 20.0}, "b")])          # b stored in a different order
    x, y = paired(df, "test_mse_clean", 100, "a", "b")
    assert np.allclose(y / x, 10.0)


def test_paired_raises_on_different_seeds():
    df = pd.concat([_df({0: 1.0, 1: 2.0}, "a"), _df({0: 1.0, 2: 2.0}, "b")])
    with pytest.raises(ValueError):
        paired(df, "test_mse_clean", 100, "a", "b")


def test_wilcoxon_table():
    rng = np.random.default_rng(0)
    seeds = range(30)
    base = rng.uniform(1, 2, 30)
    df = pd.concat([_df(dict(zip(seeds, base)), "a"), _df(dict(zip(seeds, base + 1)), "b"),
                    _df(dict(zip(seeds, base + rng.normal(0, 0.01, 30))), "c")])
    wil = make_wilcoxon(df).set_index(["method_a", "method_b"])
    assert len(wil) == 3
    assert wil.loc[("a", "b"), "significant"] and wil.loc[("a", "b"), "cliffs_delta"] < 0
    assert not wil.loc[("a", "c"), "significant"]
    assert (wil["p_holm"] >= wil["p_raw"]).all()
    assert rankings(df)[100][-1] == "b"


def test_wilcoxon_one_method_is_empty():
    wil = make_wilcoxon(_df({0: 1.0, 1: 2.0}))
    assert len(wil) == 0 and "p_holm" in wil.columns


def test_combine_and_compare():
    base = pd.concat([_df({0: 1.0, 1: 2.0, 2: 3.0}, m) for m in ("a", "b")])
    other = _df({0: 1.0, 1: 2.0, 2: 3.0}, "a")
    combined, filled = combine_with_base(other, base)
    assert filled == ["b"] and set(combined["method"]) == {"a", "b"}
    cmp = compare_methods(base, other)
    assert cmp.loc[0, "p_raw"] == 1.0 and cmp.loc[0, "cliffs_delta"] == 0.0    # identical runs
    pairs = compare_pairs(make_wilcoxon(base), make_wilcoxon(combined))
    assert pairs["same_verdict"].all()


def test_load_runs_and_medians(tmp_path):
    cfg = copy.deepcopy(CORE)
    cfg["experiment"]["name"] = "pytest"
    cfg["budget"]["max_evals"] = 200
    for method in ("cmaes", "lbfgs"):
        for seed in (0, 1):
            run_and_save(cfg, method, 25, seed, tmp_path)
    df = load_runs(tmp_path / "pytest")
    assert len(df) == 4 and {"train_mse", "test_mse_clean", "git_commit", "seed"} <= set(df.columns)
    med = make_medians(df)
    assert list(med["runs"]) == [2, 2]
    assert len(make_wilcoxon(df)) == 1
    with pytest.raises(FileNotFoundError):
        load_runs(tmp_path / "empty")


def test_each_experiment_has_its_own_output_folder():
    names = [load_config(p)["experiment"]["name"] for p in sorted(CONFIGS.glob("core*.yaml"))]
    assert "core_ntrain_sweep" in names and len(names) == len(set(names))
    assert len({processed_dir(n) for n in names}) == len(names)


def _without(cfg, *paths):
    cfg = copy.deepcopy(cfg)
    for path in paths:
        section, key = path.split(".")
        cfg[section].pop(key)
    return cfg


@pytest.mark.parametrize("g", [1, 3])
def test_gradcost_configs_differ_only_where_intended(g):
    cfg = load_config(CONFIGS / f"core_gradcost{g}.yaml")
    assert cfg["budget"]["grad_cost"] == g
    assert cfg["experiment"]["methods"] == ["lbfgs", "sgd"]
    fields = ("experiment.name", "experiment.methods", "budget.grad_cost")
    assert _without(cfg, *fields) == _without(CORE, *fields)


def test_ga_long_config_differs_only_where_intended():
    cfg = load_config(CONFIGS / "core_ga_long.yaml")
    assert cfg["budget"]["type"] == "evaluations" and cfg["budget"]["max_evals"] == 10 * CORE["budget"]["max_evals"]
    assert cfg["experiment"]["methods"] == ["ga"] and cfg["experiment"]["seeds"] == 10
    # max_seconds is only used by wall-clock budgets
    fields = ("experiment.name", "experiment.methods", "experiment.seeds", "budget.max_evals", "budget.max_seconds")
    assert _without(cfg, *fields) == _without(CORE, *fields)
