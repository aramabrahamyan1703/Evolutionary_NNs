"""Tests for Phase 5.1: selection sets (D10), dense checkpoints, early-stopping checkpoints."""

import copy

import numpy as np
import pytest

from evonn.analysis import load_runs
from evonn.budget import default_checkpoints, dense_checkpoints
from evonn.data import generate_datasets, selection_sets
from evonn.experiment import ROOT, load_config, run_and_save, run_cell

CORE = load_config(ROOT / "configs" / "core.yaml")
P5 = load_config(ROOT / "configs" / "p5_selection.yaml")


def _small(cfg, max_evals=500):
    cfg = copy.deepcopy(cfg)
    cfg["experiment"]["name"] = "pytest"
    cfg["budget"]["max_evals"] = max_evals
    return cfg


def test_selection_sets_nested_sized_and_reproducible():
    sets = selection_sets(3)
    assert {n: len(sets[n][1]) for n in sets} == {25: 25, 100: 100, 400: 400, 1600: 1600}
    assert np.array_equal(sets[100][0], sets[1600][0][:100])
    assert np.array_equal(selection_sets(3)[400][1], sets[400][1])
    assert not np.array_equal(selection_sets(4)[25][0], sets[25][0])


def test_selection_sets_independent_of_training_and_val_data():
    data = generate_datasets(3)
    X_sel = selection_sets(3)[1600][0]
    for X in (data["train"][1600][0], data["val"][0], data["test"][0]):
        assert not np.isin(X_sel[:, 0], X[:, 0]).any()


def test_dense_checkpoints():
    pts = dense_checkpoints(100_000, per_decade=10)
    assert set(default_checkpoints(100_000)) <= set(pts)          # the 1-2-5 grid is kept
    assert pts == sorted(set(pts)) and pts[0] == 1 and pts[-1] == 100_000
    assert 45 <= len(pts) <= 60
    assert dense_checkpoints(37, 10)[-1] == 37


def test_p5_config_is_core_plus_selection():
    cfg = copy.deepcopy(P5)
    assert cfg.pop("selection") == {"checkpoints_per_decade": 10, "save_thetas": True}
    assert cfg["experiment"].pop("name") == "p5_selection"
    core = copy.deepcopy(CORE)
    core["experiment"].pop("name")
    assert cfg == core


@pytest.mark.parametrize("method", ["cmaes", "ga", "lbfgs", "sgd"])
def test_selection_does_not_change_training(method):
    plain = run_cell(_small(CORE), method, n=25, seed=1)
    sel = run_cell(_small(P5), method, n=25, seed=1)
    assert sel["theta"] == plain["theta"]
    assert sel["final"]["train_mse"] == plain["final"]["train_mse"]
    assert "best_select" not in plain and "select_mse" not in plain["final"]
    # early-stopping checkpoint: lowest selection MSE over the (denser) curve, with weights
    assert len(sel["curve"]) > len(plain["curve"])
    assert all("theta" in r and "select_mse" in r for r in sel["curve"])
    assert sel["best_select"]["select_mse"] == min(r["select_mse"] for r in sel["curve"])


def test_load_runs_selection(tmp_path):
    for method in ("cmaes", "lbfgs"):
        run_and_save(_small(P5, 300), method, 25, 0, tmp_path)
    final = load_runs(tmp_path / "pytest")
    early = load_runs(tmp_path / "pytest", "best_select")
    assert (final["selected_evals"] == final["evals_used"]).all()
    assert (early["selected_evals"] <= early["evals_used"]).all()
    assert "theta" not in early.columns and "select_mse" in early.columns
    with pytest.raises(ValueError):
        load_runs(tmp_path / "pytest", "nonsense")


def test_load_runs_without_selection_block_raises(tmp_path):
    run_and_save(_small(CORE, 200), "lbfgs", 25, 0, tmp_path)
    with pytest.raises(KeyError):
        load_runs(tmp_path / "pytest", "best_select")
