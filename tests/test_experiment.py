"""Tests for evonn.experiment: result format, config handling, saving and skipping."""

import copy
import json

import numpy as np
import pytest

from evonn.experiment import ROOT, build_data, load_config, run_cell, run_and_save, result_path
from evonn.methods import METHODS
from evonn.mlp import MLP

BASE = load_config(ROOT / "configs" / "default.yaml")


def _cfg(max_evals=300):
    cfg = copy.deepcopy(BASE)
    cfg["experiment"]["name"] = "pytest"
    cfg["budget"]["max_evals"] = max_evals
    return cfg


@pytest.fixture(scope="module", params=sorted(METHODS))
def result(request):
    return run_cell(_cfg(), request.param, n=25, seed=0)


def test_top_level_keys(result):
    assert set(result) == {"meta", "budget", "final", "theta", "curve", "best_val", "restarts"}


def test_meta_and_budget(result):
    assert result["meta"]["n_train"] == 25
    assert result["meta"]["seed"] == 0
    assert result["meta"]["model"]["n_params"] == 41
    assert result["budget"]["max_evals"] == 300
    assert 300 - result["budget"]["evals_used"] < 2
    assert result["budget"]["starts"] == len(result["restarts"])


def test_final_matches_theta(result):
    model = MLP()
    data = build_data(_cfg(), 0)
    theta = np.array(result["theta"])
    assert theta.shape == (41,)
    assert result["final"]["train_mse"] == pytest.approx(model.loss(theta, *data["train"][25]))


def test_curve_ends_at_final(result):
    curve = result["curve"]
    assert [r["evals"] for r in curve] == sorted(r["evals"] for r in curve)
    assert curve[-1]["evals"] == 300
    assert curve[-1]["train_mse"] == pytest.approx(result["final"]["train_mse"])
    assert result["best_val"]["val_mse"] == min(r["val_mse"] for r in curve)


def test_json_serialisable(result):
    json.dumps(result)


def test_uses_config_data_settings():
    cfg = _cfg()
    cfg["data"]["n_train"] = [10, 25]
    cfg["data"]["noise_std"] = 0.5
    r = run_cell(cfg, "lbfgs", n=10, seed=0)
    assert r["final"]["noise_floor"] == pytest.approx(0.25)
    assert r["meta"]["config"]["data"]["noise_std"] == 0.5


def test_wall_clock_budget():
    cfg = _cfg()
    cfg["budget"]["type"] = "wall_clock"
    cfg["budget"]["max_seconds"] = 0.2
    r = run_cell(cfg, "cmaes", n=25, seed=0)
    assert r["budget"]["wall_time"] < 1.0
    assert r["budget"]["evals_used"] > 0
    assert r["curve"][-1]["evals"] == r["budget"]["evals_used"]
    assert r["curve"][-1]["train_mse"] == pytest.approx(r["final"]["train_mse"])


def test_wall_clock_budget_per_n():
    cfg = _cfg()
    cfg["budget"]["type"] = "wall_clock"
    cfg["budget"]["max_seconds"] = {25: 0.1, 100: 0.3}
    assert run_cell(cfg, "cmaes", n=25, seed=0)["budget"]["max_seconds"] == 0.1
    cfg["budget"]["max_seconds"] = {100: 0.3}
    with pytest.raises(ValueError):                    # no limit for this n
        run_cell(cfg, "cmaes", n=25, seed=0)


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        run_cell(_cfg(), "no_such_method", n=25, seed=0)


def test_save_skip_overwrite(tmp_path):
    cfg = _cfg(max_evals=100)
    path, status = run_and_save(cfg, "lbfgs", 25, 0, tmp_path)
    assert status == "done"
    assert path == result_path(cfg, "lbfgs", 25, 0, tmp_path)
    assert json.loads(path.read_text())["meta"]["method"] == "lbfgs"
    assert run_and_save(cfg, "lbfgs", 25, 0, tmp_path)[1] == "skipped"
    assert run_and_save(cfg, "lbfgs", 25, 0, tmp_path, overwrite=True)[1] == "done"
    assert not list(tmp_path.rglob("*.tmp"))
