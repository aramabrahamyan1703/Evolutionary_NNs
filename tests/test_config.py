"""configs/default.yaml must agree with the constants in evonn.data.

Until experiment.py passes config values into generate_datasets, the same
settings live in two places; this test fails as soon as they drift apart.
"""

from pathlib import Path

import yaml

from evonn import data

CFG = yaml.safe_load((Path(__file__).parents[1] / "configs" / "default.yaml").read_text())


def test_config_matches_data_constants():
    d = CFG["data"]
    assert d["noise_std"] == data.SIGMA
    assert d["input_dim"] == data.D_IN
    assert tuple(d["n_train"]) == data.TRAIN_SIZES
    assert d["n_val"] == data.N_VAL
    assert d["n_test"] == data.N_TEST
    assert d["data_seed"] == data.DATA_SEED
    assert CFG["experiment"]["seeds"] == data.N_RUNS
