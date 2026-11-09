"""Tests for evonn.seeding: streams are reproducible and independent of the data stream."""

import numpy as np

from evonn.seeding import rng, INIT, OPTIMISER


def test_stream_reproducible():
    assert np.array_equal(rng(3, INIT).uniform(size=10), rng(3, INIT).uniform(size=10))


def test_streams_differ_from_data_stream_and_each_other():
    data = np.random.default_rng(3).uniform(size=10)   # how evonn.data draws the training pool
    init = rng(3, INIT).uniform(size=10)
    opt = rng(3, OPTIMISER).uniform(size=10)
    assert not np.array_equal(data, init)
    assert not np.array_equal(data, opt)
    assert not np.array_equal(init, opt)
