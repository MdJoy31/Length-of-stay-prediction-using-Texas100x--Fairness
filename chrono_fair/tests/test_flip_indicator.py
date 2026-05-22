"""Tests for the flip indicator and Flip Hazard estimator."""
import numpy as np
from chrono_fair.flip_hazard import (kaplan_meier_curve, logrank_two_groups,
                                       restricted_mean_flip_time)


def test_km_survival_monotone_non_increasing():
    rng = np.random.default_rng(0)
    z = (rng.random(600) < 0.1).astype(int)
    km = kaplan_meier_curve(np.arange(600.0), z)
    s = km['S'].values
    assert np.all(np.diff(s) <= 1e-9)


def test_km_survival_one_when_no_events():
    km = kaplan_meier_curve(np.arange(100.0), np.zeros(100, dtype=int))
    assert abs(km['S'].iloc[-1] - 1.0) < 1e-9


def test_logrank_detects_difference():
    rng = np.random.default_rng(1)
    et = np.arange(500.0)
    a = (rng.random(500) < 0.20).astype(int)
    b = (rng.random(500) < 0.03).astype(int)
    res = logrank_two_groups(et, a, et, b)
    assert res['pvalue'] < 0.01


def test_rmft_within_horizon():
    rng = np.random.default_rng(2)
    z = (rng.random(400) < 0.1).astype(int)
    km = kaplan_meier_curve(np.arange(400.0), z)
    rmft = restricted_mean_flip_time(km, 400)
    assert 0 <= rmft <= 400
