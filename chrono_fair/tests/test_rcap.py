"""Tests for Regression Counterfactual Allocation Parity."""
import numpy as np
from chrono_fair.rcap import rank_positions, rcap_w1, rcap_w1_ci


def test_rank_positions_in_unit_interval():
    rng = np.random.default_rng(0)
    y = rng.normal(size=500)
    u = rank_positions(y, y)
    assert u.min() >= 0.0 and u.max() <= 1.0


def test_rcap_zero_for_identical_distributions():
    rng = np.random.default_rng(1)
    u = rng.random(1000)
    assert rcap_w1(u, u) == 0.0


def test_rcap_positive_for_shifted_distributions():
    rng = np.random.default_rng(2)
    u_a = rng.random(1000)
    u_b = np.clip(u_a + 0.2, 0, 1)
    assert rcap_w1(u_a, u_b) > 0.1


def test_rcap_ci_brackets_or_exceeds_point():
    rng = np.random.default_rng(3)
    u_a = rng.random(800)
    u_b = np.clip(u_a + 0.1, 0, 1)
    w1, boot_mean, lo, hi = rcap_w1_ci(u_a, u_b, n_boot=200, seed=0)
    assert lo <= hi and w1 >= 0.0 and boot_mean >= 0.0
