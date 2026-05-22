"""Tests for the anytime-valid e-process monitor."""
import numpy as np
from chrono_fair.e_process import EProcessMonitor, detection_delay


def test_no_alarm_under_null():
    """Under H0 the e-process should rarely alarm at alpha = 0.05."""
    rng = np.random.default_rng(0)
    alarms = 0
    for s in range(50):
        m = EProcessMonitor(rho0=0.05, alpha=0.05)
        for _ in range(5000):
            m.update(int(rng.random() < 0.05))
        alarms += m.alarm_at is not None
    assert alarms / 50 <= 0.10   # well within nominal tolerance


def test_alarm_under_drift():
    """A clear upward drift must raise an alarm."""
    rng = np.random.default_rng(1)
    m = EProcessMonitor(rho0=0.05, alpha=0.05)
    for _ in range(8000):
        m.update(int(rng.random() < 0.20))
    assert m.alarm_at is not None


def test_one_sided_no_alarm_when_rate_below_rho0():
    """If the true rate is below rho0 the one-sided monitor must not alarm."""
    rng = np.random.default_rng(2)
    m = EProcessMonitor(rho0=0.15, alpha=0.05)
    for _ in range(8000):
        m.update(int(rng.random() < 0.05))
    assert m.alarm_at is None


def test_detection_delay_helper():
    assert detection_delay(None, 100, 200) is None
    assert detection_delay(50, 100, 200) == 0
    assert detection_delay(150, 100, 200) == 50
