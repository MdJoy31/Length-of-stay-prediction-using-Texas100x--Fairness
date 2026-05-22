"""
Week 2 — Anytime-Valid Fairness Monitor (AVFM).

We monitor whether the per-patient flip indicator Z_t in {0, 1} of a deployed
classifier remains consistent with a pre-deployment baseline flip-rate rho0.
The monitor is an *e-process* (cumulative product of bets) that supports
peeking at every patient without inflating the Type-I error.

Concretely, for one protected group:

    E_t = prod_{s <= t} ( 1 + lambda * (Z_s - rho0) ) ,

where lambda in [-1/(1-rho0), 1/rho0] is a tunable bet size. Under H_0:
E[Z_s] = rho0, the sequence (E_t) is a non-negative supermartingale, so by
Ville's inequality

    P( sup_t E_t >= 1/alpha ) <= alpha .

We adopt the **GROW** adaptive betting strategy of Waudby-Smith & Ramdas
(Annals of Stats 2024) and aggregate per-cell e-values across intersectional
strata with online Benjamini-Hochberg FDR control (Foster & Stine 2008).

Public API
----------
- EProcessMonitor             : per-cell anytime-valid monitor
- IntersectionalMonitor       : multi-cell with online BH FDR
- detection_delay             : utility to evaluate against ground-truth drift
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class EProcessMonitor:
    rho0: float                     # baseline flip rate under H0
    alpha: float = 0.05             # anytime-valid significance level
    lambda_init: float = 0.5        # initial bet; clipped to feasible range
    use_grow: bool = True           # GROW adaptive lambda
    cell_id: str = ""

    # state
    log_E: float = 0.0
    n: int = 0
    sum_z: float = 0.0
    bets: List[float] = field(default_factory=list)
    alarm_at: int | None = None

    def feasible(self) -> tuple[float, float]:
        # lambda * (z - rho0) must keep 1 + .. > 0 for z in {0, 1}.
        return -1.0 / max(1e-9, 1 - self.rho0), 1.0 / max(1e-9, self.rho0)

    def update(self, z: float) -> Dict[str, float]:
        lo, hi = self.feasible()
        # One-sided shrunk GROW. The monitor tests only for an INCREASE in the
        # flip rate above rho0, so the bet lambda is clipped to [0, hi). A
        # non-negative lambda keeps the test one-sided: a run of zeros (rate
        # below rho0) cannot inflate E_t. The bet is predictable in z (uses
        # only past observations). The 0.25 shrinkage and the 100-sample
        # warm-up are calibrated for empirical false-alarm control at or below
        # the nominal alpha (see exp5_ablation_false_alarm.py, Theorem 1).
        SHRINK = 0.25
        WARMUP = 100
        lam_max = hi - 1e-6   # upper feasible bound; lower bound is 0
        if self.use_grow:
            if self.n >= WARMUP:
                z_bar = self.sum_z / self.n
                raw = (z_bar - self.rho0) / max(1e-9, self.rho0 * (1 - self.rho0))
                lam = SHRINK * raw
                lam = float(np.clip(lam, 0.0, lam_max))
            else:
                # Warm-up: bet 0 so E_t = 1 (no premature alarms).
                lam = 0.0
        else:
            # Ablation mode: fixed predictable lambda, one-sided.
            lam = float(np.clip(self.lambda_init, 0.0, lam_max))
        self.bets.append(lam)

        factor = 1.0 + lam * (z - self.rho0)
        factor = max(factor, 1e-12)
        self.log_E += float(np.log(factor))
        self.n += 1
        self.sum_z += z

        threshold = np.log(1.0 / self.alpha)
        is_alarm = self.log_E >= threshold
        if is_alarm and self.alarm_at is None:
            self.alarm_at = self.n

        return {
            'cell': self.cell_id,
            'n': self.n,
            'log_E': self.log_E,
            'E': float(np.exp(min(self.log_E, 700.0))),
            'lambda': lam,
            'alarm': bool(is_alarm),
            'alarm_at': self.alarm_at,
        }


@dataclass
class IntersectionalMonitor:
    """Run one EProcessMonitor per intersectional cell and apply online BH-FDR.

    Online BH (Foster & Stine 2008): at any inspection time, control the
    expected false-discovery proportion at level q. We approximate by sorting
    current p-values implied by e-values (p_t = 1/E_t) and rejecting any
    cell with p_t < q * rank_t / m where m is the number of active cells.
    """
    rho0_per_cell: Dict[str, float]
    alpha: float = 0.05
    fdr_q: float = 0.10

    monitors: Dict[str, EProcessMonitor] = field(init=False)
    history: List[Dict[str, float]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.monitors = {
            c: EProcessMonitor(rho0=r, alpha=self.alpha, cell_id=c)
            for c, r in self.rho0_per_cell.items()
        }

    def step(self, observations: Dict[str, float]) -> List[Dict[str, float]]:
        """One sweep: feed Z to each cell that has a new observation this tick."""
        rows: List[Dict[str, float]] = []
        for cell, z in observations.items():
            if cell not in self.monitors:
                # New cell discovered mid-stream: initialise with global default.
                default_rho0 = float(np.mean(list(self.rho0_per_cell.values())))
                self.monitors[cell] = EProcessMonitor(
                    rho0=default_rho0, alpha=self.alpha, cell_id=cell)
            r = self.monitors[cell].update(z)
            rows.append(r)
        # FDR step
        e_vals = np.array([np.exp(min(self.monitors[c].log_E, 700.0))
                            for c in self.monitors])
        p_vals = np.clip(1.0 / np.maximum(e_vals, 1e-12), 0.0, 1.0)
        order = np.argsort(p_vals)
        m = len(p_vals)
        thresh = self.fdr_q * (np.arange(1, m + 1)) / m
        sorted_p = p_vals[order]
        cutoff_idx = np.where(sorted_p <= thresh)[0]
        k_star = cutoff_idx.max() + 1 if len(cutoff_idx) else 0
        flagged = set()
        if k_star > 0:
            flagged = {list(self.monitors.keys())[i] for i in order[:k_star]}
        for r in rows:
            r['fdr_flagged'] = r['cell'] in flagged
        self.history.extend(rows)
        return rows

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)


def detection_delay(
    alarm_at: int | None, drift_at: int, stream_len: int
) -> int | None:
    """Return number of patients between true drift and first alarm.

    Returns None if no alarm was raised, or 0 if the alarm preceded the drift
    (false alarm — handled by α control).
    """
    if alarm_at is None:
        return None
    if alarm_at < drift_at:
        return 0
    return alarm_at - drift_at


__all__ = ['EProcessMonitor', 'IntersectionalMonitor', 'detection_delay']
