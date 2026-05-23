"""
Texas100x-style synthetic clinical stream generator.

Generates a realistic length-of-stay (LOS) classification + regression stream
matching the Texas-100X statistics reported in the parent repository
(925,128 records, ~55/45 class balance, 4 protected attributes). Each patient
carries (i) features X, (ii) protected attribute A, (iii) admission time t,
(iv) ground-truth LOS y_los and binary y_ext (extended stay > 3 days).

The synthesizer optionally injects:
  * temporal distribution drift in P(X | A) starting at a known change-point
  * sensitive-attribute-correlated label noise (aleatoric bias source)
  * model-capacity-controlled prediction error (epistemic source)

This lets us evaluate CHRONO-Fair against ground truth in a controlled way
without requiring access to the proprietary Texas Inpatient PUDF.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Tuple


RACE_LEVELS = ['White', 'Black', 'Hispanic', 'Asian/PI', 'Other']
RACE_PROBS = [0.48, 0.13, 0.30, 0.05, 0.04]   # matches Texas demographics
SEX_LEVELS = ['Female', 'Male']
SEX_PROBS = [0.55, 0.45]
ETH_LEVELS = ['Non-Hispanic', 'Hispanic']
ETH_PROBS = [0.65, 0.35]
AGE_LEVELS = ['Pediatric', 'Young Adult', 'Middle-aged', 'Elderly']
AGE_PROBS = [0.05, 0.20, 0.40, 0.35]


@dataclass
class StreamConfig:
    n: int = 50_000
    n_features: int = 12
    drift_at: int | None = None   # arrival index where drift begins (None = no drift)
    drift_magnitude: float = 0.0  # 0 = no shift; 1.0 = full mean shift in minority group
    aleatoric_bias: float = 0.0   # fraction of minority-group labels flipped
    base_los_mean: float = 4.2
    seed: int | None = 42        # set to None for fresh randomisation each call
    n_hospitals: int = 30        # number of modelled hospital sites
    start_time: pd.Timestamp = field(default_factory=lambda: pd.Timestamp("2006-01-01"))


def generate_stream(cfg: StreamConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)

    race = rng.choice(RACE_LEVELS, size=cfg.n, p=RACE_PROBS)
    sex = rng.choice(SEX_LEVELS, size=cfg.n, p=SEX_PROBS)
    eth = rng.choice(ETH_LEVELS, size=cfg.n, p=ETH_PROBS)
    age_grp = rng.choice(AGE_LEVELS, size=cfg.n, p=AGE_PROBS)
    age_yrs = rng.normal(loc=np.where(age_grp == 'Pediatric', 8,
                                np.where(age_grp == 'Young Adult', 28,
                                np.where(age_grp == 'Middle-aged', 52, 74))),
                          scale=6.0)

    X = rng.normal(size=(cfg.n, cfg.n_features))

    # Encode true clinical risk as a function of (mostly non-sensitive) X
    # and a modest, *partly justified* effect of age (older patients stay longer).
    coef = rng.normal(size=cfg.n_features) * 0.3
    risk = X @ coef + 0.02 * (age_yrs - age_yrs.mean())

    # Inject a structural disparity: Black + Hispanic patients receive systematically
    # higher predicted risk because their X is drifted by a small amount in features 0-2.
    minority_mask = np.isin(race, ['Black', 'Hispanic'])
    X[minority_mask, :3] += 0.4

    # Apply temporal drift if requested. After cfg.drift_at, P(X|minority) further shifts.
    if cfg.drift_at is not None and cfg.drift_at < cfg.n:
        drift_idx = np.arange(cfg.n) >= cfg.drift_at
        delta = cfg.drift_magnitude * rng.normal(size=(drift_idx.sum(), 3))
        X[np.where(drift_idx)[0][:, None],
          np.arange(3)] += delta * minority_mask[drift_idx][:, None]

    # Continuous length of stay (regression target)
    y_los = np.maximum(
        0.5,
        cfg.base_los_mean
        + risk
        + 0.5 * rng.standard_normal(cfg.n)
    )
    # Binary extended-stay target. The threshold is set to the 55th
    # percentile of predicted length of stay, so that ~45% of records are
    # labelled extended stay. This matches the published Texas-100X class
    # balance (~55% normal, ~45% extended) used by the parent repository.
    los_threshold = float(np.quantile(y_los, 0.55))
    y_ext = (y_los > los_threshold).astype(int)

    # Aleatoric label bias: flip a fraction of minority-group labels (e.g.,
    # historical undercoding of extended stays for under-served groups).
    if cfg.aleatoric_bias > 0:
        flip_mask = minority_mask & (
            rng.random(cfg.n) < cfg.aleatoric_bias
        )
        y_ext[flip_mask] = 1 - y_ext[flip_mask]

    # Arrival timestamps spread evenly across a 360-day deployment year, so
    # the stream covers four calendar quarters as Texas-100X (2006) does.
    minutes_per_patient = (360 * 24 * 60) / max(1, cfg.n)
    arrival = cfg.start_time + pd.to_timedelta(
        np.arange(cfg.n) * minutes_per_patient, unit='m')

    # Calendar quarter, derived from the arrival timestamp. Texas-100X covers
    # the four quarters of 2006; the synthesiser reproduces a quarter label so
    # quarter-based temporal replay can be exercised.
    quarter = arrival.quarter.astype(int)

    # Hospital assignment. Texas-100X spans 441 hospitals; the synthesiser
    # models n_hospitals sites, each with a site-specific minority share. The
    # site-to-minority-share heterogeneity reproduces the hospital case-mix
    # variation that hospital-group replay is designed to surface.
    hosp_share = rng.uniform(0.20, 0.65, size=cfg.n_hospitals)
    # Assign each patient a hospital; minority patients are routed with a
    # probability proportional to the site minority share.
    hosp = np.empty(cfg.n, dtype=int)
    for i in range(cfg.n):
        if minority_mask[i]:
            w = hosp_share
        else:
            w = 1.0 - hosp_share
        hosp[i] = rng.choice(cfg.n_hospitals, p=w / w.sum())

    df = pd.DataFrame({
        'arrival': arrival,
        'patient_id': np.arange(cfg.n),
        'quarter': quarter,
        'hospital': hosp,
        'race': race,
        'sex': sex,
        'ethnicity': eth,
        'age_group': age_grp,
        'age_years': age_yrs,
        'y_los': y_los,
        'y_ext': y_ext,
    })
    for j in range(cfg.n_features):
        df[f'x{j}'] = X[:, j]
    return df


def temporal_split(df: pd.DataFrame, train_frac: float = 0.4) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split by arrival time, not random — simulates real deployment."""
    cut = int(len(df) * train_frac)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


if __name__ == "__main__":
    cfg = StreamConfig(n=10_000, drift_at=5_000, drift_magnitude=0.8, aleatoric_bias=0.05)
    s = generate_stream(cfg)
    print(s.describe())
    print("race counts:\n", s.race.value_counts())
