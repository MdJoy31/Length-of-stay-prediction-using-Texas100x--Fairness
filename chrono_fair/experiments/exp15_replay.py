"""
Experiment 15. Temporal and hospital-group replay on Texas-100X.

Texas-100X is administrative discharge data, not a live admission feed.
This experiment reconstructs three deployment-style replay streams from it
and runs the CHRONO-Fair monitor against ADWIN and a static VFR baseline.

Part A. Quarter-based temporal replay (Texas-100X-calibrated stream).
  A 30,000-patient stream spanning the four 2006 calendar quarters is
  generated. A fairness drift is injected from the Q3 boundary onward. The
  baseline rho_0 is calibrated on Q1. Q2, Q3, and Q4 are replayed in
  arrival order. The per-quarter flip rate, e-value, and alarm quarter are
  reported.

Part B. Hospital-group replay (Texas-100X-calibrated stream).
  The same stream carries 30 hospital sites with site-specific minority
  share. The baseline rho_0 is calibrated on 10 baseline hospitals. The
  remaining 20 hospitals are replayed in ascending hospital-ID order. The
  experiment reports whether the flip rate shifts under hospital case-mix
  change and whether the monitor detects it.

Part C. Real Texas-100 record-order replay.
  The real Texas-100 file (67,330 records) is replayed in file/record
  order, since the released file carries no calendar field. The baseline
  rho_0 is calibrated on the first 30% of records; the rest is replayed.
  This is a record-order replay, not a calendar-quarter replay, and is
  labelled as such.

No result is fabricated. Part C uses the real Texas-100 file if present,
else the shipped 12,000-record real subsample.
"""
from __future__ import annotations
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from chrono_fair.data.synthesizer import generate_stream, StreamConfig
from chrono_fair.e_process import EProcessMonitor


# --------------------------------------------------------------------------
# ADWIN baseline (compact, checked every 25 patients)
# --------------------------------------------------------------------------
class ADWIN:
    def __init__(self, delta=0.05, check_every=25):
        self.delta, self.check_every = delta, check_every
        self.cs, self.n, self.alarm_at, self._since = [0], 0, None, 0

    def update(self, z, t):
        self.cs.append(self.cs[-1] + z)
        self.n += 1
        self._since += 1
        if self.n < 60 or self._since < self.check_every:
            return False
        self._since = 0
        c, n = self.cs, self.n
        for split in range(20, n - 20, max(1, n // 30)):
            mu0, mu1 = c[split] / split, (c[n] - c[split]) / (n - split)
            m = 1 / (1 / split + 1 / (n - split))
            cut = np.sqrt((2 / m) * np.log(2 * n / self.delta))
            if abs(mu0 - mu1) > cut and self.alarm_at is None:
                self.alarm_at = t
                return True
        return False


# --------------------------------------------------------------------------
# Part A. Quarter-based temporal replay
# --------------------------------------------------------------------------
def quarter_replay(out_dir, drift_quarter=3, drift_strength=0.6):
    """Quarter-based replay with a controlled verdict-process drift.

    From `drift_quarter` onward, minority patients receive an additional
    risk shift on the factual prediction only. The counterfactual does not
    receive the shift, so the counterfactual flip rate rises for minority
    patients in the drift quarters. This is a controlled drift injection on
    Texas-100X-calibrated predictions.
    """
    cfg = StreamConfig(n=30_000, seed=42, aleatoric_bias=0.05)
    df = generate_stream(cfg)
    rng = np.random.default_rng(7)
    minority = df['race'].isin(['Black', 'Hispanic']).to_numpy()
    risk = df[['x0', 'x1', 'x2']].to_numpy().sum(axis=1) + minority * 0.6
    thr = np.median(risk + 0.4 * rng.standard_normal(len(df)))
    # injected drift on the factual prediction for minority patients in the
    # drift quarters
    drift_mask = minority & (df['quarter'].to_numpy() >= drift_quarter)
    risk_factual = risk + drift_strength * drift_mask
    df['y_hat'] = (risk_factual + 0.4 * rng.standard_normal(len(df)) > thr).astype(int)
    risk_cf = risk - 0.6 * minority
    df['y_hat_cf'] = (risk_cf + 0.4 * rng.standard_normal(len(df)) > thr).astype(int)
    df['flip'] = (df['y_hat'] != df['y_hat_cf']).astype(int)

    # Calibrate rho_0 on Q1
    rho0 = float(df.loc[df['quarter'] == 1, 'flip'].mean())
    # Replay Q2..Q4 in arrival order
    replay = df[df['quarter'] >= 2].sort_values('arrival').reset_index(drop=True)
    chrono = EProcessMonitor(rho0=rho0, alpha=0.05)
    adwin = ADWIN()
    chrono_alarm = adwin_alarm = None
    for t, z in enumerate(replay['flip']):
        chrono.update(int(z))
        adwin.update(int(z), t)
        if chrono_alarm is None and chrono.alarm_at is not None:
            chrono_alarm = t
        if adwin_alarm is None and adwin.alarm_at is not None:
            adwin_alarm = t

    # per-quarter table
    rows = []
    for q in [1, 2, 3, 4]:
        sub = df[df['quarter'] == q]
        rows.append({'quarter': f'Q{q}', 'n': len(sub),
                      'flip_rate': round(float(sub['flip'].mean()), 4),
                      'role': 'baseline (rho_0)' if q == 1 else 'replay'})
    qt = pd.DataFrame(rows)
    qt.to_csv(os.path.join(out_dir, 'exp15A_quarter_replay.csv'), index=False)

    # locate the replay index where Q3 begins (drift onset quarter)
    q3_start = int((replay['quarter'] == 3).values.argmax())
    summary = {
        'baseline_rho0_Q1': round(rho0, 4),
        'chrono_alarm_replay_idx': chrono_alarm,
        'adwin_alarm_replay_idx': adwin_alarm,
        'q3_starts_at_replay_idx': q3_start,
        'chrono_detected': chrono_alarm is not None,
        'adwin_detected': adwin_alarm is not None,
    }
    print('=== Part A: Quarter-based temporal replay ===')
    print(qt.to_string(index=False))
    print(f'baseline rho_0 (Q1) = {rho0:.4f}')
    print(f'Q3 (drift quarter) starts at replay index {q3_start}')
    print(f'CHRONO-Fair alarm at replay index {chrono_alarm}')
    print(f'ADWIN alarm at replay index {adwin_alarm}')
    return df, qt, summary


# --------------------------------------------------------------------------
# Part B. Hospital-group replay
# --------------------------------------------------------------------------
def hospital_replay(df, out_dir):
    # Rank hospitals by minority share. The 10 lowest-minority-share
    # hospitals form the baseline; the 20 higher-minority-share hospitals
    # are replayed in ascending minority-share order, so the replay stream
    # carries an increasing case-mix shift.
    share = (df.groupby('hospital')['race']
               .apply(lambda s: s.isin(['Black', 'Hispanic']).mean())
               .sort_values())
    ordered = list(share.index)
    baseline_h = ordered[:10]
    replay_h = ordered[10:]
    rho0 = float(df[df['hospital'].isin(baseline_h)]['flip'].mean())

    share_rank = {h: i for i, h in enumerate(ordered)}
    replay = df[df['hospital'].isin(replay_h)].copy()
    replay['_ord'] = replay['hospital'].map(share_rank)
    replay = replay.sort_values(['_ord', 'arrival']).reset_index(drop=True)
    chrono = EProcessMonitor(rho0=rho0, alpha=0.05)
    adwin = ADWIN()
    chrono_alarm = adwin_alarm = None
    for t, z in enumerate(replay['flip']):
        chrono.update(int(z))
        adwin.update(int(z), t)
        if chrono_alarm is None and chrono.alarm_at is not None:
            chrono_alarm = t
        if adwin_alarm is None and adwin.alarm_at is not None:
            adwin_alarm = t

    rows = []
    for h in ordered:
        sub = df[df['hospital'] == h]
        rows.append({'hospital': int(h), 'n': len(sub),
                      'minority_share': round(float(sub['race'].isin(
                          ['Black', 'Hispanic']).mean()), 3),
                      'flip_rate': round(float(sub['flip'].mean()), 4),
                      'role': 'baseline' if h in baseline_h else 'replay'})
    ht = pd.DataFrame(rows)
    ht.to_csv(os.path.join(out_dir, 'exp15B_hospital_replay.csv'), index=False)
    base_fr = ht[ht.role == 'baseline']['flip_rate']
    rep_fr = ht[ht.role == 'replay']['flip_rate']
    print('\n=== Part B: Hospital-group replay ===')
    print(f'baseline rho_0 (10 hospitals) = {rho0:.4f}')
    print(f'baseline-hospital flip rate: mean {base_fr.mean():.4f}, '
           f'range {base_fr.min():.4f} to {base_fr.max():.4f}')
    print(f'replay-hospital flip rate:   mean {rep_fr.mean():.4f}, '
           f'range {rep_fr.min():.4f} to {rep_fr.max():.4f}')
    print(f'CHRONO-Fair alarm at replay index {chrono_alarm}')
    print(f'ADWIN alarm at replay index {adwin_alarm}')
    summary = {'baseline_rho0': round(rho0, 4),
                'baseline_flip_mean': round(float(base_fr.mean()), 4),
                'replay_flip_mean': round(float(rep_fr.mean()), 4),
                'chrono_alarm_idx': chrono_alarm,
                'adwin_alarm_idx': adwin_alarm}
    return ht, summary


# --------------------------------------------------------------------------
# Part C. Real Texas-100 record-order replay
# --------------------------------------------------------------------------
def real_record_replay(out_dir):
    data_dir = os.environ.get('TEXAS100_DIR', '/tmp/tx_data')
    feat = os.path.join(data_dir, 'texas_100_features.p')
    if os.path.exists(feat):
        with open(feat, 'rb') as f:
            X = np.asarray(pickle.load(f))
        with open(os.path.join(data_dir, 'texas_100_labels.p'), 'rb') as f:
            y = np.asarray(pickle.load(f))
        src = 'full Texas-100'
    else:
        sub = os.path.join(os.path.dirname(__file__), '..', 'data',
                            'texas100_real_subsample.npz')
        if not os.path.exists(sub):
            print('\n=== Part C: real Texas-100 unavailable, skipped ===')
            return None
        npz = np.load(sub)
        X, y = npz['X'].astype(np.float64), npz['y']
        src = '12,000-record real Texas-100 subsample'

    classes, counts = np.unique(y, return_counts=True)
    frequent = set(classes[np.argsort(-counts)][:50].tolist())
    y_bin = np.array([1 if v in frequent else 0 for v in y])
    n = len(y_bin)
    i_tr = int(0.70 * n)
    # train an 11-member ensemble on the first 70% (record order)
    rng = np.random.default_rng(0)
    members = []
    for k in range(11):
        idx = rng.choice(i_tr, size=i_tr, replace=True)
        m = LogisticRegression(max_iter=200, solver='liblinear')
        m.fit(X[idx], y_bin[idx])
        members.append(m)
    preds = np.stack([m.predict(X) for m in members])
    modal = (preds.mean(axis=0) >= 0.5).astype(int)
    flip = (preds != modal).mean(axis=0)
    flip_event = (flip > 0).astype(int)

    # calibrate rho_0 on first 30% record order, replay the rest
    cal = int(0.30 * n)
    rho0 = float(flip_event[:cal].mean())
    replay = flip_event[cal:]
    chrono = EProcessMonitor(rho0=max(rho0, 1e-3), alpha=0.05)
    adwin = ADWIN()
    chrono_alarm = adwin_alarm = None
    for t, z in enumerate(replay):
        chrono.update(int(z))
        adwin.update(int(z), t)
        if chrono_alarm is None and chrono.alarm_at is not None:
            chrono_alarm = t
        if adwin_alarm is None and adwin.alarm_at is not None:
            adwin_alarm = t
    print(f'\n=== Part C: real Texas-100 record-order replay ({src}) ===')
    print(f'records {n}, calibration rho_0 (first 30%) = {rho0:.4f}')
    print(f'replay-window flip rate = {replay.mean():.4f}')
    print(f'CHRONO-Fair alarm at replay index {chrono_alarm}')
    print(f'ADWIN alarm at replay index {adwin_alarm}')
    summary = {'source': src, 'n_records': int(n),
                'calibration_rho0': round(rho0, 4),
                'replay_flip_rate': round(float(replay.mean()), 4),
                'chrono_alarm_idx': chrono_alarm,
                'adwin_alarm_idx': adwin_alarm}
    pd.DataFrame([summary]).to_csv(
        os.path.join(out_dir, 'exp15C_real_record_replay.csv'), index=False)
    return summary


def main():
    out_dir = os.path.join(os.path.dirname(__file__), '..', '..',
                            'chrono_fair_figures')
    os.makedirs(out_dir, exist_ok=True)
    df, qt, qa = quarter_replay(out_dir)
    ht, hb = hospital_replay(df, out_dir)
    rc = real_record_replay(out_dir)

    # ---- figure: 3 panels ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    # Panel A: per-quarter flip rate
    colours = ['#2ecc71' if q == 'Q1' else '#e74c3c' for q in qt['quarter']]
    axes[0].bar(qt['quarter'], qt['flip_rate'], color=colours)
    axes[0].axhline(qa['baseline_rho0_Q1'], ls='--', color='black',
                     label='baseline rho_0 (Q1)')
    axes[0].set_title('(A) Quarter-based temporal replay')
    axes[0].set_ylabel('Counterfactual flip rate')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, axis='y')
    # Panel B: per-hospital flip rate vs minority share
    base = ht[ht.role == 'baseline']
    rep = ht[ht.role == 'replay']
    axes[1].scatter(base['minority_share'], base['flip_rate'],
                     c='#2ecc71', label='baseline hospitals', s=45)
    axes[1].scatter(rep['minority_share'], rep['flip_rate'],
                     c='#e74c3c', label='replay hospitals', s=45)
    axes[1].set_title('(B) Hospital-group replay')
    axes[1].set_xlabel('Hospital minority share')
    axes[1].set_ylabel('Counterfactual flip rate')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    # Panel C: real record-order replay e-process trace
    if rc is not None:
        axes[2].text(0.5, 0.62, 'Real Texas-100 record-order replay',
                      ha='center', fontsize=10, transform=axes[2].transAxes)
        axes[2].text(0.5, 0.45,
                      f"n = {rc['n_records']:,}\n"
                      f"calibration rho_0 = {rc['calibration_rho0']}\n"
                      f"replay flip rate = {rc['replay_flip_rate']}\n"
                      f"CHRONO alarm idx = {rc['chrono_alarm_idx']}\n"
                      f"ADWIN alarm idx = {rc['adwin_alarm_idx']}",
                      ha='center', va='center', fontsize=9,
                      family='monospace', transform=axes[2].transAxes)
        axes[2].set_title('(C) Real Texas-100 record-order replay')
        axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig16_replay.png'), dpi=150,
                 bbox_inches='tight')
    plt.close()
    print('\nWrote fig16_replay.png')


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    main()
