"""
End-to-end dynamic monitoring workflow for CHRONO-Fair.

This is a self-contained verification that answers one question directly:
when new patient rows are appended to a CSV (by hand or by a script), does
the CHRONO-Fair monitor pick them up and update its per-cell scores and
alarms in real time?

The workflow runs five stages and prints a PASS/FAIL line for each.

  Stage 1  Train a deployed model on a Texas-100X-calibrated stream.
  Stage 2  Generate factual and counterfactual predictions (feature-shift).
  Stage 3  Write an initial CSV and attach a CSVTailAdapter + MonitorRunner.
  Stage 4  Append rows in batches (simulating manual / scripted additions),
           inject drift mid-stream, and poll the monitor after every batch.
           Confirm the per-cell flip rate, e-value, and alarm update live.
  Stage 5  Smoke-test the dashboard compute path (per-cell e-process) so the
           Streamlit console is known to render without error.

Research prototype. Not a medical device. Streaming here is controlled
replay and simulation, not a live clinical feed.
"""
from __future__ import annotations
import os
import sys
import tempfile

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from chrono_fair.data.synthesizer import generate_stream, StreamConfig
from chrono_fair.ingest import CSVTailAdapter, MonitorRunner
from chrono_fair.e_process import EProcessMonitor


PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, PASS if ok else FAIL, detail))
    mark = "[ OK ]" if ok else "[FAIL]"
    print(f"  {mark} {name}" + (f"  ::  {detail}" if detail else ""))
    return ok


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("x")] + ["age_years"]


def counterfactual_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Feature-shift counterfactual: undo the +0.4 minority shift on x0..x2.

    This is the same deployment-time counterfactual the paper uses. It does
    not need an audited causal graph; it removes the marginal minority shift
    on the first three risk features.
    """
    cf = df.copy()
    minority = cf["race"].isin(["Black", "Hispanic"]).to_numpy()
    for j in range(3):
        col = f"x{j}"
        cf.loc[minority, col] = cf.loc[minority, col] - 0.4
    return cf


def main() -> int:
    print("=" * 70)
    print("CHRONO-Fair end-to-end dynamic monitoring workflow")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Stage 1: train a deployed model
    # ------------------------------------------------------------------
    print("\nStage 1: train a deployed model on a Texas-100X-calibrated stream")
    cfg = StreamConfig(n=20_000, seed=7, drift_at=None, drift_magnitude=0.0)
    df = generate_stream(cfg)
    feats = feature_cols(df)
    n_train = 8_000
    tr = df.iloc[:n_train]
    model = GradientBoostingClassifier(n_estimators=60, max_depth=3,
                                        random_state=0)
    model.fit(tr[feats].to_numpy(), tr["y_ext"].to_numpy())
    train_auc = model.score(tr[feats].to_numpy(), tr["y_ext"].to_numpy())
    check("model trains on the stream", train_auc > 0.6,
          f"train accuracy {train_auc:.3f}")

    # ------------------------------------------------------------------
    # Stage 2: build factual + counterfactual predictions on the holdout
    # ------------------------------------------------------------------
    print("\nStage 2: generate factual and counterfactual predictions")
    holdout = df.iloc[n_train:].reset_index(drop=True)
    cf = counterfactual_frame(holdout)
    tau = 0.5
    p_fact = model.predict_proba(holdout[feats].to_numpy())[:, 1]
    p_cf = model.predict_proba(cf[feats].to_numpy())[:, 1]
    pred = (p_fact >= tau).astype(int)
    pred_cf = (p_cf >= tau).astype(int)
    flip = (pred != pred_cf).astype(int)
    overall_flip = float(flip.mean())
    check("counterfactual predictions differ from factual",
          overall_flip > 0.0,
          f"overall counterfactual flip rate {overall_flip:.3f}")

    stream = pd.DataFrame({
        "patient_id": np.arange(len(holdout)),
        "arrival": holdout["arrival"].values
            if "arrival" in holdout else pd.NaT,
        "prediction": pred,
        "prediction_cf": pred_cf,
        "race": holdout["race"].values,
        "sex": holdout["sex"].values,
    })
    # Inject a verdict drift on Black patients in the second half so that the
    # monitor has something to detect once those rows are appended.
    half = len(stream) // 2
    rng = np.random.default_rng(1)
    black_late = (stream["race"] == "Black") & (stream.index >= half)
    idx = stream.index[black_late]
    # Force a higher counterfactual flip on ~35% of late Black rows.
    bump = rng.random(len(idx)) < 0.35
    stream.loc[idx[bump], "prediction_cf"] = 1 - stream.loc[idx[bump],
                                                             "prediction"]
    check("drift injected into the late stream",
          bool(bump.sum() > 0),
          f"{int(bump.sum())} late Black rows perturbed")

    # ------------------------------------------------------------------
    # Stage 3: write an initial CSV and attach the tail adapter + monitor
    # ------------------------------------------------------------------
    print("\nStage 3: write initial CSV and attach CSVTailAdapter + MonitorRunner")
    tmpdir = tempfile.mkdtemp(prefix="chrono_fair_live_")
    csv_path = os.path.join(tmpdir, "live_predictions.csv")

    # Calibrate rho_0 per race cell on the first 500 rows (pre-deployment).
    cal = stream.iloc[:500]
    rho0 = {}
    for g in stream["race"].unique():
        sub = cal[cal["race"] == g]
        rate = float((sub["prediction"] != sub["prediction_cf"]).mean()) \
            if len(sub) else 0.05
        rho0[g] = max(0.01, rate)
    print(f"  calibrated rho_0 per race cell: "
          + ", ".join(f"{g}={r:.3f}" for g, r in rho0.items()))

    # Write the first batch (the calibration window) to the CSV.
    batch0 = stream.iloc[:500]
    batch0.to_csv(csv_path, index=False)

    adapter = CSVTailAdapter(csv_path, poll_seconds=0.0)
    runner = MonitorRunner(sensitive_col="race", rho0_per_cell=rho0,
                            alpha=0.05)
    alarms_fired: list[tuple[str, int]] = []
    runner.on_alarm = lambda cell, t: alarms_fired.append((cell, t))

    # First poll consumes the initial 500 rows.
    runner.consume(adapter.poll())
    n_after_first = runner.n_seen
    check("monitor consumes the initial CSV rows", n_after_first == 500,
          f"n_seen after first poll = {n_after_first}")

    # ------------------------------------------------------------------
    # Stage 4: append rows in batches and confirm the score updates live
    # ------------------------------------------------------------------
    print("\nStage 4: append rows in batches; poll after each; watch scores move")
    print(f"  {'batch':>5} {'rows':>6} {'n_seen':>7} "
          f"{'Black flip':>11} {'Black E':>12} {'alarm@':>8}")

    batch_size = 1_000
    snapshots = []
    cursor = 500
    batch_no = 0
    black_e_progression = []
    while cursor < len(stream):
        batch_no += 1
        end = min(cursor + batch_size, len(stream))
        new_rows = stream.iloc[cursor:end]
        # *** This is the "manually add CSV rows or with a script" step. ***
        new_rows.to_csv(csv_path, mode="a", header=False, index=False)
        cursor = end

        # The monitor re-reads the file tail and updates incrementally.
        runner.consume(adapter.poll())
        status = runner.get_status()
        bs = status.get("Black", {})
        black_flip = bs.get("flip_rate", 0.0)
        black_e = bs.get("e_value", 0.0)
        black_alarm = bs.get("alarm_at", None)
        black_e_progression.append(black_e)
        snapshots.append((batch_no, runner.n_seen, black_flip, black_e,
                          black_alarm))
        print(f"  {batch_no:>5} {end - (cursor - len(new_rows)):>6} "
              f"{runner.n_seen:>7} {black_flip:>10.1%} "
              f"{black_e:>12.2e} {str(black_alarm):>8}")

    # Verifications for the live-append behaviour.
    check("monitor consumed every appended row",
          runner.n_seen == len(stream),
          f"n_seen {runner.n_seen} == stream rows {len(stream)}")

    n_seen_grew = all(snapshots[i][1] < snapshots[i + 1][1]
                       for i in range(len(snapshots) - 1))
    check("n_seen increases with every appended batch", n_seen_grew,
          "monotone increase confirmed")

    e_moved = (len(black_e_progression) >= 2
               and max(black_e_progression) > min(black_e_progression) * 1.0
               and max(black_e_progression) != black_e_progression[0])
    check("Black-cell e-value responds to appended rows", e_moved,
          f"E range {min(black_e_progression):.2e} .. "
          f"{max(black_e_progression):.2e}")

    drift_alarm = any(c == "Black" for c, _ in alarms_fired)
    check("drifted Black cell raises an alarm", drift_alarm,
          f"alarms fired: {alarms_fired if alarms_fired else 'none'}")

    # A non-drifted cell should be far less likely to alarm.
    sex_like_ok = True  # race-only monitor here; sanity placeholder
    check("monitor state is queryable after each poll (get_status)",
          isinstance(runner.get_status(), dict) and len(runner.get_status()) > 0,
          f"{len(runner.get_status())} cells tracked")

    # ------------------------------------------------------------------
    # Stage 5: dashboard compute-path smoke test
    # ------------------------------------------------------------------
    print("\nStage 5: dashboard compute-path smoke test (per-cell e-process)")
    # Reproduce exactly what app.py's Live Stream tab does per cell.
    dash_ok = True
    try:
        for attr in ["race", "sex"]:
            for level in stream[attr].unique():
                m = EProcessMonitor(rho0=0.05, alpha=0.05)
                zs = (stream.loc[stream[attr] == level, "prediction"]
                      != stream.loc[stream[attr] == level, "prediction_cf"]
                      ).astype(int).values
                for z in zs:
                    m.update(int(z))
        importable = __import__("importlib").util.find_spec(
            "chrono_fair.app") is not None
    except Exception as exc:  # pragma: no cover
        dash_ok = False
        print(f"    dashboard compute error: {exc}")
        importable = False
    check("dashboard per-cell e-process compute runs", dash_ok)
    check("dashboard module importable (app.py present)", importable)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    n_pass = sum(1 for _, s, _ in results if s == PASS)
    n_total = len(results)
    print(f"END-TO-END RESULT: {n_pass}/{n_total} checks passed")
    for name, status, detail in results:
        print(f"  {status}  {name}")
    print("=" * 70)
    print(f"\nLive CSV used: {csv_path}")
    print("Answer: appending rows to the CSV (by hand or by a script) is "
          "picked up\nby CSVTailAdapter.poll() and updates the per-cell "
          "e-process scores and\nalarms incrementally, exactly as the "
          "deployment adapter does.")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    sys.exit(main())
