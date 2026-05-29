"""
Two-process live demo for CHRONO-Fair.

Run a producer in one terminal and a monitor in another, pointed at the
same CSV. The producer appends prediction rows over time (this is the
"add CSV rows with a script" path). The monitor tails the CSV and prints
the per-cell flip rate, e-value, and alarm as soon as new rows land.

Terminal A (producer, appends a batch every 2 seconds):
    python -m chrono_fair.live_demo produce --csv /tmp/feed.csv \
        --interval 2 --batch 200 --drift-after 3000

Terminal B (monitor, prints live scores):
    python -m chrono_fair.live_demo monitor --csv /tmp/feed.csv \
        --attr race --rho0 0.05 --alpha 0.05

You can also append rows by hand with any tool. The monitor picks up
whatever lands in the file on its next poll.

Research prototype. Not a medical device. Controlled replay / simulation.
"""
from __future__ import annotations
import argparse
import os
import time

import numpy as np
import pandas as pd

from chrono_fair.data.synthesizer import generate_stream, StreamConfig
from chrono_fair.ingest import CSVTailAdapter, MonitorRunner


COLUMNS = ["patient_id", "arrival", "prediction", "prediction_cf", "race", "sex"]


def _build_stream(n: int, drift_after: int | None, seed: int) -> pd.DataFrame:
    """Build a prediction stream with optional drift on Black patients."""
    cfg = StreamConfig(n=n + 4000, seed=seed)
    df = generate_stream(cfg).iloc[:n].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    # Cheap factual / counterfactual proxy: base flip rate 5%, raised to 30%
    # for Black patients after the drift index.
    base = rng.random(n) < 0.05
    pred = (rng.random(n) < 0.5).astype(int)
    flip = base.copy()
    if drift_after is not None:
        late_black = (df["race"].to_numpy() == "Black") & (np.arange(n) >= drift_after)
        flip = flip | (late_black & (rng.random(n) < 0.30))
    pred_cf = np.where(flip, 1 - pred, pred)
    return pd.DataFrame({
        "patient_id": np.arange(n),
        "arrival": df["arrival"].values,
        "prediction": pred,
        "prediction_cf": pred_cf,
        "race": df["race"].values,
        "sex": df["sex"].values,
    })


def produce(args: argparse.Namespace) -> int:
    stream = _build_stream(args.n, args.drift_after, args.seed)
    # Fresh file with header only.
    stream.iloc[:0].to_csv(args.csv, index=False)
    print(f"[produce] writing to {args.csv}; {len(stream)} rows total, "
          f"batch={args.batch}, interval={args.interval}s, "
          f"drift_after={args.drift_after}")
    cursor = 0
    while cursor < len(stream):
        end = min(cursor + args.batch, len(stream))
        stream.iloc[cursor:end].to_csv(args.csv, mode="a", header=False,
                                        index=False)
        print(f"[produce] appended rows {cursor}..{end} "
              f"({end}/{len(stream)})")
        cursor = end
        if cursor < len(stream):
            time.sleep(args.interval)
    print("[produce] done")
    return 0


def monitor(args: argparse.Namespace) -> int:
    # Wait for the file to exist.
    print(f"[monitor] watching {args.csv} on attribute '{args.attr}'")
    while not os.path.exists(args.csv):
        time.sleep(0.5)
    adapter = CSVTailAdapter(args.csv, poll_seconds=args.interval)
    # Lazy rho_0: a single flat baseline for every cell (the runner will
    # initialise unseen cells with this value).
    runner = MonitorRunner(sensitive_col=args.attr,
                            rho0_per_cell={"__default__": args.rho0},
                            alpha=args.alpha)
    runner.on_alarm = lambda cell, t: print(
        f"[monitor] *** ALARM cell={cell} at patient {t} ***")
    idle = 0
    while idle < args.max_idle:
        new = list(adapter.poll())
        if new:
            idle = 0
            runner.consume(iter(new))
            status = runner.get_status()
            line = " | ".join(
                f"{c}: n={s['n']}, flip={s['flip_rate']:.1%}, "
                f"E={s['e_value']:.2e}"
                + ("" if s["alarm_at"] is None else f", ALARM@{s['alarm_at']}")
                for c, s in sorted(status.items())
                if c != "__default__")
            print(f"[monitor] n_seen={runner.n_seen}  {line}")
        else:
            idle += 1
            time.sleep(args.interval)
    print("[monitor] no new rows for a while; stopping")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="CHRONO-Fair live CSV demo")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("produce", help="append prediction rows over time")
    pp.add_argument("--csv", required=True)
    pp.add_argument("--n", type=int, default=8000)
    pp.add_argument("--batch", type=int, default=200)
    pp.add_argument("--interval", type=float, default=2.0)
    pp.add_argument("--drift-after", type=int, default=3000)
    pp.add_argument("--seed", type=int, default=7)
    pp.set_defaults(func=produce)

    pm = sub.add_parser("monitor", help="tail the CSV and print live scores")
    pm.add_argument("--csv", required=True)
    pm.add_argument("--attr", default="race")
    pm.add_argument("--rho0", type=float, default=0.05)
    pm.add_argument("--alpha", type=float, default=0.05)
    pm.add_argument("--interval", type=float, default=1.0)
    pm.add_argument("--max-idle", type=int, default=20)
    pm.set_defaults(func=monitor)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    raise SystemExit(main())
