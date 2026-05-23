"""
Live data ingestion patterns for CHRONO-Fair.

This module documents the contract a real source has to honour and provides
two reference adapters that any clinical MLOps team can wire to a live
feed. The contract is intentionally narrow: the monitor consumes one
record per arriving patient, with the fields listed in the Record schema
below. Everything else (FHIR resources, Kafka, file tails, queues, HL7v2)
is the source's responsibility to translate.

Record schema
-------------
A single ingested record is a dict with at least:

  patient_id        : str or int       unique stable identifier
  arrival           : pandas.Timestamp UTC arrival timestamp
  prediction        : 0 or 1           the model's factual binary decision
  prediction_cf     : 0 or 1           the counterfactual decision under
                                        the protected-attribute swap
  race / sex / ...  : str              the protected-attribute values
                                        (only the attributes you monitor)

Optional fields (used by RCAP and the regression Inspector outputs):

  y_los_hat         : float            predicted length of stay in days
  y_los_hat_cf      : float            counterfactual predicted LOS
  hospital          : int or str       hospital identifier
  quarter           : int 1 to 4       calendar quarter, if applicable

Two adapters
------------
- CSVTailAdapter polls a CSV file and yields any new rows. It is the
  simplest way to wire CHRONO-Fair to an MLOps export.
- QueueAdapter wraps a queue.Queue so a thread or process can push records
  into the monitor. This is the recommended pattern for Kafka, FHIR
  Subscription, or HL7v2 listeners: write a small producer that drops a
  Record into the queue, and let CHRONO-Fair consume.

The MonitorRunner
-----------------
The MonitorRunner consumes records and updates the per-cell e-process and
the optional Flip Hazard buffer. It exposes a get_status() snapshot that a
dashboard or alerting webhook can poll.

This module deliberately holds no PHI and writes no logs by default. Bring
your own logger if you want audit traces.

Research prototype. Not a medical device. Streaming examples are
controlled replay or simulation unless connected to a validated
deployment stream.
"""
from __future__ import annotations
import os
import time
import queue
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, Optional
import pandas as pd

from chrono_fair.e_process import EProcessMonitor


# ----------------------------------------------------------------------------
# Record-level contract
# ----------------------------------------------------------------------------
REQUIRED_FIELDS = ('patient_id', 'arrival', 'prediction', 'prediction_cf')


def make_record(patient_id, arrival, prediction, prediction_cf,
                 **protected_attrs) -> Dict:
    """Build a single record. The producer (your FHIR or queue adapter)
    calls this for each arriving patient. The dict is opaque to the
    monitor; only the fields it needs are read."""
    rec = {
        'patient_id': patient_id,
        'arrival': pd.Timestamp(arrival),
        'prediction': int(prediction),
        'prediction_cf': int(prediction_cf),
    }
    rec.update(protected_attrs)
    return rec


# ----------------------------------------------------------------------------
# CSV tail adapter
# ----------------------------------------------------------------------------
class CSVTailAdapter:
    """Yield new rows from a CSV file as they appear.

    The adapter remembers how many rows it has emitted and re-reads the
    file on each poll(). It is good enough for an MLOps export that
    appends predictions to a CSV every few minutes. Use the queue adapter
    if you need lower latency.
    """

    def __init__(self, path: str, poll_seconds: float = 5.0):
        self.path = path
        self.poll_seconds = poll_seconds
        self._cursor = 0

    def poll(self) -> Iterator[Dict]:
        if not os.path.exists(self.path):
            return
        df = pd.read_csv(self.path)
        if self._cursor >= len(df):
            return
        new = df.iloc[self._cursor:]
        self._cursor = len(df)
        for _, row in new.iterrows():
            yield row.to_dict()

    def follow(self) -> Iterator[Dict]:
        """Blocking generator: poll forever and yield records as they arrive."""
        while True:
            for rec in self.poll():
                yield rec
            time.sleep(self.poll_seconds)


# ----------------------------------------------------------------------------
# Queue adapter (recommended for Kafka, FHIR Subscription, HL7v2 listener)
# ----------------------------------------------------------------------------
class QueueAdapter:
    """Wrap a queue.Queue so a producer thread can push records.

    Typical wiring with a FHIR Subscription handler:

        adapter = QueueAdapter()
        def fhir_callback(bundle):
            rec = parse_bundle_to_record(bundle)   # your code
            adapter.push(rec)
        ...
        runner.consume(adapter.follow(timeout=60))
    """

    def __init__(self, maxsize: int = 10_000):
        self.q: 'queue.Queue[Dict]' = queue.Queue(maxsize=maxsize)

    def push(self, record: Dict) -> None:
        self.q.put(record, timeout=5.0)

    def follow(self, timeout: float = 60.0) -> Iterator[Dict]:
        while True:
            try:
                yield self.q.get(timeout=timeout)
            except queue.Empty:
                return


# ----------------------------------------------------------------------------
# MonitorRunner: consumes records, updates per-cell e-processes
# ----------------------------------------------------------------------------
@dataclass
class MonitorRunner:
    """Maintain per-cell e-process state across an arriving stream.

    Parameters
    ----------
    sensitive_col : the protected-attribute key on each record. Multi-
        attribute monitoring is supported by running one MonitorRunner per
        protected attribute and sharing the patient stream.
    rho0_per_cell : pre-calibrated baseline flip rate per cell.
    alpha         : anytime-valid alpha for the e-process.
    on_alarm      : optional callback fired the first time a cell crosses
                    the alarm threshold. Use this to post to Slack or
                    PagerDuty.
    """
    sensitive_col: str
    rho0_per_cell: Dict[str, float]
    alpha: float = 0.05
    on_alarm: Optional[Callable[[str, int], None]] = None

    monitors: Dict[str, EProcessMonitor] = field(init=False)
    n_seen: int = field(init=False, default=0)
    last_alarms: Dict[str, int] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.monitors = {
            c: EProcessMonitor(rho0=r, alpha=self.alpha, cell_id=c)
            for c, r in self.rho0_per_cell.items()
        }

    def update_one(self, record: Dict) -> None:
        """Update the monitor with a single record."""
        for k in REQUIRED_FIELDS:
            if k not in record:
                raise KeyError(f'record is missing required field {k!r}')
        cell = record.get(self.sensitive_col)
        if cell is None:
            return
        if cell not in self.monitors:
            # First time we have seen this cell value: initialise lazily.
            default = float(sum(self.rho0_per_cell.values())
                              / max(1, len(self.rho0_per_cell)))
            self.monitors[cell] = EProcessMonitor(
                rho0=default, alpha=self.alpha, cell_id=str(cell))
        z = int(record['prediction'] != record['prediction_cf'])
        mon = self.monitors[cell]
        mon.update(z)
        self.n_seen += 1
        if mon.alarm_at is not None and cell not in self.last_alarms:
            self.last_alarms[cell] = mon.alarm_at
            if self.on_alarm is not None:
                self.on_alarm(str(cell), mon.alarm_at)

    def consume(self, stream: Iterator[Dict]) -> None:
        """Pull records from any iterator that yields dicts."""
        for rec in stream:
            self.update_one(rec)

    def get_status(self) -> Dict[str, Dict]:
        """Snapshot of the monitor for a dashboard or alert handler."""
        import numpy as np
        out = {}
        for c, m in self.monitors.items():
            out[c] = {
                'n': m.n,
                'flip_rate': (m.sum_z / m.n) if m.n else 0.0,
                'log_E': m.log_E,
                'e_value': float(np.exp(min(m.log_E, 700.0))),
                'alarm_at': m.alarm_at,
            }
        return out


# ----------------------------------------------------------------------------
# Convenience: spin a producer thread that tails a CSV
# ----------------------------------------------------------------------------
def csv_to_queue(csv_path: str, queue_adapter: QueueAdapter,
                  poll_seconds: float = 5.0) -> threading.Thread:
    """Tail csv_path in a background thread and push each new row to the
    queue adapter. Returns the thread; call thread.start() to run it."""

    def _producer():
        tail = CSVTailAdapter(csv_path, poll_seconds=poll_seconds)
        for rec in tail.follow():
            queue_adapter.push(rec)

    t = threading.Thread(target=_producer, daemon=True)
    return t


if __name__ == '__main__':
    # Tiny self-demo: 200 synthetic records, race as the sensitive attribute.
    import numpy as np
    from chrono_fair.data.synthesizer import generate_stream, StreamConfig
    df = generate_stream(StreamConfig(n=200, seed=0))
    rng = np.random.default_rng(0)
    df['prediction'] = (rng.random(len(df)) < 0.5).astype(int)
    df['prediction_cf'] = (rng.random(len(df)) < 0.5).astype(int)

    rho0 = {'White': 0.5, 'Black': 0.5, 'Hispanic': 0.5,
             'Asian/PI': 0.5, 'Other': 0.5}
    runner = MonitorRunner(sensitive_col='race', rho0_per_cell=rho0,
                            on_alarm=lambda c, t: print(f'ALARM {c} at t={t}'))

    for _, row in df.iterrows():
        runner.update_one(row.to_dict())
    for c, s in runner.get_status().items():
        print(f'{c:9s}  n={s["n"]:3d}  rate={s["flip_rate"]:.3f}  '
               f'log E={s["log_E"]:+.2f}  alarm={s["alarm_at"]}')
