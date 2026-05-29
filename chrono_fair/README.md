# CHRONO-Fair

**Counterfactual Hazard-Rate ONline Surveillance of Fairness**

A time-resolved, anytime-valid, intersectionally-decomposed, automatically-inspected fairness monitor for clinical machine learning. Subsumes the scalar Verdict Flip Rate (VFR) as the aggregate marginal of a group-conditional Flip Hazard survival object and extends it along four axes:

1. **Time resolution** — `flip_hazard.py`: Kaplan–Meier / Nelson–Aalen / log-rank / RMFT
2. **Anytime-valid statistical control** — `e_process.py`: shrunk-GROW e-process + intersectional step-wise FDR
3. **Diagnostic attribution** — `decomposition.py`: aleatoric vs epistemic triage split of flip mass (triage signal, not causal proof)
4. **Regression coverage** — `rcap.py`: Wasserstein-1 counterfactual rank shift (LOS)

Plus:
- `inspector_agent.py` — auto-generated governance report w/ FDA PCCP + EU AI Act mapping
- `dashboard/` — live Streamlit dashboard (also rendered as a static figure)
- `experiments/` — five reproducible studies generating every paper figure
- `paper/` — full LaTeX manuscript with bibliography of Q1-only references

## Quick start

```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn statsmodels

# Run all experiments (writes chrono_fair_figures/)
python -m chrono_fair.experiments.exp0_architecture_diagram
python -m chrono_fair.experiments.exp1_detection_delay
python -m chrono_fair.experiments.exp2_decomposition_accuracy
python -m chrono_fair.experiments.exp3_flip_hazard_curves
python -m chrono_fair.experiments.exp4_rcap_los
python -m chrono_fair.experiments.exp5_ablation_false_alarm
python -m chrono_fair.dashboard.mockup
```

## Headline results

| Metric                       | CHRONO-Fair        | Best baseline                        |
|------------------------------|--------------------|--------------------------------------|
| Drift detection rate         | **50 / 50 (100%)** | ADWIN: 21 / 50 (42 %)                |
| Mean detection delay         | **856 patients**   | ADWIN: 3927 patients (4.5× slower)    |
| False-alarm rate under H₀    | **0 / 50 (0 %)**   | matches nominal α                    |
| Decomposition accuracy       | **100 %**          | n/a                                   |
| RCAP gap exposed over MAE-gap| **up to 2.5×**     | (MAE-gap is the existing baseline)    |

## Architecture

```
clinical stream (FHIR)
       │
       ▼
deployed model ──► K-member ensemble
       │                │
       ▼                ▼
Flip Hazard      Aleatoric / Epistemic
e-process        decomposition
RCAP             ───────────────►   Inspector Agent ──► dashboard
       │                                │
       └─► alarms ◄─────── FDR ─────────┘
                                        │
                                        ▼
                              regulatory mapping
                              (FDA PCCP, EU AI Act,
                               STANDING Together)
```

See `paper/chrono_fair.pdf` for the full method, theorem, experiments and references.

## Monitoring dashboard

A Streamlit console with ten tabs (Overview, Live Stream, Flip Hazard,
Anytime E-Process, RCAP, Decomposition, Texas-100X Verdicts, Robustness,
Inspector Report, Export).

```bash
pip install -r requirements.txt
python -m chrono_fair.data.build_dataset    # build data/ artefacts first
streamlit run app.py
```

Configuration defaults are in `config/default.yaml` (alpha, q, rho_0,
warm-up, drift onset, stream length). A sample input schema is in
`data/sample_schema.csv`.

Research prototype. Not a medical device. Streaming examples are controlled
replay or simulation unless connected to a validated deployment stream.

## Live monitoring and end-to-end verification

Appending rows to a prediction CSV (by hand or with a script) updates the
per-cell scores and alarms incrementally. Two tools demonstrate this.

**One-shot end-to-end check** (trains a model, generates factual and
counterfactual predictions, appends rows in batches, confirms the scores
move and an alarm fires, smoke-tests the dashboard):

```bash
python -m chrono_fair.verify_end_to_end
```

**Two-process live demo.** Run a producer that appends rows over time in
one terminal and a monitor that tails the same CSV in another. The monitor
prints the per-cell flip rate, e-value, and alarm as soon as new rows land.

```bash
# Terminal A: append a 200-row batch every 2 seconds, drift after row 3000
python -m chrono_fair.live_demo produce --csv /tmp/feed.csv \
    --interval 2 --batch 200 --drift-after 3000

# Terminal B: tail the CSV and print live scores
python -m chrono_fair.live_demo monitor --csv /tmp/feed.csv \
    --attr race --rho0 0.05 --alpha 0.05
```

You can also append rows to `/tmp/feed.csv` with any tool; the monitor
picks up whatever lands in the file on its next poll. The ingest contract
and adapters (`CSVTailAdapter`, `QueueAdapter`, `MonitorRunner`) are in
`ingest.py`.

## Tests

```bash
pip install pytest
python -m pytest tests/ -q
```

Twelve tests cover the e-process (false-alarm control, one-sided
behaviour), RCAP (rank positions, Wasserstein-1, bootstrap CI), and the
Flip Hazard estimator (monotone survival, log-rank).

## Repository layout

- `app.py` : Streamlit monitoring console
- `config/default.yaml` : monitoring configuration
- `data/` : synthetic stream, 70/15/15 splits, real audit artefacts, real
  Texas-100 subsample, data card
- `flip_hazard.py, e_process.py, decomposition.py, rcap.py,
  inspector_agent.py` : the four estimators and the Inspector report
- `experiments/` : 14 reproducible experiment scripts
- `tests/` : pytest unit tests
- `paper/` : LaTeX manuscript and bibliography
- `overleaf/` : self-contained Overleaf package
- `CHRONO_Fair_End_to_End.ipynb` : executed end-to-end notebook
