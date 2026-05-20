# CHRONO-Fair

**Counterfactual Hazard-Rate ONline Surveillance of Fairness**

A time-resolved, anytime-valid, intersectionally-decomposed, automatically-inspected fairness monitor for clinical machine learning. Subsumes the scalar Verdict Flip Rate (VFR) as the aggregate marginal of a group-conditional Flip Hazard survival object and extends it along four axes:

1. **Time resolution** — `flip_hazard.py`: Kaplan–Meier / Nelson–Aalen / log-rank / RMFT
2. **Anytime-valid statistical control** — `e_process.py`: shrunk-GROW e-process + intersectional step-wise FDR
3. **Root-cause attribution** — `decomposition.py`: aleatoric vs epistemic split of flip mass
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
