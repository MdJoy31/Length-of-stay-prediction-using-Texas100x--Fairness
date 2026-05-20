"""
CHRONO-Fair: Counterfactual Hazard-Rate ONline Surveillance of Fairness.

A time-resolved, anytime-valid, intersectionally-decomposed, automatically-
inspected fairness monitor for clinical ML. Subsumes VFR (verdict flip rate)
as the t=0 instantaneous special case and extends it along four axes:

  1. Time-resolved Flip Hazard       (flip_hazard.py)
  2. Anytime-valid e-process monitor (e_process.py)
  3. Aleatoric/Epistemic decomposition (decomposition.py)
  4. Regression Counterfactual Allocation Parity (rcap.py)
"""
__version__ = "0.1.0"
