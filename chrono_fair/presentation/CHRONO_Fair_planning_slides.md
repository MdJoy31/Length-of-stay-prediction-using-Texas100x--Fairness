# CHRONO-Fair: Planning-Presentation Slide Deck

A 12-slide planning deck covering the gap, the prior VFR-Audit work, and the CHRONO-Fair extension. Each slide is one numbered section. Bullets are speaker notes. Drop into Gamma / PowerPoint / Keynote.

---

## Slide 1: Title
**CHRONO-Fair: From Static Fairness Audits to Time-Resolved Counterfactual Monitoring in Clinical ML**

- Target: Q1 digital-health / health-informatics venue (Frontiers in Digital Health-style)
- Planning deck for the journal submission of the controlled-validation study
- Anonymous draft (double-blind compatible)

---

## Slide 2: The clinical-AI fairness problem in one slide

- Clinical machine learning models are increasingly deployed for risk scoring, length-of-stay prediction, and triage.
- Once released, they are evaluated by a thresholded group-fairness metric (e.g. disparate impact >= 0.80).
- The output of that evaluation is a binary verdict: pass / fail.
- That verdict is the object hospitals act on.

---

## Slide 3: The gap that motivates this work

- Existing tooling reports the fairness verdict on a fixed snapshot.
- Aequitas, Fairlearn, AIF360, Microsoft RAI Dashboard, Davis et al. drift study: all batch or quarterly.
- A point-estimate verdict hides whether the underlying number is stable.
- Real Texas-100X evidence (925,128 records): one race verdict reads "fair" at WTPR 0.824, yet flips to "unfair" in 4 of 20 hospital-network resamples (VFR 0.20).
- A batch audit cannot tell when, or whether, that flip recurs as new patients arrive.

---

## Slide 4: What our prior VFR-Audit study proposed

- VFR-Audit introduced the Verdict Flip Rate (VFR) as a scalar measure of static verdict-stability.
- Three audit-reliability axes:
  - Resampling fluctuation of the fairness verdict.
  - Audit-size sensitivity of the fairness verdict.
  - Cross-hospital-site agreement of the fairness verdict.
- Defined the Standard / Fair monitored-model setup at release time.
- Real Texas-100X numbers (paper_table8a):
  - Race DI: 0.654 -> 0.884 (fail -> pass)
  - Sex DI: 0.762 -> 0.933 (fail -> pass)
  - Ethnicity DI: 0.832 -> 0.962 (pass -> pass)
  - Age Group DI: 0.291 -> 0.801 (fail -> pass)
  - Std Acc 0.8777, Fair Acc 0.8248 (-5.28 pp)

---

## Slide 5: What VFR-Audit did NOT do

- VFR-Audit returns one number per audited verdict.
- No time axis.
- No alarm rule that controls false-alarm rate under repeated inspection.
- No diagnostic attribution (is the signal data-side or model-side?).
- No regression-output extension (length-of-stay rank allocation).
- These are the four gaps CHRONO-Fair targets.

---

## Slide 6: CHRONO-Fair, in one sentence

> **CHRONO-Fair contributes a post-release, time-resolved monitoring layer for thresholded fairness verdicts using a per-cell anytime-valid e-process on counterfactual flip indicators.**

- It is a monitor, not a mitigator.
- It does NOT create the Fair model. It begins after the Standard -> Fair transition.
- It can sit above any pre-release mitigation method, or above an uncorrected model.

---

## Slide 7: Architecture

- Input: patient stream, deployed model.
- Factual prediction + counterfactual prediction -> flip indicator Z_i.
- Four modules consume the flip indicator:
  1. **Flip Hazard** (Kaplan-Meier on patient-arrival index): time-to-flip view.
  2. **Per-cell anytime-valid e-process** (shrunk GROW, Ville's inequality): main alarm rule.
  3. **Diagnostic attribution** (entropy-based aleatoric / epistemic split): triage signal, not causal proof.
  4. **RCAP** (Wasserstein-1 on rank positions): length-of-stay regression extension.
- Output: Inspector report (cell, e-value, hazard ratio, triage label, recommended action) -> governance review.

---

## Slide 8: Method discipline (what we explicitly do not claim)

- Per-cell anytime-valid Type-I control: YES, under Ville's inequality.
- Step-wise Benjamini-Hochberg across cells: applied at each inspection time only; NOT a time-uniform online FDR guarantee.
- Diagnostic attribution: aleatoric / epistemic triage; NOT causal root cause.
- Counterfactual definition: feature-shift (computable at inference); weaker than SCM-based counterfactual fairness; may under-flag indirect effects.

---

## Slide 9: Evidence base

- Layer 1: real Texas-100X batch verdict-stability analysis (925,128 records).
  - 11 of 12 verdicts stable; 1 of 12 unstable; the unstable one is race-WTPR.
- Layer 2: Texas-100X-calibrated controlled streaming and retrospective replay experiments.
  - Detection delay (50 Monte-Carlo runs).
  - False-alarm calibration across alpha.
  - Sensitivity to drift magnitude.
  - Four-attribute monitoring.
  - Quarter-based and hospital-group replays.
- Layer 3: robustness and stress tests (miscalibrated baseline, small intersectional cells, label delay).
- Layer 2 and Layer 3 are controlled replay / simulation, not prospective clinical validation.

---

## Slide 10: Headline streaming result

- Detection on a 10,000-patient stream with drift onset at patient 5,000:
  - CHRONO-Fair: 50 of 50 alarms, median delay 828 patients, mean 837.
  - ADWIN (k=1): 21 of 50, mean delay 3,878.
  - ADWIN (k=25): 21 of 50, mean delay 3,927.
  - Davis-style ADWIN on fairness gap: 21 of 50, mean delay 3,904.
  - Periodic batch (n=500): 24 of 50, mean delay 499 (no peeking control).
  - Static VFR: 0 of 50.
- Empirical false-alarm rate at or below the nominal alpha for every tested alpha.

---

## Slide 11: Operational comparison

| Method | Per-patient update | Repeated-inspection control | Diagnostic output | Limitation |
|---|---|---|---|---|
| Static VFR | none after release | no | no | no time-localisation |
| Periodic batch | O(batch size) at boundaries | no | no | peeking vs latency |
| ADWIN | lightweight | drift bound, not fairness-specific | no | not designed for verdict flips |
| CHRONO-Fair | O(1) per cell | per-cell anytime-valid | yes (triage only) | controlled-replay evidence; prospective validation needed |

---

## Slide 12: Limitations and next steps

- Single empirical clinical dataset (Texas-100X).
- Streaming evidence is controlled replay / simulation, not prospective EHR.
- Feature-shift counterfactual is weaker than SCM-based counterfactual fairness.
- Small intersectional cells need >= 1,000 monitored patients for detection rate >= 0.95.
- BH layer is inspection-time cross-cell, not full online FDR.
- Diagnostic attribution is triage, not causal proof.
- CHRONO-Fair is a research prototype, not a medical device.

**Next steps:**
- Prospective EHR-stream validation on a partner-site cohort.
- Time-uniform online FDR procedure (LORD / e-BH / SAFFRON).
- Counterfactual-sensitivity study across swap, feature-shift, and SCM definitions.
- Governance-user evaluation of the Inspector report.
- Anonymised reproducibility artifact at journal-camera-ready time.

---

## Speaker notes for Q&A

- "Why VFR-Audit if you don't claim it?" -> VFR-Audit is the prior static predecessor. The Standard / Fair table is the monitored-model starting condition, not a CHRONO-Fair contribution. CHRONO-Fair begins after release.
- "Why not use full online FDR?" -> the inspection-time BH is the conservative honest choice in the present submission. LORD / e-BH is named as future work in Method 4.2 and in the Conclusion.
- "Why feature-shift counterfactual?" -> it is computable at inference time without an audited causal graph. The Limitations paragraph names this as a deliberate trade-off and SCM-based counterfactual fairness as the stronger but heavier alternative.
- "Why not deploy?" -> the Limitations and Scope statement both state that prospective EHR-stream validation has not been completed. No operational adoption claim is made.
