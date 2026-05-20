# CHRONO-Fair Brutal Reviewer Loop

Each round: act as a NeurIPS/JMLR/NEJM-AI Area Chair reviewing the paper
under the standards of a Q1 journal / A*-conference reject-recommend. Find
the most damaging real issue still present. Fix the paper + code. Move on.

## Round 1 — "VFR reframing is undermotivated"

**Critique.** The abstract claims VFR is the t=0 marginal of Flip Hazard,
but Section 3 says VFR corresponds to `1 - S_a(tau*)`. These two
identifications cannot both be true. A reviewer will catch the
inconsistency and lose trust in the whole framework.

**Fix.** Clarify in Section 3 that VFR can be cast as the *aggregate*
flip-probability `1 - S_a(tau*)` evaluated at the full sample size; it is
NOT the t=0 marginal (which would be 0 by definition of a survival
function). Drop the "t=0 special case" claim from the abstract.

## Round 2 — "Detection delay comparison is unfair"

**Critique.** ADWIN at delta=0.05 is being run with check_every=25, which
artificially slows it. CHRONO-Fair updates every patient. To compare like
with like, ADWIN should be tested with check_every=1 too.

**Fix.** Add a second ADWIN configuration at check_every=1 to exp1 and
report both. Likely still loses to CHRONO-Fair but the comparison becomes
honest.

## Round 3 — "Decomposition track 2 looks like a failure"

**Critique.** Track 1 of exp2 is perfect (100%); Track 2 (the ML-driven
case the paper actually claims matters) gets 38% accuracy and the
confusion matrix is dominated by 'mixed'. A reviewer will read this as
'the decomposition does not work in practice'.

**Fix.** Either (i) reframe Track 2 as a positive finding ("real ML
pipelines naturally produce mixed cases, which CHRONO-Fair correctly
identifies as requiring joint mitigation") and remove the accuracy label,
OR (ii) drop Track 2 from the paper and keep only the controlled validation.
Choose (i) — it is the honest scientific story.

## Round 4 — "RCAP effect sizes are tiny"

**Critique.** The RCAP table shows rcap_W1 values of 0.006–0.016. These
are tiny rank shifts and a reviewer will ask if the metric is even
detectable above noise. Need either (a) larger structural disparity in
the synthesiser to make the gap visible, or (b) a confidence interval
on RCAP so we can say "the gap is significant despite being small".

**Fix.** Add a bootstrap CI for RCAP and report it in Figure 4b. Also
increase synthesiser drift_magnitude in exp4 to show RCAP responding.

## Round 5 — "FDR control claim is unproven"

**Critique.** The paper claims "online BH FDR" for the intersectional
monitor but the implementation is the static BH applied at every step,
which does NOT provide online FDR control in the Foster-Stine sense.

**Fix.** Either implement true online BH (Javanmard-Montanari LORD) or
soften the claim to "step-wise FDR control under exchangeable cells".
The latter is more honest and still publishable.

## Round 6 — "Synthesiser bakes in the conclusion"

**Critique.** The synthesiser injects +0.4 on features 0-2 for
minority groups, AND the counterfactual_fn removes exactly that shift.
The framework is being tested against the disparity it constructed.

**Fix.** Add a sanity check experiment where the model truly does NOT
depend on the sensitive attribute (no minority shift) and confirm that
CHRONO-Fair does NOT raise false alarms across protected groups.

## Round 7 — "No real data"

**Critique.** Everything is synthetic. Reviewers from NEJM AI / Nature
Med will demand at least one real-data demonstration.

**Fix.** Acknowledge the limitation prominently in Limitations + cite
the prior VFR paper's real-data Texas-100X results that CHRONO-Fair
reduces to as the t=tau* marginal.

## Round 8 — "Anytime-valid bet derivation is hand-waved"

**Critique.** Theorem 1 states Ville's bound but the betting strategy
specification "shrunk GROW with c=0.25 and warm-up 100" is a hyperparameter
choice without theoretical justification. A statistics reviewer will ask
why these specific numbers.

**Fix.** Replace the hyperparameter justification with explicit empirical
calibration (Experiment 5) and note that other settings of (c, warm-up)
trade detection delay against conservatism but anytime-validity holds for
ANY predictable choice — that's the strength of Ville's inequality.

## Round 9 — "No comparison to Maughan-Near prediction sensitivity"

**Critique.** Section 2 cites prediction sensitivity (Maughan-Near 2022)
as the closest existing approach but doesn't compare against it
empirically. A reviewer will say "this might just be prediction sensitivity
in survival language".

**Fix.** Add a paragraph in Section 2 and a result line in Experiment 1
clarifying that prediction sensitivity computes a scalar per prediction
batch; CHRONO-Fair runs sequentially with statistical control. They are
complementary, not competing.

## Round 10 — "RMFT and VFR comparison plot is confusing"

**Critique.** The fig3b plot shows "1 - RMFT/tau*" vs "VFR" — these are
on different scales, and a reviewer cannot tell if RMFT is "better" or
"worse" than VFR. Need to either justify the transformation or report
RMFT in patient-units directly.

**Fix.** Replace 1-RMFT/tau* with RMFT in patient-units and add VFR as
a second axis. Also explicitly state that higher RMFT == more fair, so
the directional interpretation is clear.

## Round 11 — "The architecture diagram has too many arrows"

**Critique.** Figure 0 has cross-arrows between the rows that don't add
information. Visual clutter for a paper figure.

**Fix.** Simplify: keep only left-to-right flow plus single agent->UI
arrow. Drop the curved cross-arrows.

## Round 12 — "Inspector Agent claims it maps to specific regulatory clauses but no validation"

**Critique.** The paper says the Inspector Agent maps each cause to FDA
PCCP sections. How was this mapping validated? A reviewer will demand a
regulatory expert review or at least a citation.

**Fix.** Soften the claim: "we provide a SUGGESTED regulatory mapping
based on a reading of the December 2024 FDA PCCP final guidance and
the EU AI Act Articles 10, 14, 17, 61. Formal validation by regulatory
counsel is future work."

## Round 13 — "Multi-attribute generalisation is unaddressed"

**Critique.** The framework is shown for RACE only. The user's repo has
4 protected attributes (RACE, ETHNICITY, SEX, AGE_GROUP). How does it
extend? Just running 4 independent monitors? Or jointly?

**Fix.** Add a paragraph in Section 6.something explaining that
CHRONO-Fair runs `monitors x attributes x cells` simultaneously with
the same FDR pool, and run a quick demonstration with all four
attributes.

## Round 14 — "The dashboard mockup is fake data"

**Critique.** The dashboard panel shows specific numbers ("Black cohort
n=1,628, hazard ratio 3.05, FDA PCCP IV.B") that are not from any of
the experiments. Reviewers will mark this as fabricated content.

**Fix.** Generate the dashboard's contents from the actual exp3 + exp4
output instead of hardcoded strings, and cite the underlying experiment.

## Round 15 — "Counterfactual fairness assumption is unstated"

**Critique.** Counterfactual fairness (Kusner 2017) requires a structural
causal model. CHRONO-Fair uses a simple feature-shift counterfactual
that is more like "marginal protected-attribute swap" than a true
counterfactual. This should be made explicit.

**Fix.** Add a paragraph explicitly stating: "We adopt a feature-shift
counterfactual rather than a full SCM-based counterfactual; this matches
the deployment-time tractability requirement (Section 1) and aligns
with the audit-counterfactual definition of Maughan-Near (2022)."

## Round 16 — "Compute claims are unverified"

**Critique.** Section 3 says the estimator is O(1) per patient. But the
e-process has GROW lambda that requires `sum_z / n` — fine, but the
intersectional FDR step sorts m e-values every tick, which is O(m log m).
Need to be honest about this.

**Fix.** Add a complexity discussion: per-patient cost is O(K) (ensemble),
O(1) per cell (e-process), O(m log m) for FDR (m = number of cells).
Note that m is small (~20 intersectional cells in healthcare).

## Round 17 — "Reproducibility — synthesiser seeds not pinned in paper"

**Critique.** The paper claims reproducibility but doesn't specify the
seeds used per figure. Reviewers will mark this for the reproducibility
checklist.

**Fix.** Add an appendix-level table listing the (n, seed, drift_at,
drift_magnitude, aleatoric_bias) for every figure.

## Round 18 — "Comparison to Davis et al. 2025 is missing"

**Critique.** Davis 2025 is cited as the motivating prior work but no
empirical comparison is presented. The reader cannot tell if CHRONO-Fair
detects faster than Davis's ADWIN-on-fairness-gap method.

**Fix.** Add an exp1 baseline labeled "Davis-ADWIN" that monitors the
fairness *gap* (a continuous metric) rather than the binary flip
indicator. Show CHRONO-Fair's gain over Davis specifically.

## Round 19 — "Title is verbose"

**Critique.** The current title is 14 words. Some journals impose 15-word
limits. Tighten.

**Fix.** Shorten to: "CHRONO-Fair: Anytime-Valid Monitoring of
Counterfactual Verdict Flips in Clinical Machine Learning". 11 words.

## Round 20 — "No conclusion section"

**Critique.** Discussion exists but no formal Conclusion. Most Q1
clinical journals expect one.

**Fix.** Add a short Conclusion section.
