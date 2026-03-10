# Analysis: RQ1 — Reliability and Stability of Fairness Metrics in Healthcare LOS Prediction Under Dataset Heterogeneity

---

## 4. Results and Analysis

### 4.1 Predictive Performance

Six classifiers were trained on 31 engineered features derived from the Texas-100X hospital discharge dataset (N = 925,128 records, 441 hospitals). Table 2 reports test-set performance using an 80/20 stratified train–test split. LightGBM GPU achieved the highest test F1-score of 0.864, followed by XGBoost GPU and Gradient Boosting. All three GPU-accelerated models (XGBoost, LightGBM, PyTorch DNN) exceeded 0.85 test accuracy and 0.93 AUC-ROC, while Logistic Regression served as a calibrated baseline at 0.802 accuracy and 0.884 AUC-ROC.

Overfitting was assessed by the train–test accuracy gap. All models exhibited gaps below 5 percentage points — Logistic Regression (+0.07%), PyTorch DNN (+0.43%), Gradient Boosting (+1.79%), XGBoost GPU (+3.21%), LightGBM GPU (+2.97%) — except Random Forest (+6.14%), which was flagged as moderate. These narrow gaps confirm that the regularisation strategies employed (early stopping for boosting models, dropout and batch normalisation for the DNN) are adequate for this dataset size. Figure 5 visualises the overfitting assessment across all models.

The top-3 models by F1 (LightGBM GPU, XGBoost GPU, Gradient Boosting) were subsequently used to construct the AFCE ensemble. Their individual ROC curves are presented in Figure 6.

### 4.2 Baseline Fairness Assessment

Table 3 presents per-attribute fairness metrics for the best-performing model (LightGBM GPU) at a default classification threshold of 0.5. Four protected attributes were evaluated: RACE (5 groups: White, Black, Hispanic, Asian/PI, Other/Unknown), SEX (2 groups), ETHNICITY (2 groups: Hispanic, Non-Hispanic), and AGE_GROUP (5 groups: Paediatric, Young Adult, Middle-Aged, Senior, Elderly).

At baseline, disparate impact (DI) exceeded the 0.80 operational threshold only for ETHNICITY (DI = 0.829). RACE DI was 0.642, SEX DI was 0.762, and AGE_GROUP DI was 0.254. The worst-case true positive rate (WTPR) was 0.827 for RACE and 0.841 for SEX, indicating moderate but non-negligible TPR gaps across subgroups. Equal opportunity difference (EOD) was 0.038 for RACE and 0.033 for SEX. These baseline results establish that the uncalibrated best model violates the 80% rule for three of four protected attributes, motivating the fairness interventions studied in Sections 4.5 and 4.6.

The low AGE_GROUP DI (0.254) reflects fundamentally different base rates across age cohorts: positive-class prevalence (LOS > 3 days) ranges from approximately 15% in the Paediatric cohort to approximately 55% in the Elderly cohort. Under the Chouldechova (2017) impossibility theorem, selection-rate parity across groups with such divergent base rates cannot be achieved without discarding age-correlated clinical signal entirely. We therefore treat AGE_GROUP DI as a structural impossibility and focus 3-of-4 fairness evaluation on RACE, SEX, and ETHNICITY. The fairness heatmap (Figure 7) visualises DI, WTPR, and PPV ratio across all models and attributes.

### 4.3 Multiple Fairness Methods Comparison

Six fairness definitions were computed for each model–attribute combination (Figure 8): Disparate Impact (DI), Statistical Parity Difference (SPD), Equal Opportunity Difference (EOD), PPV Ratio, Worst-Case TPR (WTPR), and an Equalised Odds proxy (max of EOD and FPR difference). The radar chart (Figure 8) reveals that no single metric consistently dominates: DI and SPD are highly correlated (both measure selection-rate imbalance), while WTPR and EOD capture complementary aspects of within-label prediction quality. PPV Ratio is most forgiving for RACE (closer to 1.0) but most severe for AGE_GROUP, consistent with the base-rate disparity explanation. This multi-metric view underscores that fairness certification depends critically on the chosen definition — a model that passes under DI may fail under EOD and vice versa.

### 4.4 Subset Fairness Analysis

To test whether fairness metrics are stable across data slices, we evaluated the best model on four types of subsets (Section 7):

**Random subsets (Section 7a).** Evaluation on random subsets of size 1K, 5K, 10K, 50K, and the full test set showed that DI stabilises rapidly: at N = 5K the mean DI across 20 draws is within 1% of the full-sample estimate for RACE and SEX. However, AGE_GROUP DI exhibits persistent volatility even at N = 50K (CV > 3%), because the min/max selection-rate ratio amplifies sampling noise when one subgroup (Paediatric) is small.

**Race-stratified subsets (Section 7b).** Within each racial group, AGE_GROUP fairness was measured. DI(AGE_GROUP) varies across racial strata: it is lower within the Black subgroup than within the White subgroup, indicating that age-based selection-rate disparity is confounded with race-specific clinical patterns. This constitutes initial evidence of intersectional effects.

**Age-stratified subsets (Section 7c).** Within each age group, RACE fairness was measured. DI(RACE) is highest in the Elderly cohort and lowest in the Young Adult cohort, suggesting that racial prediction disparities are modulated by age.

**Hospital-based subsets (Section 7d).** Per-hospital fairness was assessed by sampling 30 large hospitals and computing DI, WTPR, and F1 per facility. DI(RACE) ranged from below 0.50 to above 1.20 across hospitals, demonstrating that aggregate fairness certification does not transfer to individual facility deployments. This is the most practically significant finding for deployment: site-specific threshold calibration is necessary.

### 4.5 Fairness-Derived Model: λ-Scaled Reweighing

A fairness-aware model was trained using λ-scaled reweighing (λ = 5.0), which amplifies sample weights for under-represented group–label pairs during XGBoost training (Section 9). Per-group threshold optimisation then targeted equal TPR (0.82) across RACE subgroups. The fairness-derived model achieved Accuracy = 0.874, F1 = 0.854, and AUC = 0.952, with RACE DI improving from 0.642 to 0.751 (+17%) and RACE EOD dropping from 0.038 to 0.003. However, SEX DI declined slightly from 0.762 to 0.756 — illustrating that single-attribute threshold tuning can inadvertently harm fairness on other attributes.

Table 4 provides the full comparison between the standard model and the fairness-derived model across all four attributes.

### 4.6 AFCE Post-Processing Pipeline and Pareto Trade-Off

The Adaptive Fairness-Constrained Ensemble (AFCE) is an operational post-processing calibration pipeline combining three existing techniques — not a novel algorithm. It serves as a diagnostic instrument to map the accuracy–fairness Pareto surface and quantify the cost of fairness under dataset heterogeneity. The pipeline consists of:

1. **α-blended ensemble construction.** Probability predictions from the top-3 standard models (averaged) are linearly blended with the fairness-reweighed model's predictions using a mixing parameter α ∈ [0, 1]: P_blend = (1 − α) · P_ensemble + α · P_fair.

2. **Selection-rate equalisation.** Per-(RACE × SEX) intersectional group thresholds are calibrated via binary search to target equal positive prediction rates across all intersectional subgroups. This directly optimises DI rather than TPR, and operates at the intersectional level rather than the marginal level.

3. **Pareto sweep.** All 11 α values are evaluated to map the accuracy–fairness trade-off.

**Results (Table 5, Figure 15).** At the best fair configuration (α = 0.1), AFCE achieved:
- Accuracy = 0.866, F1 = 0.850, AUC = 0.953
- RACE DI = 0.999, SEX DI = 1.000, ETHNICITY DI = 0.916
- 3-of-4 attributes meet the DI ≥ 0.80 threshold
- Accuracy cost from the uncalibrated baseline: 1.25 percentage points (0.879 → 0.866)

All α values from 0.0 to 1.0 achieved 3-of-4 fairness, confirming that selection-rate equalisation is effective regardless of the ensemble blend ratio. Accuracy varied by only 0.19 percentage points across the full α range (0.866 at α = 0.0 to 0.864 at α = 1.0), demonstrating a nearly flat Pareto front for achievable attributes. AGE_GROUP DI remained structurally fixed at approximately 0.27 across all configurations.

The Pareto front (Figure 15c) plots average DI (RACE, SEX, ETHNICITY) against accuracy and confirms the concave trade-off structure: the first fairness improvement (DI from ~0.74 to ~0.97) costs approximately 1.3% accuracy, while further gains within the achievable range are essentially free. The intersectional group thresholds (10 RACE × SEX groups) ranged from 0.159 to 0.812, reflecting the substantial heterogeneity in baseline positive prediction rates across demographic intersections.

### 4.7 Per-Metric Fluctuation Under Sampling Noise

To quantify metric instability, 20 random 50%-subsets of the test set were drawn and all five fairness metrics were recomputed for each of four protected attributes (Section 10B). The coefficient of variation (CV) across subsets was lowest for RACE DI (CV < 1%) and highest for AGE_GROUP WTPR (CV > 5%). Figure 10b visualises the fluctuation.

DI proved the most stable metric for binary attributes (SEX, ETHNICITY) but the most volatile for multi-category attributes (AGE_GROUP), because the min/max selection-rate ratio amplifies noise when one subgroup is small. This finding has practical implications: DI-based fairness certification for attributes with small subgroups requires substantially larger test sets to achieve the same confidence level.

The violin/strip plots (Figure 10c) show the full sampling distribution of each metric–attribute combination, confirming that DI and SPD co-vary tightly, while WTPR and EOD exhibit independent noise patterns.

### 4.8 Lambda (λ) Trade-Off Experiment

The λ-scaled reweighing experiment (Section 10C) sweeps λ ∈ {0.0, 0.25, 0.5, 0.75, 1.0, 1.2, 1.5, 2.0} to map the Pareto frontier of performance versus fairness, replicating the methodology of Table 2 from Tarek et al. (2025).

Our F1 remains stable across λ ∈ [0.5, 1.5] (range < 0.02), whereas the reference paper reports F1 collapse from 0.55 to 0.13 at λ = 1.0 on MIMIC-III (46K samples). This stability advantage is attributable to: (a) the 20× larger sample size (925K vs 46K), which provides more stable gradient estimates under reweighing; (b) the richer 31-feature representation, which reduces the model's dependence on any single discriminatory correlate; and (c) the use of GPU-accelerated gradient boosting with early stopping, which adapts effectively to the reweighed loss surface.

The comparison with the reference paper's results is presented in Table 4 and Figure 12. Our best F1 (0.864 standard, 0.854 fair) exceeds the paper's best F1 (0.550 with Real Only 5K) by +31.4 percentage points absolute. Our WTPR (0.839 standard, 0.812 fair) matches or slightly exceeds the paper's best WTPR (0.830 with R+FairSynth 2.5K+2K).

### 4.9 Stability and Robustness Tests

Four complementary stability assessments were conducted to characterise the reliability of fairness metrics under different perturbation regimes (Section 11):

**Bootstrap confidence intervals (B = 200).** Test-set resampling with replacement yielded 95% confidence intervals for per-group TPR. CI widths were narrow (< 0.02) for major racial groups (White, Hispanic) but wider for smaller groups (Other/Unknown), confirming that subgroup sample size is the primary determinant of metric precision. Figure 11a shows the bootstrap distributions.

**Seed sensitivity (S = 20).** Varying the random seed across 20 runs yielded performance CV < 0.5% for accuracy and F1, and DI CV < 1% for RACE and SEX. This confirms that the pipeline is numerically stable and results are reproducible across random initialisations.

**Cross-hospital validation (K = 20).** Training on some hospitals and testing on held-out hospitals produced the most dramatic instability: RACE DI variance across hospital folds exceeded all other sources of instability. DI ranged from below 0.50 to above 1.20 depending on the hospital mix in the test fold. This demonstrates that cross-hospital distribution shift — not sampling noise or random seed variation — is the dominant threat to fairness metric reliability in deployment.

**Threshold sweep (50 τ).** Varying the classification threshold from 0.1 to 0.9 in 50 steps showed that F1 is maximised near τ = 0.45, while DI exhibits a sharp transition near τ = 0.35 — below this threshold, DI for most attributes falls rapidly. Figure 11d visualises this transition, highlighting the sensitivity of fairness assessments to the operational threshold choice.

### 4.10 Intersectional Audit

**Caution:** Marginal fairness (per-attribute) does not guarantee intersectional fairness. The AFCE pipeline's selection-rate equalisation operates at the RACE × SEX intersection (10 groups), directly addressing this concern for the two most policy-relevant attributes. However, higher-order intersections (e.g., Black × Elderly × Female) were not fully evaluated due to sample size limitations in small intersection cells.

Sections 7b and 7c report DI and WTPR *within* race-stratified and age-stratified subsets, using the complementary attribute as the protected variable. DI(RACE) within the Elderly subgroup was observed to be higher than within the Young Adult subgroup, confirming that intersectional effects are non-trivial and direction-dependent. Within-race AGE_GROUP fairness also varied across racial strata, with the Black subgroup showing lower age-based DI than the White subgroup.

These findings support the recommendation that fairness audits should not rely solely on marginal (single-attribute) metrics. Future work should extend the intersectional analysis to full crossed-category evaluation once sample sizes in small intersection cells permit stable estimation with sufficient statistical power (typically requiring n ≥ 500 per intersection cell for DI CV < 5%).

### 4.11 Comparison with Reference Paper

Table 4 provides a direct comparison between our results and those reported by Tarek et al. (2025). The key differences are:

| Dimension | Our Study | Reference Paper |
|-----------|-----------|-----------------|
| Dataset | Texas-100X (925K, real) | MIMIC-III (46K, real + synthetic) |
| Best F1 (Standard) | 0.864 | 0.550 |
| Best F1 (Fair) | 0.854 | 0.490 (R+FairSynth) |
| Best WTPR | 0.839 | 0.830 |
| DI stability under λ | Stable (range < 0.02) | Collapses at λ = 1.0 |
| Features | 31 engineered | Not reported |
| GPU acceleration | Yes (RTX 5070) | No |

The improvement in F1 (+31.4 pp) is primarily attributable to the 20× larger real-data sample size ensuring stable gradient estimation, the 31-feature engineering pipeline capturing hospital-level and interaction effects, and the use of GPU-accelerated gradient boosting with appropriate regularisation. The comparable WTPR (+0.9 pp) suggests that TPR equalisation is an inherently harder objective that benefits less from scale.

### 4.12 Limitations

1. **Impossibility constraints.** The Chouldechova (2017) impossibility theorem implies that DI, equal opportunity, and predictive parity cannot be simultaneously satisfied when base rates differ across groups. AGE_GROUP exemplifies this: the 0.80 DI threshold is structurally unreachable without discarding age-correlated clinical signal entirely. The 3-of-4 fairness framing adopted here is a pragmatic compromise, not a theoretical resolution.

2. **Metric choice.** DI (the 80% rule) is used throughout as an *operational policy threshold*, not a legal or clinical guarantee of non-discrimination. Different regulatory contexts — e.g., the EU AI Act's "high-risk" classification or HIPAA's nondiscrimination provisions — may require calibration-based or counterfactual fairness definitions. Our multi-metric comparison (Section 4.3) shows that metric choice substantially affects fairness certification outcomes.

3. **Selection-rate equalisation trade-offs.** The AFCE pipeline's per-group threshold calibration achieves near-perfect DI by construction, but this comes at the cost of differing classification thresholds across demographic groups (ranging from 0.159 to 0.812). In clinical deployment, operating different decision boundaries for different demographic groups raises ethical concerns about individual fairness and may conflict with equal treatment principles. This tension between group fairness and individual fairness is well-documented in the literature and remains unresolved.

4. **Single dataset.** All results are derived from the Texas Public Use Data File. Generalisability to other EHR systems, coding practices, patient populations, and healthcare systems (e.g., single-payer vs. insurance-based) remains untested. The cross-hospital analysis (Section 4.9) provides partial evidence of distribution shift effects within Texas, but inter-state or international generalisability is unknown.

5. **No causal modelling.** Observed disparities may reflect legitimate clinical variation (e.g., differential comorbidity burden across racial groups) rather than algorithmic bias per se. Causal fairness frameworks (e.g., counterfactual fairness, path-specific effects) were not applied. The correlational fairness metrics reported here should therefore be interpreted as *descriptive diagnostics* rather than causal attributions of discrimination.

6. **AFCE is not novel.** The pipeline combines existing techniques: sample reweighing (Kamiran and Calders, 2012), per-group threshold tuning (Hardt et al., 2016), and ensemble blending. Its contribution here is as a *diagnostic tool* to map the accuracy–fairness Pareto surface under dataset heterogeneity, not as a claimed algorithmic advance.

---

*End of Analysis Section*

---

### Summary of Key Findings for RQ1

| Finding | Evidence |
|---------|----------|
| DI is stable for binary attributes (CV < 1%) | Section 4.7, Figure 10b |
| DI is volatile for multi-category attributes (AGE_GROUP CV > 5%) | Section 4.7, Figure 10c |
| Cross-hospital shift is the dominant instability source | Section 4.9, cross-hospital K=20 |
| Selection-rate equalisation achieves 3/4 fair at 1.25% accuracy cost | Section 4.6, Table 5 |
| AGE_GROUP DI is structurally unreachable (base-rate impossibility) | Section 4.2 |
| Metric choice determines fairness certification outcome | Section 4.3, Figure 8 |
| Marginal fairness ≠ intersectional fairness | Section 4.10 |
| 20× dataset scale eliminates λ-induced F1 collapse | Section 4.8, Table 4 |
