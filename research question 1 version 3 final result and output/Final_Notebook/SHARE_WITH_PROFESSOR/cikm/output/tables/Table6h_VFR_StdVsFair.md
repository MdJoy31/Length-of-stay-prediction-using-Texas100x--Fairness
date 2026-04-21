# Table 6h — VFR Before/After (Standard → Fair, 20 Hospital Clusters)

The **Verdict Flip Rate (VFR)** is the fraction of 20 hospital clusters where the per-cluster fair/unfair verdict disagrees with the cluster-mean verdict. Low VFR means the headline pass/fail result is portable across sites; high VFR means the verdict is a coin-flip in a new hospital.


Reliability classes: **High** < 10% VFR; **Moderate** 10–30%; **Unstable** > 30%. Thresholds match main.tex Table 10.


| Attribute | Metric | Std_Mean | Fair_Mean | Delta_Mean | Std_Verdict_at_mean | Fair_Verdict_at_mean | Std_Pass_k_over_N | Fair_Pass_k_over_N | Std_VFR_pct | Fair_VFR_pct | Delta_VFR | Std_Reliability | Fair_Reliability | Clusters_P2P | Clusters_P2F | Clusters_F2P | Clusters_F2F |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RACE | DI | 0.648 | 0.838 | 0.190 | Fail | Pass | 2/20 | 15/20 | 10.0 | 25.0 | 15.0 | Moderate | Moderate | 2 | 0 | 13 | 5 |
| RACE | SPD | 0.176 | 0.075 | -0.100 | Fail | Pass | 3/20 | 16/20 | 15.0 | 20.0 | 5.0 | Moderate | Moderate | 3 | 0 | 13 | 4 |
| RACE | EOPP | 0.180 | 0.138 | -0.042 | Fail | Fail | 2/20 | 7/20 | 10.0 | 35.0 | 25.0 | Moderate | Unstable | 0 | 2 | 7 | 11 |
| RACE | EOD | 0.187 | 0.176 | -0.011 | Fail | Fail | 1/20 | 2/20 | 5.0 | 10.0 | 5.0 | High | Moderate | 0 | 1 | 2 | 17 |
| RACE | TI | 0.482 | 0.467 | -0.015 | Fail | Fail | 0/20 | 0/20 | 0.0 | 0.0 | 0.0 | High | High | 0 | 0 | 0 | 20 |
| RACE | PP | 0.157 | 0.248 | 0.090 | Fail | Fail | 10/20 | 1/20 | 50.0 | 5.0 | -45.0 | Unstable | High | 1 | 9 | 0 | 10 |
| RACE | CAL | 0.217 | 0.256 | 0.038 | Fail | Fail | 0/20 | 0/20 | 0.0 | 0.0 | 0.0 | High | High | 0 | 0 | 0 | 20 |
| SEX | DI | 0.756 | 0.953 | 0.197 | Fail | Pass | 7/20 | 20/20 | 35.0 | 0.0 | -35.0 | Unstable | High | 7 | 0 | 13 | 0 |
| SEX | SPD | 0.124 | 0.020 | -0.104 | Fail | Pass | 6/20 | 20/20 | 30.0 | 0.0 | -30.0 | Unstable | High | 6 | 0 | 14 | 0 |
| SEX | EOPP | 0.049 | 0.018 | -0.031 | Pass | Pass | 18/20 | 20/20 | 10.0 | 0.0 | -10.0 | Moderate | High | 18 | 0 | 2 | 0 |
| SEX | EOD | 0.066 | 0.081 | 0.015 | Pass | Pass | 18/20 | 15/20 | 10.0 | 25.0 | 15.0 | Moderate | Moderate | 13 | 5 | 2 | 0 |
| SEX | TI | 0.502 | 0.486 | -0.016 | Fail | Fail | 0/20 | 0/20 | 0.0 | 0.0 | 0.0 | High | High | 0 | 0 | 0 | 20 |
| SEX | PP | 0.007 | 0.150 | 0.143 | Pass | Fail | 20/20 | 4/20 | 0.0 | 20.0 | 20.0 | High | Moderate | 4 | 16 | 0 | 0 |
| SEX | CAL | 0.090 | 0.108 | 0.018 | Fail | Fail | 2/20 | 2/20 | 10.0 | 10.0 | 0.0 | Moderate | Moderate | 0 | 2 | 2 | 16 |
| ETHNICITY | DI | 0.824 | 0.918 | 0.093 | Pass | Pass | 12/20 | 19/20 | 40.0 | 5.0 | -35.0 | Unstable | High | 11 | 1 | 8 | 0 |
| ETHNICITY | SPD | 0.081 | 0.038 | -0.043 | Pass | Pass | 13/20 | 19/20 | 35.0 | 5.0 | -30.0 | Unstable | High | 12 | 1 | 7 | 0 |
| ETHNICITY | EOPP | 0.040 | 0.040 | -0.001 | Pass | Pass | 18/20 | 19/20 | 10.0 | 5.0 | -5.0 | Moderate | High | 17 | 1 | 2 | 0 |
| ETHNICITY | EOD | 0.052 | 0.062 | 0.009 | Pass | Pass | 18/20 | 17/20 | 10.0 | 15.0 | 5.0 | Moderate | Moderate | 15 | 3 | 2 | 0 |
| ETHNICITY | TI | 0.486 | 0.484 | -0.002 | Fail | Fail | 0/20 | 0/20 | 0.0 | 0.0 | 0.0 | High | High | 0 | 0 | 0 | 20 |
| ETHNICITY | PP | 0.027 | 0.096 | 0.069 | Pass | Pass | 20/20 | 13/20 | 0.0 | 35.0 | 35.0 | High | Unstable | 13 | 7 | 0 | 0 |
| ETHNICITY | CAL | 0.111 | 0.145 | 0.034 | Fail | Fail | 1/20 | 1/20 | 5.0 | 5.0 | 0.0 | High | High | 0 | 1 | 1 | 18 |
| AGE_GROUP | DI | 0.254 | 0.841 | 0.587 | Fail | Pass | 0/20 | 16/20 | 0.0 | 20.0 | 20.0 | High | Moderate | 0 | 0 | 16 | 4 |
| AGE_GROUP | SPD | 0.461 | 0.072 | -0.388 | Fail | Pass | 0/20 | 16/20 | 0.0 | 20.0 | 20.0 | High | Moderate | 0 | 0 | 16 | 4 |
| AGE_GROUP | EOPP | 0.307 | 0.184 | -0.123 | Fail | Fail | 0/20 | 1/20 | 0.0 | 5.0 | 5.0 | High | High | 0 | 0 | 1 | 19 |
| AGE_GROUP | EOD | 0.307 | 0.210 | -0.098 | Fail | Fail | 0/20 | 0/20 | 0.0 | 0.0 | 0.0 | High | High | 0 | 0 | 0 | 20 |
| AGE_GROUP | TI | 0.479 | 0.481 | 0.001 | Fail | Fail | 0/20 | 0/20 | 0.0 | 0.0 | 0.0 | High | High | 0 | 0 | 0 | 20 |
| AGE_GROUP | PP | 0.097 | 0.473 | 0.377 | Pass | Fail | 13/20 | 0/20 | 35.0 | 0.0 | -35.0 | Unstable | High | 0 | 13 | 0 | 7 |
| AGE_GROUP | CAL | 0.220 | 0.315 | 0.095 | Fail | Fail | 0/20 | 0/20 | 0.0 | 0.0 | 0.0 | High | High | 0 | 0 | 0 | 20 |

## Attribute Summary — Reliability Uplift


| Attribute | Std N_fair/7 | Fair N_fair/7 | Std avg VFR | Fair avg VFR | Δ avg VFR |
| --- | --- | --- | --- | --- | --- |
| RACE | 0/7 | 2/7 | 12.9% | 13.6% | +0.7 pp |
| SEX | 3/7 | 4/7 | 13.6% | 7.9% | -5.7 pp |
| ETHNICITY | 5/7 | 5/7 | 14.3% | 10.0% | -4.3 pp |
| AGE_GROUP | 1/7 | 2/7 | 5.0% | 6.4% | +1.4 pp |

### Reading
Cells with **Fair_Verdict_at_mean = Pass AND Std_Verdict_at_mean = Fail** are the new fair verdicts earned by the intervention. Cells with **Fair_VFR < Std_VFR** indicate that the Fair pipeline does not just raise the mean — it also stabilises the verdict across hospitals, converting unstable verdicts into portable ones. This combination is how the paper should motivate the ‘reliability-through-intervention’ claim.
