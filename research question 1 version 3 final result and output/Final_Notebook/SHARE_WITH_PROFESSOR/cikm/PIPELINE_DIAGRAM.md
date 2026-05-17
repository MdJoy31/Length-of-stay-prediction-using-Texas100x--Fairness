# Pipeline Diagram — CIKM 2026 LOS Fairness

End-to-end pipeline of the FINAL notebook
(`CIKM_2026_LOS_Fairness_FINAL.ipynb`).
Renders in any Markdown viewer that supports Mermaid (GitHub, Overleaf
via `mermaid` package, VS Code with mermaid extension, JupyterLab,
Notion, Obsidian).

---

## 1. High-level pipeline (poster-style)

```mermaid
flowchart TB
    %% =========== DATA LAYER ===========
    A0[/"Texas THCIC PUDF<br/>2019–2023<br/>raw discharge files"/]
        --> A1{{"Cleaning<br/>• drop missing demo / clinical<br/>• drop LOS ≤ 0 or > 365<br/>• 925,128 records, 441 hospitals"}}

    %% =========== FEATURE LAYER ===========
    A1 --> B1[/"Protected attributes<br/>RACE • SEX • ETHNICITY • AGE_GROUP<br/>(4-bucket: Pediatric / Young / Middle / Elderly)"/]
    A1 --> B2[/"Low-card features<br/>PAT_AGE • TOTAL_CHARGES<br/>PAT_STATUS • TYPE_OF_ADMISSION<br/>SOURCE_OF_ADMISSION"/]
    A1 --> B3[/"High-card features<br/>ADMITTING_DIAGNOSIS<br/>PRINC_SURG_PROC_CODE<br/>THCIC_ID"/]
    B3 --> B3a["Bayesian-smoothed<br/>target encoding<br/>m = 10 (train-only)"]

    %% =========== SPLIT LAYER ===========
    B1 --> C0
    B2 --> C0
    B3a --> C0
    C0{{"80/20 stratified split<br/>seed = 42<br/>train n = 740,102 · test n = 185,026"}}
    C0 --> D_TRAIN[/"X_train, y_train<br/>StandardScaler.fit_transform"/]
    C0 --> D_TEST[/"X_test, y_test<br/>StandardScaler.transform"/]

    %% =========== MODELLING LAYER ===========
    D_TRAIN --> E1["12 ML models trained<br/>LR · DT · RF · GB · AdaBoost<br/>XGBoost (canonical) · LightGBM<br/>CatBoost · HistGB · Bagging<br/>Extra Trees · Stacking Ensemble"]
    E1 --> E_PROBS[/"Probabilities &amp; predictions<br/>per model on test"/]

    %% =========== FAIRNESS EVALUATION ===========
    D_TEST --> F0
    E_PROBS --> F0
    F0[["FairnessCalculator<br/>7 metrics × 4 attributes = 28 cells/model<br/>DI · SPD · EOPP · EOD · TI · PP · CAL"]]

    %% =========== STABILITY TESTS ===========
    F0 --> G1["Test 1<br/>Bootstrap B = 500<br/>(stratified, n = 10,000)"]
    F0 --> G2["Test 2<br/>Sample-size sweep<br/>n ∈ {1k … 185k}"]
    F0 --> G3["Test 3<br/>Cross-hospital GroupKFold<br/>K = 20 (by THCIC_ID)"]
    F0 --> G4["Test 4<br/>Seed-equivalent perturbation<br/>500 boot × 20 fold = 520 draws"]
    F0 --> G5["Test 5<br/>Threshold sweep<br/>τ ∈ [0.1, 0.9] step 0.05"]

    %% =========== RELIABILITY ===========
    G1 --> H_CV{{"Coefficient of Variation<br/>(per metric × attribute)"}}
    G3 --> H_KAPPA{{"Fleiss κ inter-hospital<br/>(per-metric × 4 attrs × 20 folds)"}}
    G1 --> H_VFR{{"Verdict Flip Rate<br/>per 28-cell × bootstrap"}}

    %% =========== INTERVENTION ===========
    F0 --> I1["Phase 1 · λ-scaled intersectional reweighing<br/>RACE × AGE × SEX cells, λ = 2"]
    I1 --> I2["Phase 2 · Per-age isotonic calibration"]
    I2 --> I3["Phase 3 · Per-cell α-SR/TPR/PPV<br/>grid search (master prompt)"]
    I3 --> I4["Phase 4 · Pareto selection<br/>config-4 fallback if calibration breaks all-4-DI"]

    %% =========== PER-CLUSTER VALIDATION ===========
    I4 --> J0["Per-cluster transferability<br/>20 hospital clusters<br/>local intersection cells"]

    %% =========== OUTPUTS ===========
    H_CV  --> K1[("output_final/tables/<br/>T3 … T20 + cikm_vfr_all_metrics")]
    H_KAPPA --> K1
    H_VFR --> K1
    I4    --> K1
    J0    --> K1
    F0    --> K2[("output_final/figures/<br/>F1–F5  300 dpi")]
    K1    --> L1["Manuscript-claim verification<br/>(T19, T20, REWRITE_SUMMARY.md)"]

    %% =========== STYLES ===========
    classDef data fill:#dbeafe,stroke:#1e3a8a,stroke-width:1.4px,color:#0f172a;
    classDef proc fill:#fef3c7,stroke:#78350f,stroke-width:1.4px,color:#0f172a;
    classDef test fill:#dcfce7,stroke:#14532d,stroke-width:1.4px,color:#0f172a;
    classDef inter fill:#fce7f3,stroke:#831843,stroke-width:1.4px,color:#0f172a;
    classDef out fill:#e0e7ff,stroke:#3730a3,stroke-width:1.4px,color:#0f172a;

    class A0,B1,B2,B3,D_TRAIN,D_TEST,E_PROBS data;
    class A1,B3a,C0,E1,F0 proc;
    class G1,G2,G3,G4,G5,H_CV,H_KAPPA,H_VFR test;
    class I1,I2,I3,I4,J0 inter;
    class K1,K2,L1 out;
```

---

## 2. Detailed: feature pipeline only

```mermaid
flowchart LR
    R[(Raw record)] --> F1{Protected?}
    F1 -- "RACE / SEX / ETHNICITY / AGE_GROUP" --> X[/"Excluded from X<br/>used only for fairness audit"/]
    F1 -- "all others" --> F2{High-cardinality?}
    F2 -- "ADMITTING_DIAGNOSIS<br/>PRINC_SURG_PROC_CODE<br/>THCIC_ID" --> TE["Bayesian-smoothed<br/>target encoding<br/>μ̂_k = (n_k·ȳ_k + m·ȳ) / (n_k + m)<br/>m = 10, train-fold only"]
    F2 -- "PAT_AGE • TOTAL_CHARGES<br/>PAT_STATUS • TYPE_OF_ADMISSION<br/>SOURCE_OF_ADMISSION" --> NUM["Numeric / low-card kept as-is"]
    TE --> S[StandardScaler]
    NUM --> S
    S --> XOUT[/"X_train_sc, X_test_sc<br/>11 features"/]
```

---

## 3. Detailed: intervention pipeline only

```mermaid
flowchart TB
    P0[/"XGBoost test probabilities<br/>p_i for every test record"/]
    P0 --> P1["Phase 1<br/>λ-Reweighing<br/>RACE × AGE × SEX cells<br/>λ = 2"]
    P1 -. "sample weights" .-> P2["Phase 2<br/>Per-age isotonic calibration<br/>4 calibrators"]
    P2 --> P3{"Phase 3<br/>Per-cell α-SR/TPR/PPV<br/>grid search"}
    P3 -- "α_SR ∈ {0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0}" --> P3a
    P3 -- "α_TPR ∈ {0.0, 0.3, 0.5, 0.7, 0.9, 1.0}" --> P3a
    P3 -- "α_PPV ∈ {0.0, 0.3, 0.6, 0.9}" --> P3a
    P3a["For every (RACE, AGE, SEX) cell<br/>find threshold minimising<br/>α_SR · |SR_gap|² + α_TPR · |TPR_gap|² + α_PPV · |PPV_gap|²<br/>subject to drop_limit = 0.08"]
    P3a --> P4{"Phase 4<br/>Pareto selection"}
    P4 -- "calibrated probs all-4-DI ≥ 0.80?" --> P4_OK[/"Config 4 (calibrated)"/]
    P4 -- "no" --> P4_FB[/"Config 3 (uncalibrated fallback)"/]
    P4_OK --> OUT[(Final fair predictions)]
    P4_FB --> OUT
```

---

## 4. Stability-test layer — data flow

```mermaid
flowchart LR
    M[(Trained XGBoost<br/>+ test set)] --> T1["Test 1<br/>500 stratified bootstrap<br/>resamples of size 10,000"]
    M --> T2["Test 2<br/>Sample-size grid<br/>{1k, 5k, 10k, 50k, 100k, 185k}"]
    M --> T3["Test 3<br/>20 GroupKFold splits<br/>(group = THCIC_ID)"]
    M --> T5["Test 5<br/>17 thresholds × 4 attrs"]

    T1 --> R1[("Per-cell verdict distribution<br/>VFR + 95% CI")]
    T2 --> R2[("CV vs n curves<br/>min-sample thresholds T9")]
    T3 --> R3[("Per-fold fairness<br/>+ Fleiss κ T11")]
    T1 -. combine .- T3
    T1 --> R4[("520 seed-equivalent draws<br/>= Test 4 surrogate")]
    T3 --> R4
    T5 --> R5[("Threshold-fairness curves")]
```

---

## How to render in your manuscript / IDE

* **VS Code**: install the *Markdown Preview Mermaid Support* extension, open this file, press `Ctrl+Shift+V`.
* **GitHub README**: paste the ` ```mermaid … ``` ` block — renders natively.
* **Overleaf / LaTeX**: convert to PNG/SVG with [mermaid-cli](https://github.com/mermaid-js/mermaid-cli):
  ```bash
  npx -p @mermaid-js/mermaid-cli mmdc -i PIPELINE_DIAGRAM.md -o PIPELINE_DIAGRAM.png -t default -b transparent
  ```
* **JupyterLab**: install `jupyterlab-mermaid` and render Markdown cells with `mermaid` fences.
