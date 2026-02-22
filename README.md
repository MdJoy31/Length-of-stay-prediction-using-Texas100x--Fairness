# Length-of-Stay Prediction & Fairness Metrics Reliability Analysis
## Texas-100X Hospital Discharge Dataset

**Author:** Md Rakibul Islam Joy
**Supervisors:** Dr. Caslon Chua, Dr. Viet Vo
**Institution:** Swinburne University of Technology
**Program:** Masters by Research

---

## Research Question

**"How reliable are fairness metrics when evaluating healthcare prediction models across different data volumes, model architectures, and fairness interventions?"**

## Key Results

| Model | Accuracy | F1-Score | AUC-ROC |
|-------|----------|----------|---------|
| Logistic Regression | 0.802 | 0.784 | 0.884 |
| Random Forest | 0.862 | 0.848 | 0.941 |
| Gradient Boosting | 0.874 | 0.858 | 0.950 |
| XGBoost (GPU) | 0.878 | 0.863 | 0.953 |
| **LightGBM (GPU)** | **0.879** | **0.864** | **0.954** |
| PyTorch DNN (GPU) | 0.857 | 0.843 | 0.938 |
| **Stacking Ensemble** | **0.879** | **0.865** | **0.954** |

**vs Reference Paper (Tarek et al. 2025):** F1 improved from 0.550 → 0.865 (+57%)

### Fairness Results (RACE attribute)
| Model | DI | WTPR | SPD | EOD |
|-------|----|------|-----|-----|
| Standard (LightGBM) | 0.64 | 0.84 | 0.08 | 0.07 |
| Fair (λ=5.0 reweighing) | 0.68 | 0.81 | 0.07 | 0.05 |

---

## Project Structure

```
├── LOS_Prediction_Standard.ipynb     # Clean notebook (standard comments)
├── LOS_Prediction_Detailed.ipynb     # Detailed notebook (extensive explanations)
├── Fairness_Analysis_Complete.ipynb   # Original analysis notebook
├── README.md
│
├── data/
│   ├── texas_100x.csv                # Main dataset (925,128 records)
│   ├── texas_100x_features.p
│   ├── texas_100x_labels.p
│   └── texas_100x_feature_desc.p
│
├── presentation/
│   ├── weekly_review.md              # Supervisor presentation content
│   └── weekly_review.pptx            # PowerPoint (7 slides)
│
├── scripts/                          # Pipeline scripts
│   ├── run_all.py
│   ├── script1_data_preprocessing.py
│   ├── script2_model_training.py
│   ├── script3_fairness_metrics.py
│   ├── script4a_bootstrap_stability.py
│   ├── script4b_sample_size.py
│   ├── script4c_cross_hospital.py
│   ├── script4d_seed_sensitivity.py
│   ├── script4e_threshold_sweep.py
│   └── script5_final_report.py
│
├── figures/                          # Generated plots
├── results/                          # Saved model results
└── report/                           # Generated reports
```

---

## Dataset

**Texas-100X Hospital Discharge Data**
- 925,128 patient records from Texas hospitals
- 12 features including demographics, diagnosis, procedure codes
- Binary target: Length of Stay > 3 days (Extended Stay)
- ~55% Normal / ~45% Extended (balanced)

## Models

- **6 ML models:** LR, RF, GB, XGBoost (GPU), LightGBM (GPU), PyTorch DNN (GPU)
- **Stacking Ensemble:** 5-fold OOF with LGB+XGB+HistGB base → meta-learner
- **Hyperparameters:** Optimized via 50-agent parallel search

## Feature Engineering

- **Target encoding** with Bayesian smoothing for diagnosis, procedure, hospital
- **Hospital-level features** (+3.3% F1 improvement)
- **Interaction features** (age×charges, diagnosis×procedure, etc.)
- **35+ engineered features** from 12 original columns

## Fairness Analysis

- **6 metrics:** Disparate Impact, WTPR, SPD, EOD, PPV Ratio, Equalized Odds
- **4 protected attributes:** Race, Ethnicity, Sex, Age Group
- **9-size subset stability:** 1K, 2K, 5K, 10K, 25K, 50K, 100K, 200K, Full
- **Lambda-reweighing:** Improves DI by ~6% with only ~1% F1 cost

## How to Run

### Notebooks (Recommended)
Open either notebook in VS Code or Jupyter:
- `LOS_Prediction_Standard.ipynb` — Clean, professional analysis
- `LOS_Prediction_Detailed.ipynb` — Detailed with extensive explanations

### Requirements
```bash
pip install numpy pandas scikit-learn xgboost lightgbm torch matplotlib seaborn tqdm
```

### Hardware
- GPU recommended (NVIDIA RTX series)
- ~8GB VRAM for GPU-accelerated training
- ~16GB RAM for dataset processing

---

## Reference

Tarek et al. (2025) — "Reliability of Fairness Metrics under Synthetic Augmentation" (CHASE '25)

## Contact

Md Rakibul Islam Joy
Swinburne University of Technology
