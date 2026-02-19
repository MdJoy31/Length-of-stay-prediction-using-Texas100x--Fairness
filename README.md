# 🏥 Texas-100X Fairness Metrics Reliability Analysis

## Q1 Journal Publication Project

**Author:** Md Jannatul Rakib Joy  
**Supervisors:** Dr. Caslon Chua, Dr. Viet Vo  
**Institution:** Swinburne University of Technology

---

## 📋 Research Question

**"How reliable are fairness metrics when evaluating healthcare prediction models?"**

This project tests whether standard fairness metrics produce consistent, trustworthy results across different experimental conditions in healthcare AI.

---

## 📁 Your Data Files

You have the **FULL Texas-100X dataset** (925,128 records):
```
./data/
├── texas_100x.csv           (36 MB)  ← Main dataset
├── texas_100x_features.p    (79 MB)  ← Preprocessed features
├── texas_100x_labels.p      (7 MB)   ← Labels
└── texas_100x_feature_desc.p (51 KB) ← Feature descriptions
```

---

## 🔬 Complete Testing Framework

### 5 Fairness Metrics
| # | Metric | What It Tests |
|---|--------|---------------|
| 1 | Demographic Parity | Equal selection rates |
| 2 | Equalized Odds | Equal TPR + FPR |
| 3 | Equal Opportunity | Equal TPR (sensitivity) |
| 4 | Predictive Parity | Equal precision |
| 5 | Calibration (ECE) | Probability accuracy |

### 5 Stability Tests
| # | Test | Parameters | What It Measures |
|---|------|------------|------------------|
| 1 | Bootstrap | B=1,000 | Sampling uncertainty |
| 2 | Sample Size | N=10K→Full | Data requirement |
| 3 | Cross-Hospital | K=50 folds | Site heterogeneity |
| 4 | Seed Sensitivity | S=50 | Split variance |
| 5 | Threshold Sweep | τ=99 | Threshold sensitivity |

### 4 Protected Attributes (13 Subgroups)
- **RACE:** White, Black, Hispanic, Asian, Other
- **ETHNICITY:** Hispanic, Non-Hispanic
- **SEX:** Male, Female
- **AGE_GROUP:** Pediatric, Adult, Middle-aged, Elderly

### 4 ML Models
- Logistic Regression
- Random Forest
- Gradient Boosting
- Neural Network

---

## 🚀 How to Run

### Option 1: Run Everything
```bash
cd scripts
python run_all.py
```

### Option 2: Run Step by Step
```bash
python script1_data_preprocessing.py     # ~2 min
python script2_model_training.py         # ~10 min
python script3_fairness_metrics.py       # ~5 min
python script4a_bootstrap_stability.py   # ~2 hours
python script4b_sample_size.py           # ~1 hour
python script4c_cross_hospital.py        # ~2 hours
python script4d_seed_sensitivity.py      # ~2 hours
python script4e_threshold_sweep.py       # ~30 min
python script5_final_report.py           # ~5 min
```

**Total Runtime: ~8-10 hours** (run overnight)

---

## 📊 Output Files

After completion:
```
./figures/
├── fairness_metrics_*.png (4 per attribute)
├── fairness_disparity_heatmap.png
├── bootstrap_ci_forest_plot.png
├── sample_size_convergence.png
├── cross_hospital_boxplot.png
├── seed_sensitivity_violin.png
├── threshold_sweep_*.png
└── final_dashboard.png

./tables/
├── fairness_metrics_by_subgroup.csv
├── bootstrap_confidence_intervals.csv
├── sample_size_convergence.csv
├── cross_hospital_heterogeneity.csv
├── seed_sensitivity_statistics.csv
└── comprehensive_comparison.csv

./results/
├── fairness_results.pkl
├── bootstrap_*.pkl
├── sample_size_*.pkl
├── cross_hospital_*.pkl
├── seed_*.pkl
└── threshold_*.pkl
```

---

## 🤖 FIRST PROMPT FOR YOUR AGENT

Copy and paste this to start:

```
I'm starting a fairness analysis project on the Texas-100X hospital discharge dataset.

DATA FILES (in ./data/):
- texas_100x.csv (36 MB, 925K records) - main dataset
- texas_100x_features.p - preprocessed features
- texas_100x_labels.p - labels
- texas_100x_feature_desc.p - feature descriptions

RESEARCH QUESTION: How reliable are fairness metrics in healthcare prediction models?

TASK: Length of Stay prediction (LOS > 3 days = Extended Stay)

PROTECTED ATTRIBUTES: RACE, SEX, ETHNICITY, AGE

Please:
1. First, explore the data files to understand their structure
2. Then run script1_data_preprocessing.py to start the pipeline
3. Show me the data summary and what protected attributes were found

The scripts are in ./scripts/ folder.
```

---

## 📈 Expected Results

This analysis will produce:
- **~1.4 million** individual fairness measurements
- **95% confidence intervals** for all metrics
- **Heterogeneity statistics** (I²) across hospitals
- **Convergence curves** showing data requirements
- **Publication-ready figures** for Q1 journal submission

---

## 🎯 Target Venues

- FAccT (ACM Fairness, Accountability, Transparency)
- CHIL (Conference on Health, Inference, Learning)
- Nature Machine Intelligence
- JAMIA

---

## 📧 Contact

Md Jannatul Rakib Joy  
Swinburne University of Technology
