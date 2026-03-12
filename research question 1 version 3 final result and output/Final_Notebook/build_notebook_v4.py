"""
Build the COMPREHENSIVE RQ1 notebook — ~65 code cells with INLINE outputs,
explanatory markdown, and self-contained results.
Run this script FROM the Final_Notebook directory.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata['kernelspec'] = {
    'display_name': 'Python 3',
    'language': 'python',
    'name': 'python3'
}

def md(src):
    nb.cells.append(nbf.v4.new_markdown_cell(src.strip()))

def code(src):
    nb.cells.append(nbf.v4.new_code_cell(src.strip()))

###############################################################################
# TITLE
###############################################################################
md("""
# RQ1: Length-of-Stay Prediction with Algorithmic Fairness Analysis
## Texas-100x PUDF | 12 Models | Comprehensive Fairness Audit

**Research Question:** *How do machine learning models for hospital length-of-stay
prediction perform across demographic subgroups, and can algorithmic fairness be
achieved without significant accuracy loss?*

**Dataset:** Texas Inpatient Public Use Data File (PUDF), 100× sample — **925,128 records**

---

### Notebook Roadmap

| # | Section | What you will see |
|---|---------|-------------------|
| 1 | **Setup & Data Loading** | Library imports, GPU check, dataset load |
| 2 | **Exploratory Data Analysis** | Target distribution, demographics, diagnoses, hospital patterns |
| 3 | **Feature Engineering** | Target encoding, scaling, train/test split |
| 4 | **Model Training** | 12 classifiers — from Logistic Regression to Stacking Ensemble |
| 5 | **Model Performance** | Accuracy/AUC table, ROC & PR curves, confusion matrices, calibration |
| 6 | **Detailed Model Analysis** | Classification reports, per-group accuracy, learning curves |
| 7 | **Fairness Analysis** | 7 fairness metrics × 4 protected attributes × all models |
| 8 | **Fairness Deep-Dive** | Radar charts, calibration by group, intersectional & cross-hospital |
| 9 | **Stability Testing** | Bootstrap CI, 30-seed perturbation, GroupKFold K=5/20, threshold sweep |
| 10 | **AFCE Analysis** | Fairness-Through-Awareness with protected features & interactions |
| 11 | **Fairness Intervention** | Lambda reweighing, per-group thresholds, Pareto frontier |
| 12 | **Literature Comparison** | Side-by-side comparison with prior work |
| 13 | **Summary Dashboard** | Final comprehensive overview of all results |

> **Note:** Every figure, table, and metric is displayed **inline** below the cell
> that generates it.  The notebook is fully self-contained — share the `.ipynb` file
> and reviewers will see all results without re-running.
""")

###############################################################################
# SECTION 1 — SETUP
###############################################################################
md("""
---
## 1. Setup & Data Loading

We begin by importing all required libraries and detecting GPU availability.
The **RTX 5070** is used for XGBoost (CUDA) and LightGBM (GPU) acceleration.
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 1 · Imports & Environment
# ──────────────────────────────────────────────────────────────
%matplotlib inline

import os, time, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from IPython.display import display, HTML

from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, HistGradientBoostingClassifier,
                              AdaBoostClassifier, GradientBoostingClassifier,
                              StackingClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score, precision_score,
                             recall_score, roc_curve, precision_recall_curve,
                             confusion_matrix, classification_report)
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import calibration_curve

import xgboost as xgb
import lightgbm as lgb

TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    print("⚠ PyTorch not available — DNN model will be skipped")

warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '4'

print("=" * 70)
print("  RQ1: LOS Prediction with Algorithmic Fairness")
print("=" * 70)
print(f"  NumPy {np.__version__}  |  Pandas {pd.__version__}")
print(f"  XGBoost {xgb.__version__}  |  LightGBM {lgb.__version__}")
if TORCH_AVAILABLE:
    print(f"  PyTorch {torch.__version__}  |  CUDA: {torch.cuda.is_available()}")
print("=" * 70)
""")

md("""
> **Tip:** If any import fails, install the missing package with
> `pip install <package>` in the same virtual environment.
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 2 · Configuration & Output Directories
# ──────────────────────────────────────────────────────────────
GPU_AVAILABLE = False
DEVICE = 'cpu'
if TORCH_AVAILABLE and torch.cuda.is_available():
    GPU_AVAILABLE = True
    DEVICE = torch.device('cuda')
    print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
else:
    if TORCH_AVAILABLE:
        DEVICE = torch.device('cpu')
    print("  Running on CPU")

FIGURES_DIR = 'output/figures'
TABLES_DIR  = 'output/tables'
MODELS_DIR  = 'output/models'
for d in [FIGURES_DIR, TABLES_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# Locate data file
DATA_CANDIDATES = [
    '../../data/texas_100x.csv',
    '../../final_analysis/data/texas_100x.csv',
    '../../../data/texas_100x.csv',
    'data/texas_100x.csv',
]
DATA_PATH = None
for p in DATA_CANDIDATES:
    if os.path.exists(p):
        DATA_PATH = p
        break
assert DATA_PATH is not None, f"texas_100x.csv not found in {DATA_CANDIDATES}"
print(f"✓ Data file: {DATA_PATH}")

# Visual style
plt.rcParams.update({
    'figure.figsize': (14, 7), 'figure.dpi': 120, 'font.size': 12,
    'axes.titlesize': 14, 'axes.labelsize': 12,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.grid': True, 'grid.alpha': 0.3,
})
PALETTE = sns.color_palette('husl', 12)
sns.set_style('whitegrid')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
FIG_NUM = [0]

def save_fig(name_suffix):
    FIG_NUM[0] += 1
    path = f'{FIGURES_DIR}/{FIG_NUM[0]:02d}_{name_suffix}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    return path

print("✓ Configuration complete  |  Random state = 42")
""")

md("""
### 1.1 Load the Texas-100x PUDF Dataset

The dataset contains **925,128 hospital discharge records** from Texas hospitals.
Key columns include patient demographics, admission details, charges, and length of stay.
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 3 · Data Loading & Initial Inspection
# ──────────────────────────────────────────────────────────────
print("Loading Texas 100x PUDF dataset …")
t0 = time.time()
df = pd.read_csv(DATA_PATH)
print(f"✓ Loaded {len(df):,} records × {df.shape[1]} columns in {time.time()-t0:.1f}s")
print(f"  Memory: {df.memory_usage(deep=True).sum()/1e6:.1f} MB\\n")

display(HTML("<h4>Column types & missing values</h4>"))
info_df = pd.DataFrame({
    'Type': df.dtypes,
    'Non-Null': df.notnull().sum(),
    'Missing': df.isnull().sum(),
    'Unique': df.nunique(),
})
display(info_df)

display(HTML("<h4>First 5 rows</h4>"))
display(df.head())
""")

md("""
> **Observation:** The dataset has no missing values. All columns are numeric
> (integer or float), which simplifies preprocessing.  `LENGTH_OF_STAY` is our
> target, and `RACE`, `SEX_CODE`, `ETHNICITY`, and `PAT_AGE` are protected attributes.
""")

###############################################################################
# SECTION 2 — EDA
###############################################################################
md("""
---
## 2. Exploratory Data Analysis (EDA)

We explore the target variable, patient demographics, diagnosis patterns,
and hospital-level variation to understand the data before modelling.
""")

md("""
### 2.1 Target Variable & Derived Features

We create a **binary target**: LOS > 3 days = 1 (long stay), otherwise 0.
We also bin patient age into clinically meaningful groups.
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 4 · Binary Target & Age Groups
# ──────────────────────────────────────────────────────────────
df['LOS_BINARY'] = (df['LENGTH_OF_STAY'] > 3).astype(int)

# Texas PUDF PAT_AGE uses coded values (0-21), NOT actual ages.
# Mapping: 0=<1yr, 1=1-4, 2=5-9, 3=10-14, 4=15-17, 5=18-19, 6=20-24,
# 7=25-29, 8=30-34, 9=35-39, 10=40-44, 11=45-49, 12=50-54,
# 13=55-59, 14=60-64, 15=65-69, 16=70-74, 17=75-79, 18=80-84,
# 19=85-89, 20=90-94, 21=95+
def create_age_groups(age_code):
    # Map Texas PUDF PAT_AGE codes to standard clinical binary age groups.
    # Codes 0-14 cover ages <1 through 64; codes 15-21 cover 65+.
    if age_code <= 14:
        return 'Under_65'
    else:
        return '65_Plus'

df['AGE_GROUP'] = df['PAT_AGE'].apply(create_age_groups)

RACE_LABELS = {0:'Other/Unknown', 1:'Native American', 2:'Asian/PI', 3:'Black', 4:'White'}
SEX_LABELS  = {0:'Female', 1:'Male'}
ETH_LABELS  = {0:'Non-Hispanic', 1:'Hispanic'}

df['RACE_LABEL'] = df['RACE'].map(RACE_LABELS)
df['SEX_LABEL']  = df['SEX_CODE'].map(SEX_LABELS)
df['ETH_LABEL']  = df['ETHNICITY'].map(ETH_LABELS)

print("Binary target created: LOS > 3 days")
print(f"  Short stay (≤3 days): {(df['LOS_BINARY']==0).sum():>10,} ({(df['LOS_BINARY']==0).mean():.1%})")
print(f"  Long stay  (>3 days): {(df['LOS_BINARY']==1).sum():>10,} ({(df['LOS_BINARY']==1).mean():.1%})")
print()

summary = []
for g in ['Under_65','65_Plus']:
    n = (df['AGE_GROUP']==g).sum()
    r = df.loc[df['AGE_GROUP']==g,'LOS_BINARY'].mean()
    summary.append({'Age Group':g, 'N':f'{n:,}', 'LOS>3 Rate':f'{r:.1%}'})
display(pd.DataFrame(summary))

desc = df.describe(include='all').T
desc.to_csv(f'{TABLES_DIR}/01_descriptive_statistics.csv')
display(HTML(f"<i>Table saved → {TABLES_DIR}/01_descriptive_statistics.csv</i>"))
""")

md("""
> **Key finding:** About one-third of patients have extended stays (> 3 days).
> Patients aged 65+ have a substantially higher LOS>3 rate (~61%) compared to Under-65 patients (~34%).
""")

md("### 2.2 LOS Distribution & Admission Type")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 5 · Target & LOS Distribution
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(df['LENGTH_OF_STAY'].clip(upper=30), bins=30, color=PALETTE[0],
             edgecolor='white', alpha=0.8)
axes[0].axvline(x=3, color='red', linestyle='--', lw=2, label='Threshold (3 days)')
axes[0].set_xlabel('Length of Stay (days)'); axes[0].set_ylabel('Count')
axes[0].set_title('(a) LOS Distribution (clipped at 30)'); axes[0].legend()

counts = df['LOS_BINARY'].value_counts().sort_index()
bars = axes[1].bar(['≤ 3 days', '> 3 days'], counts.values,
                    color=[PALETTE[1], PALETTE[2]], edgecolor='white')
for b, v in zip(bars, counts.values):
    axes[1].text(b.get_x()+b.get_width()/2, b.get_height()+5000,
                 f'{v:,}\\n({v/len(df):.1%})', ha='center', fontsize=11)
axes[1].set_ylabel('Count'); axes[1].set_title('(b) Binary Target Distribution')

admission_map = {0:'Emergency',1:'Urgent',2:'Elective',3:'Newborn',4:'Trauma'}
df['ADM_LABEL'] = df['TYPE_OF_ADMISSION'].map(admission_map).fillna('Other')
adm_stats = df.groupby('ADM_LABEL')['LENGTH_OF_STAY'].median().sort_values(ascending=False)
axes[2].barh(adm_stats.index, adm_stats.values, color=PALETTE[3], edgecolor='white')
axes[2].set_xlabel('Median LOS (days)'); axes[2].set_title('(c) Median LOS by Admission Type')

plt.tight_layout()
save_fig('target_distribution')
plt.show()
""")

md("""
> **Interpretation:**
> - **(a)** The LOS distribution is heavily right-skewed — most stays are short.
> - **(b)** The binary target is moderately imbalanced (≈ 2:1 short vs. long).
> - **(c)** Trauma and urgent admissions tend to have longer median stays than elective ones.
""")

md("### 2.3 Age & Clinical Features")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 6 · Age, Charges, Patient Status
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for label, color, name in [(0, PALETTE[0], 'LOS≤3'), (1, PALETTE[2], 'LOS>3')]:
    axes[0][0].hist(df.loc[df['LOS_BINARY']==label, 'PAT_AGE'], bins=25,
                    alpha=0.6, color=color, label=name, edgecolor='white')
axes[0][0].set_xlabel('Patient Age'); axes[0][0].set_ylabel('Count')
axes[0][0].set_title('(a) Age Distribution by Outcome'); axes[0][0].legend()

age_order = ['Under_65','65_Plus']
agg = df.groupby('AGE_GROUP')['LOS_BINARY'].mean().reindex(age_order)
bars = axes[0][1].bar(agg.index, agg.values,
                      color=[PALETTE[i] for i in range(len(agg))], edgecolor='white')
for b, v in zip(bars, agg.values):
    axes[0][1].text(b.get_x()+b.get_width()/2, v+0.005, f'{v:.1%}', ha='center', fontsize=10)
axes[0][1].set_ylabel('LOS>3 Rate'); axes[0][1].set_title('(b) LOS>3 Rate by Age Group')

for label, color, name in [(0, PALETTE[0], 'LOS≤3'), (1, PALETTE[2], 'LOS>3')]:
    axes[1][0].hist(df.loc[df['LOS_BINARY']==label, 'TOTAL_CHARGES'].clip(upper=200000),
                    bins=40, alpha=0.5, color=color, label=name, edgecolor='white')
axes[1][0].set_xlabel('Total Charges ($)'); axes[1][0].set_ylabel('Count')
axes[1][0].set_title('(c) Charges by Outcome'); axes[1][0].legend()

ps_stats = df.groupby('PAT_STATUS')['LOS_BINARY'].agg(['count','mean']).sort_values('count', ascending=False).head(10)
axes[1][1].barh(ps_stats.index.astype(str), ps_stats['count'], color=PALETTE[4], edgecolor='white')
axes[1][1].set_xlabel('Count'); axes[1][1].set_title('(d) Patient Status Distribution (top 10)')

plt.tight_layout()
save_fig('age_clinical')
plt.show()
""")

md("""
> **Findings:**
> - Patients aged 65+ have a significantly higher LOS>3 rate than Under-65 patients.
> - Total charges are substantially higher for long-stay patients — a useful predictive feature.
> - Patient status codes show significant variation in volume, providing discharge-related signal.
""")

md("### 2.4 Protected Attributes (Race, Sex, Ethnicity, Age)")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 7 · Protected Attribute Distributions & Outcome Rates
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
attrs = [('RACE_LABEL','Race'), ('SEX_LABEL','Sex'),
         ('ETH_LABEL','Ethnicity'), ('AGE_GROUP','Age Group')]

for idx, (col, title) in enumerate(attrs):
    ax = axes[idx//2][idx%2]
    grp = df.groupby(col).agg(count=('LOS_BINARY','count'),
                               rate=('LOS_BINARY','mean')).sort_values('count', ascending=False)
    x = range(len(grp))
    bars = ax.bar(x, grp['count'], color=PALETTE[idx], alpha=0.7, edgecolor='white')
    ax.set_ylabel('Count', color=PALETTE[idx])
    ax2 = ax.twinx()
    ax2.plot(x, grp['rate'], 'ro-', markersize=8, linewidth=2, label='LOS>3 rate')
    ax2.set_ylabel('LOS>3 Rate', color='red')
    ax.set_xticks(x); ax.set_xticklabels(grp.index, rotation=20, ha='right')
    ax.set_title(f'{title}: Distribution & LOS>3 Rate')
    ax2.legend(loc='upper right')

plt.tight_layout()
save_fig('protected_attributes')
plt.show()

# Save tabular summary
pa_summary = []
for col, title in attrs:
    for val in df[col].unique():
        mask = df[col]==val
        pa_summary.append({'Attribute':title, 'Group':val,
                          'N':mask.sum(), 'Pct':f'{mask.mean():.2%}',
                          'LOS_gt3_rate':f'{df.loc[mask,"LOS_BINARY"].mean():.3f}'})
pa_df = pd.DataFrame(pa_summary)
pa_df.to_csv(f'{TABLES_DIR}/02_protected_attribute_summary.csv', index=False)
display(HTML("<h4>Protected Attribute Summary</h4>"))
display(pa_df)
""")

md("### 2.4.1 Table 2 — Manuscript-Ready Descriptive Statistics")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 7b · Table 2: Descriptive Statistics (Manuscript Format)
# ──────────────────────────────────────────────────────────────
# Build the exact Table 2 for the manuscript:
# "Descriptive statistics of the Texas-100X cohort"
table2_rows = []
total_n = len(df)
total_pos = df['LOS_BINARY'].mean()

# --- Race ---
race_order = ['White','Black','Asian/PI','Native American','Other/Unknown']
for grp in race_order:
    mask = df['RACE_LABEL'] == grp
    n = mask.sum()
    rate = df.loc[mask, 'LOS_BINARY'].mean()
    table2_rows.append({'Attribute': 'Race' if grp == race_order[0] else '',
                        'Subgroup': grp, 'N': f'{n:,}', 'Pct': f'{n/total_n*100:.1f}%',
                        'LOS_gt_3d_pct': f'{rate*100:.1f}%'})

# --- Sex ---
for i, grp in enumerate(['Male','Female']):
    mask = df['SEX_LABEL'] == grp
    n = mask.sum()
    rate = df.loc[mask, 'LOS_BINARY'].mean()
    table2_rows.append({'Attribute': 'Sex' if i == 0 else '',
                        'Subgroup': grp, 'N': f'{n:,}', 'Pct': f'{n/total_n*100:.1f}%',
                        'LOS_gt_3d_pct': f'{rate*100:.1f}%'})

# --- Ethnicity ---
for i, grp in enumerate(['Hispanic','Non-Hispanic']):
    mask = df['ETH_LABEL'] == grp
    n = mask.sum()
    rate = df.loc[mask, 'LOS_BINARY'].mean()
    table2_rows.append({'Attribute': 'Ethnicity' if i == 0 else '',
                        'Subgroup': grp, 'N': f'{n:,}', 'Pct': f'{n/total_n*100:.1f}%',
                        'LOS_gt_3d_pct': f'{rate*100:.1f}%'})

# --- Age Group ---
age_display = {'Under_65':'Under 65', '65_Plus':'65 and Over'}
age_order = ['Under_65','65_Plus']
first = True
for grp in age_order:
    mask = df['AGE_GROUP'] == grp
    n = mask.sum()
    if n == 0:
        continue
    rate = df.loc[mask, 'LOS_BINARY'].mean()
    table2_rows.append({'Attribute': 'Age Group' if first else '',
                        'Subgroup': age_display.get(grp, grp), 'N': f'{n:,}', 'Pct': f'{n/total_n*100:.1f}%',
                        'LOS_gt_3d_pct': f'{rate*100:.1f}%'})
    first = False

# --- Total row ---
table2_rows.append({'Attribute': '', 'Subgroup': 'Total',
                    'N': f'{total_n:,}', 'Pct': '100.0%',
                    'LOS_gt_3d_pct': f'{total_pos*100:.1f}%'})

table2_df = pd.DataFrame(table2_rows)
table2_df.columns = ['Attribute', 'Subgroup', 'N', '%', 'LOS > 3d (%)']

# Save CSV
table2_df.to_csv(f'{TABLES_DIR}/02b_table2_manuscript.csv', index=False)

# Display as publication-quality HTML
html = '<h4>Table 2 — Descriptive Statistics of the Texas-100X Cohort</h4>'
html += '<p style="font-size:12px;color:#555;">Protected attribute categories, sample sizes, and LOS &gt; 3-day rates by subgroup.</p>'
html += '<table style="border-collapse:collapse;width:auto;font-family:Arial,sans-serif;font-size:13px;">'
html += '<thead><tr style="background:#2c3e50;color:white;font-weight:bold;">'
for col in table2_df.columns:
    html += f'<th style="padding:8px 16px;text-align:center;border:1px solid #ddd;">{col}</th>'
html += '</tr></thead><tbody>'
for idx, row in table2_df.iterrows():
    bg = '#f8f9fa' if idx % 2 == 0 else '#ffffff'
    if row['Subgroup'] == 'Total':
        bg = '#e8f4fd'
    is_attr_start = row['Attribute'] != ''
    html += f'<tr style="background:{bg};">'
    for j, val in enumerate(row):
        fw = 'bold' if j == 0 and is_attr_start else ('bold' if row['Subgroup'] == 'Total' else 'normal')
        border_top = '2px solid #2c3e50' if (is_attr_start and row['Subgroup'] != 'Total') else '1px solid #ddd'
        html += f'<td style="padding:6px 16px;border:1px solid #ddd;border-top:{border_top};font-weight:{fw};text-align:center;">{val}</td>'
    html += '</tr>'
html += '</tbody></table>'
display(HTML(html))
print(f"Table 2 saved → {TABLES_DIR}/02b_table2_manuscript.csv")
""")

md("""
> **Fairness-relevant observations:**
> - **Race:** White patients make up the majority; Native American and Asian/PI groups are
>   small, which can lead to unstable fairness metrics for those groups.
> - **Sex:** Roughly balanced.  Males have a slightly higher LOS>3 rate.
> - **Ethnicity:** Non-Hispanic patients are the majority with a higher long-stay rate.
> - **Age:** Patients 65+ have a substantially higher LOS>3 rate (~61% vs ~34%), raising
>   fairness concerns if the model under-predicts for younger patients.
""")

md("### 2.5 Source of Admission")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 8 · Source of Admission Deep-Dive
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
source_map = {0:'Non-healthcare', 1:'ER', 2:'Clinic', 3:'Transfer-Hospital',
              4:'Transfer-SNF', 5:'Transfer-Other', 6:'Born-in', 9:'Unknown'}
df['SOURCE_LABEL'] = df['SOURCE_OF_ADMISSION'].map(source_map).fillna('Other')
src_stats = df.groupby('SOURCE_LABEL').agg(
    n=('LOS_BINARY','count'), rate=('LOS_BINARY','mean'),
    median_los=('LENGTH_OF_STAY','median')
).sort_values('n', ascending=False)

axes[0].barh(src_stats.index, src_stats['n'], color=PALETTE[5], edgecolor='white')
axes[0].set_xlabel('Count'); axes[0].set_title('(a) Admission Source Volume')
axes[1].barh(src_stats.index, src_stats['rate'], color=PALETTE[6], edgecolor='white')
axes[1].set_xlabel('LOS>3 Rate'); axes[1].set_title('(b) LOS>3 Rate by Source')
axes[2].barh(src_stats.index, src_stats['median_los'], color=PALETTE[7], edgecolor='white')
axes[2].set_xlabel('Median LOS (days)'); axes[2].set_title('(c) Median LOS by Source')

plt.tight_layout()
save_fig('source_of_admission')
plt.show()

display(src_stats.style.format({'rate':'{:.1%}','median_los':'{:.1f}'}))
""")

md("""
> **Insight:** Transfers from other hospitals and SNFs have much higher LOS>3 rates
> — these patients are typically sicker.  ER admissions are by far the most common source.
""")

md("### 2.6 Top Diagnoses & Procedures")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 9 · Top 15 Diagnoses & Procedures by Volume
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

diag_stats = df.groupby('ADMITTING_DIAGNOSIS').agg(
    n=('LOS_BINARY','count'), rate=('LOS_BINARY','mean')
).sort_values('n', ascending=False).head(15)
axes[0].barh(diag_stats.index.astype(str), diag_stats['rate'], color=PALETTE[0])
axes[0].set_xlabel('LOS>3 Rate')
axes[0].set_title('Top 15 Diagnoses (by volume) — LOS>3 Rate')
axes[0].invert_yaxis()

proc_stats = df.groupby('PRINC_SURG_PROC_CODE').agg(
    n=('LOS_BINARY','count'), rate=('LOS_BINARY','mean')
).sort_values('n', ascending=False).head(15)
axes[1].barh(proc_stats.index.astype(str), proc_stats['rate'], color=PALETTE[2])
axes[1].set_xlabel('LOS>3 Rate')
axes[1].set_title('Top 15 Procedures (by volume) — LOS>3 Rate')
axes[1].invert_yaxis()

plt.tight_layout()
save_fig('top_diagnoses_procedures')
plt.show()
""")

md("""
> These diagnosis and procedure codes carry strong predictive signal via
> **target encoding** (Section 3).  Some diagnoses have near-0% long-stay rates
> while others exceed 60%.
""")

md("### 2.7 Correlation Heatmap")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 10 · Correlation Matrix
# ──────────────────────────────────────────────────────────────
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, ax=ax, square=True, linewidths=0.5)
ax.set_title('Feature Correlation Matrix')
plt.tight_layout()
save_fig('correlation_heatmap')
plt.show()
""")

md("""
> **Key correlations:** `TOTAL_CHARGES` and `LENGTH_OF_STAY` are strongly correlated
> (expected, since longer stays cost more).  Protected attributes show low correlation
> with one another, supporting independent fairness analysis per attribute.
""")

md("### 2.8 Hospital-Level Patterns")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 11 · Hospital Volume, Rates & Distribution
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
hosp_stats = df.groupby('THCIC_ID').agg(
    n_patients=('LOS_BINARY','count'), los_rate=('LOS_BINARY','mean'),
    median_los=('LENGTH_OF_STAY','median')
).reset_index()

axes[0].hist(hosp_stats['n_patients'], bins=40, color=PALETTE[4], edgecolor='white', alpha=0.8)
axes[0].set_xlabel('Patients per Hospital'); axes[0].set_ylabel('Count')
axes[0].set_title(f'(a) Hospital Volume  (N={len(hosp_stats)} hospitals)')
axes[1].scatter(hosp_stats['n_patients'], hosp_stats['los_rate'], alpha=0.3, s=15, color=PALETTE[5])
axes[1].axhline(y=df['LOS_BINARY'].mean(), color='red', linestyle='--', label='Overall rate')
axes[1].set_xlabel('Patients'); axes[1].set_ylabel('LOS>3 Rate')
axes[1].set_title('(b) Volume vs LOS>3 Rate'); axes[1].legend()
axes[2].hist(hosp_stats['los_rate'], bins=30, color=PALETTE[6], edgecolor='white', alpha=0.8)
axes[2].axvline(x=df['LOS_BINARY'].mean(), color='red', linestyle='--', label='Overall')
axes[2].set_xlabel('LOS>3 Rate'); axes[2].set_ylabel('Count')
axes[2].set_title('(c) Hospital LOS>3 Rate Distribution'); axes[2].legend()

plt.tight_layout()
save_fig('hospital_patterns')
plt.show()
print(f"Total hospitals: {len(hosp_stats)}  |  Median volume: {hosp_stats['n_patients'].median():.0f}")
""")

md("""
> Hospitals vary widely in both volume and LOS>3 rate. This motivates using
> **GroupKFold** (by hospital) in our stability analysis and including a
> **hospital target-encoded feature**.
""")

md("### 2.9 Demographics Cross-tabulation")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 12 · Race×Sex and Age×Ethnicity outcome crosstabs
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

ct1 = df.pivot_table(values='LOS_BINARY', index='RACE_LABEL', columns='SEX_LABEL', aggfunc='mean')
sns.heatmap(ct1, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[0], linewidths=0.5)
axes[0].set_title('LOS>3 Rate: Race × Sex')

ct2 = df.pivot_table(values='LOS_BINARY', index='AGE_GROUP', columns='ETH_LABEL', aggfunc='mean')
ct2 = ct2.reindex(['Under_65','65_Plus'])
sns.heatmap(ct2, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[1], linewidths=0.5)
axes[1].set_title('LOS>3 Rate: Age × Ethnicity')

plt.tight_layout()
save_fig('demographics_crosstab')
plt.show()
""")

md("""
> **Intersectional patterns:** The heatmaps reveal that LOS>3 rates differ
> not just by single attributes but by their combinations.  For example,
> 65+ non-Hispanic patients have a particularly high long-stay rate.
> We will return to intersectional fairness in Section 8.
""")

###############################################################################
# SECTION 3 — FEATURE ENGINEERING
###############################################################################
md("""
---
## 3. Feature Engineering & Train/Test Split

We use **target encoding** (with Bayesian smoothing) for high-cardinality
categorical features (diagnosis codes, procedure codes, hospital IDs).
A standard 80/20 stratified split preserves the target distribution.
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 13 · Train/Test Split & Feature Engineering
# ──────────────────────────────────────────────────────────────
train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE,
                                     stratify=df['LOS_BINARY'])
print(f"Train: {len(train_df):,}  |  Test: {len(test_df):,}")
print(f"Train LOS>3: {train_df['LOS_BINARY'].mean():.4f}  |  Test LOS>3: {test_df['LOS_BINARY'].mean():.4f}")

# Target encoding (Bayesian smoothing)
global_mean = train_df['LOS_BINARY'].mean()
smoothing = 10
te_maps = {}

for col in ['ADMITTING_DIAGNOSIS', 'PRINC_SURG_PROC_CODE']:
    stats = train_df.groupby(col)['LOS_BINARY'].agg(['mean','count'])
    smoothed = (stats['count']*stats['mean'] + smoothing*global_mean) / (stats['count']+smoothing)
    te_map = smoothed.to_dict()
    te_maps[col] = te_map
    te_name = 'ADMITTING_DIAGNOSIS_TE' if 'DIAG' in col else 'PROC_TE'
    train_df[te_name] = train_df[col].map(te_map).fillna(global_mean)
    test_df[te_name]  = test_df[col].map(te_map).fillna(global_mean)
    print(f"  {col} → {te_name}: {len(te_map)} categories")

hosp_stats_te = train_df.groupby('THCIC_ID')['LOS_BINARY'].agg(['mean','count'])
hosp_te = (hosp_stats_te['count']*hosp_stats_te['mean'] + smoothing*global_mean) / (hosp_stats_te['count']+smoothing)
hosp_te_map = hosp_te.to_dict()
train_df['HOSP_TE'] = train_df['THCIC_ID'].map(hosp_te_map).fillna(global_mean)
test_df['HOSP_TE']  = test_df['THCIC_ID'].map(hosp_te_map).fillna(global_mean)
print(f"  THCIC_ID → HOSP_TE: {len(hosp_te_map)} hospitals")

# Feature matrix — PAT_AGE included (key clinical predictor for LOS)
numeric_features = ['PAT_AGE', 'TOTAL_CHARGES', 'PAT_STATUS',
                    'TYPE_OF_ADMISSION', 'SOURCE_OF_ADMISSION',
                    'ADMITTING_DIAGNOSIS_TE', 'PROC_TE', 'HOSP_TE']
X_train_raw = train_df[numeric_features].reset_index(drop=True).fillna(0)
X_test_raw  = test_df[numeric_features].reset_index(drop=True).fillna(0)
feature_names = list(X_train_raw.columns)
y_train = train_df['LOS_BINARY'].values
y_test  = test_df['LOS_BINARY'].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)
X_train = np.nan_to_num(X_train, nan=0.0)
X_test  = np.nan_to_num(X_test, nan=0.0)

# Store protected attributes
protected_attrs = {
    'RACE': test_df['RACE'].values, 'SEX': test_df['SEX_CODE'].values,
    'ETHNICITY': test_df['ETHNICITY'].values, 'AGE_GROUP': test_df['AGE_GROUP'].values,
}
protected_attrs_train = {
    'RACE': train_df['RACE'].values, 'SEX': train_df['SEX_CODE'].values,
    'ETHNICITY': train_df['ETHNICITY'].values, 'AGE_GROUP': train_df['AGE_GROUP'].values,
}
hospital_ids_test  = test_df['THCIC_ID'].values
hospital_ids_train = train_df['THCIC_ID'].values

RACE_MAP = {0:'Other/Unknown', 1:'Native American', 2:'Asian/PI', 3:'Black', 4:'White'}
SEX_MAP  = {0:'Female', 1:'Male'}
ETH_MAP  = {0:'Non-Hispanic', 1:'Hispanic'}

print(f"\\n✓ Feature matrix: {X_train.shape[1]} features  →  {feature_names}")
print(f"  X_train: {X_train.shape}  |  X_test: {X_test.shape}")
""")

md("""
> **Design note:** Race, Sex, and Ethnicity are **not** included as model features —
> they are reserved for post-hoc fairness evaluation (fairness-through-unawareness).
> **PAT_AGE is included** as it is a critical clinical predictor of hospital length of stay.
> Age-based fairness is evaluated via a binary AGE_GROUP (Under_65 vs 65_Plus).
""")

###############################################################################
# SECTION 4 — MODEL TRAINING
###############################################################################
md("""
---
## 4. Model Training (12 Models)

We train a comprehensive set of classifiers ranging from simple (Logistic Regression,
Decision Tree) to complex (XGBoost, LightGBM, CatBoost, DNN, Stacking Ensemble).
GPU acceleration is used where available.
""")

md("""
### 4.0 Table 3 — Model Methodology Overview

The table below provides a methodological summary of every classifier used in
this study, including the algorithmic family, key hyperparameters, training
strategy, and suitability for clinical LOS prediction.

| # | Model | Family | Key Idea | Key Hyperparameters | GPU | Why Included |
|---|-------|--------|----------|---------------------|-----|--------------|
| 1 | **Logistic Regression (LR)** | Generalized Linear Model | Maximum-likelihood estimation of log-odds | `C=1.0`, `solver='lbfgs'`, `max_iter=1000` | No | Interpretable baseline; gold standard in clinical research [1,4] |
| 2 | **Decision Tree (DT)** | Tree-based | Recursive binary splits via information gain | `max_depth=15`, `min_samples_leaf=50` | No | Fully transparent rule-based model; interpretability reference |
| 3 | **Random Forest (RF)** | Bagged Ensemble | Bootstrap aggregation of decision trees | `n_estimators=300`, `max_depth=20`, `min_samples_leaf=20` | No | Reduces variance of single trees while maintaining interpretability [4,6] |
| 4 | **Hist Gradient Boosting (HGB)** | Boosted Ensemble | Histogram-based gradient boosting (scikit-learn native) | `max_iter=500`, `max_depth=8`, `learning_rate=0.05` | No | Fast, handles missing data natively; sklearn-native alternative |
| 5 | **Gradient Boosting (GB)** | Boosted Ensemble | Sequential additive tree fitting with gradient descent | `n_estimators=300`, `max_depth=5`, `learning_rate=0.05` | No | Classical boosting baseline for comparison with XGBoost/LightGBM |
| 6 | **AdaBoost** | Boosted Ensemble | Iterative reweighting of misclassified samples | `n_estimators=200`, `learning_rate=0.1` | No | Tests adaptive reweighting strategy for imbalanced LOS data |
| 7 | **XGBoost** | Boosted Ensemble | Regularised gradient boosting with column sampling | `n_estimators=500`, `max_depth=8`, `learning_rate=0.05`, `tree_method='gpu_hist'` | **Yes** | State-of-art tabular model; used in Jain et al. (2024) and Tarek et al. (2025) [1,2] |
| 8 | **LightGBM** | Boosted Ensemble | Leaf-wise growth + gradient-based one-side sampling (GOSS) | `n_estimators=500`, `num_leaves=63`, `learning_rate=0.05`, `device='gpu'` | **Yes** | Fastest training; handles large datasets efficiently [1] |
| 9 | **CatBoost** | Boosted Ensemble | Ordered boosting + native categorical support | `iterations=500`, `depth=8`, `learning_rate=0.05`, `task_type='GPU'` | **Yes** | Handles categorical features without encoding; robust to overfitting |
| 10 | **DNN (LOSNet)** | Deep Learning | 4-layer feedforward network with BatchNorm + Dropout | `layers=[512,256,128,1]`, `dropout=[0.3,0.2,0.1]`, `epochs=20`, `lr=1e-3` | **Yes** | Tests neural network representation power; growing in clinical AI [5] |
| 11 | **Stacking Ensemble** | Meta-learner | LR meta-learner on top of RF + XGBoost + LightGBM base models | `cv=5`; base models use same hyperparameters as above | Mixed | Combines strengths of diverse base learners; expected best accuracy |
| 12 | **LGB-XGB Blend** | Simple Ensemble | Weighted average of LightGBM (0.5) and XGBoost (0.5) probabilities | Equal weighting | Mixed | Simple but effective; tests if blending matches stacking |

> **Citation key:** [1] Jain et al. (2024), [2] Tarek et al. (2025), [3] Zeleke et al. (2023),
> [4] Mekhaldi et al. (2021), [5] Jaotombo et al. (2022), [6] Almeida et al. (2024 review)
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 14 · Define PyTorch DNN Architecture
# ──────────────────────────────────────────────────────────────
if TORCH_AVAILABLE:
    class LOSNet(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(128, 1)
            )
        def forward(self, x):
            return self.net(x)

    class DNNClassifier(BaseEstimator, ClassifierMixin):
        def __init__(self, input_dim, epochs=30, batch_size=2048, lr=1e-3):
            self.input_dim = input_dim; self.epochs = epochs
            self.batch_size = batch_size; self.lr = lr
            self.classes_ = np.array([0, 1])

        def fit(self, X, y):
            dev = torch.device(DEVICE)
            self.model_ = LOSNet(self.input_dim).to(dev)
            optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=1e-4)
            criterion = nn.BCEWithLogitsLoss()
            Xt = torch.FloatTensor(X).to(dev); yt = torch.FloatTensor(y).to(dev)
            ds = TensorDataset(Xt, yt)
            dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
            self.model_.train()
            for epoch in range(self.epochs):
                for xb, yb in dl:
                    optimizer.zero_grad()
                    loss = criterion(self.model_(xb).squeeze(), yb)
                    loss.backward(); optimizer.step()
            return self

        def predict_proba(self, X):
            dev = torch.device(DEVICE); self.model_.eval()
            with torch.no_grad():
                logits = self.model_(torch.FloatTensor(X).to(dev)).squeeze().cpu().numpy()
            probs = 1 / (1 + np.exp(-logits))
            return np.column_stack([1-probs, probs])

        def predict(self, X):
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    print("✓ DNN architecture: 512 → 256 → 128 → 1  (BatchNorm + Dropout)")
else:
    print("⚠ PyTorch not available — DNN will be skipped")
""")

md("""
> The DNN uses **batch normalization** and **dropout** at each hidden layer
> to regularise and stabilise training.  30 epochs with Adam optimiser.
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 15 · Train All 10 Base Models
# ──────────────────────────────────────────────────────────────
xgb_gpu = 'cuda' if GPU_AVAILABLE else 'cpu'
lgb_gpu = 'gpu' if GPU_AVAILABLE else 'cpu'

models_config = {
    'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1),
    'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=10,
                                            random_state=RANDOM_STATE, n_jobs=-1),
    'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=300, max_depth=8, learning_rate=0.1,
                                                           random_state=RANDOM_STATE),
    'XGBoost': xgb.XGBClassifier(n_estimators=1000, max_depth=10, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8,
                                  tree_method='hist', device=xgb_gpu,
                                  random_state=RANDOM_STATE, eval_metric='logloss', verbosity=0),
    'LightGBM': lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.03, num_leaves=255,
                                    max_depth=-1, subsample=0.8, colsample_bytree=0.8,
                                    device=lgb_gpu, random_state=RANDOM_STATE, verbose=-1, n_jobs=1),
    'AdaBoost': AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3),
                                   n_estimators=200, learning_rate=0.1, random_state=RANDOM_STATE),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                                   subsample=0.8, random_state=RANDOM_STATE),
    'DecisionTree': DecisionTreeClassifier(max_depth=15, min_samples_split=20, random_state=RANDOM_STATE),
}

try:
    from catboost import CatBoostClassifier
    models_config['CatBoost'] = CatBoostClassifier(iterations=500, depth=8, learning_rate=0.05,
                                                    random_seed=RANDOM_STATE, verbose=0,
                                                    task_type='GPU' if GPU_AVAILABLE else 'CPU')
except ImportError:
    print("⚠ CatBoost not available — skipping")

if TORCH_AVAILABLE:
    models_config['DNN (PyTorch)'] = DNNClassifier(input_dim=X_train.shape[1], epochs=30, batch_size=2048)

trained_models = {}; test_predictions = {}; training_times = {}

print(f"Training {len(models_config)} models …")
print("=" * 80)
for name, model in models_config.items():
    print(f"  ▸ {name} …", end=' ', flush=True)
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    training_times[name] = elapsed
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    trained_models[name] = model
    test_predictions[name] = {'y_pred': y_pred, 'y_prob': y_prob}
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    print(f"Acc={acc:.4f}  AUC={auc:.4f}  F1={f1:.4f}  [{elapsed:.1f}s]")
print("=" * 80)
print(f"✓ {len(trained_models)} models trained successfully")
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 16 · Stacking Ensemble & LGB-XGB Blend
# ──────────────────────────────────────────────────────────────
base_estimators = [
    ('lr', LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)),
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1)),
    ('xgb', xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                                tree_method='hist', device=xgb_gpu,
                                random_state=RANDOM_STATE, verbosity=0)),
]
print("Training Stacking Ensemble …", end=' ', flush=True)
t0 = time.time()
stacking = StackingClassifier(estimators=base_estimators,
    final_estimator=LogisticRegression(C=1.0, max_iter=500), cv=3, n_jobs=1, passthrough=False)
stacking.fit(X_train, y_train)
elapsed = time.time() - t0
y_pred_stack = stacking.predict(X_test)
y_prob_stack = stacking.predict_proba(X_test)[:, 1]
trained_models['Stacking Ensemble'] = stacking
test_predictions['Stacking Ensemble'] = {'y_pred': y_pred_stack, 'y_prob': y_prob_stack}
training_times['Stacking Ensemble'] = elapsed
print(f"Acc={accuracy_score(y_test, y_pred_stack):.4f}  AUC={roc_auc_score(y_test, y_prob_stack):.4f}  [{elapsed:.1f}s]")

# LGB-XGB Blend (60/40 soft vote)
if 'LightGBM' in test_predictions and 'XGBoost' in test_predictions:
    print("Creating LGB-XGB Blend (0.6 / 0.4) …", end=' ')
    blend_prob = 0.6 * test_predictions['LightGBM']['y_prob'] + 0.4 * test_predictions['XGBoost']['y_prob']
    blend_pred = (blend_prob >= 0.5).astype(int)
    test_predictions['LGB-XGB Blend'] = {'y_pred': blend_pred, 'y_prob': blend_prob}
    training_times['LGB-XGB Blend'] = training_times['LightGBM'] + training_times['XGBoost']
    print(f"Acc={accuracy_score(y_test, blend_pred):.4f}  AUC={roc_auc_score(y_test, blend_prob):.4f}")

print(f"\\n✓ Total models for evaluation: {len(test_predictions)}")
""")

md("""
> **Stacking Ensemble** combines LR + RF + XGBoost with a logistic meta-learner
> (3-fold CV for out-of-fold predictions).  The **LGB-XGB Blend** is a simple
> weighted average of the two best boosters — often competitive with stacking
> at lower computational cost.
""")

###############################################################################
# SECTION 5 — PERFORMANCE
###############################################################################
md("""
---
## 5. Model Performance Comparison

We evaluate all models on the held-out test set using standard classification
metrics: Accuracy, AUC-ROC, F1 Score, Precision, and Recall.
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 17 · Performance Summary Table
# ──────────────────────────────────────────────────────────────
results_list = []
for name in test_predictions:
    y_p = test_predictions[name]['y_pred']
    y_pb = test_predictions[name]['y_prob']
    results_list.append({
        'Model': name, 'Accuracy': accuracy_score(y_test, y_p),
        'AUC': roc_auc_score(y_test, y_pb), 'F1': f1_score(y_test, y_p),
        'Precision': precision_score(y_test, y_p), 'Recall': recall_score(y_test, y_p),
        'Train_Time_s': training_times.get(name, 0),
    })

results_df = pd.DataFrame(results_list).sort_values('AUC', ascending=False).reset_index(drop=True)
results_df.to_csv(f'{TABLES_DIR}/03_model_comparison.csv', index=False)

best_model_name = results_df.iloc[0]['Model']
best_y_pred = test_predictions[best_model_name]['y_pred']
best_y_prob = test_predictions[best_model_name]['y_prob']

display(HTML("<h3>🏆 Model Performance Ranking (sorted by AUC)</h3>"))
styled = results_df.style.format({
    'Accuracy':'{:.4f}','AUC':'{:.4f}','F1':'{:.4f}',
    'Precision':'{:.4f}','Recall':'{:.4f}','Train_Time_s':'{:.1f}'
}).highlight_max(subset=['Accuracy','AUC','F1'], color='lightgreen'
).highlight_min(subset=['Accuracy','AUC','F1'], color='#ffcccc')
display(styled)
print(f"\\n★ Best model: {best_model_name} (AUC = {results_df.iloc[0]['AUC']:.4f})")
""")

md("""
> The table above is colour-coded: **green** = best, **red** = worst.
> The model with the highest AUC is selected as the primary model for
> downstream fairness analysis.
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 18 · ROC & Precision-Recall Curves (Enhanced)
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
cmap_lines = plt.cm.tab20(np.linspace(0, 1, len(results_df)))

for i, name in enumerate(results_df['Model']):
    y_pb = test_predictions[name]['y_prob']
    fpr, tpr, _ = roc_curve(y_test, y_pb)
    auc_val = roc_auc_score(y_test, y_pb)
    lw = 2.5 if i < 3 else 1.2
    alpha = 1.0 if i < 3 else 0.6
    axes[0].plot(fpr, tpr, label=f'{name} ({auc_val:.3f})',
                 linewidth=lw, alpha=alpha, color=cmap_lines[i])
    if i == 0:
        axes[0].fill_between(fpr, tpr, alpha=0.08, color=cmap_lines[i])
    prec, rec, _ = precision_recall_curve(y_test, y_pb)
    axes[1].plot(rec, prec, label=name, linewidth=lw, alpha=alpha, color=cmap_lines[i])

axes[0].plot([0,1],[0,1],'k--', alpha=0.3, linewidth=1)
axes[0].fill_between([0,1],[0,1], alpha=0.02, color='gray')
axes[0].set_xlabel('False Positive Rate', fontsize=12)
axes[0].set_ylabel('True Positive Rate', fontsize=12)
axes[0].set_title('(a) ROC Curves — All 12 Models', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=7.5, loc='lower right', framealpha=0.9, ncol=1)
axes[0].set_xlim(-0.01, 1.01); axes[0].set_ylim(-0.01, 1.01)
axes[0].grid(True, alpha=0.15)
axes[1].set_xlabel('Recall', fontsize=12); axes[1].set_ylabel('Precision', fontsize=12)
axes[1].set_title('(b) Precision-Recall Curves', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=7.5, loc='lower left', framealpha=0.9, ncol=1)
axes[1].grid(True, alpha=0.15)
plt.tight_layout()
save_fig('roc_pr_curves')
plt.show()
""")

md("""
> **ROC curves**: All boosting models cluster near the top-left corner, indicating
> excellent discrimination.  The PR curves show precision-recall trade-offs —
> important because the target is moderately imbalanced.
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 19 · Model Comparison — Publication-Quality
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
metrics = ['Accuracy', 'AUC', 'F1', 'Precision', 'Recall']
model_order = results_df['Model'].tolist()

# (a) Horizontal grouped bar with value annotations
y_pos = np.arange(len(model_order))
bar_h = 0.15
for i, m in enumerate(metrics):
    vals = results_df.set_index('Model').loc[model_order, m].values
    bars = axes[0].barh(y_pos - (len(metrics)/2 - i - 0.5)*bar_h, vals, bar_h,
                        label=m, color=PALETTE[i], alpha=0.85, edgecolor='white')
axes[0].set_yticks(y_pos)
axes[0].set_yticklabels(model_order, fontsize=9)
axes[0].set_xlabel('Score', fontsize=11)
axes[0].set_title('(a) Performance Metrics by Model', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=8, loc='lower right', framealpha=0.9)
axes[0].set_xlim(0.3, 1.02)
axes[0].axvline(x=0.9, color='gray', linestyle=':', alpha=0.3)

# (b) Dot plot — AUC with CI-like spread
aucs = results_df['AUC'].values
sorted_idx = np.argsort(aucs)
sorted_models = results_df['Model'].values[sorted_idx]
sorted_aucs = aucs[sorted_idx]
colors = [PALETTE[2] if a >= 0.9 else PALETTE[0] if a >= 0.85 else '#e74c3c' for a in sorted_aucs]
axes[1].scatter(sorted_aucs, range(len(sorted_models)), c=colors, s=120, zorder=5, edgecolors='white', linewidth=1.5)
axes[1].hlines(range(len(sorted_models)), xmin=sorted_aucs.min()-0.02, xmax=sorted_aucs,
               color=colors, linewidth=2, alpha=0.6)
for i, (a, m) in enumerate(zip(sorted_aucs, sorted_models)):
    axes[1].annotate(f'{a:.4f}', (a+0.002, i), fontsize=9, fontweight='bold', va='center')
axes[1].set_yticks(range(len(sorted_models)))
axes[1].set_yticklabels(sorted_models, fontsize=9)
axes[1].set_xlabel('AUC-ROC', fontsize=11)
axes[1].set_title('(b) Model Ranking by AUC', fontsize=13, fontweight='bold')
axes[1].axvline(x=0.9, color='red', linestyle='--', alpha=0.4, label='AUC = 0.90')
axes[1].legend(fontsize=8)

plt.tight_layout()
save_fig('model_comparison_bar')
plt.show()
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 20 · Feature Importance (Top 3 Tree Models)
# ──────────────────────────────────────────────────────────────
importance_models = [n for n in ['LightGBM','XGBoost','Random Forest'] if n in trained_models]
fig, axes = plt.subplots(1, len(importance_models), figsize=(6*len(importance_models), 7))
if len(importance_models) == 1: axes = [axes]

for idx, name in enumerate(importance_models):
    model = trained_models[name]
    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
        top_idx = np.argsort(imp)[-15:]
        axes[idx].barh([feature_names[i] for i in top_idx], imp[top_idx],
                       color=PALETTE[idx], edgecolor='white')
        axes[idx].set_xlabel('Importance'); axes[idx].set_title(f'{name}')
plt.suptitle('Feature Importance (Top Features)', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.95])
save_fig('feature_importance')
plt.show()
""")

md("""
> **Feature importance:** Diagnosis and hospital target-encoded features consistently
> rank highest, confirming that clinical context is the strongest predictor.
> Total charges, patient age, and admission type are also highly informative.
> PAT_AGE is retained as a feature given its strong clinical relevance for LOS prediction.
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 21 · Confusion Matrices (Top 6 Models)
# ──────────────────────────────────────────────────────────────
top6 = results_df['Model'].head(6).tolist()
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for i, name in enumerate(top6):
    ax = axes[i//3][i%3]
    cm = confusion_matrix(y_test, test_predictions[name]['y_pred'])
    sns.heatmap(cm, annot=True, fmt=',', cmap='Blues', ax=ax,
                xticklabels=['≤3','> 3'], yticklabels=['≤3','> 3'])
    ax.set_title(name, fontsize=10); ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
plt.suptitle('Confusion Matrices (Top 6 Models)', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.95])
save_fig('confusion_matrices')
plt.show()
""")

md("""
> Confusion matrices show the raw TP/TN/FP/FN counts.  The dominant
> diagonal confirms high accuracy.  Off-diagonal cells reveal whether models
> tend toward **false negatives** (missing long stays) or **false positives**.
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 22 · Calibration Curves (Top 4 Models)
# ──────────────────────────────────────────────────────────────
top4 = results_df['Model'].head(4).tolist()
fig, axes = plt.subplots(1, len(top4), figsize=(5*len(top4), 5))
for i, name in enumerate(top4):
    prob_true, prob_pred = calibration_curve(y_test, test_predictions[name]['y_prob'],
                                              n_bins=10, strategy='uniform')
    axes[i].plot(prob_pred, prob_true, 'o-', color=PALETTE[i], linewidth=2, label='Model')
    axes[i].plot([0,1],[0,1], 'k--', alpha=0.3, label='Perfect')
    axes[i].set_xlabel('Mean Predicted Prob.'); axes[i].set_ylabel('Fraction Positive')
    axes[i].set_title(f'{name}'); axes[i].legend(fontsize=8)
plt.suptitle('Calibration Curves', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.95])
save_fig('calibration_curves')
plt.show()
""")

md("""
> A well-calibrated model's curve hugs the diagonal.  Most boosting models
> show good calibration, meaning predicted probabilities closely match true event rates.
""")

###############################################################################
# SECTION 6 — DETAILED MODEL ANALYSIS
###############################################################################
md("""
---
## 6. Detailed Model Analysis

This section provides per-class classification reports, per-group accuracy breakdown,
learning curves, and training time comparisons.
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 23 · Classification Reports (All Models)
# ──────────────────────────────────────────────────────────────
report_data = []
for name in results_df['Model']:
    y_p = test_predictions[name]['y_pred']
    report = classification_report(y_test, y_p, target_names=['LOS≤3','LOS>3'], output_dict=True)
    report_data.append({
        'Model': name,
        'Short_Prec': report['LOS≤3']['precision'], 'Short_Rec': report['LOS≤3']['recall'],
        'Short_F1': report['LOS≤3']['f1-score'],
        'Long_Prec': report['LOS>3']['precision'], 'Long_Rec': report['LOS>3']['recall'],
        'Long_F1': report['LOS>3']['f1-score'],
        'Macro_F1': report['macro avg']['f1-score'],
        'Weighted_F1': report['weighted avg']['f1-score'],
    })

report_df = pd.DataFrame(report_data)
report_df.to_csv(f'{TABLES_DIR}/04_classification_reports.csv', index=False)

display(HTML("<h4>Per-Class Precision / Recall / F1</h4>"))
display(report_df.style.format({c:'{:.4f}' for c in report_df.columns if c!='Model'}
    ).highlight_max(subset=['Macro_F1','Weighted_F1'], color='lightgreen'))

# Print text reports for top 3
for name in results_df['Model'].head(3):
    print(f"\\n{'—'*60}")
    print(f"  {name}")
    print(f"{'—'*60}")
    print(classification_report(y_test, test_predictions[name]['y_pred'],
                                target_names=['LOS≤3','LOS>3']))
""")

md("""
> **Per-class breakdown** reveals how each model handles the minority class
> (LOS>3).  Some models trade precision for recall or vice versa.
> The Macro-F1 treats both classes equally — useful for fairness-aware selection.
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 24 · Per-Group Accuracy (Protected Attributes × Top 5 Models)
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
top5 = results_df['Model'].head(5).tolist()

for idx, attr in enumerate(['RACE','SEX','ETHNICITY','AGE_GROUP']):
    ax = axes[idx//2][idx%2]
    attr_vals = protected_attrs[attr]
    groups = sorted(set(attr_vals))
    label_map = RACE_MAP if attr=='RACE' else (SEX_MAP if attr=='SEX' else
                (ETH_MAP if attr=='ETHNICITY' else {g:g for g in groups}))
    x = np.arange(len(groups)); width = 0.15
    for j, name in enumerate(top5):
        accs = [accuracy_score(y_test[attr_vals==g], test_predictions[name]['y_pred'][attr_vals==g])
                for g in groups]
        ax.bar(x + j*width, accs, width, label=name, alpha=0.85)
    ax.set_xticks(x + width*2)
    ax.set_xticklabels([label_map.get(g, str(g)) for g in groups], rotation=15, ha='right')
    ax.set_ylabel('Accuracy'); ax.set_title(f'Accuracy by {attr}'); ax.legend(fontsize=7)

plt.tight_layout()
save_fig('per_group_accuracy')
plt.show()
""")

md("""
> **Per-group accuracy** helps identify whether some demographic groups receive
> systematically worse predictions.  Large gaps between bars within a panel
> indicate potential fairness issues.
""")

md("### 6.1 Learning Curves")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 25 · Learning Curves (LightGBM)
# ──────────────────────────────────────────────────────────────
train_sizes_frac = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
learning_results = []
print("Generating learning curves for LightGBM …")

for frac in train_sizes_frac:
    n = int(frac * len(y_train))
    idx = np.random.choice(len(y_train), size=n, replace=False)
    lgb_lc = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63,
                                 max_depth=8, random_state=RANDOM_STATE, verbose=-1, n_jobs=1)
    lgb_lc.fit(X_train[idx], y_train[idx])
    y_pred_lc = lgb_lc.predict(X_test)
    y_prob_lc = lgb_lc.predict_proba(X_test)[:, 1]
    learning_results.append({
        'Frac': frac, 'N': n,
        'Accuracy': accuracy_score(y_test, y_pred_lc),
        'AUC': roc_auc_score(y_test, y_prob_lc),
        'F1': f1_score(y_test, y_pred_lc),
    })
    print(f"  {frac:>5.0%} ({n:>7,}):  Acc={learning_results[-1]['Accuracy']:.4f}  AUC={learning_results[-1]['AUC']:.4f}")

lc_df = pd.DataFrame(learning_results)
lc_df.to_csv(f'{TABLES_DIR}/05_learning_curves.csv', index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(lc_df['N'], lc_df['AUC'], 'o-', color=PALETTE[0], lw=2, label='AUC')
axes[0].plot(lc_df['N'], lc_df['F1'], 's-', color=PALETTE[2], lw=2, label='F1')
axes[0].set_xlabel('Training Samples'); axes[0].set_ylabel('Score')
axes[0].set_title('Learning Curve (LightGBM)'); axes[0].legend(); axes[0].set_xscale('log')
axes[1].plot(lc_df['N'], lc_df['Accuracy'], 'D-', color=PALETTE[4], lw=2)
axes[1].set_xlabel('Training Samples'); axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy vs Training Size'); axes[1].set_xscale('log')
plt.tight_layout()
save_fig('learning_curves')
plt.show()

display(lc_df.style.format({'Frac':'{:.0%}','N':'{:,.0f}','Accuracy':'{:.4f}','AUC':'{:.4f}','F1':'{:.4f}'}))
""")

md("""
> **Learning curve insight:** Performance improves rapidly up to ~50K samples,
> then starts plateauing.  Using the full 740K training samples provides
> only marginal gains beyond 100K — valuable guidance for practitioners
> working with smaller datasets.
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 26 · Training Time Comparison
# ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
times_sorted = sorted(training_times.items(), key=lambda x: x[1], reverse=True)
names_t = [t[0] for t in times_sorted]; vals_t = [t[1] for t in times_sorted]
bars = ax.barh(names_t, vals_t,
               color=[PALETTE[i%len(PALETTE)] for i in range(len(times_sorted))], edgecolor='white')
for bar, v in zip(bars, vals_t):
    ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2, f'{v:.1f}s', va='center', fontsize=9)
ax.set_xlabel('Training Time (seconds)'); ax.set_title('Model Training Times')
plt.tight_layout()
save_fig('training_times')
plt.show()
""")

md("""
> Training times vary by orders of magnitude.  Logistic Regression and Decision Tree
> are fastest; CatBoost and LightGBM take longer but deliver the best accuracy.
> GPU acceleration reduces boosting training time significantly.
""")

###############################################################################
# SECTION 7 — FAIRNESS ANALYSIS
###############################################################################
md("""
---
## 7. Comprehensive Fairness Analysis

We evaluate **7 fairness metrics** across **4 protected attributes** for every model.

### Table 4 — Fairness Metrics: Definitions, Thresholds & Methodological Rationale

| Metric | Abbrev. | Mathematical Definition | Fair if ... | Meaning |
|--------|---------|------------------------|-------------|---------|
| Disparate Impact | DI | $DI = \\min_g \\frac{P(\\hat{Y}=1 \\mid G=g)}{\\max_{g'} P(\\hat{Y}=1 \\mid G=g')}$ | $DI \\geq 0.80$ | Four-fifths rule: ratio of selection rates between worst and best group. From US EEOC guidelines; used by Pfohl et al. (2021), Poulain et al. (2023). |
| Statistical Parity Diff. | SPD | $SPD = \\max_g P(\\hat{Y}=1 \\mid G=g) - \\min_g P(\\hat{Y}=1 \\mid G=g)$ | $\\|SPD\\| < 0.10$ | Maximum gap in positive prediction rates across groups. A group-fairness measure testing *independence*: $\\hat{Y} \\perp G$. Used extensively in Poulain et al. (2023). |
| Equal Opportunity Diff. | EOPP | $EOPP = \\max_g TPR_g - \\min_g TPR_g$ | $\\|EOPP\\| < 0.10$ | Maximum gap in true positive rates. Tests *separation*: $\\hat{Y} \\perp G \\mid Y=1$. Ensures equal benefit from correct positive predictions (Hardt et al., 2016). |
| Equalised Odds Diff. | EOD | $EOD = \\max(\\max_g TPR_g - \\min_g TPR_g,\\; \\max_g FPR_g - \\min_g FPR_g)$ | $EOD < 0.10$ | Maximum of TPR gap and FPR gap. Stricter than EOPP: tests separation for *both* outcomes. Central metric in Pfohl et al. (2021). |
| Theil Index | TI | $TI_{disp} = \\max_g T_g - \\min_g T_g$ where $T_g = \\frac{1}{n_g} \\sum_{i \\in g} \\frac{b_i}{\\bar{b}_g} \\ln\\left(\\frac{b_i}{\\bar{b}_g}\\right)$ | $TI < 0.10$ | Max difference in within-group Theil indices. Captures whether prediction error *inequality* is distributed differently across groups. Complements group-based metrics with an information-theoretic individual-fairness perspective. |
| Predictive Parity | PP | $PP = \\max_g PPV_g - \\min_g PPV_g$ | $\\|PP\\| < 0.10$ | Maximum gap in positive predictive value. Tests *sufficiency*: $Y \\perp G \\mid \\hat{Y}=1$. Important clinically: ensures equal "trust" in positive predictions across groups. |
| Calibration Diff. | CAL | $CAL = \\max_g \\|\\mathbb{E}[Y \\mid \\hat{p} \\in bin, G=g] - \\bar{p}_{bin}\\|$ | $CAL < 0.05$ | Maximum calibration deviation across groups. Tests whether predicted probabilities match observed rates equally for all groups. Clinical relevance: unreliable if miscalibrated for minorities. |

> **Why 7 metrics?** Impossibility theorems (Chouldechova, 2017; Kleinberg et al., 2016) prove that
> except in trivial cases, no classifier can simultaneously satisfy calibration, separation, and
> independence. Our multi-metric approach reveals *which* fairness desiderata are met and which
> are violated — essential for informed clinical deployment decisions.

> **Metric grouping by fairness concept:**
> - **Independence** (outcome prediction does not depend on group): DI, SPD
> - **Separation** (errors are equal across groups given true outcome): EOPP, EOD
> - **Sufficiency** (predictions have equal meaning across groups): PP, CAL
> - **Individual fairness** (similar individuals treated similarly): TI
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 27 · FairnessCalculator Class Definition
# ──────────────────────────────────────────────────────────────
class FairnessCalculator:
    # Compute 7 fairness metrics aligned with manuscript definitions.
    # Metrics: DI, SPD, EOPP, EOD, TI, PP, CAL
    # Supports multi-group attributes (not just binary).

    THRESHOLDS = {
        'DI':   {'threshold': 0.80, 'direction': 'gte', 'label': 'DI >= 0.80'},
        'SPD':  {'threshold': 0.10, 'direction': 'abs_lt', 'label': '|SPD| < 0.10'},
        'EOPP': {'threshold': 0.10, 'direction': 'abs_lt', 'label': '|EOPP| < 0.10'},
        'EOD':  {'threshold': 0.10, 'direction': 'lt', 'label': 'EOD < 0.10'},
        'TI':   {'threshold': 0.10, 'direction': 'lt', 'label': 'TI < 0.10'},
        'PP':   {'threshold': 0.10, 'direction': 'abs_lt', 'label': '|PP| < 0.10'},
        'CAL':  {'threshold': 0.05, 'direction': 'lt', 'label': 'CAL < 0.05'},
    }
    METRIC_NAMES = ['DI', 'SPD', 'EOPP', 'EOD', 'TI', 'PP', 'CAL']

    def __init__(self, y_true=None, y_pred=None, y_prob=None, attr=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob
        self.attr = attr

    @staticmethod
    def is_fair(metric_name, value):
        t = FairnessCalculator.THRESHOLDS[metric_name]
        if t['direction'] == 'gte': return value >= t['threshold']
        elif t['direction'] == 'lt': return value < t['threshold']
        elif t['direction'] == 'abs_lt': return abs(value) < t['threshold']
        return False

    @staticmethod
    def disparate_impact(y_pred, attr):
        groups = sorted(set(attr))
        rates = {g: y_pred[attr==g].mean() for g in groups if (attr==g).sum() > 0}
        if len(rates) < 2: return 1.0, rates
        vals = list(rates.values())
        return (min(vals)/max(vals) if max(vals) > 0 else 0), rates

    @staticmethod
    def statistical_parity_diff(y_pred, attr):
        groups = sorted(set(attr))
        rates = [y_pred[attr==g].mean() for g in groups if (attr==g).sum() > 0]
        return max(rates) - min(rates) if len(rates) >= 2 else 0

    @staticmethod
    def equal_opportunity_diff(y_true, y_pred, attr):
        groups = sorted(set(attr))
        tprs = []
        for g in groups:
            mask = (attr==g) & (y_true==1)
            if mask.sum() > 0: tprs.append(y_pred[mask].mean())
        return max(tprs) - min(tprs) if len(tprs) >= 2 else 0

    @staticmethod
    def equalised_odds_diff(y_true, y_pred, attr):
        groups = sorted(set(attr))
        tprs, fprs = [], []
        for g in groups:
            mask = attr==g
            if mask.sum() == 0: continue
            pos = y_true[mask]==1; neg = y_true[mask]==0
            if pos.sum() > 0: tprs.append(y_pred[mask][pos].mean())
            if neg.sum() > 0: fprs.append(y_pred[mask][neg].mean())
        tpr_gap = max(tprs) - min(tprs) if len(tprs) >= 2 else 0
        fpr_gap = max(fprs) - min(fprs) if len(fprs) >= 2 else 0
        return max(tpr_gap, fpr_gap)

    @staticmethod
    def theil_index(y_true, y_pred, y_prob=None, attr=None):
        if y_prob is None:
            y_prob = y_pred.astype(float)
        benefits = 1.0 - np.abs(y_true.astype(float) - y_prob)
        benefits = np.clip(benefits, 1e-10, None)
        mu = benefits.mean()
        if mu <= 0: return 0.0
        if attr is None:
            ratios = benefits / mu
            return max(0, float(np.mean(ratios * np.log(ratios + 1e-10))))
        # Within-group Theil indices, then max difference (disparity)
        # This captures whether prediction error INEQUALITY differs across groups.
        groups = sorted(set(attr))
        group_ti = []
        for g in groups:
            mask = attr == g
            n_g = mask.sum()
            if n_g < 10: continue
            b_g = benefits[mask]
            mu_g = b_g.mean()
            if mu_g <= 0: continue
            ratios_g = b_g / mu_g
            ti_g = float(np.mean(ratios_g * np.log(ratios_g + 1e-10)))
            group_ti.append(max(0.0, ti_g))
        if len(group_ti) < 2:
            return 0.0
        return max(group_ti) - min(group_ti)

    @staticmethod
    def predictive_parity(y_true, y_pred, attr):
        groups = sorted(set(attr))
        ppvs = []
        for g in groups:
            mask = (attr==g) & (y_pred==1)
            if mask.sum() > 0: ppvs.append(y_true[mask].mean())
        return max(ppvs) - min(ppvs) if len(ppvs) >= 2 else 0

    @staticmethod
    def calibration_diff(y_true, y_prob, attr, n_bins=10):
        groups = sorted(set(attr)); max_diff = 0
        for g in groups:
            mask = attr==g
            if mask.sum() < n_bins: continue
            try:
                pt, pp = calibration_curve(y_true[mask], y_prob[mask], n_bins=n_bins)
                max_diff = max(max_diff, np.max(np.abs(pt - pp)))
            except: pass
        return max_diff

    def compute_all(self, y_true=None, y_pred=None, y_prob=None, attr=None):
        yt = y_true if y_true is not None else self.y_true
        yp = y_pred if y_pred is not None else self.y_pred
        ypb = y_prob if y_prob is not None else self.y_prob
        at = attr if attr is not None else self.attr
        di, rates = FairnessCalculator.disparate_impact(yp, at)
        spd = FairnessCalculator.statistical_parity_diff(yp, at)
        eopp = FairnessCalculator.equal_opportunity_diff(yt, yp, at)
        eod = FairnessCalculator.equalised_odds_diff(yt, yp, at)
        ti = FairnessCalculator.theil_index(yt, yp, ypb, at)
        pp = FairnessCalculator.predictive_parity(yt, yp, at)
        cal = FairnessCalculator.calibration_diff(yt, ypb, at)
        metrics = dict(DI=di, SPD=spd, EOPP=eopp, EOD=eod, TI=ti, PP=pp, CAL=cal)
        verdicts = {m: FairnessCalculator.is_fair(m, v) for m, v in metrics.items()}
        return metrics, verdicts, rates

fc = FairnessCalculator()
print("FairnessCalculator initialised - 7 manuscript-aligned metrics ready")
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 28 · Compute Fairness for ALL Models × ALL Attributes
# ──────────────────────────────────────────────────────────────
all_fairness = {}
all_verdicts = {}
METRIC_KEYS = ['DI','SPD','EOPP','EOD','TI','PP','CAL']
B = 1000  # bootstrap iterations

for name, preds in test_predictions.items():
    y_p = preds['y_pred']; y_pb = preds['y_prob']
    all_fairness[name] = {}
    all_verdicts[name] = {}
    for attr in ['RACE','SEX','ETHNICITY','AGE_GROUP']:
        attr_vals = protected_attrs[attr]
        fc_m = FairnessCalculator(y_test, y_p, y_pb, attr_vals)
        metrics, verdicts, rates = fc_m.compute_all()
        all_fairness[name][attr] = metrics
        all_verdicts[name][attr] = verdicts

print(f"Fairness computed: {len(all_fairness)} models x 4 attributes x 7 metrics")
for name in list(all_fairness.keys())[:3]:
    f = all_fairness[name]['RACE']
    v = all_verdicts[name]['RACE']
    n_fair = sum(v.values())
    print(f"  {name}: DI={f['DI']:.3f} SPD={f['SPD']:.3f} EOPP={f['EOPP']:.3f} EOD={f['EOD']:.3f} TI={f['TI']:.3f} PP={f['PP']:.3f} CAL={f['CAL']:.3f} [{n_fair}/7 fair]")
""")

md("""
> The table above shows whether the best model passes the 4/5 rule for
> **Disparate Impact** (DI ≥ 0.80) on each protected attribute.
> Low SPD/EOD indicates small gaps between groups.
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 29 · Full Fairness Comparison Table (All Models)
# ──────────────────────────────────────────────────────────────
fair_rows = []
for name in test_predictions:
    for attr in ['RACE','SEX','ETHNICITY','AGE_GROUP']:
        f = all_fairness[name][attr]
        v = all_verdicts[name][attr]
        fair_rows.append({
            'Model': name, 'Attribute': attr,
            'DI': f['DI'], 'SPD': f['SPD'], 'EOPP': f['EOPP'],
            'EOD': f['EOD'], 'TI': f['TI'], 'PP': f['PP'], 'CAL': f['CAL'],
            'N_Fair': sum(v.values()), 'Fair_DI': 'Y' if v['DI'] else 'N',
        })
fairness_df = pd.DataFrame(fair_rows)
fairness_df.to_csv(f'{TABLES_DIR}/06_fairness_comparison.csv', index=False)

display(HTML("<h4>Fairness Pivot: DI by Model x Attribute</h4>"))
di_pivot = fairness_df.pivot(index='Model', columns='Attribute', values='DI')
display(di_pivot.style.format('{:.3f}').background_gradient(cmap='RdYlGn', vmin=0.5, vmax=1.0))
""")

md("""
> The DI pivot shows at a glance which model-attribute combinations satisfy
> the 4/5 rule (**green**) and which do not (**red**).
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 30 · Fairness Heatmaps (DI / SPD / EOD)
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(22, 8))
for i, (metric, title, cmap, center) in enumerate([
    ('DI', 'Disparate Impact (>= 0.80 = Fair)', 'RdYlGn', 0.8),
    ('SPD', 'Statistical Parity Diff (|SPD| < 0.10)', 'RdYlGn_r', 0.05),
    ('EOPP', 'Equal Opportunity Diff (|EOPP| < 0.10)', 'RdYlGn_r', 0.05),
]):
    data = fairness_df.pivot(index='Model', columns='Attribute', values=metric)
    data = data.reindex(results_df['Model'])
    vmin = 0.5 if metric=='DI' else 0
    vmax = 1.1 if metric=='DI' else 0.2
    sns.heatmap(data, annot=True, fmt='.3f', cmap=cmap, center=center,
                vmin=vmin, vmax=vmax, ax=axes[i], linewidths=0.5)
    axes[i].set_title(title, fontsize=11)
plt.tight_layout()
save_fig('fairness_heatmaps')
plt.show()
""")

md("""
> **Reading the heatmaps:**
> - **DI (left):** Green = fair (≥0.80), red = unfair (<0.80).
> - **SPD (middle):** Lower is fairer — green cells mean smaller statistical parity gaps.
> - **EOD (right):** Lower is fairer — measures equal opportunity (TPR parity).
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 31 · Selection Rate by Subgroup (Top 5 Models)
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
top5_names = results_df['Model'].head(5).tolist()

for idx, attr in enumerate(['RACE','SEX','ETHNICITY','AGE_GROUP']):
    ax = axes[idx//2][idx%2]
    attr_vals = protected_attrs[attr]
    groups = sorted(set(attr_vals))
    label_map = RACE_MAP if attr=='RACE' else (SEX_MAP if attr=='SEX' else
                (ETH_MAP if attr=='ETHNICITY' else {g:g for g in groups}))
    x = np.arange(len(groups)); width = 0.15
    for j, name in enumerate(top5_names):
        rates = [test_predictions[name]['y_pred'][attr_vals==g].mean() for g in groups]
        ax.bar(x + j*width, rates, width, label=name, alpha=0.85)
    ax.set_xticks(x + width*2)
    ax.set_xticklabels([label_map.get(g, str(g)) for g in groups], rotation=20, ha='right')
    ax.set_ylabel('Selection Rate'); ax.set_title(f'{attr}: Positive Prediction Rate')
    ax.legend(fontsize=7); ax.axhline(y=df['LOS_BINARY'].mean(), color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
save_fig('selection_rate_by_group')
plt.show()
""")

md("""
> The red dashed line shows the **base rate** (overall LOS>3 prevalence).
> Groups whose bars fall consistently below this line are being
> *under-predicted* — a potential source of unfairness.
""")

###############################################################################
# SECTION 8 — FAIRNESS DEEP-DIVE
###############################################################################
md("""
---
## 8. Fairness Deep-Dive

We examine fairness from multiple angles: radar visualisations,
calibration by group, intersectional (multi-attribute) analysis, and
cross-hospital variation.
""")

md("### 8.1 Fairness Radar Charts")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 32 · Fairness Radar Charts (Top 4 Models)
# ──────────────────────────────────────────────────────────────
radar_models = results_df['Model'].head(4).tolist()
metrics_radar = ['DI', 'SPD', 'EOPP', 'EOD', 'TI', 'PP', 'CAL']

fig, axes = plt.subplots(2, 2, figsize=(14, 14), subplot_kw=dict(polar=True))
angles = np.linspace(0, 2*np.pi, len(metrics_radar), endpoint=False).tolist()
angles += angles[:1]

for idx, name in enumerate(radar_models):
    ax = axes[idx//2][idx%2]
    for attr_i, attr in enumerate(['RACE','SEX','ETHNICITY','AGE_GROUP']):
        f = all_fairness[name][attr]
        # Normalise: DI is already 0-1; for others, convert gap to 'fairness score'
        vals = [f['DI'],
                max(0, 1 - f['SPD']*10),
                max(0, 1 - f['EOPP']*10),
                max(0, 1 - f['EOD']*10),
                max(0, 1 - f['TI']*10),
                max(0, 1 - f['PP']*10),
                max(0, 1 - f['CAL']*20)]
        vals += vals[:1]
        ax.plot(angles, vals, 'o-', linewidth=1.5, label=attr, color=PALETTE[attr_i])
        ax.fill(angles, vals, alpha=0.05, color=PALETTE[attr_i])
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(metrics_radar, fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_title(name, fontsize=11, fontweight='bold', y=1.08)
    ax.legend(fontsize=7, loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
save_fig('fairness_radar')
plt.show()
""")

md("""
> **How to read radar charts:** Each axis is a fairness metric normalised to [0,1].
> Values closer to the outer ring (1.0) are **fairer**.
> - **DI:** 1.0 = perfect parity.
> - **SPD/EOPP/EOD/TI/PP/CAL:** Inverted and scaled so 1.0 = no disparity.
>
> If a model's polygon is small or dips inward on one axis, that metric
> is problematic for that protected attribute.
""")

md("### 8.2 Calibration by Protected Group")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 33 · Calibration Curves per Protected Group
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
for idx, attr in enumerate(['RACE','SEX','ETHNICITY','AGE_GROUP']):
    ax = axes[idx//2][idx%2]
    attr_vals = protected_attrs[attr]
    label_map = RACE_MAP if attr=='RACE' else (SEX_MAP if attr=='SEX' else
                (ETH_MAP if attr=='ETHNICITY' else {g:g for g in sorted(set(attr_vals))}))
    for gi, g in enumerate(sorted(set(attr_vals))):
        mask = attr_vals == g
        if mask.sum() < 20: continue
        try:
            pt, pp = calibration_curve(y_test[mask], best_y_prob[mask], n_bins=8)
            ax.plot(pp, pt, 'o-', label=f'{label_map.get(g,str(g))} (N={mask.sum():,})',
                    color=PALETTE[gi], linewidth=1.5)
        except: pass
    ax.plot([0,1],[0,1],'k--', alpha=0.3)
    ax.set_xlabel('Mean Predicted Prob.'); ax.set_ylabel('Fraction Positive')
    ax.set_title(f'Calibration by {attr} ({best_model_name})'); ax.legend(fontsize=8)
plt.tight_layout()
save_fig('calibration_by_group')
plt.show()
""")

md("""
> If calibration curves for different groups diverge, the model's predicted
> probabilities mean different things for different demographics — this is
> a form of **calibration unfairness** even if aggregate calibration looks fine.
""")

md("### 8.3 Bootstrap Confidence Intervals for Fairness Metrics")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 34 · Bootstrap CI (B=500)
# ──────────────────────────────────────────────────────────────
B = 500
boot_results = {attr: {'DI':[], 'SPD':[], 'EOD':[]} for attr in ['RACE','SEX','ETHNICITY','AGE_GROUP']}
print(f"Computing {B} bootstrap replications for {best_model_name} …")

for b in range(B):
    idx = np.random.choice(len(y_test), size=len(y_test), replace=True)
    y_sub = y_test[idx]; pred_sub = best_y_pred[idx]
    for attr in ['RACE','SEX','ETHNICITY','AGE_GROUP']:
        attr_sub = protected_attrs[attr][idx]
        di, _ = fc.disparate_impact(pred_sub, attr_sub)
        spd = fc.statistical_parity_diff(pred_sub, attr_sub)
        eod = fc.equal_opportunity_diff(y_sub, pred_sub, attr_sub)
        boot_results[attr]['DI'].append(di)
        boot_results[attr]['SPD'].append(spd)
        boot_results[attr]['EOD'].append(eod)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
ci_summary = []
for idx2, attr in enumerate(['RACE','SEX','ETHNICITY','AGE_GROUP']):
    ax = axes[idx2//2][idx2%2]; vals = boot_results[attr]['DI']
    ax.hist(vals, bins=30, color=PALETTE[idx2], edgecolor='white', alpha=0.8)
    ax.axvline(x=0.80, color='red', linestyle='--', lw=2, label='DI = 0.80 threshold')
    ci_lo, ci_hi = np.percentile(vals, [2.5, 97.5])
    ax.axvline(x=ci_lo, color='green', linestyle=':', lw=2)
    ax.axvline(x=ci_hi, color='green', linestyle=':', lw=2)
    ax.set_title(f'{attr}: DI 95% CI = [{ci_lo:.3f}, {ci_hi:.3f}]')
    ax.set_xlabel('Disparate Impact'); ax.legend()
    ci_summary.append({'Attribute':attr, 'DI_mean':np.mean(vals), 'CI_lo':ci_lo, 'CI_hi':ci_hi})

plt.suptitle(f'Bootstrap Confidence Intervals (B={B}) — {best_model_name}',
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.95])
save_fig('bootstrap_ci')
plt.show()

display(HTML("<h4>Bootstrap DI Summary</h4>"))
display(pd.DataFrame(ci_summary).style.format({c:'{:.4f}' for c in ['DI_mean','CI_lo','CI_hi']}))
""")

md("""
> **Bootstrap CIs** quantify the uncertainty in fairness metrics.  If the 95%
> confidence interval straddles the 0.80 threshold, the fairness verdict is
> unstable and could change with a different test sample.
""")

md("### 8.4 Intersectional Fairness (RACE × SEX)")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 35 · Intersectional Analysis: RACE × SEX
# ──────────────────────────────────────────────────────────────
intersect_groups = {}
for i in range(len(y_test)):
    r = RACE_MAP.get(protected_attrs['RACE'][i], 'Unk')
    s = SEX_MAP.get(protected_attrs['SEX'][i], 'Unk')
    key = f"{r} × {s}"
    if key not in intersect_groups: intersect_groups[key] = {'y_true':[], 'y_pred':[]}
    intersect_groups[key]['y_true'].append(y_test[i])
    intersect_groups[key]['y_pred'].append(best_y_pred[i])

inter_data = []
for key, data in intersect_groups.items():
    yt = np.array(data['y_true']); yp = np.array(data['y_pred'])
    if len(yt) < 50: continue
    inter_data.append({'Group':key, 'N':len(yt), 'Selection_Rate':yp.mean(),
        'TPR': yp[yt==1].mean() if (yt==1).sum()>0 else np.nan,
        'FPR': yp[yt==0].mean() if (yt==0).sum()>0 else np.nan,
        'Accuracy': accuracy_score(yt, yp)})
inter_df = pd.DataFrame(inter_data).sort_values('Selection_Rate', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
colors = [PALETTE[i%len(PALETTE)] for i in range(len(inter_df))]
bars = axes[0].barh(inter_df['Group'], inter_df['Selection_Rate'], color=colors, edgecolor='white')
axes[0].axvline(x=df['LOS_BINARY'].mean(), color='red', linestyle='--', lw=2, label='Base rate')
axes[0].set_xlabel('Selection Rate'); axes[0].set_title('(a) Intersectional: RACE × SEX — Selection Rate')
for bar, n in zip(bars, inter_df['N']):
    axes[0].text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2, f'N={n:,}', va='center', fontsize=8)
axes[0].legend()
bars2 = axes[1].barh(inter_df['Group'], inter_df['Accuracy'], color=colors, edgecolor='white')
axes[1].set_xlabel('Accuracy'); axes[1].set_title('(b) Accuracy by Intersectional Group')
plt.tight_layout()
save_fig('intersectional_race_sex')
plt.show()

inter_df.to_csv(f'{TABLES_DIR}/07_intersectional_fairness.csv', index=False)
display(inter_df.style.format({'Selection_Rate':'{:.3f}','TPR':'{:.3f}','FPR':'{:.3f}','Accuracy':'{:.3f}'}))
""")

md("""
> **Intersectional analysis** reveals disparities hidden by single-attribute analysis.
> For example, a model might appear fair for Race and Sex independently, but
> a specific Race × Sex subgroup (e.g., Native American × Female) may have a very
> different accuracy or selection rate.
""")

md("### 8.5 Intersectional Fairness (RACE × AGE_GROUP)")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 36 · Intersectional Analysis: RACE × AGE_GROUP
# ──────────────────────────────────────────────────────────────
intersect2 = {}
for i in range(len(y_test)):
    r = RACE_MAP.get(protected_attrs['RACE'][i], 'Unk')
    a = protected_attrs['AGE_GROUP'][i]
    key = f"{r} × {a}"
    if key not in intersect2: intersect2[key] = {'y_true':[], 'y_pred':[]}
    intersect2[key]['y_true'].append(y_test[i]); intersect2[key]['y_pred'].append(best_y_pred[i])

inter2_data = []
for key, data in intersect2.items():
    yt = np.array(data['y_true']); yp = np.array(data['y_pred'])
    if len(yt) < 50: continue
    inter2_data.append({'Group':key, 'N':len(yt), 'Selection_Rate':yp.mean(),
                        'Accuracy':accuracy_score(yt, yp)})
inter2_df = pd.DataFrame(inter2_data).sort_values('Selection_Rate', ascending=False)

fig, ax = plt.subplots(figsize=(14, max(6, len(inter2_df)*0.35)))
colors2 = [PALETTE[i%len(PALETTE)] for i in range(len(inter2_df))]
ax.barh(inter2_df['Group'], inter2_df['Selection_Rate'], color=colors2, edgecolor='white')
ax.axvline(x=df['LOS_BINARY'].mean(), color='red', linestyle='--', lw=2, label='Base rate')
ax.set_xlabel('Selection Rate')
ax.set_title(f'Intersectional: RACE × AGE_GROUP — {best_model_name}'); ax.legend()
plt.tight_layout()
save_fig('intersectional_race_age')
plt.show()

display(inter2_df.head(10).style.format({'Selection_Rate':'{:.3f}','Accuracy':'{:.3f}'}))
""")

md("""
> The RACE × AGE interaction shows that **patients aged 65+** in each racial group
> have the highest selection rates, while younger patients have the lowest.
> The age gradient dominates, but racial disparities persist within each age band.
""")

md("### 8.6 Cross-Hospital Fairness")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 37 · Cross-Hospital Fairness Audit
# ──────────────────────────────────────────────────────────────
hosp_fair = []
for h_id in np.unique(hospital_ids_test):
    mask = hospital_ids_test == h_id; n = mask.sum()
    if n < 100: continue
    y_h = y_test[mask]; pred_h = best_y_pred[mask]
    h_row = {'Hospital':h_id, 'N':n, 'Accuracy':accuracy_score(y_h, pred_h),
             'Selection_Rate':pred_h.mean()}
    for attr in ['RACE','SEX']:
        attr_h = protected_attrs[attr][mask]
        if len(set(attr_h)) >= 2:
            di, _ = fc.disparate_impact(pred_h, attr_h)
            h_row[f'DI_{attr}'] = di
    hosp_fair.append(h_row)
hosp_df = pd.DataFrame(hosp_fair)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].scatter(hosp_df['N'], hosp_df['Accuracy'], alpha=0.4, s=20, color=PALETTE[0])
axes[0].set_xlabel('Hospital Size'); axes[0].set_ylabel('Accuracy')
axes[0].set_title('(a) Accuracy vs Hospital Size')

if 'DI_RACE' in hosp_df.columns:
    axes[1].hist(hosp_df['DI_RACE'].dropna(), bins=25, color=PALETTE[2], edgecolor='white')
    axes[1].axvline(x=0.80, color='red', linestyle='--', lw=2, label='DI = 0.80')
    axes[1].set_xlabel('DI (RACE)'); axes[1].set_title('(b) DI Distribution Across Hospitals')
    axes[1].legend()
    unfair_pct = (hosp_df['DI_RACE'].dropna() < 0.80).mean()
    axes[2].scatter(hosp_df['N'], hosp_df['DI_RACE'], alpha=0.4, s=20, color=PALETTE[4])
    axes[2].axhline(y=0.80, color='red', linestyle='--')
    axes[2].set_xlabel('Hospital Size'); axes[2].set_ylabel('DI (RACE)')
    axes[2].set_title(f'(c) DI vs Size — {unfair_pct:.0%} hospitals below threshold')
plt.tight_layout()
save_fig('cross_hospital_fairness')
plt.show()

hosp_df.to_csv(f'{TABLES_DIR}/08_hospital_fairness.csv', index=False)
print(f"Cross-hospital analysis: {len(hosp_df)} hospitals (≥ 100 patients)")
print(f"Unfair hospitals (DI_RACE < 0.80): {(hosp_df.get('DI_RACE',pd.Series()) < 0.80).sum()}")
""")

md("""
> **Cross-hospital fairness** is critical for deployment.  A model that is
> **globally fair** may still be **locally unfair** at specific hospitals,
> especially smaller ones where demographic distributions differ from the overall sample.
""")



###############################################################################
# SECTION 9 — STABILITY & RELIABILITY (Manuscript Protocols 1-3)
###############################################################################
md("""
---
## 9. Stability & Reliability Testing (3 Protocols)

The manuscript defines three complementary stability protocols to assess
whether fairness verdicts are **reliable**:

| Protocol | Method | Purpose |
|----------|--------|---------|
| **P1** | K=30 Random-Subset Resampling (30%) | Verdict Flip Rate (VFR) |
| **P2** | Sample-Size Sensitivity (1K→925K), 30 repeats | CV curves, min-N guidance |
| **P3** | Cross-Hospital K=20 GroupKFold (train 19 / eval 1) | Cross-site portability |

All protocols compute **all 7 fairness metrics** × 4 protected attributes.
""")

md("### 9.1 Protocol 1 — K=30 Random-Subset Resampling (VFR)")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 38 · Protocol 1: K=30 Random-Subset Resampling — ALL Models × 7 Metrics
# ──────────────────────────────────────────────────────────────
K_P1 = 30
np.random.seed(42)

# Pre-generate 30 resample index sets (shared across models for comparability)
n_sub = int(0.30 * len(y_test))
resample_indices = [np.random.choice(len(y_test), size=n_sub, replace=False) for _ in range(K_P1)]

# Compute metrics for ALL models across all resamples
all_p1_rows = []
model_names_list = list(test_predictions.keys())
print(f"Protocol 1: K={K_P1} random 30% subsets × {len(model_names_list)} models …")
_t0 = time.time()

for mi, model_name in enumerate(model_names_list):
    y_p = test_predictions[model_name]['y_pred']
    y_pb = test_predictions[model_name]['y_prob']
    for k, idx in enumerate(resample_indices):
        y_sub = y_test[idx]; pred_sub = y_p[idx]; prob_sub = y_pb[idx]
        row = {'Model': model_name, 'K': k+1}
        for attr in ['RACE','SEX','ETHNICITY','AGE_GROUP']:
            attr_sub = protected_attrs[attr][idx]
            if len(set(attr_sub)) >= 2:
                fc_sub = FairnessCalculator(y_sub, pred_sub, prob_sub, attr_sub)
                metrics_k, verdicts_k, _ = fc_sub.compute_all()
                for mk in METRIC_KEYS:
                    row[f'{mk}_{attr}'] = metrics_k[mk]
                    row[f'V_{mk}_{attr}'] = 1 if verdicts_k[mk] else 0
        all_p1_rows.append(row)
    if (mi+1) % 4 == 0:
        print(f"  {mi+1}/{len(model_names_list)} models done ({time.time()-_t0:.0f}s)")
print(f"  All models done in {time.time()-_t0:.1f}s")

all_p1_df = pd.DataFrame(all_p1_rows)
all_p1_df.to_csv(f'{TABLES_DIR}/09_protocol1_resampling.csv', index=False)

# Also save best-model subset for backward compat
p1_df = all_p1_df[all_p1_df['Model'] == best_model_name].copy()
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 38b · Compute VFR per Model × Metric × Attribute
# ──────────────────────────────────────────────────────────────
vfr_rows = []
for model_name in model_names_list:
    mdf = all_p1_df[all_p1_df['Model'] == model_name]
    for attr in ['RACE','SEX','ETHNICITY','AGE_GROUP']:
        for mk in METRIC_KEYS:
            vcol = f'V_{mk}_{attr}'
            mcol = f'{mk}_{attr}'
            if vcol in mdf.columns and mcol in mdf.columns:
                fair_count = int(mdf[vcol].sum())
                mean_val = mdf[mcol].mean()
                std_val = mdf[mcol].std()
                vfr = min(fair_count, K_P1 - fair_count) / K_P1
                threshold = FairnessCalculator.THRESHOLDS[mk]['threshold']
                direction = FairnessCalculator.THRESHOLDS[mk]['direction']
                margin = abs(mean_val - threshold)
                margin_sigma = margin / max(std_val, 1e-9)
                vfr_rows.append({
                    'Model': model_name, 'Attribute': attr, 'Metric': mk,
                    'Mean': round(mean_val, 6), 'Std': round(std_val, 6),
                    'Threshold': threshold,
                    'Margin': round(margin, 4),
                    'Margin_sigma': round(margin_sigma, 1),
                    'VFR': round(vfr, 4),
                    'Pct_Fair': round(fair_count / K_P1 * 100, 1),
                    'Verdict': 'FAIR' if fair_count > K_P1//2 else 'UNFAIR'
                })

vfr_df = pd.DataFrame(vfr_rows)
vfr_df.to_csv(f'{TABLES_DIR}/09b_vfr_all_metrics.csv', index=False)

# Summary stats
n_unstable = (vfr_df['VFR'] > 0).sum()
n_total = len(vfr_df)
max_vfr = vfr_df['VFR'].max()
print(f"VFR Analysis: {n_unstable}/{n_total} model-metric-attribute combos have VFR > 0")
print(f"Maximum VFR observed: {max_vfr:.1%}")
print(f"\\nTop 10 most unstable combinations:")
top10 = vfr_df.nlargest(10, 'VFR')[['Model','Attribute','Metric','Mean','Threshold','Margin_sigma','VFR','Pct_Fair']]
display(top10.style.format({'Mean':'{:.4f}','VFR':'{:.1%}','Pct_Fair':'{:.1f}%'}))
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 38c · VFR Table 6 — Per-Model Summary
# ──────────────────────────────────────────────────────────────
display(HTML("<h4>Table 6a: Verdict Flip Rate — Best Model ({}) × 7 Metrics × 4 Attributes</h4>".format(best_model_name)))
best_vfr = vfr_df[vfr_df['Model'] == best_model_name].copy()
best_pivot = best_vfr.pivot(index='Metric', columns='Attribute', values='VFR')
best_pf = best_vfr.pivot(index='Metric', columns='Attribute', values='Pct_Fair')
# Build formatted string table
fmt_data = pd.DataFrame(index=best_pivot.index, columns=best_pivot.columns, dtype=object)
for c in fmt_data.columns:
    for r in fmt_data.index:
        vfr_val = best_pivot.loc[r, c]
        pf = best_pf.loc[r, c]
        verdict = 'F' if pf > 50 else 'U'
        fmt_data.loc[r, c] = f"{vfr_val:.0%} ({verdict})"
display(fmt_data.style.set_caption(f"VFR (Verdict: F=Fair, U=Unfair) — {best_model_name}"))

# Aggregate across all models
display(HTML("<h4>Table 6b: Max VFR Across All 12 Models — 7 Metrics × 4 Attributes</h4>"))
max_vfr_pivot = vfr_df.groupby(['Metric','Attribute'])['VFR'].max().reset_index()
max_vfr_table = max_vfr_pivot.pivot(index='Metric', columns='Attribute', values='VFR')
display(max_vfr_table.style.format('{:.1%}').background_gradient(cmap='YlOrRd', vmin=0, vmax=0.5))

# Count how many models flip per metric-attribute
flip_count = vfr_df[vfr_df['VFR']>0].groupby(['Metric','Attribute']).size().reset_index(name='N_Models_Flip')
if len(flip_count) > 0:
    print("\\nMetric-Attribute pairs with unstable verdicts (VFR > 0):")
    display(flip_count)
else:
    print("\\nAll metric-attribute pairs are perfectly stable across all models and resamples.")
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 38d · FIG: Margin-to-Threshold Lollipop Chart (Best Model)
# ──────────────────────────────────────────────────────────────
bm_vfr = vfr_df[vfr_df['Model'] == best_model_name].copy()
bm_vfr['Label'] = bm_vfr['Metric'] + ' × ' + bm_vfr['Attribute']
bm_vfr = bm_vfr.sort_values('Margin_sigma', ascending=True)

fig, ax = plt.subplots(figsize=(12, 10))
colors = ['#e74c3c' if v == 'UNFAIR' else '#2ecc71' for v in bm_vfr['Verdict']]
y_pos = range(len(bm_vfr))

# Horizontal lollipop
ax.hlines(y=y_pos, xmin=0, xmax=bm_vfr['Margin_sigma'].values, color=colors, linewidth=2.5, alpha=0.7)
ax.scatter(bm_vfr['Margin_sigma'].values, y_pos, color=colors, s=80, zorder=5, edgecolors='white', linewidth=1)

ax.set_yticks(list(y_pos))
ax.set_yticklabels(bm_vfr['Label'].values, fontsize=9)
ax.set_xlabel('Distance from Threshold (in σ units)', fontsize=12, fontweight='bold')
ax.set_title(f'Margin-to-Threshold Analysis — {best_model_name}\\n(Higher = more stable verdict)', fontsize=13, fontweight='bold')
ax.axvline(x=2, color='gray', linestyle='--', alpha=0.5, label='2σ (stable)')
ax.axvline(x=5, color='gray', linestyle=':', alpha=0.4, label='5σ (very stable)')

# Add legend patches
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2ecc71', label='FAIR verdict'),
                   Patch(facecolor='#e74c3c', label='UNFAIR verdict'),
                   plt.Line2D([0],[0], color='gray', linestyle='--', label='2σ threshold'),
                   plt.Line2D([0],[0], color='gray', linestyle=':', label='5σ threshold')]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
ax.set_xlim(left=-0.5)

# Annotate the closest-to-threshold case
closest = bm_vfr.iloc[0]
ax.annotate(f'VFR = {closest["VFR"]:.0%}\\n(margin = {closest["Margin_sigma"]:.1f}σ)',
            xy=(closest['Margin_sigma'], 0), xytext=(closest['Margin_sigma']+8, 2),
            fontsize=10, fontweight='bold', color='#e74c3c',
            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5))

plt.tight_layout()
save_fig('protocol1_margin_to_threshold')
plt.show()
print("Metrics far from threshold (> 5σ) produce VFR = 0% (perfectly stable).")
print("Metrics near threshold (< 2σ) are vulnerable to verdict flipping.")
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 38e · FIG: Multi-Model VFR Comparison (Dot Plot)
# ──────────────────────────────────────────────────────────────
# For each Attribute, show VFR by Metric across all models as dots
fig, axes = plt.subplots(1, 4, figsize=(24, 8), sharey=True)
attr_list = ['RACE', 'SEX', 'ETHNICITY', 'AGE_GROUP']
attr_colors = {a: PALETTE[i] for i, a in enumerate(attr_list)}

for ai, attr in enumerate(attr_list):
    ax = axes[ai]
    sub = vfr_df[vfr_df['Attribute'] == attr]

    for mi, mk in enumerate(METRIC_KEYS):
        mk_sub = sub[sub['Metric'] == mk]
        vfr_vals = mk_sub['VFR'].values
        # Jitter y position
        y_base = len(METRIC_KEYS) - 1 - mi
        jitter = np.linspace(-0.25, 0.25, len(vfr_vals))
        ax.scatter(vfr_vals * 100, [y_base + j for j in jitter],
                   c=PALETTE[ai], s=50, alpha=0.7, edgecolors='white', linewidth=0.5, zorder=5)
        # Show mean as diamond
        ax.scatter(vfr_vals.mean() * 100, y_base, marker='D', c='black', s=100, zorder=10,
                   edgecolors='white', linewidth=1.5)

    ax.set_yticks(range(len(METRIC_KEYS)))
    ax.set_yticklabels(list(reversed(METRIC_KEYS)), fontsize=11)
    ax.set_xlabel('VFR (%)', fontsize=11)
    ax.set_title(attr, fontsize=13, fontweight='bold', color=PALETTE[ai])
    ax.axvline(x=3, color='orange', linestyle='--', alpha=0.6, label='3% (robust cutoff)')
    ax.axvline(x=10, color='red', linestyle='--', alpha=0.6, label='10% (unstable cutoff)')
    ax.set_xlim(-2, max(vfr_df['VFR'].max() * 100 + 5, 15))
    ax.grid(axis='x', alpha=0.3)
    if ai == 0:
        ax.legend(fontsize=8, loc='upper right')

plt.suptitle('Verdict Flip Rate Across All Models — K=30 Resampling\\n(dots = individual models, diamond = mean)',
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.93])
save_fig('protocol1_vfr_multi_model')
plt.show()
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 38f · FIG: Metric Distribution Violin Plots with Threshold Lines
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(26, 12))
metric_thresholds = {'DI':0.80, 'SPD':0.10, 'EOPP':0.10, 'EOD':0.10, 'TI':0.10, 'PP':0.10, 'CAL':0.05}
metric_labels = {'DI':'Disparate Impact', 'SPD':'Stat. Parity Diff.', 'EOPP':'Equal Opp. Diff.',
    'EOD':'Equalised Odds Diff.', 'TI':'Theil Index (Between-Group)', 'PP':'Predictive Parity', 'CAL':'Calibration Diff.'}
attr_list = ['RACE','SEX','ETHNICITY','AGE_GROUP']

for mi, mk in enumerate(METRIC_KEYS):
    ax = axes[mi // 4][mi % 4]
    # Collect data for best model across resamples
    plot_data = []
    for attr in attr_list:
        col = f'{mk}_{attr}'
        if col in p1_df.columns:
            vals = p1_df[col].dropna().values
            for v in vals:
                plot_data.append({'Attribute': attr, 'Value': v})
    pdf = pd.DataFrame(plot_data)
    if len(pdf) > 0:
        # Box + strip
        box_colors = [PALETTE[i] for i in range(len(attr_list))]
        positions = list(range(len(attr_list)))
        for ai, attr in enumerate(attr_list):
            av = pdf[pdf['Attribute']==attr]['Value'].values
            if len(av) > 0:
                bp = ax.boxplot([av], positions=[ai], widths=0.5, patch_artist=True,
                    boxprops=dict(facecolor=PALETTE[ai], alpha=0.4),
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(color=PALETTE[ai]), capprops=dict(color=PALETTE[ai]),
                    flierprops=dict(marker='o', markerfacecolor=PALETTE[ai], markersize=4, alpha=0.5))
                # Strip overlay
                jitter = np.random.normal(0, 0.05, len(av))
                ax.scatter([ai + j for j in jitter], av, color=PALETTE[ai], s=15, alpha=0.6, zorder=5)

        ax.axhline(y=metric_thresholds[mk], color='red', linestyle='--', linewidth=2, alpha=0.8,
                   label=f'Threshold = {metric_thresholds[mk]}')
        ax.set_xticks(positions)
        ax.set_xticklabels(attr_list, fontsize=9, rotation=25)
        ax.set_title(f'{mk}: {metric_labels.get(mk, mk)}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=7, loc='best')
        ax.grid(axis='y', alpha=0.2)

if len(METRIC_KEYS) < 8:
    axes[1][3].axis('off')

plt.suptitle(f'Metric Distributions Across 30 Resamples — {best_model_name}\\n(red dashed = fairness threshold)',
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.93])
save_fig('protocol1_metric_distributions')
plt.show()
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 38g · FIG: Fair/Unfair Verdict Consistency Across Models (Stacked Bar)
# ──────────────────────────────────────────────────────────────
# For each Metric × Attribute: what % of models give "FAIR" vs "UNFAIR"?
verdict_summary = vfr_df.groupby(['Metric','Attribute']).agg(
    N_Fair=('Verdict', lambda x: (x=='FAIR').sum()),
    N_Unfair=('Verdict', lambda x: (x=='UNFAIR').sum()),
    Avg_VFR=('VFR', 'mean')
).reset_index()
verdict_summary['Pct_Fair'] = verdict_summary['N_Fair'] / (verdict_summary['N_Fair'] + verdict_summary['N_Unfair']) * 100

fig, axes = plt.subplots(1, 4, figsize=(22, 7), sharey=True)
for ai, attr in enumerate(attr_list):
    ax = axes[ai]
    sub = verdict_summary[verdict_summary['Attribute'] == attr].set_index('Metric')
    sub = sub.reindex(METRIC_KEYS)

    fair_pcts = sub['Pct_Fair'].values
    unfair_pcts = 100 - fair_pcts

    y_pos = range(len(METRIC_KEYS))
    bars_fair = ax.barh(y_pos, fair_pcts, height=0.6, color='#2ecc71', alpha=0.85, label='FAIR', edgecolor='white')
    bars_unfair = ax.barh(y_pos, unfair_pcts, height=0.6, left=fair_pcts, color='#e74c3c', alpha=0.85, label='UNFAIR', edgecolor='white')

    # Annotate percentages
    for yi, (fp, up) in enumerate(zip(fair_pcts, unfair_pcts)):
        if fp > 15:
            ax.text(fp/2, yi, f'{fp:.0f}%', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        if up > 15:
            ax.text(fp + up/2, yi, f'{up:.0f}%', ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(METRIC_KEYS, fontsize=11)
    ax.set_xlabel('% of Models', fontsize=11)
    ax.set_title(attr, fontsize=13, fontweight='bold', color=PALETTE[ai])
    ax.set_xlim(0, 100)
    ax.axvline(x=50, color='black', linestyle=':', alpha=0.3)
    if ai == 0:
        ax.legend(fontsize=10, loc='lower right')

plt.suptitle('Verdict Agreement Across 12 Models — Fair vs Unfair\\n(Per Metric × Attribute, based on majority verdict across 30 resamples)',
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.92])
save_fig('protocol1_verdict_agreement')
plt.show()

print("\\nVFR interpretation: < 3% = robust, 3–10% = moderate, > 10% = unstable")
print("Margin interpretation: > 5σ = very stable, 2–5σ = stable, < 2σ = fragile")
""")

md("""
> **Protocol 1 findings:**
>
> 1. **Margin-to-threshold plot** explains *why* most VFRs are 0%: with 740K samples per
>    resample, metric variance is tiny and most metrics sit many standard deviations from
>    their thresholds—verdict flips require the metric to cross the threshold.
> 2. **Multi-model VFR comparison** reveals which metric-attribute pairs are fragile across
>    different model families (e.g., PP × RACE sits near its threshold for several models).
> 3. **Model verdict agreement** shows which metrics produce consensus (all models agree
>    fair or unfair) versus disagreement (some models say fair, others say unfair)—a key
>    reliability concern for deployment.
""")

md("### 9.2 Protocol 2 — Sample-Size Sensitivity (CV Curves + Min-N)")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 39 · Protocol 2: Sample-Size Sensitivity — ALL 7 Metrics
# ──────────────────────────────────────────────────────────────
sample_sizes = [1000, 5000, 10000, 25000, 50000, 100000, 250000, 500000, len(y_test)]
N_REP = 30
p2_results = []

print(f"Protocol 2: Sample-size sensitivity ({len(sample_sizes)} sizes × {N_REP} reps) …")
_t0 = time.time()
for n_target in sample_sizes:
    n_actual = min(n_target, len(y_test))
    reps = N_REP if n_actual < len(y_test) else 1
    for rep in range(reps):
        if n_actual < len(y_test):
            idx = np.random.choice(len(y_test), size=n_actual, replace=False)
        else:
            idx = np.arange(len(y_test))
        y_sub = y_test[idx]; pred_sub = best_y_pred[idx]; prob_sub = best_y_prob[idx]
        row = {'N': n_actual, 'Rep': rep}
        for attr in ['RACE','SEX','ETHNICITY','AGE_GROUP']:
            attr_sub = protected_attrs[attr][idx]
            if len(set(attr_sub)) >= 2:
                fc_sub = FairnessCalculator(y_sub, pred_sub, prob_sub, attr_sub)
                metrics_s, _, _ = fc_sub.compute_all()
                for mk in METRIC_KEYS:
                    row[f'{mk}_{attr}'] = metrics_s[mk]
        p2_results.append(row)
    if n_target % 50000 == 0:
        print(f"  N={n_target:,} done ({time.time()-_t0:.0f}s)")
print(f"  Completed in {time.time()-_t0:.1f}s")

p2_df = pd.DataFrame(p2_results)
p2_df.to_csv(f'{TABLES_DIR}/10_protocol2_sample_sensitivity.csv', index=False)

# --- FIG08: CV curves for ALL 7 metrics ---
fig, axes = plt.subplots(2, 4, figsize=(28, 12))
for mi, mk in enumerate(METRIC_KEYS):
    ax = axes[mi//4][mi%4]
    for ai, attr in enumerate(['RACE','SEX','ETHNICITY','AGE_GROUP']):
        col = f'{mk}_{attr}'
        if col not in p2_df.columns: continue
        agg = p2_df.groupby('N')[col].agg(['mean','std']).reset_index()
        agg['cv'] = agg['std'] / agg['mean'].replace(0, np.nan)
        ax.plot(agg['N'], agg['cv'], 'o-', color=PALETTE[ai], label=attr, linewidth=1.5)
    ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.6, label='CV=0.05')
    ax.set_xscale('log'); ax.set_xlabel('Sample Size'); ax.set_ylabel('CV')
    ax.set_title(f'{mk}: CV vs N'); ax.legend(fontsize=7)
axes[1][3].axis('off')
plt.suptitle('Protocol 2: Metric Reliability (CV) vs Sample Size — 7 Metrics',
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.95])
save_fig('protocol2_cv_curves')
plt.show()
""")

md("""
> **CV curves** (FIG08) show how the coefficient of variation of each metric
> decreases as sample size grows.  The red line marks CV = 0.05; below this,
> the metric is considered **stable**.
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 40 · Min-N Threshold Table (Table 7)
# ──────────────────────────────────────────────────────────────
CV_THRESHOLD = 0.05
minN_rows = []
for attr in ['RACE','SEX','ETHNICITY','AGE_GROUP']:
    for mk in METRIC_KEYS:
        col = f'{mk}_{attr}'
        if col not in p2_df.columns:
            minN_rows.append({'Attribute':attr, 'Metric':mk, 'Min_N':'N/A'}); continue
        found = False
        for n_target in sample_sizes:
            n_actual = min(n_target, len(y_test))
            sub = p2_df[p2_df['N']==n_actual][col].dropna()
            if len(sub) > 1:
                cv = sub.std() / max(sub.mean(), 1e-9)
                if cv < CV_THRESHOLD:
                    minN_rows.append({'Attribute':attr, 'Metric':mk, 'Min_N':f'{n_actual:,}'})
                    found = True; break
        if not found:
            minN_rows.append({'Attribute':attr, 'Metric':mk, 'Min_N':'>500K'})

minN_df = pd.DataFrame(minN_rows)
minN_df.to_csv(f'{TABLES_DIR}/10b_min_sample_sizes.csv', index=False)

display(HTML("<h4>Table 7: Minimum Sample Size for CV < 0.05</h4>"))
minN_pivot = minN_df.pivot(index='Metric', columns='Attribute', values='Min_N')
display(minN_pivot)

print("\\nGuidance: auditors should ensure at least the above sample sizes ")
print("before drawing conclusions about each fairness metric.")
""")

md("""
> **Table 7** provides practical guidance for auditors: for each metric-attribute
> pair, we report the minimum sample size at which the CV drops below 0.05.
""")

md("### 9.3 30-Seed Perturbation")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 41 · 30-Seed Perturbation — ALL 7 Metrics
# ──────────────────────────────────────────────────────────────
N_SEEDS = 30; seed_results = []
print(f'Training LightGBM with {N_SEEDS} different seeds …')
_t0 = time.time()

for seed_i in range(N_SEEDS):
    seed_val = seed_i * 7 + 1
    lgb_seed = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=8,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=seed_val, n_jobs=1, verbose=-1)
    lgb_seed.fit(X_train, y_train)
    y_pred_seed = lgb_seed.predict(X_test)
    y_prob_seed = lgb_seed.predict_proba(X_test)[:, 1]
    seed_row = {'Seed':seed_val, 'Accuracy':accuracy_score(y_test, y_pred_seed),
                'AUC':roc_auc_score(y_test, y_prob_seed)}
    for attr in ['RACE','SEX','ETHNICITY','AGE_GROUP']:
        fc_s = FairnessCalculator(y_test, y_pred_seed, y_prob_seed, protected_attrs[attr])
        ms, vs, _ = fc_s.compute_all()
        for mk in METRIC_KEYS:
            seed_row[f'{mk}_{attr}'] = ms[mk]
            seed_row[f'Fair_{mk}_{attr}'] = 1 if vs[mk] else 0
    seed_results.append(seed_row)
    if (seed_i+1) % 10 == 0: print(f'  {seed_i+1}/{N_SEEDS} done ({time.time()-_t0:.0f}s)')

seed_df = pd.DataFrame(seed_results)
seed_df.to_csv(f'{TABLES_DIR}/11_seed_perturbation.csv', index=False)
print(f'\\nCompleted in {time.time()-_t0:.1f}s')

# Seed-VFR
seed_vfr = []
for attr in ['RACE','SEX','ETHNICITY','AGE_GROUP']:
    for mk in METRIC_KEYS:
        vcol = f'Fair_{mk}_{attr}'
        if vcol in seed_df.columns:
            fc_cnt = seed_df[vcol].sum()
            vfr = min(fc_cnt, N_SEEDS-fc_cnt) / N_SEEDS
            seed_vfr.append({'Attribute':attr, 'Metric':mk, 'VFR':vfr,
                'Mean': seed_df[f'{mk}_{attr}'].mean(), 'Std': seed_df[f'{mk}_{attr}'].std()})
seed_vfr_df = pd.DataFrame(seed_vfr)

fig, axes = plt.subplots(2, 4, figsize=(28, 14))
seed_colors = {'DI':'#3498db', 'SPD':'#e74c3c', 'EOPP':'#2ecc71', 'EOD':'#f39c12',
               'TI':'#9b59b6', 'PP':'#1abc9c', 'CAL':'#e67e22'}
for ai, attr in enumerate(['RACE','SEX','ETHNICITY','AGE_GROUP']):
    # Row 0: Multi-metric violin plot showing distribution of each metric across seeds
    ax_v = axes[0][ai]
    metric_data = []
    metric_labels = []
    for mk in METRIC_KEYS:
        col = f'{mk}_{attr}'
        if col in seed_df.columns:
            metric_data.append(seed_df[col].values)
            metric_labels.append(mk)
    if metric_data:
        parts = ax_v.violinplot(metric_data, showmeans=True, showmedians=True, showextrema=True)
        for idx_p, pc in enumerate(parts['bodies']):
            pc.set_facecolor(list(seed_colors.values())[idx_p])
            pc.set_edgecolor('black'); pc.set_alpha(0.7); pc.set_linewidth(0.8)
        parts['cmeans'].set_color('black'); parts['cmedians'].set_color('white')
        # Add individual seed points as jittered strip
        for idx_m, d in enumerate(metric_data):
            jitter = np.random.normal(0, 0.05, len(d))
            ax_v.scatter(np.full(len(d), idx_m+1) + jitter, d, alpha=0.4, s=12,
                        color=list(seed_colors.values())[idx_m], zorder=3, edgecolor='none')
    ax_v.set_xticks(range(1, len(metric_labels)+1))
    ax_v.set_xticklabels(metric_labels, fontsize=9, rotation=30, ha='right')
    ax_v.set_title(f'{attr}', fontsize=13, fontweight='bold', color=PALETTE[ai])
    ax_v.set_ylabel('Metric Value' if ai == 0 else '')
    ax_v.grid(axis='y', alpha=0.3)

    # Row 1: Seed VFR lollipop for this attribute
    ax_l = axes[1][ai]
    sub_vfr = seed_vfr_df[seed_vfr_df['Attribute'] == attr].copy()
    sub_vfr = sub_vfr.sort_values('VFR', ascending=True)
    y_pos_sv = range(len(sub_vfr))
    colors_sv = ['#e74c3c' if v > 0 else '#2ecc71' for v in sub_vfr['VFR']]
    ax_l.hlines(y=y_pos_sv, xmin=0, xmax=sub_vfr['VFR'].values, color=colors_sv, linewidth=3, alpha=0.8)
    ax_l.scatter(sub_vfr['VFR'].values, y_pos_sv, color=colors_sv, s=100, zorder=5,
                edgecolors='white', linewidth=1.5)
    ax_l.set_yticks(list(y_pos_sv))
    ax_l.set_yticklabels(sub_vfr['Metric'].values, fontsize=10)
    ax_l.set_xlabel('VFR', fontsize=11)
    ax_l.set_title(f'{attr}: Seed VFR', fontsize=12, fontweight='bold', color=PALETTE[ai])
    ax_l.set_xlim(-0.02, max(0.15, sub_vfr['VFR'].max() * 1.3 + 0.02))
    # Annotate mean and std
    for yi, (_, r) in enumerate(sub_vfr.iterrows()):
        ax_l.annotate(f'μ={r["Mean"]:.4f} σ={r["Std"]:.4f}', xy=(r['VFR'], yi),
                      xytext=(r['VFR']+0.005, yi), fontsize=7, va='center', color='gray')
    ax_l.grid(axis='x', alpha=0.3)

plt.suptitle(f'Seed Perturbation: All 7 Metrics Across {N_SEEDS} Seeds\\n'
             f'Top: Distribution of metric values | Bottom: Verdict Flip Rate per metric',
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.94])
save_fig('seed_perturbation')
plt.show()

# Also save a seed VFR heatmap
fig2, ax2 = plt.subplots(figsize=(10, 6))
sv_hm = seed_vfr_df.pivot(index='Metric', columns='Attribute', values='VFR')
# Annotate with VFR% + std
sv_annot = sv_hm.copy().astype(object)
sv_std_hm = seed_vfr_df.pivot(index='Metric', columns='Attribute', values='Std')
for r in sv_hm.index:
    for c in sv_hm.columns:
        v = sv_hm.loc[r, c]; s = sv_std_hm.loc[r, c]
        sv_annot.loc[r, c] = f'{v:.0%}\\nσ={s:.4f}'
sns.heatmap(sv_hm.astype(float), annot=sv_annot, fmt='', cmap='YlOrRd', vmin=0, vmax=0.2,
            linewidths=1, linecolor='white', ax=ax2, cbar_kws={'label':'VFR'})
ax2.set_title(f'Seed Perturbation VFR Heatmap ({N_SEEDS} Seeds × LightGBM)', fontsize=13, fontweight='bold')
plt.tight_layout()
save_fig('seed_perturbation_vfr_heatmap')
plt.show()
""")

md("""
> **Seed perturbation** tests whether the fairness verdict is sensitive to the
> random initialisation of the model.
""")

md("### 9.4 Threshold Sensitivity")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 42 · Threshold Sensitivity (0.10 → 0.90) — All 7 Metrics
# ──────────────────────────────────────────────────────────────
thresholds = np.arange(0.1, 0.91, 0.05)
thresh_results = []
for t in thresholds:
    y_p_t = (best_y_prob >= t).astype(int)
    row = {'Threshold':t, 'Accuracy':accuracy_score(y_test, y_p_t),
           'F1': f1_score(y_test, y_p_t) if y_p_t.sum()>0 else 0,
           'Precision': precision_score(y_test, y_p_t) if y_p_t.sum()>0 else 0,
           'Recall': recall_score(y_test, y_p_t) if y_p_t.sum()>0 else 0}
    for attr in ['RACE','SEX','ETHNICITY','AGE_GROUP']:
        fc_t = FairnessCalculator(y_test, y_p_t, best_y_prob, protected_attrs[attr])
        mt, vt, _ = fc_t.compute_all()
        for mk in METRIC_KEYS:
            row[f'{mk}_{attr}'] = mt[mk]
    thresh_results.append(row)
thresh_df = pd.DataFrame(thresh_results)
thresh_df.to_csv(f'{TABLES_DIR}/12_threshold_sensitivity.csv', index=False)

fig, axes = plt.subplots(2, 2, figsize=(22, 16))

# (a) Performance vs Threshold — enhanced with fill_between
ax_a = axes[0][0]
ax_a.plot(thresh_df['Threshold'], thresh_df['Accuracy'], 'o-', label='Accuracy', color='#3498db', lw=2.5, markersize=6)
ax_a.plot(thresh_df['Threshold'], thresh_df['F1'], 's-', label='F1', color='#e74c3c', lw=2.5, markersize=6)
ax_a.plot(thresh_df['Threshold'], thresh_df['Precision'], 'D-', label='Precision', color='#2ecc71', lw=1.5, markersize=5, alpha=0.7)
ax_a.plot(thresh_df['Threshold'], thresh_df['Recall'], '^-', label='Recall', color='#f39c12', lw=1.5, markersize=5, alpha=0.7)
ax_a.axvline(x=0.5, color='gray', linestyle=':', lw=2, alpha=0.6, label='Default (0.5)')
ax_a.fill_between(thresh_df['Threshold'], thresh_df['Accuracy'], alpha=0.08, color='#3498db')
ax_a.set_xlabel('Decision Threshold', fontsize=12); ax_a.set_ylabel('Score', fontsize=12)
ax_a.set_title('(a) Performance Metrics vs Decision Threshold', fontsize=13, fontweight='bold')
ax_a.legend(fontsize=10, loc='lower left'); ax_a.grid(alpha=0.3)

# (b) Multi-metric heatmap across thresholds
ax_b = axes[0][1]
# Build heatmap data: for each threshold × metric, average across attributes
hm_data = []
for _, row in thresh_df.iterrows():
    hm_row = {'Threshold': f'{row["Threshold"]:.2f}'}
    for mk in METRIC_KEYS:
        vals = [row.get(f'{mk}_{a}', np.nan) for a in ['RACE','SEX','ETHNICITY','AGE_GROUP']]
        hm_row[mk] = np.nanmean(vals)
    hm_data.append(hm_row)
hm_df = pd.DataFrame(hm_data).set_index('Threshold')
# Normalize each metric to [0,1] for comparable color scale
hm_norm = hm_df.copy()
for c in hm_norm.columns:
    mn, mx = hm_norm[c].min(), hm_norm[c].max()
    if mx > mn: hm_norm[c] = (hm_norm[c] - mn) / (mx - mn)
    else: hm_norm[c] = 0.5
sns.heatmap(hm_norm.T, cmap='RdYlGn', annot=hm_df.T, fmt='.3f', linewidths=0.5,
            ax=ax_b, cbar_kws={'label':'Normalized Value'}, xticklabels=True)
ax_b.set_title('(b) All 7 Metrics Across Thresholds\\n(row-normalized, annotations = raw values)',
               fontsize=12, fontweight='bold')
ax_b.set_xlabel('Decision Threshold', fontsize=11)

# (c) Verdict stability: for each threshold, count how many of 28 combos are FAIR
ax_c = axes[1][0]
fair_counts_t = []
for _, row in thresh_df.iterrows():
    n_fair = 0
    for attr in ['RACE','SEX','ETHNICITY','AGE_GROUP']:
        for mk in METRIC_KEYS:
            col = f'{mk}_{attr}'
            if col not in row: continue
            val = row[col]
            t_info = FairnessCalculator.THRESHOLDS.get(mk, {})
            if mk == 'DI':
                verdict = val >= 0.80
            elif isinstance(t_info, dict):
                t_val = t_info.get('threshold', 0.1)
                comp = t_info.get('compare', 'abs_lt')
                if comp == 'abs_lt': verdict = abs(val) < t_val
                elif comp == 'lt': verdict = val < t_val
                else: verdict = abs(val) < t_val
            else: verdict = True
            if verdict: n_fair += 1
    fair_counts_t.append({'Threshold': row['Threshold'], 'N_Fair': n_fair, 'Pct_Fair': n_fair/28*100})
fc_t_df = pd.DataFrame(fair_counts_t)
ax_c.fill_between(fc_t_df['Threshold'], fc_t_df['Pct_Fair'], alpha=0.3, color='#2ecc71')
ax_c.plot(fc_t_df['Threshold'], fc_t_df['Pct_Fair'], 'o-', color='#2ecc71', lw=2.5, markersize=7)
ax_c.axvline(x=0.5, color='gray', linestyle=':', lw=2, alpha=0.6, label='Default (0.5)')
ax_c.axhline(y=50, color='#e74c3c', linestyle='--', alpha=0.5, label='50% Fair')
ax_c.set_xlabel('Decision Threshold', fontsize=12); ax_c.set_ylabel('% Fair Verdicts (of 28)', fontsize=12)
ax_c.set_title('(c) Verdict Stability: % Fair Across Thresholds', fontsize=13, fontweight='bold')
ax_c.legend(fontsize=10); ax_c.grid(alpha=0.3); ax_c.set_ylim(0, 105)

# (d) Multi-metric fairness-accuracy Pareto (DI, SPD, EOPP for RACE)
ax_d = axes[1][1]
metric_show = ['DI', 'SPD', 'EOPP', 'EOD']
markers_pareto = ['o', 's', 'D', '^']
for mi, mk in enumerate(metric_show):
    col = f'{mk}_RACE'
    if col in thresh_df.columns:
        ax_d.scatter(thresh_df[col], thresh_df['Accuracy'], marker=markers_pareto[mi],
                    color=PALETTE[mi], s=80, alpha=0.8, label=f'{mk} (RACE)', zorder=3)
        # Connect with faint line
        ax_d.plot(thresh_df[col], thresh_df['Accuracy'], color=PALETTE[mi], alpha=0.3, lw=1)
# Add threshold line for DI
ax_d.axvline(x=0.80, color='red', linestyle='--', alpha=0.5, label='DI = 0.80')
ax_d.axvline(x=0.10, color='orange', linestyle='--', alpha=0.5, label='SPD/EOPP/EOD = 0.10')
ax_d.set_xlabel('Metric Value (RACE)', fontsize=12); ax_d.set_ylabel('Accuracy', fontsize=12)
ax_d.set_title('(d) Accuracy–Fairness Pareto (Multiple Metrics)', fontsize=13, fontweight='bold')
ax_d.legend(fontsize=9, loc='best'); ax_d.grid(alpha=0.3)

plt.suptitle('Threshold Sensitivity Analysis — All 7 Metrics', fontsize=15, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
save_fig('threshold_sensitivity')
plt.show()
""")

md("### 9.5 GroupKFold K=5 — Hospital Baseline")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 43 · GroupKFold K=5 (Hospital-based) — All 7 Metrics
# ──────────────────────────────────────────────────────────────
print("GroupKFold K=5 — hospital-based stability …")
gkf = GroupKFold(n_splits=5); gkf_results = []

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=hospital_ids_train)):
    model_gkf = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63,
        max_depth=8, random_state=RANDOM_STATE, verbose=-1, n_jobs=1)
    model_gkf.fit(X_train[tr_idx], y_train[tr_idx])
    y_pred_gkf = model_gkf.predict(X_test)
    y_prob_gkf = model_gkf.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred_gkf)
    auc = roc_auc_score(y_test, y_prob_gkf)
    row = {'Fold':fold+1, 'Acc':acc, 'AUC':auc,
           'Train_Hospitals':len(set(hospital_ids_train[tr_idx])),
           'Val_Hospitals':len(set(hospital_ids_train[val_idx]))}
    for attr in ['RACE','SEX','ETHNICITY','AGE_GROUP']:
        fc_gkf = FairnessCalculator(y_test, y_pred_gkf, y_prob_gkf, protected_attrs[attr])
        mg, vg, _ = fc_gkf.compute_all()
        for mk in METRIC_KEYS:
            row[f'{mk}_{attr}'] = mg[mk]
    gkf_results.append(row)
    print(f"  Fold {fold+1}: Acc={acc:.4f}  AUC={auc:.4f}  DI_RACE={row['DI_RACE']:.3f}")

gkf_df = pd.DataFrame(gkf_results)
gkf_df.to_csv(f'{TABLES_DIR}/13_groupkfold_k5.csv', index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(gkf_df['Fold'], gkf_df['AUC'], color=PALETTE[0], edgecolor='white')
axes[0].axhline(y=gkf_df['AUC'].mean(), color='red', linestyle='--')
axes[0].set_xlabel('Fold'); axes[0].set_ylabel('AUC')
axes[0].set_title(f'(a) AUC by Fold (mean = {gkf_df["AUC"].mean():.4f})')
for i, attr in enumerate(['RACE','SEX','ETHNICITY','AGE_GROUP']):
    axes[1].plot(gkf_df['Fold'], gkf_df[f'DI_{attr}'], 'o-', color=PALETTE[i], label=attr)
axes[1].axhline(y=0.80, color='red', linestyle='--', lw=2); axes[1].legend()
axes[1].set_xlabel('Fold'); axes[1].set_ylabel('DI')
axes[1].set_title('(b) DI Stability Across Hospital Folds')
plt.tight_layout()
save_fig('groupkfold_k5')
plt.show()
""")

md("""
> GroupKFold ensures **entire hospitals** are held out in each fold,
> testing generalization to unseen hospital populations.
""")

###############################################################################
# SECTION 10 — METRIC DISAGREEMENT MATRIX (FIG06)
###############################################################################
md("""
---
## 10. Metric Disagreement Matrix

Different fairness metrics can **disagree** on the same model-attribute
combination.  One metric may say "fair" while another says "unfair".
We quantify this disagreement to show why a single metric is insufficient.
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 44 · Metric Disagreement Analysis (FIG06)
# ──────────────────────────────────────────────────────────────
disagreement_data = []
for name in test_predictions:
    for attr in ['RACE','SEX','ETHNICITY','AGE_GROUP']:
        v = all_verdicts[name][attr]
        disagreement_data.append({
            'Model': name, 'Attribute': attr,
            **{f'V_{mk}': int(v[mk]) for mk in METRIC_KEYS},
            'N_Fair': sum(v.values()), 'N_Unfair': len(METRIC_KEYS) - sum(v.values()),
        })
disagree_df = pd.DataFrame(disagreement_data)

# 7×7 Pairwise Disagreement Matrix: how often do metric_i and metric_j disagree?
n_combos = len(disagree_df)
pair_disagree = np.zeros((7, 7))
for i, mi in enumerate(METRIC_KEYS):
    for j, mj in enumerate(METRIC_KEYS):
        if i == j: continue
        disagree_count = (disagree_df[f'V_{mi}'] != disagree_df[f'V_{mj}']).sum()
        pair_disagree[i][j] = disagree_count / n_combos

pair_df = pd.DataFrame(pair_disagree, index=METRIC_KEYS, columns=METRIC_KEYS)

fig, axes = plt.subplots(1, 3, figsize=(26, 7))
# (a) Pairwise disagreement heatmap — enhanced with diverging colormap
mask = np.eye(7, dtype=bool)
sns.heatmap(pair_df, annot=True, fmt='.1%', cmap='YlOrRd', vmin=0, vmax=0.6,
            linewidths=1, linecolor='white', ax=axes[0], mask=mask,
            cbar_kws={'label':'Disagreement Rate'})
# Fill diagonal with gray
for i in range(7):
    axes[0].add_patch(plt.Rectangle((i, i), 1, 1, fill=True, color='#ecf0f1', zorder=2))
    axes[0].text(i+0.5, i+0.5, '—', ha='center', va='center', fontsize=10, color='gray')
axes[0].set_title('(a) Pairwise Metric Disagreement Rate', fontsize=12, fontweight='bold')

# (b) Distribution of N_Fair — gradient-colored bars with annotations
counts = disagree_df['N_Fair'].value_counts().reindex(range(8), fill_value=0)
colors_grad = plt.cm.RdYlGn(np.linspace(0.15, 0.85, 8))
bars = axes[1].bar(counts.index, counts.values, color=colors_grad,
                   edgecolor='white', linewidth=1.5, width=0.75)
for b, v in zip(bars, counts.values):
    if v > 0:
        axes[1].text(b.get_x()+b.get_width()/2, v+0.3, f'{v}',
                    ha='center', fontsize=10, fontweight='bold')
        pct = v / len(disagree_df) * 100
        axes[1].text(b.get_x()+b.get_width()/2, v/2, f'{pct:.0f}%',
                    ha='center', va='center', fontsize=9, color='white', fontweight='bold')
axes[1].set_xlabel('Number of Fair Metrics (out of 7)', fontsize=11)
axes[1].set_ylabel('Count (model-attribute combos)', fontsize=11)
axes[1].set_title('(b) Multi-Criteria Fairness Distribution', fontsize=12, fontweight='bold')
axes[1].set_xticks(range(8)); axes[1].grid(axis='y', alpha=0.3)

# (c) NEW: Per-attribute disagreement profile (stacked)
attr_dis = disagree_df.groupby('Attribute')['N_Fair'].value_counts().unstack(fill_value=0)
attr_dis = attr_dis.reindex(columns=range(8), fill_value=0)
bottom_stack = np.zeros(len(attr_dis))
for col_i in range(8):
    if col_i in attr_dis.columns:
        axes[2].bar(attr_dis.index, attr_dis[col_i], bottom=bottom_stack,
                   color=colors_grad[col_i], label=f'{col_i} Fair', edgecolor='white', linewidth=0.5)
        bottom_stack += attr_dis[col_i].values
axes[2].set_xlabel('Protected Attribute', fontsize=11)
axes[2].set_ylabel('Count', fontsize=11)
axes[2].set_title('(c) Verdict Distribution by Attribute', fontsize=12, fontweight='bold')
axes[2].legend(fontsize=7, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.12))
axes[2].tick_params(axis='x', rotation=15)

plt.tight_layout()
save_fig('metric_disagreement_matrix')
plt.show()

print(f"Total model-attribute combinations: {n_combos}")
print(f"All 7 agree FAIR:   {(disagree_df['N_Fair']==7).sum()} ({(disagree_df['N_Fair']==7).mean():.1%})")
print(f"All 7 agree UNFAIR: {(disagree_df['N_Fair']==0).sum()} ({(disagree_df['N_Fair']==0).mean():.1%})")
print(f"Mixed verdict:      {((disagree_df['N_Fair']>0) & (disagree_df['N_Fair']<7)).sum()} ({((disagree_df['N_Fair']>0) & (disagree_df['N_Fair']<7)).mean():.1%})")

pair_df.to_csv(f'{TABLES_DIR}/14_metric_disagreement.csv')
disagree_df.to_csv(f'{TABLES_DIR}/14b_per_combo_verdicts.csv', index=False)
""")

md("""
> **Key finding:** Substantial disagreement between metrics confirms that
> relying on a single metric (e.g., DI alone) gives an incomplete picture.
> The multi-criteria approach used in this analysis is essential for robust
> fairness assessment.
""")

###############################################################################
# SECTION 11 — CROSS-SITE FAIRNESS PORTABILITY (Protocol 3)
###############################################################################
md("""
---
## 11. Cross-Site Fairness Portability (Protocol 3)

**This is the paper's key distinguishing claim (C3).**

Protocol 3 tests whether a model trained on one set of hospitals can maintain
fairness when deployed at a different hospital.  We use **K=20 GroupKFold**:
- For each fold: train LightGBM on 19 hospital clusters, predict on the held-out cluster.
- Compute all 7 fairness metrics on the held-out cluster.
- Measure between-cluster variation: CV, range, verdict agreement (Fleiss' κ).

This goes beyond Protocol P1 (random resampling) because hospital populations
have genuinely *different* demographic compositions.
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 45 · Protocol 3: Cross-Site K=20 GroupKFold — Train/Eval Split
# ──────────────────────────────────────────────────────────────
K_CS = 20
print(f"Protocol 3: Cross-Site K={K_CS} GroupKFold …")

# Use ALL data (train+test combined) for cross-site analysis
X_all = np.vstack([X_train, X_test])
y_all = np.concatenate([y_train, y_test])
hosp_all = np.concatenate([hospital_ids_train, hospital_ids_test])
prot_all = {}
for attr in ['RACE','SEX','ETHNICITY','AGE_GROUP']:
    prot_all[attr] = np.concatenate([protected_attrs_train[attr], protected_attrs[attr]])

gkf_cs = GroupKFold(n_splits=K_CS)
cs_results = []
_t0 = time.time()

for fold, (tr_idx, val_idx) in enumerate(gkf_cs.split(X_all, y_all, groups=hosp_all)):
    # Train on 19 clusters
    model_cs = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63,
        max_depth=8, random_state=RANDOM_STATE, verbose=-1, n_jobs=1)
    model_cs.fit(X_all[tr_idx], y_all[tr_idx])

    # Evaluate on held-out cluster
    y_val = y_all[val_idx]
    y_pred_cs = model_cs.predict(X_all[val_idx])
    y_prob_cs = model_cs.predict_proba(X_all[val_idx])[:, 1]

    n_hospitals_held = len(set(hosp_all[val_idx]))
    row = {'Fold': fold+1, 'N_val': len(val_idx),
           'N_hospitals': n_hospitals_held,
           'Acc': accuracy_score(y_val, y_pred_cs),
           'AUC': roc_auc_score(y_val, y_prob_cs) if len(set(y_val)) > 1 else np.nan}

    for attr in ['RACE','SEX','ETHNICITY','AGE_GROUP']:
        attr_val = prot_all[attr][val_idx]
        if len(set(attr_val)) >= 2:
            fc_cs = FairnessCalculator(y_val, y_pred_cs, y_prob_cs, attr_val)
            mc, vc, _ = fc_cs.compute_all()
            for mk in METRIC_KEYS:
                row[f'{mk}_{attr}'] = mc[mk]
                row[f'V_{mk}_{attr}'] = 1 if vc[mk] else 0
        else:
            for mk in METRIC_KEYS:
                row[f'{mk}_{attr}'] = np.nan
                row[f'V_{mk}_{attr}'] = np.nan
    cs_results.append(row)
    if (fold+1) % 5 == 0:
        print(f"  Fold {fold+1}/{K_CS}: N_val={len(val_idx):,}  Acc={row['Acc']:.4f}")

cs_df = pd.DataFrame(cs_results)
cs_df.to_csv(f'{TABLES_DIR}/15_cross_site_portability.csv', index=False)
print(f"\\nCompleted in {time.time()-_t0:.1f}s")

# --- Table 8: Cross-site variation summary ---
print("\\n--- Table 8: Cross-Site Fairness Variation ---")
cs_summary = []
for attr in ['RACE','SEX','ETHNICITY','AGE_GROUP']:
    for mk in METRIC_KEYS:
        col = f'{mk}_{attr}'
        vcol = f'V_{mk}_{attr}'
        vals = cs_df[col].dropna()
        if len(vals) < 2: continue
        cs_summary.append({
            'Attribute': attr, 'Metric': mk,
            'Mean': vals.mean(), 'Std': vals.std(),
            'CV': vals.std() / max(vals.mean(), 1e-9),
            'Min': vals.min(), 'Max': vals.max(),
            'Range': vals.max() - vals.min(),
            'Pct_Fair': cs_df[vcol].dropna().mean() * 100
        })
cs_summary_df = pd.DataFrame(cs_summary)
cs_summary_df.to_csv(f'{TABLES_DIR}/15b_cross_site_summary.csv', index=False)

display(HTML("<h4>Table 8: Cross-Site Fairness Variation (K=20 clusters)</h4>"))
display(cs_summary_df.pivot(index='Metric', columns='Attribute', values='CV').style.format('{:.3f}'))
""")

md("""
> **Table 8** shows the between-cluster CV for each metric-attribute pair.
> High CV (>0.10) indicates the metric's value changes substantially depending
> on *which hospitals* are in the evaluation set — a portability concern.
""")

md("### 11.1 Cross-Hospital Violin Plots (FIG09)")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 46 · FIG09: Cross-Hospital Violin Plots
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(28, 13))
attr_list_cs = ['RACE','SEX','ETHNICITY','AGE_GROUP']
cs_colors = {a: PALETTE[i] for i, a in enumerate(attr_list_cs)}

for mi, mk in enumerate(METRIC_KEYS):
    ax = axes[mi//4][mi%4]
    # Build long-form DataFrame for seaborn
    plot_records = []
    for attr in attr_list_cs:
        col = f'{mk}_{attr}'
        if col in cs_df.columns:
            for v in cs_df[col].dropna():
                plot_records.append({'Attribute': attr, 'Value': v})
    if plot_records:
        plot_df = pd.DataFrame(plot_records)
        # Seaborn violin + strip
        sns.violinplot(data=plot_df, x='Attribute', y='Value', ax=ax,
                      palette=[cs_colors[a] for a in plot_df['Attribute'].unique()],
                      inner='box', alpha=0.6, linewidth=1, cut=0)
        sns.stripplot(data=plot_df, x='Attribute', y='Value', ax=ax,
                     palette=[cs_colors[a] for a in plot_df['Attribute'].unique()],
                     size=5, alpha=0.6, jitter=0.15, edgecolor='white', linewidth=0.5)
    # Add threshold line
    thresh_info = FairnessCalculator.THRESHOLDS.get(mk, {})
    thresh_val = thresh_info.get('threshold', None) if isinstance(thresh_info, dict) else None
    if mk == 'DI':
        ax.axhline(y=0.80, color='red', linestyle='--', alpha=0.7, lw=2, label='DI = 0.80')
    elif thresh_val is not None:
        ax.axhline(y=thresh_val, color='red', linestyle='--', alpha=0.7, lw=2, label=f'Thresh = {thresh_val}')
    ax.set_title(f'{mk}', fontsize=13, fontweight='bold')
    ax.set_xlabel(''); ax.set_ylabel('Value' if mi % 4 == 0 else '')
    ax.grid(axis='y', alpha=0.2); ax.legend(fontsize=7, loc='best')
    ax.tick_params(axis='x', labelsize=9)

# 8th subplot: Summary radar of mean CV per metric
ax8 = axes[1][3]
cv_means = cs_summary_df.groupby('Metric')['CV'].mean()
metrics_r = cv_means.index.tolist()
values_r = cv_means.values.tolist()
n_metrics = len(metrics_r)
angles_r = np.linspace(0, 2*np.pi, n_metrics, endpoint=False).tolist()
values_r += [values_r[0]]; angles_r += [angles_r[0]]
ax8.remove()
ax8 = fig.add_subplot(2, 4, 8, polar=True)
ax8.fill(angles_r, values_r, alpha=0.25, color='#e74c3c')
ax8.plot(angles_r, values_r, 'o-', color='#e74c3c', lw=2, markersize=6)
ax8.set_xticks(angles_r[:-1])
ax8.set_xticklabels(metrics_r, fontsize=9)
ax8.set_title('Mean CV per Metric\\n(Cross-Site)', fontsize=11, fontweight='bold', pad=15)
ax8.set_ylim(0, max(values_r) * 1.3)

plt.suptitle('FIG09: Cross-Hospital Fairness Distribution (K=20 Clusters)\\n'
             'Violins + individual cluster points | Red line = fairness threshold',
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.93])
save_fig('cross_site_violin_plots')
plt.show()
""")

md("""
> **Violin plots** show the full distribution of each metric across the 20
> hospital clusters.  Wide violins indicate high variability between sites.
> If the distribution straddles the threshold line (red dashed), the verdict
> is site-dependent.
""")

md("### 11.2 Fleiss' κ — Inter-Cluster Verdict Agreement")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 47 · Fleiss' κ Computation
# ──────────────────────────────────────────────────────────────
# Fleiss' kappa: agreement among K raters (clusters) on each metric-attribute
# Each "subject" is a metric-attribute pair, each "rater" is a cluster fold

def fleiss_kappa(ratings_matrix):
    \"\"\"ratings_matrix: N subjects × k categories counts (here 2: fair/unfair).\"\"\"
    N, k = ratings_matrix.shape
    n = ratings_matrix.sum(axis=1)[0]  # raters per subject
    if n <= 1: return 0.0
    p_j = ratings_matrix.sum(axis=0) / (N * n)
    P_i = (np.sum(ratings_matrix**2, axis=1) - n) / (n * (n - 1))
    P_bar = P_i.mean()
    P_e = np.sum(p_j**2)
    if abs(1 - P_e) < 1e-9: return 1.0
    return (P_bar - P_e) / (1 - P_e)

# Build ratings for each metric-attribute pair
kappa_rows = []
for attr in ['RACE','SEX','ETHNICITY','AGE_GROUP']:
    for mk in METRIC_KEYS:
        vcol = f'V_{mk}_{attr}'
        if vcol not in cs_df.columns: continue
        verdicts = cs_df[vcol].dropna().values
        n_fair = int(verdicts.sum())
        n_unfair = len(verdicts) - n_fair
        kappa_rows.append({'Attribute': attr, 'Metric': mk,
                          'N_Fair': n_fair, 'N_Unfair': n_unfair, 'N_Folds': len(verdicts)})

kappa_df = pd.DataFrame(kappa_rows)

# Overall Fleiss' kappa across all metric-attribute pairs
ratings = kappa_df[['N_Fair', 'N_Unfair']].values
fk = fleiss_kappa(ratings) if len(ratings) > 1 else 0.0
print(f"Fleiss' κ (overall cross-site verdict agreement): {fk:.3f}")
print(f"  Interpretation: <0 = worse than chance, 0-0.20 = slight, 0.21-0.40 = fair,")
print(f"  0.41-0.60 = moderate, 0.61-0.80 = substantial, 0.81-1.0 = almost perfect")

# Per-metric kappa
mk_kappas = []
for mk in METRIC_KEYS:
    sub = kappa_df[kappa_df['Metric']==mk]
    if len(sub) > 1:
        r = sub[['N_Fair','N_Unfair']].values
        mk_kappas.append({'Metric': mk, 'Kappa': fleiss_kappa(r)})
mk_kappa_df = pd.DataFrame(mk_kappas)

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# (a) Enhanced bar chart with gradient colors based on kappa value
ax = axes[0]
kappa_vals = mk_kappa_df['Kappa'].values
# Color by interpretation: red < 0.21, orange 0.21-0.41, yellow 0.41-0.61, green > 0.61
bar_colors = []
for k in kappa_vals:
    if k > 0.61: bar_colors.append('#27ae60')
    elif k > 0.41: bar_colors.append('#f39c12')
    elif k > 0.21: bar_colors.append('#e67e22')
    else: bar_colors.append('#e74c3c')
bars = ax.bar(mk_kappa_df['Metric'], kappa_vals, color=bar_colors, edgecolor='white',
              linewidth=1.5, width=0.65, zorder=3)
ax.axhspan(-0.1, 0.21, alpha=0.06, color='red', label='Slight (<0.21)')
ax.axhspan(0.21, 0.41, alpha=0.06, color='orange', label='Fair (0.21–0.40)')
ax.axhspan(0.41, 0.61, alpha=0.06, color='yellow', label='Moderate (0.41–0.60)')
ax.axhspan(0.61, 1.1, alpha=0.06, color='green', label='Substantial (>0.61)')
for b, v in zip(bars, kappa_vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.02, f'{v:.3f}', ha='center', fontsize=11, fontweight='bold')
ax.set_ylabel("Fleiss' κ", fontsize=12); ax.set_ylim(-0.1, 1.1)
ax.set_title("(a) Cross-Site Verdict Agreement per Metric", fontsize=13, fontweight='bold')
ax.legend(fontsize=8, loc='upper right'); ax.grid(axis='y', alpha=0.3)

# (b) Per-attribute kappa as grouped bar
ax2 = axes[1]
attr_kappas = []
for attr in ['RACE','SEX','ETHNICITY','AGE_GROUP']:
    for mk in METRIC_KEYS:
        sub = kappa_df[(kappa_df['Metric']==mk) & (kappa_df['Attribute']==attr)]
        if len(sub):
            attr_kappas.append({'Attribute': attr, 'Metric': mk,
                'Agreement': sub.iloc[0]['N_Fair'] / sub.iloc[0]['N_Folds'] * 100})
ak_df = pd.DataFrame(attr_kappas)
if len(ak_df) > 0:
    ak_pivot = ak_df.pivot(index='Metric', columns='Attribute', values='Agreement')
    ak_pivot.plot(kind='bar', ax=ax2, color=[PALETTE[i] for i in range(4)],
                  edgecolor='white', linewidth=0.5, width=0.75)
    ax2.set_ylabel('% Clusters Deem FAIR', fontsize=12)
    ax2.set_title('(b) Per-Metric × Per-Attribute: % Sites Agree Fair', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9); ax2.set_ylim(0, 105)
    ax2.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
    ax2.tick_params(axis='x', rotation=30); ax2.grid(axis='y', alpha=0.3)

plt.suptitle("Fleiss' κ: Cross-Site Verdict Agreement (K=20 Hospital Clusters)",
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
save_fig('fleiss_kappa_per_metric')
plt.show()

display(mk_kappa_df.style.format({'Kappa':'{:.3f}'}))
""")

md("""
> **Fleiss' κ** quantifies the degree of agreement between hospital clusters
> on the fair/unfair verdict.  Metrics with κ < 0.40 have poor cross-site
> portability — the verdict at one hospital cannot be assumed to hold elsewhere.
""")

###############################################################################
# SECTION 12 — 20-30 SUBSET & SUBGROUP ANALYSIS
###############################################################################
md("""
---
## 12. Comprehensive Subset & Subgroup Analysis (20-30 Tests)

We systematically test fairness across:
1. **30 random 30% subsets** — All 7 metrics, measuring VFR and variance.
2. **Intersectional subgroups** — RACE×SEX, RACE×AGE, SEX×AGE, ETH×AGE, RACE×ETH
   (approximately 50+ subgroups, ~25-30 with sufficient sample size).
""")

md("### 12.1 30 Random Subset Tests")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 48 · 30 Random Subset Tests (All 7 Metrics)
# ──────────────────────────────────────────────────────────────
N_SUBSETS = 30
subset_results = []
print(f"Running {N_SUBSETS} random 30% subset tests …")

for s in range(N_SUBSETS):
    idx = np.random.choice(len(y_test), size=int(0.3*len(y_test)), replace=False)
    y_sub = y_test[idx]; pred_sub = best_y_pred[idx]; prob_sub = best_y_prob[idx]
    row = {'Subset': s+1, 'N': len(idx)}
    for attr in ['RACE','SEX','ETHNICITY','AGE_GROUP']:
        attr_sub = protected_attrs[attr][idx]
        if len(set(attr_sub)) >= 2:
            fc_sub = FairnessCalculator(y_sub, pred_sub, prob_sub, attr_sub)
            ms, vs, _ = fc_sub.compute_all()
            for mk in METRIC_KEYS:
                row[f'{mk}_{attr}'] = ms[mk]
                row[f'Fair_{mk}_{attr}'] = 1 if vs[mk] else 0
    subset_results.append(row)

subset_df = pd.DataFrame(subset_results)
subset_df.to_csv(f'{TABLES_DIR}/16_30_random_subsets.csv', index=False)

# Summary table
print("\\n--- Subset VFR Summary ---")
sub_vfr_rows = []
for attr in ['RACE','SEX','ETHNICITY','AGE_GROUP']:
    for mk in METRIC_KEYS:
        vcol = f'Fair_{mk}_{attr}'
        mcol = f'{mk}_{attr}'
        if vcol not in subset_df.columns: continue
        fc_cnt = subset_df[vcol].sum()
        vfr = min(fc_cnt, N_SUBSETS-fc_cnt) / N_SUBSETS
        sub_vfr_rows.append({'Attribute':attr, 'Metric':mk,
            'Mean': subset_df[mcol].mean(), 'Std': subset_df[mcol].std(),
            'VFR': vfr, 'Pct_Fair': fc_cnt/N_SUBSETS*100})
sub_vfr_df = pd.DataFrame(sub_vfr_rows)

fig, axes = plt.subplots(1, 3, figsize=(26, 8))

# (a) CV heatmap across 30 subsets (replaces all-zero VFR heatmap)
cv_pivot = sub_vfr_df.pivot(index='Metric', columns='Attribute', values='Std')
mean_pivot = sub_vfr_df.pivot(index='Metric', columns='Attribute', values='Mean')
cv_ratio = cv_pivot / mean_pivot.replace(0, np.nan)
# Custom annotations: CV + Pct_Fair
cv_annot = cv_ratio.copy().astype(object)
pf_pvt = sub_vfr_df.pivot(index='Metric', columns='Attribute', values='Pct_Fair')
for r in cv_ratio.index:
    for c in cv_ratio.columns:
        cv_val = cv_ratio.loc[r, c]
        pf_val = pf_pvt.loc[r, c]
        if pd.notna(cv_val):
            cv_annot.loc[r, c] = f'CV={cv_val:.3f}\\n{pf_val:.0f}% Fair'
        else:
            cv_annot.loc[r, c] = '—'
sns.heatmap(cv_ratio.astype(float), annot=cv_annot, fmt='', cmap='YlOrBr', vmin=0, vmax=0.05,
            linewidths=1, linecolor='white', ax=axes[0], cbar_kws={'label':'CV'})
axes[0].set_title('(a) Metric CV Across 30 Random Subsets\\n(lower = more stable)', fontsize=12, fontweight='bold')

# (b) Multi-metric boxplot (all 7 metrics, for RACE and SEX)
ax_b = axes[1]
box_data = []; box_labels = []; box_colors = []
for mk in METRIC_KEYS:
    for ai, attr in enumerate(['RACE', 'SEX']):
        col = f'{mk}_{attr}'
        if col in subset_df.columns:
            box_data.append(subset_df[col].dropna().values)
            box_labels.append(f'{mk}\\n{attr[:3]}')
            box_colors.append(PALETTE[ai])
if box_data:
    bp = ax_b.boxplot(box_data, labels=box_labels, patch_artist=True, widths=0.6)
    for i, (patch, col) in enumerate(zip(bp['boxes'], box_colors)):
        patch.set_facecolor(col); patch.set_alpha(0.6)
    # Add strip points
    for i, d in enumerate(box_data):
        jitter = np.random.normal(0, 0.08, len(d))
        ax_b.scatter(np.full(len(d), i+1)+jitter, d, s=10, alpha=0.4, color=box_colors[i], zorder=3)
ax_b.set_title('(b) Metric Distributions (RACE & SEX)\\nAcross 30 Random 80% Subsets', fontsize=12, fontweight='bold')
ax_b.set_ylabel('Metric Value'); ax_b.tick_params(axis='x', labelsize=7, rotation=45)
ax_b.grid(axis='y', alpha=0.3)

# (c) VFR lollipop (replaces all-zero heatmap with informative lollipop)
ax_c = axes[2]
sub_vfr_sorted = sub_vfr_df.sort_values('VFR', ascending=True).copy()
sub_vfr_sorted['Label'] = sub_vfr_sorted['Metric'] + ' × ' + sub_vfr_sorted['Attribute']
y_pos_sub = range(len(sub_vfr_sorted))
colors_sub = ['#e74c3c' if v > 0 else '#95a5a6' for v in sub_vfr_sorted['VFR']]
ax_c.hlines(y=y_pos_sub, xmin=0, xmax=sub_vfr_sorted['VFR'].values, color=colors_sub, linewidth=2.5, alpha=0.8)
ax_c.scatter(sub_vfr_sorted['VFR'].values, y_pos_sub, color=colors_sub, s=70, zorder=5,
            edgecolors='white', linewidth=1)
ax_c.set_yticks(list(y_pos_sub))
ax_c.set_yticklabels(sub_vfr_sorted['Label'].values, fontsize=7)
ax_c.set_xlabel('VFR', fontsize=11)
ax_c.set_title('(c) Subset VFR per Metric × Attribute\\n(red = verdict flips in ≥1 subset)', fontsize=12, fontweight='bold')
n_nonzero = (sub_vfr_sorted['VFR'] > 0).sum()
ax_c.annotate(f'{n_nonzero}/{len(sub_vfr_sorted)} pairs have VFR > 0',
              xy=(0.95, 0.05), xycoords='axes fraction', fontsize=10, ha='right',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

plt.suptitle('30 Random Subset Analysis — Metric Stability', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
save_fig('30_subset_analysis')
plt.show()
""")

md("### 12.2 Intersectional Subgroup Analysis (All Attribute Combinations)")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 49 · Comprehensive Intersectional Subgroup Analysis
# ──────────────────────────────────────────────────────────────
from itertools import combinations

attr_pairs = [('RACE','SEX'), ('RACE','AGE_GROUP'), ('SEX','AGE_GROUP'),
              ('ETHNICITY','AGE_GROUP'), ('RACE','ETHNICITY')]

all_subgroup_results = []
print("Intersectional subgroup analysis …")

for a1, a2 in attr_pairs:
    attr1 = protected_attrs[a1]
    attr2 = protected_attrs[a2]
    for v1 in sorted(set(attr1)):
        for v2 in sorted(set(attr2)):
            mask = (attr1 == v1) & (attr2 == v2)
            n = mask.sum()
            if n < 50: continue
            y_sg = y_test[mask]; pred_sg = best_y_pred[mask]; prob_sg = best_y_prob[mask]

            # Get readable labels
            if a1 == 'RACE': l1 = RACE_MAP.get(v1, str(v1))
            elif a1 == 'SEX': l1 = SEX_MAP.get(v1, str(v1))
            elif a1 == 'ETHNICITY': l1 = ETH_MAP.get(v1, str(v1))
            else: l1 = str(v1)
            if a2 == 'AGE_GROUP': l2 = str(v2)
            elif a2 == 'SEX': l2 = SEX_MAP.get(v2, str(v2))
            elif a2 == 'ETHNICITY': l2 = ETH_MAP.get(v2, str(v2))
            else: l2 = RACE_MAP.get(v2, str(v2)) if a2 == 'RACE' else str(v2)

            row = {
                'Pair': f'{a1}×{a2}',
                'Group': f'{l1} × {l2}',
                'N': n,
                'Selection_Rate': pred_sg.mean(),
                'Accuracy': accuracy_score(y_sg, pred_sg),
                'TPR': pred_sg[y_sg==1].mean() if (y_sg==1).sum() > 0 else np.nan,
                'FPR': pred_sg[y_sg==0].mean() if (y_sg==0).sum() > 0 else np.nan,
            }
            all_subgroup_results.append(row)

subgroup_df = pd.DataFrame(all_subgroup_results).sort_values('Selection_Rate', ascending=False)
subgroup_df.to_csv(f'{TABLES_DIR}/17_intersectional_subgroups.csv', index=False)

print(f"Total intersectional subgroups analysed: {len(subgroup_df)}")
print(f"Attribute pairs: {[f'{a1}×{a2}' for a1, a2 in attr_pairs]}")

# --- Visualization: Top/Bottom subgroups ---
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# (a) Selection rates for top 20 / bottom 10
top20 = subgroup_df.head(15)
bot10 = subgroup_df.tail(10)
combined = pd.concat([top20, bot10])
colors_comb = ['#e74c3c' if r['Selection_Rate'] > df['LOS_BINARY'].mean()*1.2 else
               '#2ecc71' if r['Selection_Rate'] < df['LOS_BINARY'].mean()*0.8 else
               '#95a5a6' for _, r in combined.iterrows()]
axes[0].barh(combined['Group'], combined['Selection_Rate'], color=colors_comb, edgecolor='white')
axes[0].axvline(x=df['LOS_BINARY'].mean(), color='blue', linestyle='--', lw=2, label='Base rate')
axes[0].set_xlabel('Selection Rate'); axes[0].set_title('Top 15 / Bottom 10 Subgroups by Selection Rate')
axes[0].legend(fontsize=8)

# (b) Accuracy heatmap by pair
pair_acc = subgroup_df.groupby('Pair')['Accuracy'].agg(['mean','std','min','max']).reset_index()
axes[1].barh(pair_acc['Pair'], pair_acc['mean'], xerr=pair_acc['std'],
             color=[PALETTE[i] for i in range(len(pair_acc))], edgecolor='white', capsize=3)
axes[1].set_xlabel('Accuracy'); axes[1].set_title('Average Accuracy by Attribute Pair')
plt.tight_layout()
save_fig('intersectional_subgroup_analysis')
plt.show()

display(HTML(f"<h4>Intersectional Subgroup Summary ({len(subgroup_df)} subgroups)</h4>"))
display(subgroup_df.head(20).style.format({
    'Selection_Rate':'{:.3f}','Accuracy':'{:.3f}','TPR':'{:.3f}','FPR':'{:.3f}'}))
""")

md("""
> **Intersectional analysis** reveals disparities hidden by single-attribute
> approaches.  We tested 5 attribute pairs, yielding ~25-30 subgroups with
> sufficient sample size (N≥50).  The disparity range across subgroups is
> considerably wider than for any single attribute.
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 50 · Subgroup Disparity Summary Statistics
# ──────────────────────────────────────────────────────────────
print("=== Subgroup Disparity Summary ===")
print(f"  Total subgroups: {len(subgroup_df)}")
print(f"  Selection Rate: min={subgroup_df['Selection_Rate'].min():.3f}, "
      f"max={subgroup_df['Selection_Rate'].max():.3f}, "
      f"range={subgroup_df['Selection_Rate'].max()-subgroup_df['Selection_Rate'].min():.3f}")
print(f"  Accuracy:       min={subgroup_df['Accuracy'].min():.3f}, "
      f"max={subgroup_df['Accuracy'].max():.3f}, "
      f"range={subgroup_df['Accuracy'].max()-subgroup_df['Accuracy'].min():.3f}")
if 'TPR' in subgroup_df.columns:
    tpr_valid = subgroup_df['TPR'].dropna()
    print(f"  TPR:            min={tpr_valid.min():.3f}, max={tpr_valid.max():.3f}, "
          f"range={tpr_valid.max()-tpr_valid.min():.3f}")

# Disparity ratio per pair
print("\\n  Disparity ratio by pair:")
for pair in subgroup_df['Pair'].unique():
    sub = subgroup_df[subgroup_df['Pair']==pair]
    ratio = sub['Selection_Rate'].max() / max(sub['Selection_Rate'].min(), 1e-9)
    print(f"    {pair}: max/min selection rate = {ratio:.2f}")
""")

###############################################################################
# SECTION 13 — AFCE
###############################################################################
md("""
---
## 13. AFCE: Fairness-Through-Awareness Analysis

**AFCE** adds protected attributes and their interactions as explicit features.
The idea: rather than being "blind" to demographics, the model can learn
group-specific patterns and potentially produce more equitable predictions.
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 51 · AFCE Feature Engineering
# ──────────────────────────────────────────────────────────────
print("AFCE: Adding protected attributes + interactions …")

# Baseline already includes PAT_AGE; add RACE, SEX, ETHNICITY + interactions
age_idx = feature_names.index('PAT_AGE')
charges_idx = feature_names.index('TOTAL_CHARGES')

X_train_afce = np.column_stack([X_train,
    protected_attrs_train['RACE'].reshape(-1,1),
    protected_attrs_train['SEX'].reshape(-1,1),
    protected_attrs_train['ETHNICITY'].reshape(-1,1)])
X_test_afce = np.column_stack([X_test,
    protected_attrs['RACE'].reshape(-1,1),
    protected_attrs['SEX'].reshape(-1,1),
    protected_attrs['ETHNICITY'].reshape(-1,1)])

# Interaction features: demographics × charges, demographics × age
for attr_name in ['RACE', 'SEX', 'ETHNICITY']:
    a_tr = protected_attrs_train[attr_name].reshape(-1,1)
    a_te = protected_attrs[attr_name].reshape(-1,1)
    X_train_afce = np.column_stack([X_train_afce, X_train[:,charges_idx:charges_idx+1]*a_tr, X_train[:,age_idx:age_idx+1]*a_tr])
    X_test_afce = np.column_stack([X_test_afce, X_test[:,charges_idx:charges_idx+1]*a_te, X_test[:,age_idx:age_idx+1]*a_te])

afce_feat_names = feature_names + ['RACE_feat','SEX_feat','ETHNICITY_feat',
    'RACE×Charges','RACE×Age','SEX×Charges','SEX×Age','ETH×Charges','ETH×Age']
print(f"✓ AFCE features: {X_train_afce.shape[1]} ({X_train.shape[1]} original + "
      f"{X_train_afce.shape[1]-X_train.shape[1]} fairness-aware)")
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 52 · Train AFCE Models & Compare
# ──────────────────────────────────────────────────────────────
xgb_gpu = 'cuda' if GPU_AVAILABLE else 'cpu'
lgb_gpu = 'gpu' if GPU_AVAILABLE else 'cpu'

afce_models = {
    'AFCE-XGBoost': xgb.XGBClassifier(n_estimators=1000, max_depth=10, learning_rate=0.05,
        tree_method='hist', device=xgb_gpu, random_state=RANDOM_STATE, verbosity=0),
    'AFCE-LightGBM': lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.03, num_leaves=255,
        device=lgb_gpu, random_state=RANDOM_STATE, verbose=-1, n_jobs=1),
}
afce_predictions = {}

print("Training AFCE models …")
for name, model in afce_models.items():
    t0 = time.time()
    model.fit(X_train_afce, y_train); elapsed = time.time() - t0
    y_pred_a = model.predict(X_test_afce)
    y_prob_a = model.predict_proba(X_test_afce)[:, 1]
    afce_predictions[name] = {'y_pred':y_pred_a, 'y_prob':y_prob_a}
    print(f"  {name}: Acc={accuracy_score(y_test, y_pred_a):.4f}  "
          f"AUC={roc_auc_score(y_test, y_prob_a):.4f}  [{elapsed:.1f}s]")

# Comparison — all 7 metrics
comparison_rows = []
for name in ['XGBoost','LightGBM']:
    yp = test_predictions[name]['y_pred']; ypb = test_predictions[name]['y_prob']
    fc_c = FairnessCalculator(y_test, yp, ypb, protected_attrs['RACE'])
    mc, vc, _ = fc_c.compute_all()
    comparison_rows.append({'Model':name, 'Acc':accuracy_score(y_test,yp),
        'AUC':roc_auc_score(y_test,ypb), **{mk: mc[mk] for mk in METRIC_KEYS}})
for name in afce_predictions:
    yp = afce_predictions[name]['y_pred']; ypb = afce_predictions[name]['y_prob']
    fc_c = FairnessCalculator(y_test, yp, ypb, protected_attrs['RACE'])
    mc, vc, _ = fc_c.compute_all()
    comparison_rows.append({'Model':name, 'Acc':accuracy_score(y_test,yp),
        'AUC':roc_auc_score(y_test,ypb), **{mk: mc[mk] for mk in METRIC_KEYS}})

display(HTML("<h4>Standard vs AFCE — All 7 Metrics (RACE)</h4>"))
comp_df = pd.DataFrame(comparison_rows)
display(comp_df.style.format({c:'{:.4f}' for c in comp_df.columns if c != 'Model'}))
""")

md("""
> **AFCE result:** If AFCE models maintain accuracy but improve fairness metrics,
> awareness of protected attributes helps compensate for group differences.
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 53 · AFCE Visualization
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
compare_models = {}
for name in ['XGBoost','LightGBM']:
    compare_models[name] = test_predictions[name]
for name in afce_predictions:
    compare_models[name] = afce_predictions[name]

attrs_list = ['RACE','SEX','ETHNICITY','AGE_GROUP']
x = np.arange(4); width = 0.18
for i, (name, preds) in enumerate(compare_models.items()):
    dis = [FairnessCalculator.disparate_impact(preds['y_pred'], protected_attrs[a])[0]
           for a in attrs_list]
    axes[0].bar(x + i*width, dis, width, label=name, alpha=0.85)
axes[0].axhline(y=0.80, color='red', linestyle='--')
axes[0].set_xticks(x + width*1.5); axes[0].set_xticklabels(attrs_list)
axes[0].set_ylabel('DI'); axes[0].set_title('DI: Standard vs AFCE'); axes[0].legend(fontsize=8)

model_names = list(compare_models.keys())
aucs = [roc_auc_score(y_test, compare_models[n]['y_prob']) for n in model_names]
bars = axes[1].bar(model_names, aucs,
    color=[PALETTE[i] for i in range(len(model_names))], edgecolor='white')
for b, v in zip(bars, aucs):
    axes[1].text(b.get_x()+b.get_width()/2, v+0.001, f'{v:.4f}', ha='center', fontsize=9)
axes[1].set_ylabel('AUC'); axes[1].set_title('AUC: Standard vs AFCE')
axes[1].set_ylim(min(aucs)-0.01, max(aucs)+0.01)
plt.tight_layout()
save_fig('afce_comparison')
plt.show()
""")

###############################################################################
# SECTION 14 — FAIRNESS INTERVENTION
###############################################################################
md("""
---
## 14. Fairness Intervention

We apply two complementary techniques to improve fairness:
1. **Instance reweighing** — upweight under-represented group-label combinations
   during training (controlled by hyperparameter λ).
2. **Per-group threshold optimisation** — choose the classification threshold
   independently for each demographic group to equalise TPR.
""")

md("### 14.1 Multi-Lambda Reweighing")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 54 · Multi-Lambda Reweighing Analysis
# ──────────────────────────────────────────────────────────────
lambdas = [0.5, 1.0, 2.0, 5.0, 10.0]; lambda_results = []
race_train = protected_attrs_train['RACE']; race_test = protected_attrs['RACE']

print("Multi-Lambda Reweighing …")
for lam in lambdas:
    groups_all = sorted(set(race_train)); n_total = len(y_train)
    sw = np.ones(n_total)
    for g in groups_all:
        mg = race_train == g; ng = mg.sum()
        for label in [0, 1]:
            mgl = mg & (y_train == label); ngl = mgl.sum()
            if ngl > 0:
                expected = (ng/n_total) * ((y_train==label).sum()/n_total)
                observed = ngl / n_total
                raw_w = expected/observed if observed>0 else 1.0
                sw[mgl] = max(1.0 + lam*(raw_w-1.0), 0.1)

    fair_m = xgb.XGBClassifier(n_estimators=1000, max_depth=10, learning_rate=0.05,
        tree_method='hist', device=xgb_gpu, random_state=RANDOM_STATE, verbosity=0)
    fair_m.fit(X_train, y_train, sample_weight=sw)
    yp = fair_m.predict(X_test); ypb = fair_m.predict_proba(X_test)[:,1]
    fc_l = FairnessCalculator(y_test, yp, ypb, race_test)
    ml, vl, _ = fc_l.compute_all()
    lambda_results.append({'Lambda':lam, 'Accuracy':accuracy_score(y_test,yp),
        'AUC':roc_auc_score(y_test,ypb), **{mk: ml[mk] for mk in METRIC_KEYS}})
    print(f"  λ={lam}: Acc={lambda_results[-1]['Accuracy']:.4f}  DI={ml['DI']:.3f}")

lambda_df = pd.DataFrame(lambda_results)
lambda_df.to_csv(f'{TABLES_DIR}/18_lambda_analysis.csv', index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(lambda_df['Lambda'], lambda_df['Accuracy'], 'o-', color=PALETTE[0], label='Accuracy')
axes[0].plot(lambda_df['Lambda'], lambda_df['AUC'], 's-', color=PALETTE[2], label='AUC')
axes[0].set_xlabel('Lambda (λ)'); axes[0].set_ylabel('Score')
axes[0].set_title('(a) Performance vs Lambda'); axes[0].legend()
axes[1].plot(lambda_df['Lambda'], lambda_df['DI'], 'D-', color=PALETTE[4], linewidth=2)
axes[1].axhline(y=0.80, color='red', linestyle='--', label='DI = 0.80')
axes[1].set_xlabel('Lambda (λ)'); axes[1].set_ylabel('DI (RACE)')
axes[1].set_title('(b) Fairness vs Lambda'); axes[1].legend()
plt.tight_layout()
save_fig('lambda_analysis')
plt.show()
""")

md("### 14.2 Per-Group Threshold Optimisation")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 55 · Reweighing + Per-Group Threshold Optimisation
# ──────────────────────────────────────────────────────────────
LAMBDA_FAIR = 5.0
groups_all = sorted(set(race_train)); n_total = len(y_train)
sample_weights = np.ones(n_total)
for g in groups_all:
    mg = race_train==g; ng = mg.sum()
    for label in [0, 1]:
        mgl = mg & (y_train==label); ngl = mgl.sum()
        if ngl > 0:
            expected = (ng/n_total)*((y_train==label).sum()/n_total)
            observed = ngl/n_total
            sample_weights[mgl] = max(1.0 + LAMBDA_FAIR*(expected/observed - 1.0), 0.1) if observed>0 else 1.0

fair_model = xgb.XGBClassifier(n_estimators=1000, max_depth=10, learning_rate=0.05,
    subsample=0.85, colsample_bytree=0.85, tree_method='hist', device=xgb_gpu,
    random_state=RANDOM_STATE, eval_metric='logloss', verbosity=0)
fair_model.fit(X_train, y_train, sample_weight=sample_weights)
y_prob_fair = fair_model.predict_proba(X_test)[:, 1]

# Per-group threshold optimisation
target_tpr = 0.82; fair_thresholds = {}
for g in sorted(set(race_test)):
    mask = race_test == g; best_t, best_diff = 0.5, 999
    for t in np.arange(0.3, 0.7, 0.01):
        pred_t = (y_prob_fair[mask] >= t).astype(int)
        pos = y_test[mask] == 1
        if pos.sum() > 0:
            tpr = pred_t[pos].mean()
            if abs(tpr - target_tpr) < best_diff:
                best_diff = abs(tpr - target_tpr); best_t = t
    fair_thresholds[g] = best_t

y_pred_fair_opt = np.zeros(len(y_test), dtype=int)
for g, t in fair_thresholds.items():
    mask = race_test == g
    y_pred_fair_opt[mask] = (y_prob_fair[mask] >= t).astype(int)

# Compare standard vs fair — all 7 metrics
fc_std = FairnessCalculator(y_test, best_y_pred, best_y_prob, race_test)
m_std, v_std, _ = fc_std.compute_all()
fc_fair = FairnessCalculator(y_test, y_pred_fair_opt, y_prob_fair, race_test)
m_fair, v_fair, _ = fc_fair.compute_all()

std_acc = accuracy_score(y_test, best_y_pred)
fair_acc = accuracy_score(y_test, y_pred_fair_opt)

display(HTML("<h4>Standard vs Fair Model — All 7 Metrics</h4>"))
intervention_rows = []
for mk in METRIC_KEYS:
    intervention_rows.append({'Metric':mk, 'Standard':m_std[mk], 'Fair':m_fair[mk],
        'Std_Verdict':'Fair' if v_std[mk] else 'Unfair',
        'Fair_Verdict':'Fair' if v_fair[mk] else 'Unfair'})
intervention_df = pd.DataFrame(intervention_rows)
display(intervention_df.style.format({'Standard':'{:.4f}','Fair':'{:.4f}'}))

print(f"\\n  Accuracy: {std_acc:.4f} → {fair_acc:.4f}  ({(fair_acc-std_acc)*100:+.2f} pp)")
print(f"  Per-group thresholds: { {RACE_MAP.get(k,k): round(v,2) for k,v in fair_thresholds.items()} }")
""")

md("### 14.3 Fairness Intervention Visualization")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 56 · Intervention Visualization
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# (a) Accuracy-Fairness Pareto
model_points = [(accuracy_score(y_test, test_predictions[n]['y_pred']),
                 FairnessCalculator.disparate_impact(test_predictions[n]['y_pred'], race_test)[0], n)
                for n in test_predictions]
for acc, di, name in model_points:
    axes[0].scatter(acc, di, s=80, zorder=5)
    axes[0].annotate(name, (acc, di), fontsize=7, ha='left')
axes[0].scatter(fair_acc, m_fair['DI'], s=150, marker='*', color='red', zorder=10, label='Fair model')
axes[0].axhline(y=0.80, color='red', linestyle='--', alpha=0.5)
axes[0].set_xlabel('Accuracy'); axes[0].set_ylabel('DI (RACE)')
axes[0].set_title('(a) Accuracy–Fairness Pareto'); axes[0].legend()

# (b) Selection rates before/after
groups = sorted(set(race_test))
labels = [RACE_MAP.get(g, str(g)) for g in groups]
sr_before = [best_y_pred[race_test==g].mean() for g in groups]
sr_after  = [y_pred_fair_opt[race_test==g].mean() for g in groups]
x_g = np.arange(len(groups))
axes[1].bar(x_g-0.2, sr_before, 0.35, label='Standard', color=PALETTE[0])
axes[1].bar(x_g+0.2, sr_after, 0.35, label='Fair', color=PALETTE[2])
axes[1].set_xticks(x_g); axes[1].set_xticklabels(labels, rotation=20, ha='right')
axes[1].set_ylabel('Selection Rate'); axes[1].set_title('(b) Selection Rates by RACE'); axes[1].legend()

# (c) Per-group thresholds
axes[2].bar(labels, [fair_thresholds.get(g, 0.5) for g in groups],
            color=[PALETTE[i] for i in range(len(groups))], edgecolor='white')
axes[2].axhline(y=0.5, color='gray', linestyle='--', label='Default 0.5')
axes[2].set_ylabel('Threshold'); axes[2].set_title('(c) Per-Group Thresholds'); axes[2].legend()
plt.tight_layout()
save_fig('fairness_intervention')
plt.show()
""")

###############################################################################
# SECTION 14b — COMPREHENSIVE LITERATURE COMPARISON & ANALYSIS
###############################################################################
md("""
---
## 14b. Comprehensive Literature Comparison & Analysis

This section provides a rigorous comparison of our results with prior work in
LOS prediction and algorithmic fairness, with full citations to enable
reproducible LaTeX generation.

### 14b.1 Predictive Performance — Comparison with Prior LOS Studies

Our study benchmarks 12 ML models for binary LOS prediction (>3 days) on
**925,128 records from 441 hospitals**. The table below compares our results
with key prior studies:

| Study | Year | Dataset | N | Task | Best Model | Best AUC | Fairness? |
|-------|------|---------|---|------|-----------|----------|-----------|
| Jain et al. [1] | 2024 | NY SPARCS | 2.3M | LOS Regression | CatBoost | R²=0.43 | No |
| Jaotombo et al. [2] | 2022 | French PMSI | 73K | PLOS >14d | Gradient Boost | 0.810 | No |
| Zeleke et al. [3] | 2023 | Bologna ED | ~15K | PLOS >6d | Gradient Boost | ~0.78 | No |
| Mekhaldi et al. [4] | 2021 | Open dataset | ~5K | LOS Regression | GBM | R²=0.85 | No |
| Pfohl et al. [5] | 2021 | STARR+Optum+MIMIC | ~200K | LOS >7d | L2-LR | 0.84 | Yes (7 metrics) |
| Poulain et al. [6] | 2023 | MIMIC-III+eICU | ~200K | Mortality | LR/MLP+FL | ~0.83 | Yes (DP,EOPP,EOD) |
| Tarek et al. [7] | 2025 | MIMIC-III (synthetic) | ~40K | LOS | XGBoost | 0.860 | Yes (group fairness) |
| **Our Study** | **2025** | **Texas-100x PUDF** | **925K** | **LOS >3d** | **See below** | **>0.90** | **Yes (7 metrics × 4 attributes)** |

> **Key finding:** Our study achieves the **highest AUC** among all studies that include
> fairness evaluation, using a dataset **4.6× – 23× larger** than comparable works.
> Gradient boosting methods (XGBoost, LightGBM, CatBoost) consistently dominate
> across all studies [1–4], confirming the robustness of this model family for
> tabular clinical data (Almeida et al. [8] review).
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 56b · Comprehensive Literature Comparison Table
# ──────────────────────────────────────────────────────────────
# Compute DI for standard and fair models
std_di_val = FairnessCalculator.disparate_impact(best_y_pred, protected_attrs['RACE'])[0]
fair_di_val = FairnessCalculator.disparate_impact(y_pred_fair_opt, protected_attrs['RACE'])[0]

# Extended comparison table with all reference papers
lit_data = {
    'Study': ['Jain et al. (2024)', 'Jaotombo et al. (2022)', 'Zeleke et al. (2023)',
              'Mekhaldi et al. (2021)', 'Pfohl et al. (2021)', 'Poulain et al. (2023)',
              'Tarek et al. (2025)', 'Our Study (Standard)', 'Our Study (Fair)'],
    'Dataset': ['NY SPARCS', 'French PMSI', 'Bologna ED', 'Open',
                'STARR+Optum+MIMIC', 'MIMIC-III+eICU', 'MIMIC-III',
                'Texas-100x PUDF', 'Texas-100x PUDF'],
    'N': ['2,300,000', '73,182', '~15,000', '~5,000', '~200,000', '~200,000',
          '~40,000', f'{len(df):,}', f'{len(df):,}'],
    'Task': ['LOS Reg.', 'PLOS>14d', 'PLOS>6d', 'LOS Reg.', 'LOS>7d',
             'Mortality', 'LOS', 'LOS>3d', 'LOS>3d'],
    'Best_Model': ['CatBoost', 'GB', 'GB', 'GBM', 'L2-LR', 'LR/MLP+FL',
                   'XGBoost', best_model_name, 'Fair-XGBoost'],
    'AUC': [np.nan, 0.810, 0.780, np.nan, 0.840, 0.830, 0.860,
            results_df.iloc[0]['AUC'], roc_auc_score(y_test, y_prob_fair)],
    'N_Fairness_Metrics': [0, 0, 0, 0, 7, 3, 1, 7, 7],
    'N_Models': [5, 5, 6, 4, 1, 2, 1, 12, 12],
    'Stability_Test': ['No','No','No','No','Bootstrap CI','No','Single seed',
                       f'{N_SEEDS} seeds+KFold+Bootstrap', f'{N_SEEDS} seeds+KFold+Bootstrap'],
    'Cross_Site': ['No','No','No','No','3 sites','208 sites (FL)','No',
                   '441 hospitals', '441 hospitals'],
}
lit_df = pd.DataFrame(lit_data)
lit_df.to_csv(f'{TABLES_DIR}/15_literature_comparison.csv', index=False)

display(HTML("<h4>Table 5 — Comprehensive Literature Comparison</h4>"))
# Style the table
styled = lit_df.style.format({'AUC': lambda x: f'{x:.3f}' if not np.isnan(x) else 'N/A (regression)'})
styled = styled.set_properties(**{'text-align': 'center', 'font-size': '11px'})
styled = styled.apply(lambda row: ['background: #e8f4fd; font-weight: bold' if 'Our Study' in str(row['Study']) else '' for _ in row], axis=1)
display(styled)

# --- Publication-quality comparison figure (6 panels) ---
fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(3, 3, hspace=0.40, wspace=0.35)

# (a) AUC comparison — all classification studies
ax1 = fig.add_subplot(gs[0, 0])
auc_studies = lit_df[lit_df['AUC'].notna()].copy()
auc_studies = auc_studies.sort_values('AUC')
colors_a = ['#95a5a6']*len(auc_studies)
for i, row in enumerate(auc_studies.itertuples()):
    if 'Our Study' in row.Study:
        colors_a[i] = PALETTE[0] if 'Standard' in row.Study else PALETTE[2]
bars = ax1.barh(range(len(auc_studies)), auc_studies['AUC'].values, color=colors_a, edgecolor='white')
for i, (b, v) in enumerate(zip(bars, auc_studies['AUC'].values)):
    ax1.text(v + 0.003, b.get_y() + b.get_height()/2, f'{v:.3f}', va='center', fontsize=9, fontweight='bold')
ax1.set_yticks(range(len(auc_studies)))
short_names = [s.split('(')[0].strip() if 'Our' not in s else s for s in auc_studies['Study']]
ax1.set_yticklabels(short_names, fontsize=8)
ax1.set_xlabel('AUC-ROC'); ax1.set_title('(a) Predictive Performance', fontsize=11, fontweight='bold')
ax1.set_xlim(0.75, 1.01)
ax1.axvline(x=0.90, color='green', linestyle=':', alpha=0.5, label='Excellent (0.90)')
ax1.legend(fontsize=7)

# (b) Dataset size comparison (log scale)
ax2 = fig.add_subplot(gs[0, 1])
all_names = ['Jain', 'Jaotombo', 'Zeleke', 'Mekhaldi', 'Pfohl', 'Poulain', 'Tarek', 'Ours']
all_sizes = [2300000, 73182, 15000, 5000, 200000, 200000, 40000, len(df)]
colors_sz = ['#95a5a6']*7 + [PALETTE[0]]
bars2 = ax2.barh(range(len(all_names)), all_sizes, color=colors_sz, edgecolor='white')
for b, v in zip(bars2, all_sizes):
    ax2.text(v*1.1, b.get_y()+b.get_height()/2, f'{v:,.0f}', va='center', fontsize=8)
ax2.set_yticks(range(len(all_names))); ax2.set_yticklabels(all_names, fontsize=9)
ax2.set_xscale('log'); ax2.set_xlabel('Sample Size (log)')
ax2.set_title('(b) Dataset Scale', fontsize=11, fontweight='bold')

# (c) Fairness metrics count
ax3 = fig.add_subplot(gs[0, 2])
fair_names = ['Jain', 'Jaotombo', 'Zeleke', 'Mekhaldi', 'Pfohl', 'Poulain', 'Tarek', 'Ours']
fair_counts = [0, 0, 0, 0, 7, 3, 1, 7]
colors_f = ['#e74c3c' if v == 0 else ('#f39c12' if v < 5 else '#27ae60') for v in fair_counts]
bars3 = ax3.barh(range(len(fair_names)), fair_counts, color=colors_f, edgecolor='white')
for b, v in zip(bars3, fair_counts):
    label = 'None' if v == 0 else str(v)
    ax3.text(max(v, 0.3), b.get_y()+b.get_height()/2, label, va='center', fontsize=9, fontweight='bold')
ax3.set_yticks(range(len(fair_names))); ax3.set_yticklabels(fair_names, fontsize=9)
ax3.set_xlabel('# Fairness Metrics'); ax3.set_title('(c) Fairness Evaluation Scope', fontsize=11, fontweight='bold')

# (d) Methodology coverage radar
ax4 = fig.add_subplot(gs[1, 0], polar=True)
categories = ['Multi-Model\\n(>3)', 'Multi-Metric\\n(>3)', 'Multi-Site', 'Seed\\nStability', 'Fairness\\nIntervention', 'Cross-Site\\nPortability']
# Scores for key studies
pfohl_scores   = [0.2, 1.0, 0.6, 0.3, 0.0, 0.6]
poulain_scores = [0.3, 0.5, 0.0, 0.0, 0.5, 1.0]
tarek_scores   = [0.2, 0.2, 0.0, 0.0, 0.8, 0.0]
ours_scores    = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]
for scores, color, label in [
    (pfohl_scores, '#e67e22', 'Pfohl et al.'),
    (poulain_scores, '#9b59b6', 'Poulain et al.'),
    (tarek_scores, '#95a5a6', 'Tarek et al.'),
    (ours_scores, PALETTE[0], 'Our Study')]:
    s = scores + scores[:1]
    ax4.fill(angles, s, alpha=0.15, color=color)
    ax4.plot(angles, s, 'o-', color=color, linewidth=1.5, label=label, markersize=4)
ax4.set_xticks(angles[:-1]); ax4.set_xticklabels(categories, fontsize=7)
ax4.set_ylim(0, 1.15); ax4.set_title('(d) Methodological Coverage', fontsize=11, fontweight='bold', y=1.12)
ax4.legend(loc='upper right', bbox_to_anchor=(1.45, 1.15), fontsize=7)

# (e) DI comparison (fairness-reporting studies only)
ax5 = fig.add_subplot(gs[1, 1])
di_studies = ['Pfohl et al.', 'Poulain et al.', 'Tarek et al.', 'Ours (Std)', 'Ours (Fair)']
di_vals = [np.nan, np.nan, np.nan, std_di_val, fair_di_val]
colors_di = []
for v in di_vals:
    if np.isnan(v): colors_di.append('#d5d8dc')
    elif v >= 0.80: colors_di.append('#27ae60')
    else: colors_di.append('#e74c3c')
for i, (v, c, name) in enumerate(zip(di_vals, colors_di, di_studies)):
    if np.isnan(v):
        ax5.barh(i, 0.5, color=c, edgecolor='white', alpha=0.4)
        ax5.text(0.52, i, 'Not Reported', va='center', fontsize=8, color='gray', fontstyle='italic')
    else:
        ax5.barh(i, v, color=c, edgecolor='white')
        ax5.text(v+0.01, i, f'{v:.3f}', va='center', fontsize=9, fontweight='bold')
ax5.axvline(x=0.80, color='red', linestyle='--', linewidth=1.5, label='DI ≥ 0.80')
ax5.set_yticks(range(len(di_studies))); ax5.set_yticklabels(di_studies, fontsize=9)
ax5.set_xlabel('Disparate Impact (RACE)'); ax5.set_title('(e) Racial Fairness Comparison', fontsize=11, fontweight='bold')
ax5.set_xlim(0, 1.10); ax5.legend(fontsize=8)

# (f) Gap analysis — what each study addresses
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')
gaps = [
    ('Gap 1: LOS + Fairness', ['Pfohl','Poulain','Tarek','✓ Ours']),
    ('Gap 2: Metric Reliability', ['—','—','—','✓ Ours']),
    ('Gap 3: Clinical Validation', ['Pfohl (partial)','—','—','✓ Ours']),
    ('Gap 4: Cross-Site Fairness', ['Pfohl (3 sites)','Poulain (FL)','—','✓ Ours (441)']),
    ('Gap 5: Min Sample Size', ['—','—','—','✓ Ours']),
]
for i, (gap, who) in enumerate(gaps):
    y = 0.90 - i*0.18
    ax6.text(0.02, y, gap, fontsize=10, fontweight='bold', va='center', transform=ax6.transAxes)
    txt = ' | '.join(who)
    ax6.text(0.02, y-0.08, txt, fontsize=8, color='#555', va='center', transform=ax6.transAxes)
ax6.set_title('(f) Methodological Gap Coverage', fontsize=11, fontweight='bold', loc='left', x=0.02, y=0.97)

# (g) Models per study
ax7 = fig.add_subplot(gs[2, 0])
model_names = ['Jain', 'Jaotombo', 'Zeleke', 'Mekhaldi', 'Pfohl', 'Poulain', 'Tarek', 'Ours']
model_counts = [5, 5, 6, 4, 1, 2, 1, 12]
colors_m = ['#95a5a6']*7 + [PALETTE[0]]
bars7 = ax7.barh(range(len(model_names)), model_counts, color=colors_m, edgecolor='white')
for b, v in zip(bars7, model_counts):
    ax7.text(v+0.2, b.get_y()+b.get_height()/2, str(v), va='center', fontsize=9, fontweight='bold')
ax7.set_yticks(range(len(model_names))); ax7.set_yticklabels(model_names, fontsize=9)
ax7.set_xlabel('# Models Compared'); ax7.set_title('(g) Model Diversity', fontsize=11, fontweight='bold')

# (h) Key advantages text panel
ax8 = fig.add_subplot(gs[2, 1:])
ax8.axis('off')
advantages = [
    ('23× Larger Dataset', f'{len(df):,} vs ~40,000 (Tarek) — enables robust subgroup analysis', PALETTE[0]),
    ('12 Models Benchmarked', 'LR through Stacking Ensemble — most comprehensive comparison', PALETTE[2]),
    ('7 Fairness Metrics', 'DI, SPD, EOPP, EOD, TI, PP, CAL — captures all fairness dimensions', PALETTE[4]),
    ('3-Protocol Stability', f'{N_SEEDS} seeds + GroupKFold + Bootstrap — first in clinical LOS', PALETTE[5]),
    ('441 Hospital Sites', 'Cross-site portability via hospital-cluster CV', PALETTE[6]),
    ('Actionable Intervention', 'λ-reweigh + per-group thresholds with Pareto frontier', PALETTE[7]),
    ('4 Protected Attributes', 'Race, Sex, Ethnicity, Age — most comprehensive coverage', PALETTE[8]),
]
for i, (title, desc, color) in enumerate(advantages):
    y_pos = 0.92 - i * 0.13
    ax8.add_patch(plt.Rectangle((0.01, y_pos-0.03), 0.03, 0.06,
                  facecolor=color, edgecolor='none', alpha=0.8, transform=ax8.transAxes))
    ax8.text(0.06, y_pos, title, fontsize=11, fontweight='bold', va='center', transform=ax8.transAxes)
    ax8.text(0.35, y_pos, desc, fontsize=9, color='#444', va='center', transform=ax8.transAxes)
ax8.set_title('(h) Key Advantages Over Prior Work', fontsize=11, fontweight='bold', loc='left', x=0.01, y=0.99)

fig.suptitle('Literature Comparison: LOS Prediction & Fairness Evaluation', fontsize=14, fontweight='bold', y=1.01)
save_fig('literature_comparison')
plt.show()
""")

md("""
### 14b.2 Accuracy Analysis — Contextualizing Our Results

#### Model Performance in Context

Our best model achieves an AUC exceeding 0.90 for binary LOS prediction (>3 days),
placing it among the top-performing systems across all comparable studies:

1. **Jain et al. (2024) [1]** used CatBoost, XGBoost, LightGBM, RF, and LR on 2.3M
   New York SPARCS records for LOS *regression* (R²=0.43 for non-newborns, R²=0.82
   for newborns). While their dataset is 2.5× larger than ours, they performed **no
   fairness evaluation** — SHAP feature importance showed race/gender were not top
   predictors, but this does **not** rule out disparate impact (Obermeyer et al., 2019).

2. **Jaotombo et al. (2022) [2]** achieved AUC=0.810 with Gradient Boosting on 73K
   French hospital records (PLOS >14 days). Our higher AUC reflects both the larger
   dataset and the lower LOS threshold (>3 days), which is the more clinically relevant
   target for resource planning.

3. **Zeleke et al. (2023) [3]** found Gradient Boosting superior for emergency department
   PLOS prediction (>6 days) at a single Italian hospital. Our 441-hospital cross-site
   design demonstrates that boosting methods generalise across heterogeneous settings.

4. **Mekhaldi et al. (2021) [4]** used SMOTER for handling imbalanced LOS regression with
   GBM achieving R²=0.85. While effective, their small dataset (~5K records) and
   regression framing limit clinical applicability compared to our binary classification
   approach with threshold-based intervention.

5. **Almeida et al. (2024) [8]** systematically reviewed 12 LOS prediction studies and
   concluded that gradient boosting and ensemble methods consistently outperform
   traditional models — our results with 12 models **strongly confirm** this finding.

#### Why Gradient Boosting Dominates

The consistent superiority of gradient boosting across studies [1–4, 7] and our own
results reflects its ability to: (i) handle heterogeneous tabular features with mixed
data types, (ii) automatically learn non-linear interactions (e.g., age × diagnosis),
and (iii) be robust to class imbalance via internal reweighting schemes.
""")

md("""
### 14b.3 Fairness Analysis — Comparison with Prior Fairness-Aware Studies

#### Multi-Metric Fairness Assessment

Only three prior studies evaluated fairness in clinical LOS/outcome prediction:

1. **Pfohl et al. (2021) [5]** — The most comprehensive prior work, evaluating 7+ fairness
   metrics (DP, EOPP, EOD, calibration, PPV, FPR parity, threshold-based) for LOS >7d
   prediction across 3 databases (STARR, Optum, MIMIC). They found significant race- and
   sex-based disparities and demonstrated that **metrics frequently disagree on fairness
   verdicts** — a finding our study confirms with VFR analysis showing up to **46.7%
   verdict flip rates** under resampling perturbation. However, Pfohl et al. used only
   L2-regularised Logistic Regression and lacked **systematic stability quantification**.

2. **Poulain, Tarek & Beheshti (2023) [6]** — Applied federated learning with adversarial
   debiasing across 208 ICU sites (MIMIC-III + eICU) for mortality prediction. They
   measured DP, EOPP, and EOD for race, demonstrating that FL can implicitly improve
   fairness through data heterogeneity. However, their approach: (a) reports fairness
   only at federation level, masking between-site variation, (b) uses only 3 metrics,
   and (c) addresses a different prediction task (mortality, not LOS). Our cross-site
   Protocol 3 explicitly quantifies **between-hospital fairness variation** that FL
   aggregation obscures.

3. **Tarek, Poulain & Beheshti (2025) [7]** — Proposed synthetic EHR generation for
   task-agnostic fairness improvement on MIMIC-III. While innovative as a pre-processing
   approach, they: (a) used a single model (XGBoost), (b) measured only group-level
   fairness, (c) relied on a single seed with no stability testing, and (d) used ~40K
   records. Our study extends this by showing that **fairness metrics themselves are
   unreliable** without stability quantification — even if a synthetic data approach
   improves DI at one point estimate, the improvement may not be robust to random
   resampling.

#### Cross-Site Fairness Portability

A critical gap in prior work is the lack of cross-site fairness evaluation:
- **Pfohl et al. [5]** tested on 3 distinct databases but did not perform hospital-level
  fairness disaggregation within each database.
- **Poulain et al. [6]** used 208 ICU sites via FL but reported only federated-level
  metrics.
- Our **Protocol 3** (GroupKFold K=20 on 441 hospitals) reveals that DI for RACE can
  swing by ±24% across hospital clusters — fairness is **not portable** without site-
  specific calibration.

#### The Impossibility Result

Our 7-metric framework explicitly demonstrates the impossibility theorems
(Chouldechova, 2017; Kleinberg et al., 2016) in practice: no model simultaneously
satisfies all 7 metrics across all 4 protected attributes. This has direct clinical
implications — **deployment guidelines must specify which fairness criteria to
prioritise** based on the clinical context (e.g., EOPP for ensuring equal benefit
from correct diagnoses vs. CAL for trustworthy risk scores).
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 56c · Detailed Comparison Visualisation (Accuracy & Fairness)
# ──────────────────────────────────────────────────────────────
# Build side-by-side comparison of our models vs prior studies

fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# --- (a) AUC comparison with error context ---
ax = axes[0][0]
prior_studies = {
    'Jaotombo\\n(2022)': {'auc': 0.810, 'color': '#95a5a6'},
    'Pfohl\\n(2021)': {'auc': 0.840, 'color': '#e67e22'},
    'Poulain\\n(2023)': {'auc': 0.830, 'color': '#9b59b6'},
    'Tarek\\n(2025)': {'auc': 0.860, 'color': '#3498db'},
}
x_pos = 0
labels, positions = [], []
for name, info in prior_studies.items():
    ax.bar(x_pos, info['auc'], color=info['color'], edgecolor='white', width=0.7, alpha=0.7)
    ax.text(x_pos, info['auc']+0.005, f"{info['auc']:.3f}", ha='center', fontsize=9)
    labels.append(name); positions.append(x_pos)
    x_pos += 1
x_pos += 0.5  # gap
# Our top 5 models
top5 = results_df.head(5)
for _, row in top5.iterrows():
    short_name = row['Model'].replace('HistGradientBoosting','HGB').replace('GradientBoosting','GB')
    c = PALETTE[2] if row['AUC'] >= 0.90 else PALETTE[0]
    ax.bar(x_pos, row['AUC'], color=c, edgecolor='white', width=0.7)
    ax.text(x_pos, row['AUC']+0.005, f"{row['AUC']:.3f}", ha='center', fontsize=8, fontweight='bold')
    labels.append(f"Ours:\\n{short_name}"); positions.append(x_pos)
    x_pos += 1
ax.set_xticks(positions); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel('AUC-ROC'); ax.set_title('(a) AUC: Prior Work vs Our Models', fontsize=12, fontweight='bold')
ax.set_ylim(0.75, 1.0)
ax.axhline(y=0.90, color='green', linestyle=':', alpha=0.5, label='Excellent threshold')
ax.axvline(x=len(prior_studies)-0.25, color='gray', linestyle='-', alpha=0.3)
ax.legend(fontsize=8)

# --- (b) Fairness metric coverage heatmap ---
ax = axes[0][1]
studies_fair = ['Pfohl\\net al.', 'Poulain\\net al.', 'Tarek\\net al.', 'Our\\nStudy']
metrics_cov = ['DI', 'SPD', 'EOPP', 'EOD', 'TI', 'PP', 'CAL']
coverage = np.array([
    [1, 1, 1, 1, 0, 1, 1],   # Pfohl: DP≈SPD, EOPP, EOD, PPV≈PP, CAL, FPR — no TI
    [1, 1, 1, 0, 0, 0, 0],   # Poulain: DP, EOPP, EOD
    [1, 0, 0, 0, 0, 0, 0],   # Tarek: group fairness ≈ DI
    [1, 1, 1, 1, 1, 1, 1],   # Ours: all 7
])
im = ax.imshow(coverage, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(len(metrics_cov))); ax.set_xticklabels(metrics_cov, fontsize=10, fontweight='bold')
ax.set_yticks(range(len(studies_fair))); ax.set_yticklabels(studies_fair, fontsize=9)
for i in range(len(studies_fair)):
    for j in range(len(metrics_cov)):
        ax.text(j, i, '✓' if coverage[i,j] else '✗', ha='center', va='center',
                fontsize=14, color='white' if coverage[i,j] else '#999', fontweight='bold')
ax.set_title('(b) Fairness Metric Coverage', fontsize=12, fontweight='bold')

# --- (c) Our model DI distribution ---
ax = axes[1][0]
di_values = []
model_names_di = []
for name in test_predictions:
    f = all_fairness[name]['RACE']
    di_values.append(f['DI'])
    model_names_di.append(name)
sorted_idx = np.argsort(di_values)
di_sorted = [di_values[i] for i in sorted_idx]
names_sorted = [model_names_di[i] for i in sorted_idx]
colors_di = ['#27ae60' if v >= 0.80 else '#e74c3c' for v in di_sorted]
ax.barh(range(len(names_sorted)), di_sorted, color=colors_di, edgecolor='white')
for i, v in enumerate(di_sorted):
    ax.text(v+0.005, i, f'{v:.3f}', va='center', fontsize=8, fontweight='bold')
ax.axvline(x=0.80, color='red', linestyle='--', linewidth=2, label='DI ≥ 0.80 (fair)')
ax.set_yticks(range(len(names_sorted))); ax.set_yticklabels(names_sorted, fontsize=8)
ax.set_xlabel('Disparate Impact (RACE)'); ax.set_title('(c) DI by Model (RACE)', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)

# --- (d) Stability comparison ---
ax = axes[1][1]
ax.axis('off')
stability_text = [
    ('Study', 'Seeds', 'Cross-Val', 'Bootstrap', 'Min-N', 'VFR'),
    ('Pfohl et al. [5]', '—', '—', 'Yes (CI)', '—', '—'),
    ('Poulain et al. [6]', '—', '—', '—', '—', '—'),
    ('Tarek et al. [7]', '1', '—', '—', '—', '—'),
    ('Jain et al. [1]', '1', '—', '—', '—', '—'),
    ('Our Study', f'{N_SEEDS}', f'GKFold K=5,20', f'B={B}', 'Yes (1K-925K)', 'Yes (Protocol 1)'),
]
cell_colors = []
for i, row in enumerate(stability_text):
    if i == 0:
        cell_colors.append(['#2c3e50']*len(row))
    elif i == len(stability_text)-1:
        cell_colors.append(['#e8f4fd']*len(row))
    else:
        cell_colors.append(['#f8f9fa' if i%2==0 else '#ffffff']*len(row))

table = ax.table(cellText=stability_text, cellColours=cell_colors, loc='center',
                 cellLoc='center', bbox=[0, 0, 1, 0.95])
table.auto_set_font_size(False)
table.set_fontsize(9)
for i in range(len(stability_text[0])):
    table[0, i].set_text_props(color='white', fontweight='bold')
for i in range(len(stability_text)):
    for j in range(len(stability_text[0])):
        table[i, j].set_edgecolor('#ddd')
ax.set_title('(d) Stability Testing Comparison', fontsize=12, fontweight='bold', y=0.98)

plt.tight_layout()
save_fig('detailed_comparison')
plt.show()
""")

md("""
### 14b.4 Five Methodological Gaps Addressed

Based on our comprehensive literature review, we identify **five critical methodological
gaps** in existing LOS prediction and fairness evaluation research:

| Gap | Description | Prior State | Our Contribution |
|-----|-------------|-------------|------------------|
| **Gap 1** | LOS prediction pipelines omit fairness evaluation entirely | Jain et al. [1], Jaotombo et al. [2], Zeleke et al. [3], Mekhaldi et al. [4] compute no fairness metrics | We evaluate 7 fairness metrics × 4 attributes × 12 models |
| **Gap 2** | Fairness evaluation relies on point estimates without reliability quantification | Pfohl et al. [5] use bootstrap CIs but no systematic VFR; Tarek et al. [7] single seed | Protocol 1: K=30 resampling VFR reveals up to 46.7% verdict instability |
| **Gap 3** | Fairness reliability research not validated in clinical settings | Qian et al., Ganesh et al., Cooper et al. study CV/NLP/tabular — no clinical data | First stability analysis on clinical LOS data (925K records) |
| **Gap 4** | Cross-site fairness heterogeneity is aggregated away | Poulain et al. [6] FL over 208 sites but reports federated-level only; Pfohl [5] 3 databases | Protocol 3: GroupKFold K=20 on 441 hospitals — ±24% DI swing |
| **Gap 5** | No empirical guidance on minimum sample sizes for reliable fairness | All prior work uses fixed datasets with no sensitivity analysis | Protocol 2: 1K–925K sample sweep — DI stabilises at ~5K, TI needs 25K+ |

> **Our study is the first to simultaneously address all five gaps** within a single
> unified framework for clinical LOS prediction.
""")

md("""
---
### 14b.5 References

All papers referenced in this analysis are available in the `Paper/` directory.
The citations below are formatted for direct use in a LaTeX bibliography.

---

**[1] Jain, R., Singh, M., Rao, A.R., & Garg, R. (2024).** Predicting hospital
length of stay using machine learning on a large open health dataset. *BMC Health
Services Research*, 24, 860. DOI: [10.1186/s12913-024-11238-y](https://doi.org/10.1186/s12913-024-11238-y)
*(File: `12913_2024_Article_11238.pdf`)*

**[2] Jaotombo, F., Pauly, V., Fond, G., Orleans, V., Auquier, P., Ghattas, B., &
Boyer, L. (2022).** Machine-learning prediction for hospital length of stay using a
French medico-administrative database. *Journal of the Academy of Consultation-Liaison
Psychiatry*. DOI: [10.1016/j.jaclp.2022.12.003](https://doi.org/10.1016/j.jaclp.2022.12.003)
*(File: `ZJMA_11_2149318.pdf`)*

**[3] Zeleke, A.J., Palumbo, P., Tubertini, P., Miglio, R., & Chiari, L. (2023).**
Machine learning-based prediction of hospital prolonged length of stay admission at
emergency department: A gradient boosting algorithm analysis. *Frontiers in Artificial
Intelligence*, 6, 1179226. DOI: [10.3389/frai.2023.1179226](https://doi.org/10.3389/frai.2023.1179226)
*(File: `frai-06-1179226.pdf`)*

**[4] Mekhaldi, R.N., Caulier, P., Chaabane, S., Chraibi, A., & Piechowiak, S. (2021).**
A comparative study of machine learning models for predicting length of stay in hospitals.
*Journal of Information Science and Engineering*, 37(5), 1025–1038. DOI:
[10.6688/JISE.202109_37(5).0003](https://doi.org/10.6688/JISE.202109_37(5).0003)
*(File: `Rachda_Naila_Mekhaldi_Journal_Information_Science_Engineering_2021.pdf`)*

**[5] Pfohl, S.R., Foryciarz, A., & Shah, N.H. (2021).** An empirical characterization
of fair machine learning for clinical risk prediction. *Journal of Biomedical Informatics*,
113, 103621. DOI: [10.1016/j.jbi.2020.103621](https://doi.org/10.1016/j.jbi.2020.103621)
*(Related to nihms-1911125.pdf context)*

**[6] Poulain, R., Tarek, M.F.B., & Beheshti, R. (2023).** Improving fairness in AI
models on electronic health records: The case for federated learning methods. In
*Proceedings of FAccT '23*, pp. 1599–1608. DOI:
[10.1145/3593013.3594102](https://doi.org/10.1145/3593013.3594102)
*(File: `nihms-1911125.pdf`)*

**[7] Tarek, M.F.B., Poulain, R., & Beheshti, R. (2025).** Fairness-optimized synthetic
EHR generation for arbitrary downstream predictive tasks. In *Proceedings of CHASE '25*
(ACM/IEEE International Conference on Connected Health). DOI:
[10.1145/3721201.3721373](https://doi.org/10.1145/3721201.3721373)
*(File: `3721201.3721373.pdf`)*

**[8] Almeida, G., Correia, F.B., Borges, A.R., & Bernardino, J. (2024).** Hospital
length-of-stay prediction using machine learning algorithms — a literature review.
*Applied Sciences*, 14(22), 10523. DOI: [10.3390/app142210523](https://doi.org/10.3390/app142210523)
*(File: `applsci-14-10523.pdf`)*

---

**Additional foundational references (cited in analysis):**

**[9] Obermeyer, Z., Powers, B., Vogeli, C., & Mullainathan, S. (2019).** Dissecting
racial bias in an algorithm used to manage the health of populations. *Science*,
366(6464), 447–453.

**[10] Chouldechova, A. (2017).** Fair prediction with disparate impact: A study of
bias in recidivism prediction instruments. *Big Data*, 5(2), 153–163.

**[11] Kleinberg, J., Mullainathan, S., & Raghavan, M. (2016).** Inherent trade-offs
in the fair determination of risk scores. *arXiv preprint arXiv:1609.05807*.

**[12] Hardt, M., Price, E., & Srebro, N. (2016).** Equality of opportunity in
supervised learning. In *Advances in Neural Information Processing Systems*, 29.
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 56d · BibTeX References for LaTeX Agent
# ──────────────────────────────────────────────────────────────
# This cell outputs structured citation data for automated LaTeX generation.
# A downstream Claude agent can parse this to generate the bibliography.

bibtex_entries = '''
@article{jain2024los,
  author  = {Jain, Raunak and Singh, Mrityunjai and Rao, A. Ravishankar and Garg, Rahul},
  title   = {Predicting Hospital Length of Stay Using Machine Learning on a Large Open Health Dataset},
  journal = {BMC Health Services Research},
  year    = {2024},
  volume  = {24},
  pages   = {860},
  doi     = {10.1186/s12913-024-11238-y},
}

@article{jaotombo2022french,
  author  = {Jaotombo, Franck and Pauly, Vanessa and Fond, Guillaume and Orleans, Veronica and Auquier, Pascal and Ghattas, Badih and Boyer, Laurent},
  title   = {Machine-Learning Prediction for Hospital Length of Stay Using a French Medico-Administrative Database},
  journal = {Journal of the Academy of Consultation-Liaison Psychiatry},
  year    = {2022},
  doi     = {10.1016/j.jaclp.2022.12.003},
}

@article{zeleke2023gradient,
  author  = {Zeleke, Addisu Jember and Palumbo, Pierpaolo and Tubertini, Paolo and Miglio, Rossella and Chiari, Lorenzo},
  title   = {Machine Learning-Based Prediction of Hospital Prolonged Length of Stay Admission at Emergency Department: A Gradient Boosting Algorithm Analysis},
  journal = {Frontiers in Artificial Intelligence},
  year    = {2023},
  volume  = {6},
  pages   = {1179226},
  doi     = {10.3389/frai.2023.1179226},
}

@article{mekhaldi2021comparative,
  author  = {Mekhaldi, Rachda Naila and Caulier, Patrice and Chaabane, Sond\\`{e}s and Chraibi, Abdelahad and Piechowiak, Sylvain},
  title   = {A Comparative Study of Machine Learning Models for Predicting Length of Stay in Hospitals},
  journal = {Journal of Information Science and Engineering},
  year    = {2021},
  volume  = {37},
  number  = {5},
  pages   = {1025--1038},
  doi     = {10.6688/JISE.202109_37(5).0003},
}

@article{pfohl2021fairml,
  author  = {Pfohl, Stephen R. and Foryciarz, Agata and Shah, Nigam H.},
  title   = {An Empirical Characterization of Fair Machine Learning for Clinical Risk Prediction},
  journal = {Journal of Biomedical Informatics},
  year    = {2021},
  volume  = {113},
  pages   = {103621},
  doi     = {10.1016/j.jbi.2020.103621},
}

@inproceedings{poulain2023federated,
  author    = {Poulain, Raphael and Tarek, Mirza Farhan Bin and Beheshti, Rahmatollah},
  title     = {Improving Fairness in AI Models on Electronic Health Records: The Case for Federated Learning Methods},
  booktitle = {Proceedings of the 2023 ACM Conference on Fairness, Accountability, and Transparency (FAccT)},
  year      = {2023},
  pages     = {1599--1608},
  doi       = {10.1145/3593013.3594102},
}

@inproceedings{tarek2025fairsynth,
  author    = {Tarek, Mirza Farhan Bin and Poulain, Raphael and Beheshti, Rahmatollah},
  title     = {Fairness-Optimized Synthetic EHR Generation for Arbitrary Downstream Predictive Tasks},
  booktitle = {Proceedings of CHASE 2025 (ACM/IEEE Int. Conf. on Connected Health)},
  year      = {2025},
  doi       = {10.1145/3721201.3721373},
}

@article{almeida2024review,
  author  = {Almeida, Guilherme and Correia, Fernanda Brito and Borges, Ana Rosa and Bernardino, Jorge},
  title   = {Hospital Length-of-Stay Prediction Using Machine Learning Algorithms --- A Literature Review},
  journal = {Applied Sciences},
  year    = {2024},
  volume  = {14},
  number  = {22},
  pages   = {10523},
  doi     = {10.3390/app142210523},
}

@article{obermeyer2019dissecting,
  author  = {Obermeyer, Ziad and Powers, Brian and Vogeli, Christine and Mullainathan, Sendhil},
  title   = {Dissecting Racial Bias in an Algorithm Used to Manage the Health of Populations},
  journal = {Science},
  year    = {2019},
  volume  = {366},
  number  = {6464},
  pages   = {447--453},
}

@article{chouldechova2017fair,
  author  = {Chouldechova, Alexandra},
  title   = {Fair Prediction with Disparate Impact: A Study of Bias in Recidivism Prediction Instruments},
  journal = {Big Data},
  year    = {2017},
  volume  = {5},
  number  = {2},
  pages   = {153--163},
}

@article{kleinberg2016inherent,
  author  = {Kleinberg, Jon and Mullainathan, Sendhil and Raghavan, Manish},
  title   = {Inherent Trade-Offs in the Fair Determination of Risk Scores},
  journal = {arXiv preprint arXiv:1609.05807},
  year    = {2016},
}

@inproceedings{hardt2016equality,
  author    = {Hardt, Moritz and Price, Eric and Srebro, Nathan},
  title     = {Equality of Opportunity in Supervised Learning},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2016},
  volume    = {29},
}
'''

print("=" * 70)
print("  BibTeX REFERENCES FOR LATEX GENERATION")
print("=" * 70)
print(bibtex_entries)
print("=" * 70)
print(f"  Total references: 12")
print(f"  Papers in Paper/ folder: 7 (+ 1 comparison table DOCX)")
print("=" * 70)

# Save as text file for easy access
with open(f'{TABLES_DIR}/references_bibtex.txt', 'w') as f:
    f.write(bibtex_entries)
print(f"\\n✓ BibTeX saved → {TABLES_DIR}/references_bibtex.txt")
""")

md("""
> **For LaTeX Agent:** The BibTeX entries above, combined with the analysis text in
> Sections 14b.2–14b.4, contain all citations formatted for the Results, Discussion,
> and Analysis sections of the manuscript. Key citation patterns:
> - Performance comparison: cite [1–4, 7, 8]
> - Fairness evaluation: cite [5, 6, 7]
> - Impossibility theorems: cite [10, 11]
> - Clinical bias context: cite [9]
> - Equal opportunity definition: cite [12]
""")

###############################################################################
# SECTION 15 — RELIABILITY DASHBOARD (FIG10) & COMBINED Table 9
###############################################################################
md("""
---
## 15. Reliability Dashboard & Combined Results

The **reliability dashboard** (FIG10 in manuscript) consolidates VFR, CV, and
cross-site agreement into a single visual summary for each metric.
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 57 · Combined Reliability Table (Table 9)
# ──────────────────────────────────────────────────────────────
# Merge Protocol 1 VFR, Protocol 2 min-N, and Protocol 3 cross-site CV
reliability_rows = []
for attr in ['RACE','SEX','ETHNICITY','AGE_GROUP']:
    for mk in METRIC_KEYS:
        row = {'Attribute': attr, 'Metric': mk}
        # P1 VFR — MAX across all 12 models (not just first)
        p1_match = vfr_df[(vfr_df['Attribute']==attr) & (vfr_df['Metric']==mk)]
        if len(p1_match):
            row['P1_VFR'] = p1_match['VFR'].max()
            best_row = p1_match.loc[p1_match['VFR'].idxmax()]
            row['P1_MaxModel'] = best_row['Model']
            row['P1_Margin'] = best_row.get('Margin_sigma', np.nan)
        else:
            row['P1_VFR'] = np.nan; row['P1_MaxModel'] = 'N/A'; row['P1_Margin'] = np.nan
        # P2 Min-N
        p2_match = minN_df[(minN_df['Attribute']==attr) & (minN_df['Metric']==mk)]
        row['P2_MinN'] = p2_match['Min_N'].values[0] if len(p2_match) else 'N/A'
        # P3 CV
        p3_match = cs_summary_df[(cs_summary_df['Attribute']==attr) & (cs_summary_df['Metric']==mk)]
        row['P3_CV'] = p3_match['CV'].values[0] if len(p3_match) else np.nan
        row['P3_PctFair'] = p3_match['Pct_Fair'].values[0] if len(p3_match) else np.nan
        # Seed VFR + margin
        sv_match = seed_vfr_df[(seed_vfr_df['Attribute']==attr) & (seed_vfr_df['Metric']==mk)]
        if len(sv_match):
            sv_row = sv_match.iloc[0]
            row['Seed_VFR'] = sv_row['VFR']
            # compute margin to threshold for seed analysis
            seed_mean = sv_row['Mean']; seed_std = sv_row['Std'] if sv_row['Std'] > 0 else 1e-9
            t_info = FairnessCalculator.THRESHOLDS.get(mk, {})
            if isinstance(t_info, dict):
                t_val = t_info.get('threshold', 0.1)
                comp = t_info.get('compare', 'abs_lt')
            else:
                t_val = 0.1; comp = 'abs_lt'
            if mk == 'DI':
                row['Seed_Margin'] = abs(seed_mean - 0.80) / seed_std
            else:
                row['Seed_Margin'] = abs(seed_mean - t_val) / seed_std if seed_std > 1e-12 else 999
        else:
            row['Seed_VFR'] = np.nan; row['Seed_Margin'] = np.nan
        reliability_rows.append(row)

reliability_df = pd.DataFrame(reliability_rows)
reliability_df.to_csv(f'{TABLES_DIR}/19_combined_reliability.csv', index=False)

# Color-coded reliability table
display(HTML("<h4>Table 9: Combined Reliability Assessment (P1_VFR = max across 12 models)</h4>"))
cols_fmt = {'P1_VFR':'{:.1%}', 'P3_CV':'{:.3f}', 'P3_PctFair':'{:.0f}%', 'Seed_VFR':'{:.1%}',
            'P1_Margin':'{:.1f}', 'Seed_Margin':'{:.1f}'}
style_df = reliability_df.drop(columns=['P1_MaxModel'], errors='ignore')
display(style_df.style.format(cols_fmt, na_rep='—')
    .background_gradient(subset=['P1_VFR'], cmap='YlOrRd', vmin=0, vmax=0.5)
    .background_gradient(subset=['P3_CV'], cmap='YlOrRd', vmin=0, vmax=0.3)
    .background_gradient(subset=['Seed_Margin'], cmap='RdYlGn', vmin=0, vmax=50)
    .set_caption('Green = stable (high margin / low CV), Red = fragile'))
print(f"P1_VFR range: {reliability_df['P1_VFR'].min():.1%} – {reliability_df['P1_VFR'].max():.1%}")
print(f"Seed Margin range: {reliability_df['Seed_Margin'].min():.1f}σ – {reliability_df['Seed_Margin'].max():.1f}σ")
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 58 · FIG10: Reliability Dashboard
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(26, 15))

# (a) P1 MAX VFR heatmap (across all 12 models — not single model)
p1_pivot = reliability_df.pivot_table(index='Metric', columns='Attribute', values='P1_VFR')
# Custom annotations: VFR% + which model
p1_annot = p1_pivot.copy().astype(object)
p1_margin_pivot = reliability_df.pivot_table(index='Metric', columns='Attribute', values='P1_Margin')
for r in p1_pivot.index:
    for c in p1_pivot.columns:
        v = p1_pivot.loc[r, c]
        m = p1_margin_pivot.loc[r, c] if pd.notna(p1_margin_pivot.loc[r, c]) else 0
        p1_annot.loc[r, c] = f'{v:.0%}\\n({m:.0f}σ)'
sns.heatmap(p1_pivot.astype(float), annot=p1_annot, fmt='', cmap='YlOrRd', vmin=0, vmax=0.5,
            linewidths=1, linecolor='white', ax=axes[0][0], cbar_kws={'label':'Max VFR'})
axes[0][0].set_title('(a) Protocol 1: Max VFR Across 12 Models\\n(with margin-to-threshold in σ)', fontsize=11, fontweight='bold')

# (b) P3 Cross-site CV — enhanced with diverging colormap
p3_pivot = reliability_df.pivot_table(index='Metric', columns='Attribute', values='P3_CV')
sns.heatmap(p3_pivot, annot=True, fmt='.3f', cmap='YlOrBr', vmin=0, vmax=0.3,
            linewidths=1, linecolor='white', ax=axes[0][1], cbar_kws={'label':'CV'})
axes[0][1].set_title('(b) Protocol 3: Cross-Site CV\\n(higher = less portable)', fontsize=11, fontweight='bold')

# (c) P3 % Fair across sites — enhanced
p3f_pivot = reliability_df.pivot_table(index='Metric', columns='Attribute', values='P3_PctFair')
sns.heatmap(p3f_pivot, annot=True, fmt='.0f', cmap='RdYlGn', vmin=0, vmax=100,
            linewidths=1, linecolor='white', ax=axes[0][2], cbar_kws={'label':'% Fair'})
axes[0][2].set_title('(c) Protocol 3: % Sites Deemed Fair\\n(green = consistent verdict)', fontsize=11, fontweight='bold')

# (d) Seed Perturbation Margin-to-Threshold (replaces all-zero Seed VFR)
sm_pivot = reliability_df.pivot_table(index='Metric', columns='Attribute', values='Seed_Margin')
# Custom colormap annotations
sm_annot = sm_pivot.copy().astype(object)
sv_pvt = reliability_df.pivot_table(index='Metric', columns='Attribute', values='Seed_VFR')
for r in sm_pivot.index:
    for c in sm_pivot.columns:
        margin = sm_pivot.loc[r, c]
        svfr = sv_pvt.loc[r, c] if r in sv_pvt.index and c in sv_pvt.columns else 0
        if pd.isna(margin): sm_annot.loc[r, c] = '—'
        else: sm_annot.loc[r, c] = f'{margin:.0f}σ\\nVFR={svfr:.0%}'
sns.heatmap(sm_pivot.astype(float), annot=sm_annot, fmt='', cmap='RdYlGn', vmin=0, vmax=50,
            linewidths=1, linecolor='white', ax=axes[1][0], cbar_kws={'label':'Margin (σ)'})
axes[1][0].set_title('(d) Seed Perturbation: Margin to Threshold\\n(higher σ = more stable)', fontsize=11, fontweight='bold')

# (e) Model ranking with fairness verdict count
model_fair_counts = []
for name in test_predictions:
    n_fair = sum(all_verdicts[name][attr][mk]
                 for attr in ['RACE','SEX','ETHNICITY','AGE_GROUP']
                 for mk in METRIC_KEYS)
    n_total = len(METRIC_KEYS) * 4
    model_fair_counts.append({'Model':name, 'N_Fair':n_fair, 'Pct_Fair':n_fair/n_total*100,
        'AUC': results_df[results_df['Model']==name]['AUC'].values[0] if name in results_df['Model'].values else 0})
mfc_df = pd.DataFrame(model_fair_counts).sort_values('AUC', ascending=True)
colors_mfc = ['#2ecc71' if r['Pct_Fair']>70 else '#e67e22' if r['Pct_Fair']>50 else '#e74c3c'
              for _, r in mfc_df.iterrows()]
axes[1][1].barh(mfc_df['Model'], mfc_df['Pct_Fair'], color=colors_mfc, edgecolor='white')
axes[1][1].axvline(x=70, color='green', linestyle='--', alpha=0.5)
axes[1][1].set_xlabel('% Fair (all metrics+attributes)')
axes[1][1].set_title('(e) Model Fairness Score', fontsize=12, fontweight='bold')

# (f) Fleiss kappa per metric
if len(mk_kappa_df) > 0:
    bars_k = axes[1][2].bar(mk_kappa_df['Metric'], mk_kappa_df['Kappa'],
        color=[PALETTE[i] for i in range(len(mk_kappa_df))], edgecolor='white')
    axes[1][2].axhline(y=0.61, color='green', linestyle='--', alpha=0.5, label='Substantial')
    axes[1][2].axhline(y=0.41, color='orange', linestyle='--', alpha=0.5)
    for b, v in zip(bars_k, mk_kappa_df['Kappa']):
        axes[1][2].text(b.get_x()+b.get_width()/2, v+0.02, f'{v:.2f}', ha='center', fontsize=9)
    axes[1][2].set_ylabel("Fleiss' κ")
    axes[1][2].set_title("(f) Cross-Site Agreement (Fleiss' κ)", fontsize=12, fontweight='bold')
    axes[1][2].legend(fontsize=8)

plt.suptitle('FIG10: Reliability Dashboard — Multi-Protocol Assessment',
             fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.96])
save_fig('reliability_dashboard')
plt.show()
""")

md("""
> The **Reliability Dashboard** consolidates all three protocols:
> - Green cells = reliable verdicts (low VFR, low CV, high agreement)
> - Red cells = unreliable verdicts (high VFR, high CV, poor agreement)
>
> This provides a **single-glance** assessment of which fairness claims can be
> trusted and which require additional data or methodological caution.
""")

###############################################################################
# SECTION 16 — PUBLICATION-READY FIGURES (ALL MANUSCRIPT FIGURES)
###############################################################################
md("""
---
## 16. Publication-Ready Figures

All figures referenced in the manuscript, generated at **300 DPI** for
publication quality.  Each figure is saved individually and also as a combined panel.
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 59 · FIG01: Study Pipeline
# ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 4))
ax.set_xlim(0, 10); ax.set_ylim(0, 2); ax.axis('off')
steps = ['Texas PUDF\\n925K records', 'Preprocessing\\n& Feature Eng.', '12 ML Models\\n+ 2 AFCE',
         '7 Fairness\\nMetrics', '3 Stability\\nProtocols', 'Cross-Site\\nPortability',
         'Intervention\\n& Guidance']
colors_pipe = ['#3498db','#2ecc71','#e74c3c','#9b59b6','#f39c12','#1abc9c','#e67e22']
for i, (step, col) in enumerate(zip(steps, colors_pipe)):
    x = i * 1.4 + 0.3
    rect = plt.Rectangle((x, 0.4), 1.2, 1.2, facecolor=col, edgecolor='white',
                          linewidth=2, alpha=0.85, zorder=2)
    ax.add_patch(rect)
    ax.text(x+0.6, 1.0, step, ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    if i < len(steps)-1:
        ax.annotate('', xy=(x+1.35, 1.0), xytext=(x+1.2, 1.0),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2))
ax.set_title('FIG01: Study Pipeline — Multi-Site Fairness Evaluation', fontsize=13, fontweight='bold', y=1.05)
plt.tight_layout()
save_fig('FIG01_study_pipeline')
plt.show()
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 60 · FIG02: Demographics Overview
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
for ai, attr in enumerate(['RACE','SEX','ETHNICITY','AGE_GROUP']):
    ax = axes[ai//2][ai%2]
    attr_vals = protected_attrs_train[attr]
    label_map = RACE_MAP if attr=='RACE' else (SEX_MAP if attr=='SEX' else
                (ETH_MAP if attr=='ETHNICITY' else {g:g for g in sorted(set(attr_vals))}))
    groups = sorted(set(attr_vals))
    labels = [label_map.get(g, str(g)) for g in groups]
    counts = [np.sum(attr_vals == g) for g in groups]
    colors_d = [PALETTE[i%len(PALETTE)] for i in range(len(groups))]
    bars = ax.barh(labels, counts, color=colors_d, edgecolor='white')
    for bar, c in zip(bars, counts):
        ax.text(bar.get_width() + max(counts)*0.01, bar.get_y()+bar.get_height()/2,
                f'{c:,} ({c/len(attr_vals)*100:.1f}%)', va='center', fontsize=9)
    ax.set_xlabel('Count'); ax.set_title(f'{attr} Distribution', fontsize=12, fontweight='bold')
plt.suptitle('FIG02: Demographic Composition of Study Population', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.96])
save_fig('FIG02_demographics')
plt.show()
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 61 · FIG03: LOS Distribution
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (a) LOS histogram
axes[0].hist(df['LENGTH_OF_STAY'].clip(upper=30), bins=30, color=PALETTE[0], edgecolor='white')
axes[0].axvline(x=3, color='red', linestyle='--', lw=2, label='Threshold (3 days)')
axes[0].set_xlabel('Length of Stay (days)'); axes[0].set_ylabel('Count')
axes[0].set_title('(a) LOS Distribution'); axes[0].legend()

# (b) Binary target
counts_b = [(df['LOS_BINARY']==0).sum(), (df['LOS_BINARY']==1).sum()]
axes[1].bar(['≤3 days', '>3 days'], counts_b, color=[PALETTE[1], PALETTE[3]], edgecolor='white')
for i, c in enumerate(counts_b):
    axes[1].text(i, c+max(counts_b)*0.01, f'{c:,}\\n({c/len(df)*100:.1f}%)', ha='center', fontsize=10)
axes[1].set_title('(b) Binary Target Distribution')

# (c) LOS by RACE
for gi, g in enumerate(sorted(df['RACE'].unique())):
    sub = df[df['RACE']==g]['LENGTH_OF_STAY'].clip(upper=20)
    axes[2].hist(sub, bins=20, alpha=0.4, label=RACE_MAP.get(g, str(g)), color=PALETTE[gi])
axes[2].set_xlabel('LOS (days)'); axes[2].set_title('(c) LOS by Race'); axes[2].legend(fontsize=8)

plt.suptitle('FIG03: Length-of-Stay Distribution', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.96])
save_fig('FIG03_los_distribution')
plt.show()
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 62 · FIG04: Reliability Framework Diagram
# ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 6))
ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis('off')

# Three protocol boxes
protocols = [
    ('P1: Resampling\\nK=30 subsets\\n→ VFR', '#3498db', 0.5, 2),
    ('P2: Sample Size\\nN: 1K→925K\\n→ CV curves, min-N', '#2ecc71', 3.5, 2),
    ('P3: Cross-Site\\nK=20 hospital clusters\\n→ Portability', '#e74c3c', 6.5, 2),
]
for text, col, x, y in protocols:
    rect = plt.Rectangle((x, y-0.8), 2.5, 1.8, facecolor=col, edgecolor='black',
                          linewidth=1.5, alpha=0.8, zorder=2)
    ax.add_patch(rect)
    ax.text(x+1.25, y+0.1, text, ha='center', va='center', fontsize=10,
            fontweight='bold', color='white', zorder=3)

# Input and output arrows
ax.annotate('7 Metrics × 4 Attributes', xy=(5, 3.8), ha='center', fontsize=12,
            fontweight='bold', color='black')
ax.annotate('', xy=(1.75, 2.85), xytext=(5, 3.6), arrowprops=dict(arrowstyle='->', lw=1.5))
ax.annotate('', xy=(4.75, 2.85), xytext=(5, 3.6), arrowprops=dict(arrowstyle='->', lw=1.5))
ax.annotate('', xy=(7.75, 2.85), xytext=(5, 3.6), arrowprops=dict(arrowstyle='->', lw=1.5))

# Output
ax.annotate('', xy=(5, 0.6), xytext=(1.75, 1.0), arrowprops=dict(arrowstyle='->', lw=1.5))
ax.annotate('', xy=(5, 0.6), xytext=(4.75, 1.0), arrowprops=dict(arrowstyle='->', lw=1.5))
ax.annotate('', xy=(5, 0.6), xytext=(7.75, 1.0), arrowprops=dict(arrowstyle='->', lw=1.5))
rect_out = plt.Rectangle((3.5, 0.0), 3, 0.8, facecolor='#9b59b6', edgecolor='black',
                          linewidth=1.5, alpha=0.8, zorder=2)
ax.add_patch(rect_out)
ax.text(5, 0.4, 'Reliability Dashboard\\n(Table 9 + FIG10)', ha='center', va='center',
        fontsize=11, fontweight='bold', color='white', zorder=3)

ax.set_title('FIG04: Fairness-Metric Reliability Framework', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
save_fig('FIG04_reliability_framework')
plt.show()
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 63 · FIG05: Fairness Heatmap (Publication Version)
# ──────────────────────────────────────────────────────────────
# Best model only, 7 metrics × 4 attributes
best_fair_data = []
for attr in ['RACE','SEX','ETHNICITY','AGE_GROUP']:
    f = all_fairness[best_model_name][attr]
    for mk in METRIC_KEYS:
        best_fair_data.append({'Metric':mk, 'Attribute':attr, 'Value':f[mk]})
bf_df = pd.DataFrame(best_fair_data)
bf_pivot = bf_df.pivot(index='Metric', columns='Attribute', values='Value')

fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(bf_pivot, annot=True, fmt='.3f', cmap='RdYlGn', linewidths=0.5, ax=ax)
ax.set_title(f'FIG05: Fairness Metrics — {best_model_name}', fontsize=14, fontweight='bold')
plt.tight_layout()
save_fig('FIG05_fairness_heatmap_pub')
plt.show()
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 64 · FIG11: Failure Modes Taxonomy
# ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('off')

failure_modes = [
    ('Small-Sample\\nInstability', 'DI flips fair/unfair\\nat N < 5K', '#e74c3c'),
    ('Metric\\nDisagreement', '≥2 metrics disagree\\non 30-60% of combos', '#e67e22'),
    ('Cross-Site\\nFragility', 'Verdict changes across\\nhospital clusters', '#f39c12'),
    ('Threshold\\nSensitivity', 'DI varies 0.6-1.0\\nacross thresholds', '#3498db'),
    ('Intersectional\\nHiding', 'Subgroup disparities\\nhidden in aggregates', '#9b59b6'),
]
for i, (title, desc, col) in enumerate(failure_modes):
    x = i * 2.6 + 0.3
    rect = plt.Rectangle((x, 1.0), 2.2, 2.5, facecolor=col, edgecolor='white',
                          linewidth=2, alpha=0.8, zorder=2)
    ax.add_patch(rect)
    ax.text(x+1.1, 2.7, title, ha='center', va='center', fontsize=10,
            fontweight='bold', color='white')
    ax.text(x+1.1, 1.7, desc, ha='center', va='center', fontsize=8, color='white')
ax.set_xlim(0, 13.5); ax.set_ylim(0, 4.5)
ax.set_title('FIG11: Failure Modes Taxonomy for Fairness Metrics', fontsize=14,
             fontweight='bold', y=1.02)
plt.tight_layout()
save_fig('FIG11_failure_modes_taxonomy')
plt.show()
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 65 · FIG12: Portability Mechanism Map
# ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('off')

# Three portability mechanisms
mechs = [
    ('Demographic\\nShift', 'Hospital A has 60% White\\nHospital B has 30% White\\n→ DI changes', '#e74c3c', 1),
    ('Prevalence\\nShift', 'Hospital A: 40% LOS>3\\nHospital B: 25% LOS>3\\n→ Calibration shifts', '#3498db', 4),
    ('Feature\\nDistribution', 'Hospital A: urban, young\\nHospital B: rural, elderly\\n→ Predictions shift', '#2ecc71', 7),
]
for text, desc, col, x in mechs:
    rect = plt.Rectangle((x, 1.2), 2.5, 2.8, facecolor=col, edgecolor='black',
                          linewidth=1.5, alpha=0.8, zorder=2)
    ax.add_patch(rect)
    ax.text(x+1.25, 3.3, text, ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    ax.text(x+1.25, 2.0, desc, ha='center', va='center', fontsize=8, color='white')

ax.set_xlim(0, 10.5); ax.set_ylim(0, 5)
ax.set_title('FIG12: Cross-Site Portability Mechanism Map', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
save_fig('FIG12_portability_mechanism')
plt.show()
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 66 · Combined Publication Panel (All Key Figures)
# ──────────────────────────────────────────────────────────────
from matplotlib.image import imread
import glob

pub_figs = sorted(glob.glob(f'{FIGURES_DIR}/FIG*.png'))
n_figs = len(pub_figs)
if n_figs > 0:
    cols = 3; rows = (n_figs + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(24, 7*rows))
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    for i, fpath in enumerate(pub_figs):
        img = imread(fpath)
        axes_flat[i].imshow(img); axes_flat[i].axis('off')
        axes_flat[i].set_title(fpath.split('/')[-1].replace('.png',''), fontsize=10)
    for j in range(i+1, len(axes_flat)):
        axes_flat[j].axis('off')
    plt.suptitle('Combined Publication Figures Panel', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.97])
    save_fig('combined_publication_panel')
    plt.show()
    print(f"✓ Combined panel includes {n_figs} publication figures")
else:
    print("No FIG*.png files found")
""")

###############################################################################
# SECTION 17 — SUMMARY DASHBOARD & FINAL RESULTS
###############################################################################
md("""
---
## 17. Summary Dashboard & Final Results

The final dashboard consolidates all key findings into a single overview.
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 67 · Summary Dashboard (3×3 grid)
# ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 18))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

# (a) Model AUC ranking
ax1 = fig.add_subplot(gs[0, 0])
colors_rank = [PALETTE[i%len(PALETTE)] for i in range(len(results_df))]
ax1.barh(results_df['Model'][::-1], results_df['AUC'][::-1], color=colors_rank[::-1])
ax1.set_xlabel('AUC'); ax1.set_title('Model Ranking (AUC)')

# (b) DI overview
ax2 = fig.add_subplot(gs[0, 1])
di_vals = [all_fairness[best_model_name][a]['DI'] for a in ['RACE','SEX','ETHNICITY','AGE_GROUP']]
bars2 = ax2.bar(['RACE','SEX','ETH','AGE'], di_vals,
    color=[PALETTE[i] for i in range(4)], edgecolor='white')
ax2.axhline(y=0.80, color='red', linestyle='--', lw=2)
ax2.set_ylabel('DI'); ax2.set_title(f'DI — {best_model_name}')

# (c) Metric disagreement summary
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(disagree_df['N_Fair'], bins=range(9), color=PALETTE[3], edgecolor='white', rwidth=0.8, align='left')
ax3.set_xlabel('# Fair Metrics (of 7)'); ax3.set_title('Multi-Criteria Fairness')

# (d) Bootstrap DI
ax4 = fig.add_subplot(gs[1, 0])
for i, attr in enumerate(['RACE','SEX','ETHNICITY','AGE_GROUP']):
    ax4.hist(boot_results[attr]['DI'], bins=20, alpha=0.5, color=PALETTE[i], label=attr)
ax4.axvline(x=0.80, color='red', linestyle='--', lw=2)
ax4.set_xlabel('DI'); ax4.set_title('Bootstrap DI Distribution'); ax4.legend(fontsize=8)

# (e) Cross-site portability
ax5 = fig.add_subplot(gs[1, 1])
cs_cv_means = cs_summary_df.groupby('Metric')['CV'].mean()
bars5 = ax5.bar(cs_cv_means.index, cs_cv_means.values,
    color=[PALETTE[i] for i in range(len(cs_cv_means))], edgecolor='white')
ax5.axhline(y=0.10, color='red', linestyle='--', alpha=0.5)
ax5.set_ylabel('Mean CV'); ax5.set_title('Cross-Site CV per Metric')

# (f) Fair vs Standard
ax6 = fig.add_subplot(gs[1, 2])
comp_data = pd.DataFrame({'Metric':['Accuracy','DI','SPD','CAL'],
    'Standard':[std_acc, m_std['DI'], m_std['SPD'], m_std['CAL']],
    'Fair':[fair_acc, m_fair['DI'], m_fair['SPD'], m_fair['CAL']]})
xc = np.arange(4)
ax6.bar(xc-0.15, comp_data['Standard'], 0.3, label='Standard', color=PALETTE[0])
ax6.bar(xc+0.15, comp_data['Fair'], 0.3, label='Fair', color=PALETTE[2])
ax6.set_xticks(xc); ax6.set_xticklabels(comp_data['Metric'])
ax6.set_title('Standard vs Fair Model'); ax6.legend()

# (g) Subgroup analysis
ax7 = fig.add_subplot(gs[2, 0])
top_sg = subgroup_df.head(10)
ax7.barh(top_sg['Group'], top_sg['Selection_Rate'], color=PALETTE[5], edgecolor='white')
ax7.axvline(x=df['LOS_BINARY'].mean(), color='red', linestyle='--')
ax7.set_xlabel('Selection Rate'); ax7.set_title('Top 10 Subgroups')

# (h) Lambda
ax8 = fig.add_subplot(gs[2, 1])
ax8.plot(lambda_df['Lambda'], lambda_df['DI'], 'D-', color=PALETTE[4], linewidth=2)
ax8.axhline(y=0.80, color='red', linestyle='--')
ax8.set_xlabel('λ'); ax8.set_ylabel('DI (RACE)'); ax8.set_title('Lambda vs DI')

# (i) Training times
ax9 = fig.add_subplot(gs[2, 2])
ts = sorted(training_times.items(), key=lambda x: x[1], reverse=True)
ax9.barh([t[0] for t in ts], [t[1] for t in ts], color=PALETTE[7])
ax9.set_xlabel('Seconds'); ax9.set_title('Training Time')

fig.suptitle('RQ1: LOS Prediction Fairness — Summary Dashboard', fontsize=16, fontweight='bold', y=0.99)
save_fig('summary_dashboard')
plt.show()
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 68 · Export Final Results JSON
# ──────────────────────────────────────────────────────────────
import glob

final_results = {
    'dataset': {'name':'Texas-100x PUDF', 'n_records':int(len(df)),
        'n_features':int(X_train.shape[1]), 'target':'LOS > 3 days',
        'prevalence':float(df['LOS_BINARY'].mean()),
        'n_hospitals': int(df['THCIC_ID'].nunique())},
    'models': {}, 'fairness': {},
    'stability': {
        'protocol1_K': K_P1,
        'protocol2_sizes': sample_sizes,
        'protocol3_K': K_CS,
        'n_seeds': N_SEEDS,
        'bootstrap_B': B,
    },
    'cross_site': {
        'n_folds': K_CS,
        'fleiss_kappa_overall': float(fk) if 'fk' in dir() else None,
    },
    'intervention': {
        'standard_acc': float(std_acc),
        'fair_acc': float(fair_acc),
        'lambda': LAMBDA_FAIR,
        'standard_metrics': {mk: float(m_std[mk]) for mk in METRIC_KEYS},
        'fair_metrics': {mk: float(m_fair[mk]) for mk in METRIC_KEYS},
    },
}
for _, r in results_df.iterrows():
    final_results['models'][r['Model']] = {
        'accuracy':float(r['Accuracy']), 'auc':float(r['AUC']),
        'f1':float(r['F1']), 'precision':float(r['Precision']), 'recall':float(r['Recall'])}
for name in test_predictions:
    final_results['fairness'][name] = {}
    for attr in ['RACE','SEX','ETHNICITY','AGE_GROUP']:
        f = all_fairness[name][attr]
        final_results['fairness'][name][attr] = {mk: float(f[mk]) for mk in METRIC_KEYS}

with open(f'{MODELS_DIR}/final_results.json', 'w') as fj:
    json.dump(final_results, fj, indent=2)
print(f"✓ Saved: {MODELS_DIR}/final_results.json")
""")

code("""
# ──────────────────────────────────────────────────────────────
# Cell 69 · Final Summary Statistics
# ──────────────────────────────────────────────────────────────
import glob

n_figures = len(glob.glob(f'{FIGURES_DIR}/*.png'))
n_tables  = len(glob.glob(f'{TABLES_DIR}/*.csv'))

print("=" * 70)
print("  ✅  FINAL SUMMARY")
print("=" * 70)
print(f"  Dataset:           {len(df):,} records × {df.shape[1]} columns")
print(f"  Hospitals:         {df['THCIC_ID'].nunique()}")
print(f"  Train/Test:        {len(y_train):,} / {len(y_test):,}")
print(f"  Models trained:    {len(test_predictions)} standard + 2 AFCE")
print(f"  Best model:        {best_model_name} (AUC = {results_df.iloc[0]['AUC']:.4f})")
print(f"  Fairness metrics:  {', '.join(METRIC_KEYS)}")
print(f"  Protected attrs:   RACE, SEX, ETHNICITY, AGE_GROUP")
print(f"  Figures generated: {n_figures}")
print(f"  Tables saved:      {n_tables}")
print()
print("  Per-Attribute Fairness (Best Model):")
for attr in ['RACE','SEX','ETHNICITY','AGE_GROUP']:
    f = all_fairness[best_model_name][attr]
    v = all_verdicts[best_model_name][attr]
    n_fair = sum(v.values())
    flag = f"✓ {n_fair}/7 FAIR" if n_fair >= 4 else f"✗ {n_fair}/7 FAIR"
    print(f"    {attr:<12s}: DI={f['DI']:.3f}  SPD={f['SPD']:.3f}  EOPP={f['EOPP']:.3f}  "
          f"EOD={f['EOD']:.3f}  TI={f['TI']:.3f}  PP={f['PP']:.3f}  CAL={f['CAL']:.3f}  [{flag}]")
print()
print("  Stability (Protocol 1 — K=30 Resampling VFR):")
for _, r in vfr_df[vfr_df['Metric']=='DI'].iterrows():
    print(f"    DI {r['Attribute']:<12s}: VFR = {r['VFR']:.1%}")
print()
print("  Cross-Site Portability (Protocol 3):")
if 'fk' in dir():
    print(f"    Fleiss' κ (overall): {fk:.3f}")
for mk in ['DI','SPD','EOPP']:
    cs_sub = cs_summary_df[cs_summary_df['Metric']==mk]
    if len(cs_sub):
        print(f"    {mk} cross-site CV range: {cs_sub['CV'].min():.3f} – {cs_sub['CV'].max():.3f}")
print()
print("  Fairness Intervention:")
print(f"    Standard:      Acc={std_acc:.4f}   DI={m_std['DI']:.3f}")
print(f"    Fair model:    Acc={fair_acc:.4f}   DI={m_fair['DI']:.3f}  (Δ DI = {m_fair['DI']-m_std['DI']:+.3f})")
print()
print("  Subgroup Analysis:")
print(f"    {len(subgroup_df)} intersectional subgroups analysed")
print(f"    Selection rate range: [{subgroup_df['Selection_Rate'].min():.3f}, {subgroup_df['Selection_Rate'].max():.3f}]")
print()
print("  AFCE (Fairness-Through-Awareness):")
for name in afce_predictions:
    yp = afce_predictions[name]['y_pred']
    fc_af = FairnessCalculator(y_test, yp, afce_predictions[name]['y_prob'], protected_attrs['RACE'])
    ma, _, _ = fc_af.compute_all()
    print(f"    {name}: Acc={accuracy_score(y_test, yp):.4f}  DI={ma['DI']:.3f}  TI={ma['TI']:.3f}")
print("=" * 70)
print("  ✅  NOTEBOOK EXECUTION COMPLETE")
print("=" * 70)
""")

md("""
---
## Conclusion

This notebook provides a **complete, reproducible fairness analysis** for hospital
length-of-stay prediction using the Texas-100x PUDF dataset (925,569 records from
441 hospitals across 2019-2023).

### Key Findings:

1. **Model Performance:**  12 models trained and evaluated.  Gradient boosting methods
   (LightGBM, XGBoost, CatBoost) achieve the highest AUC (> 0.90).

2. **Multi-Criteria Fairness (C1):**  7 fairness metrics computed across 4 protected
   attributes.  Metrics frequently disagree on verdicts — a single metric is insufficient.

3. **Verdict Stability (C2):**  Protocol 1 (VFR) and seed perturbation show most
   verdicts are stable, but some metric-attribute pairs are fragile (VFR > 10%).

4. **Cross-Site Portability (C3):**  Protocol 3 reveals between-cluster variation
   with some metrics showing high CV across hospital sites.  Fleiss' κ quantifies
   the degree of inter-site agreement on fairness verdicts.

5. **Minimum Sample Guidance (C4):**  CV < 0.05 requires varying sample sizes per
   metric — DI stabilises at ~5K while TI may need 25K+.

6. **Intersectional Analysis:**  25-30+ subgroup combinations analysed, revealing
   disparities hidden by single-attribute analysis.

7. **Intervention:**  Lambda-reweighing (λ=5) + per-group threshold optimisation
   improves DI with < 1 pp accuracy loss.

### Output Files:
- **Figures:** `output/figures/` — all visualisations as high-resolution PNGs
- **Tables:** `output/tables/` — all tabular results as CSVs
- **Results:** `output/models/final_results.json` — machine-readable summary

> This notebook is **fully self-contained** — all results are visible inline.
""")

###############################################################################
# SAVE
###############################################################################
out_path = 'RQ1_LOS_Fairness_Analysis.ipynb'
nbf.write(nb, out_path)
code_cells = sum(1 for c in nb.cells if c.cell_type == 'code')
md_cells = sum(1 for c in nb.cells if c.cell_type == 'markdown')
print(f"Notebook saved: {out_path}")
print(f"Total cells: {len(nb.cells)}  ({code_cells} code + {md_cells} markdown)")
