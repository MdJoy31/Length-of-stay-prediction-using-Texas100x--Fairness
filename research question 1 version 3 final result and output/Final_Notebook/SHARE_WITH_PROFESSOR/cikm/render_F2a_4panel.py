"""
F2a as a 2×2 grid with FOUR panels:
  (a) Race distribution
  (b) Sex × Ethnicity stacked
  (c) Age group + LOS>3 positive-rate overlay
  (d) Per-hospital record-count histogram (log-y) with median = 686 line
"""
import pandas as pd, numpy as np, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

ROOT = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm")
FIG_OUT = ROOT / "paper_images" / "revisions"
FIG_OUT2 = ROOT / "output_final" / "figures" / "revisions"
FIG_OUT.mkdir(parents=True, exist_ok=True)
FIG_OUT2.mkdir(parents=True, exist_ok=True)
DATA = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\data\texas_100x.csv")

mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['savefig.dpi'] = 220

df = pd.read_csv(DATA, usecols=['LENGTH_OF_STAY','RACE','ETHNICITY','SEX_CODE','PAT_AGE','THCIC_ID'])
df['LOS_BINARY'] = (df['LENGTH_OF_STAY'] > 3).astype(int)
def age_grp(a):
    if a <= 4: return 'Pediatric'
    if a <= 9: return 'Young Adult'
    if a <= 14: return 'Middle-Aged'
    return 'Elderly'
df['AGE_BUCKET'] = df['PAT_AGE'].apply(age_grp)
RACE_LABEL = {0:'AIAN', 1:'Asian/PI', 2:'Black', 3:'White', 4:'Other'}
ETH_LABEL  = {0:'Hispanic', 1:'Non-Hispanic'}
SEX_LABEL  = {1:'Male', 0:'Female'}

fig, axes = plt.subplots(2, 2, figsize=(7.5, 7.5))

# (a) Race distribution
ax = axes[0, 0]
race_counts = df['RACE'].value_counts().sort_index()
race_labels = [RACE_LABEL.get(int(r), str(r)) for r in race_counts.index]
ax.bar(range(len(race_counts)), race_counts.values, color='#3a78b8', edgecolor='black', linewidth=0.4)
ax.set_xticks(range(len(race_counts))); ax.set_xticklabels(race_labels, rotation=20, ha='right', fontsize=9)
ax.set_ylabel('Records', fontsize=9)
ax.set_title('(a) Race distribution', fontsize=10, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for i, v in enumerate(race_counts.values):
    ax.text(i, v + max(race_counts.values)*0.01, f'{v:,}', ha='center', fontsize=9.5)

# (b) Sex × Ethnicity stacked
ax = axes[0, 1]
ct = pd.crosstab(df['SEX_CODE'].map(SEX_LABEL), df['ETHNICITY'].map(ETH_LABEL))
ct = ct[['Hispanic', 'Non-Hispanic']]
ct.plot(kind='bar', stacked=True, ax=ax, color=['#e07b39','#f4ca7c'], edgecolor='black', linewidth=0.4, width=0.6)
ax.set_xticklabels(ct.index, rotation=0, fontsize=9)
ax.set_xlabel(''); ax.set_ylabel('Records', fontsize=9)
ax.set_title('(b) Sex × Ethnicity', fontsize=9, fontweight='bold')
ax.legend(loc='upper left', fontsize=8, frameon=True, framealpha=0.95)
ax.grid(axis='y', alpha=0.3)
for i, sex_lbl in enumerate(ct.index):
    total = ct.loc[sex_lbl].sum()
    ax.text(i, total + max(ct.values.max(axis=1).sum(), ct.values.flatten().max())*0.01,
            f'{total:,}', ha='center', fontsize=9.5, fontweight='bold')

# (c) Age group + LOS positive rate
ax = axes[1, 0]
order = ['Pediatric','Young Adult','Middle-Aged','Elderly']
counts = df['AGE_BUCKET'].value_counts().reindex(order)
pos_rate = df.groupby('AGE_BUCKET')['LOS_BINARY'].mean().reindex(order)
ax.bar(range(len(order)), counts.values, color='#5fa55a', edgecolor='black', linewidth=0.4, label='Records')
ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=15, ha='right', fontsize=9)
ax.set_ylabel('Records', fontsize=9)
ax.set_title('(c) Age group + LOS>3 positive rate', fontsize=10, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax2 = ax.twinx()
ax2.plot(range(len(order)), pos_rate.values, marker='o', markersize=8, color='#c0392b', linewidth=2.2, label='Pos. rate')
# Labels positioned at fixed Y near top of panel, with leader line to marker if needed
for i, v in enumerate(pos_rate.values):
    label_y = 0.92  # fixed near top of right-axis range
    ax2.text(i, label_y, f'{v:.2f}', ha='center', va='center', fontsize=9.5, color='#c0392b', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='#c0392b', linewidth=0.8, alpha=0.95))
ax2.set_ylim(0, 1.0); ax2.set_ylabel('LOS>3 positive rate', color='#c0392b', fontsize=9)
ax2.tick_params(axis='y', colors='#c0392b')

# (d) Per-hospital record-count histogram (log-y)
ax = axes[1, 1]
hosp_vol = df.groupby('THCIC_ID').size().values
median_v = float(np.median(hosp_vol))
ax.hist(hosp_vol, bins=80, color='#7b6cc4', edgecolor='black', linewidth=0.3, log=True)
ax.axvline(median_v, color='#c0392b', linestyle='--', linewidth=2.0, label=f'Median = {int(median_v)}')
ax.set_xlabel('Records per hospital', fontsize=9)
ax.set_ylabel('Number of hospitals (log)', fontsize=9)
ax.set_title('(d) Per-hospital record-count distribution', fontsize=10, fontweight='bold')
ax.legend(fontsize=10, frameon=True, loc='upper right')
ax.grid(axis='y', alpha=0.3, which='both')

plt.suptitle('F2a · Texas-100X cohort composition\n(925,128 records · 441 hospitals)',
             fontsize=10, fontweight='bold', y=1.00)
plt.tight_layout()
for path in [FIG_OUT, FIG_OUT2]:
    plt.savefig(path / 'F2a_demographics.png', bbox_inches='tight', dpi=220)
plt.close()
print('F2a saved')
