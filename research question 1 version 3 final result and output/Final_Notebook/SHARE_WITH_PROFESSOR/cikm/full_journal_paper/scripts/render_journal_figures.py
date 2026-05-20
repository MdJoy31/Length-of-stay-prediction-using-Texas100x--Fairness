"""
Five journal-specific figures rendered from journal_summary.csv and the
per-experiment VFR tables. All captions use academic wording (no informal
project nicknames).

Output filenames:
  F_protocol_comparison.png         AUROC / accuracy cost / min(DI) across 4 protocols
  F_leakage_diagnostic.png          AUROC-drop leakage diagnostic + DI preservation
  F_per_attribute_di.png            Per-attribute DI on AUDIT across 3 reruns
  F_vfr_landscape.png               7x4 VFR heatmaps for the 3 reruns
  F_seed_reproducibility.png        seed=42 vs seed=123 reconciliation
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

ROOT = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\full_journal_paper")
OUT = ROOT / "journal_figures"
OUT.mkdir(parents=True, exist_ok=True)
TAB = ROOT / "output" / "tables"

mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['savefig.dpi'] = 300

summary = pd.read_csv(TAB / "journal_summary.csv")
e1 = summary[summary['experiment'] == '70_15_15'].iloc[0]
e2 = summary[summary['experiment'] == 'admission_only'].iloc[0]
e3 = summary[summary['experiment'] == 'seed_reproducibility'].iloc[0]

# Reference: single-split (80/20, in-sample tuning) figures from the main manuscript
REF = dict(AUROC=0.9528, Acc_std=0.8776, Acc_fair=0.8352, Acc_cost_pp=4.24,
           DI_Race=0.801, DI_Sex=0.932, DI_Eth=1.000, DI_Age=0.800,
           VFR_stable=21)

# ---------------------------------------------------------------------
# Protocol comparison: 4 bars × 3 panels
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0))
labels = ['Single-split\n(in-sample\ntuning)', 'Stratified\n70/15/15\n(held-out)',
          'Admission-\nonly\nablation', 'Reproducibility\n(seed = 123)']
xpos = np.arange(4)
colors = ['#888888', '#3a78b8', '#e07b39', '#5fa55a']

ax = axes[0]
aurocs = [REF['AUROC'], e1.AUROC_standard, e2.AUROC_standard, e3.AUROC_standard]
ax.bar(xpos, aurocs, color=colors, edgecolor='black', linewidth=0.5)
for i, v in enumerate(aurocs):
    ax.text(i, v + 0.005, f'{v:.4f}', ha='center', fontsize=9, fontweight='bold')
ax.set_ylim(0.83, 0.97); ax.set_ylabel('AUROC', fontsize=10, fontweight='bold')
ax.set_xticks(xpos); ax.set_xticklabels(labels, fontsize=8.5)
ax.set_title('(a) AUROC on audit cohort', fontsize=10.5, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

ax = axes[1]
costs = [REF['Acc_cost_pp'], e1.Acc_cost_pp, e2.Acc_cost_pp, e3.Acc_cost_pp]
ax.bar(xpos, costs, color=colors, edgecolor='black', linewidth=0.5)
for i, v in enumerate(costs):
    ax.text(i, v + 0.1, f'{v:.2f}', ha='center', fontsize=9, fontweight='bold')
ax.axhline(5.0, color='red', linestyle='--', linewidth=1.2, alpha=0.7, label='5 pp reference')
ax.set_ylim(0, 6.5); ax.set_ylabel('Accuracy cost (pp)', fontsize=10, fontweight='bold')
ax.set_xticks(xpos); ax.set_xticklabels(labels, fontsize=8.5)
ax.set_title('(b) Accuracy cost of the fairness intervention', fontsize=10.5, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.grid(axis='y', alpha=0.3)

ax = axes[2]
min_dis = [
    min(REF['DI_Race'], REF['DI_Sex'], REF['DI_Eth'], REF['DI_Age']),
    min(e1.DI_Race, e1.DI_Sex, e1.DI_Eth, e1.DI_Age),
    min(e2.DI_Race, e2.DI_Sex, e2.DI_Eth, e2.DI_Age),
    min(e3.DI_Race, e3.DI_Sex, e3.DI_Eth, e3.DI_Age),
]
ax.bar(xpos, min_dis, color=colors, edgecolor='black', linewidth=0.5)
for i, v in enumerate(min_dis):
    ax.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
ax.axhline(0.80, color='red', linestyle='--', linewidth=1.2, alpha=0.7, label='Four-fifths rule (DI = 0.80)')
ax.set_ylim(0.75, 0.92); ax.set_ylabel('min(DI) across 4 protected attributes', fontsize=10, fontweight='bold')
ax.set_xticks(xpos); ax.set_xticklabels(labels, fontsize=8.5)
ax.set_title('(c) Minimum DI on audit', fontsize=10.5, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.grid(axis='y', alpha=0.3)

plt.suptitle('Comparison of validation protocols on Texas-100X audit cohort',
             fontsize=11.5, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUT / 'F_protocol_comparison.png', bbox_inches='tight', dpi=300)
plt.close()
print('F_protocol_comparison.png saved')

# ---------------------------------------------------------------------
# Leakage diagnostic
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
ax = axes[0]
xpos = np.arange(2)
labels2 = ['Full\n8 features', 'Admission-only\n6 features']
aurocs2 = [e1.AUROC_standard, e2.AUROC_standard]
ax.bar(xpos, aurocs2, color=['#3a78b8', '#e07b39'], edgecolor='black', linewidth=0.5, width=0.5)
for i, v in enumerate(aurocs2):
    ax.text(i, v + 0.005, f'{v:.4f}', ha='center', fontsize=11, fontweight='bold')
drop = e1.AUROC_standard - e2.AUROC_standard
ax.annotate('', xy=(1, e2.AUROC_standard + 0.01), xytext=(0, e1.AUROC_standard + 0.01),
            arrowprops=dict(arrowstyle='->', color='red', lw=2.0))
ax.text(0.5, e1.AUROC_standard + 0.03,
        f'AUROC drop = {drop:.4f}\n(leakage diagnostic score)',
        ha='center', fontsize=10, fontweight='bold', color='red',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='red', linewidth=0.8))
ax.set_ylim(0.83, 1.00); ax.set_ylabel('AUROC on audit partition', fontsize=10, fontweight='bold')
ax.set_xticks(xpos); ax.set_xticklabels(labels2, fontsize=10, fontweight='bold')
ax.set_title('(a) AUROC-drop feature-leakage diagnostic', fontsize=10.5, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

ax = axes[1]
attr_names = ['Race', 'Sex', 'Ethnicity', 'Age']
di_full = [e1.DI_Race, e1.DI_Sex, e1.DI_Eth, e1.DI_Age]
di_adm  = [e2.DI_Race, e2.DI_Sex, e2.DI_Eth, e2.DI_Age]
x = np.arange(4); width = 0.36
ax.bar(x - width/2, di_full, width, color='#3a78b8', edgecolor='black', linewidth=0.5, label='Full 8 features')
ax.bar(x + width/2, di_adm,  width, color='#e07b39', edgecolor='black', linewidth=0.5, label='Admission-only')
for i in range(4):
    ax.text(i - width/2, di_full[i] + 0.005, f'{di_full[i]:.3f}', ha='center', fontsize=8)
    ax.text(i + width/2, di_adm[i] + 0.005, f'{di_adm[i]:.3f}', ha='center', fontsize=8)
ax.axhline(0.80, color='red', linestyle='--', linewidth=1.2, alpha=0.7, label='Four-fifths rule')
ax.set_xticks(x); ax.set_xticklabels(attr_names, fontsize=10, fontweight='bold')
ax.set_ylim(0.75, 1.02); ax.set_ylabel('Disparate Impact', fontsize=10, fontweight='bold')
ax.set_title('(b) Disparate Impact preserved under admission-only',
             fontsize=10.5, fontweight='bold')
ax.legend(fontsize=8.5, loc='lower right')
ax.grid(axis='y', alpha=0.3)

plt.suptitle('Feature-leakage diagnostic on the audit partition',
             fontsize=11.5, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUT / 'F_leakage_diagnostic.png', bbox_inches='tight', dpi=300)
plt.close()
print('F_leakage_diagnostic.png saved')

# ---------------------------------------------------------------------
# Per-attribute DI across 3 reruns
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 4.5))
attrs = ['Race', 'Sex', 'Ethnicity', 'Age']
exp_labels = ['Stratified 70/15/15 (seed=42)',
              'Admission-only ablation',
              'Reproducibility (seed=123)']
exp_data = np.array([
    [e1.DI_Race, e1.DI_Sex, e1.DI_Eth, e1.DI_Age],
    [e2.DI_Race, e2.DI_Sex, e2.DI_Eth, e2.DI_Age],
    [e3.DI_Race, e3.DI_Sex, e3.DI_Eth, e3.DI_Age],
])
n_grp = 3; width = 0.20
x = np.arange(4)
colors = ['#3a78b8', '#e07b39', '#5fa55a']
for i in range(n_grp):
    offset = (i - 1) * width
    ax.bar(x + offset, exp_data[i], width, color=colors[i], edgecolor='black',
           linewidth=0.4, label=exp_labels[i])
    for j in range(4):
        ax.text(j + offset, exp_data[i, j] + 0.005, f'{exp_data[i, j]:.3f}',
                ha='center', fontsize=8, fontweight='bold')
ax.axhline(0.80, color='red', linestyle='--', linewidth=1.5, alpha=0.85, label='Four-fifths rule')
ax.set_xticks(x); ax.set_xticklabels(attrs, fontsize=11, fontweight='bold')
ax.set_ylabel('Disparate Impact on audit partition', fontsize=10, fontweight='bold')
ax.set_ylim(0.75, 1.02)
ax.set_title('Per-attribute Disparate Impact across three independent reruns',
             fontsize=11.5, fontweight='bold')
ax.legend(fontsize=9, loc='lower right', ncol=2)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / 'F_per_attribute_di.png', bbox_inches='tight', dpi=300)
plt.close()
print('F_per_attribute_di.png saved')

# ---------------------------------------------------------------------
# VFR landscape (7 metrics × 4 attrs × 3 experiments)
# ---------------------------------------------------------------------
METRIC_ORDER = ['DI', 'SPD', 'EOPP', 'EOD', 'TI', 'PP', 'CAL']
ATTR_ORDER = ['RACE', 'SEX', 'ETHNICITY', 'AGE_GROUP']
ATTR_LBL = ['Race', 'Sex', 'Ethnicity', 'Age']

vfr_paths = [
    (TAB / 'journal_T13_vfr_70_15_15.csv',           'Stratified 70/15/15 (seed=42)'),
    (TAB / 'journal_T13_vfr_admission_only.csv',     'Admission-only ablation'),
    (TAB / 'journal_T13_vfr_seed_reproducibility.csv', 'Reproducibility (seed=123)'),
]
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
for ax_i, (path, title) in enumerate(vfr_paths):
    df_v = pd.read_csv(path)
    g = np.full((len(METRIC_ORDER), len(ATTR_ORDER)), np.nan)
    for i, m in enumerate(METRIC_ORDER):
        for j, a in enumerate(ATTR_ORDER):
            row = df_v[(df_v['metric'] == m) & (df_v['attribute'] == a)]
            if len(row):
                g[i, j] = float(row['vfr'].iloc[0])
    ax = axes[ax_i]
    im = ax.imshow(g, cmap='RdYlGn_r', vmin=0.0, vmax=0.5, aspect='auto')
    ax.set_xticks(range(4)); ax.set_xticklabels(ATTR_LBL, fontsize=10, fontweight='bold')
    ax.set_yticks(range(7)); ax.set_yticklabels(METRIC_ORDER, fontsize=10, fontweight='bold')
    for i in range(7):
        for j in range(4):
            if not np.isnan(g[i, j]):
                col = 'white' if g[i, j] > 0.30 else 'black'
                ax.text(j, i, f'{g[i, j]:.2f}', ha='center', va='center',
                        fontsize=8.5, fontweight='bold', color=col)
    n_stable = int((g <= 0.10).sum())
    ax.set_title(f'{title}\n{n_stable}/28 cells with VFR ≤ 0.10',
                 fontsize=10, fontweight='bold')

cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
cbar.set_label('Verdict-Flip-Rate (0 = stable, 0.5 = coin-flip)', fontsize=9, fontweight='bold')
plt.suptitle('Verdict-Flip-Rate landscape on audit partition (3 independent reruns)',
             fontsize=11.5, fontweight='bold', y=1.04)
plt.savefig(OUT / 'F_vfr_landscape.png', bbox_inches='tight', dpi=300)
plt.close()
print('F_vfr_landscape.png saved')

# ---------------------------------------------------------------------
# Independent-seed reproducibility
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
metrics_names = ['AUROC\nstandard', 'AUROC\nfair', 'Accuracy\nstandard', 'Accuracy\nfair', 'Accuracy\ncost (pp)']
e1_vals = [e1.AUROC_standard, e1.AUROC_fair, e1.Acc_standard, e1.Acc_fair, e1.Acc_cost_pp]
e3_vals = [e3.AUROC_standard, e3.AUROC_fair, e3.Acc_standard, e3.Acc_fair, e3.Acc_cost_pp]
ax = axes[0]
x = np.arange(5); width = 0.36
ax.bar(x - width/2, e1_vals, width, color='#3a78b8', edgecolor='black', linewidth=0.4, label='seed=42')
ax.bar(x + width/2, e3_vals, width, color='#5fa55a', edgecolor='black', linewidth=0.4, label='seed=123')
for i in range(5):
    ax.text(i - width/2, e1_vals[i] + (0.005 if e1_vals[i] < 5 else 0.1), f'{e1_vals[i]:.3f}' if e1_vals[i] < 5 else f'{e1_vals[i]:.2f}', ha='center', fontsize=8)
    ax.text(i + width/2, e3_vals[i] + (0.005 if e3_vals[i] < 5 else 0.1), f'{e3_vals[i]:.3f}' if e3_vals[i] < 5 else f'{e3_vals[i]:.2f}', ha='center', fontsize=8, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(metrics_names, fontsize=9.5)
ax.set_title('(a) Predictive performance and accuracy cost', fontsize=10.5, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 6.0)

ax = axes[1]
e1_di = [e1.DI_Race, e1.DI_Sex, e1.DI_Eth, e1.DI_Age]
e3_di = [e3.DI_Race, e3.DI_Sex, e3.DI_Eth, e3.DI_Age]
x = np.arange(4); width = 0.36
ax.bar(x - width/2, e1_di, width, color='#3a78b8', edgecolor='black', linewidth=0.4, label='seed=42')
ax.bar(x + width/2, e3_di, width, color='#5fa55a', edgecolor='black', linewidth=0.4, label='seed=123')
for i in range(4):
    ax.text(i - width/2, e1_di[i] + 0.005, f'{e1_di[i]:.3f}', ha='center', fontsize=8)
    ax.text(i + width/2, e3_di[i] + 0.005, f'{e3_di[i]:.3f}', ha='center', fontsize=8, fontweight='bold')
ax.axhline(0.80, color='red', linestyle='--', linewidth=1.2, alpha=0.7, label='Four-fifths rule')
ax.set_xticks(x); ax.set_xticklabels(['Race', 'Sex', 'Ethnicity', 'Age'], fontsize=10, fontweight='bold')
ax.set_ylim(0.75, 1.02); ax.set_ylabel('Disparate Impact', fontsize=10, fontweight='bold')
ax.set_title('(b) Per-attribute Disparate Impact', fontsize=10.5, fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
ax.grid(axis='y', alpha=0.3)

plt.suptitle('Independent-seed reproducibility check (RANDOM_STATE = 42 vs 123)',
             fontsize=11.5, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUT / 'F_seed_reproducibility.png', bbox_inches='tight', dpi=300)
plt.close()
print('F_seed_reproducibility.png saved')

print(f'\nAll 5 journal figures saved to: {OUT}')

# Remove the previous J*-prefixed files so the folder has one canonical set
for old in ['J1_protocol_comparison.png', 'J2_leakage_diagnostic.png',
            'J3_di_grid_bulletproof.png', 'J4_vfr_three_experiments.png',
            'J5_seed_reproducibility.png']:
    p = OUT / old
    if p.exists():
        p.unlink()
        print(f'  removed previous file: {old}')
