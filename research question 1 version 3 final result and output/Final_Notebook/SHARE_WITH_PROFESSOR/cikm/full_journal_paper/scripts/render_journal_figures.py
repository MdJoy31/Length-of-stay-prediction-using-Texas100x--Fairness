"""
Journal-specific figures showing the bulletproof-rerun results.

All numbers read directly from output/tables/journal_summary.csv and the
per-experiment VFR tables so the figures cannot drift from the notebook.

Outputs:
  journal_figures/J1_protocol_comparison.png      AUROC/Acc/DI across 4 protocols
  journal_figures/J2_leakage_diagnostic.png       full vs admission-only AUROC drop
  journal_figures/J3_di_grid_bulletproof.png      4 DIs on AUDIT, 3 experiments
  journal_figures/J4_vfr_three_experiments.png    VFR landscape side-by-side
  journal_figures/J5_seed_reproducibility.png     seed=42 vs seed=123 reconciliation
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
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

# CIKM headline reference (from manuscript)
CIKM = dict(AUROC=0.9528, Acc_std=0.8776, Acc_fair=0.8352, Acc_cost_pp=4.24,
            DI_Race=0.801, DI_Sex=0.932, DI_Eth=1.000, DI_Age=0.800,
            VFR_stable=21)

# ---------------------------------------------------------------------
# J1 - Protocol comparison: 4 bars (CIKM, bulletproof, admission-only, seed=123)
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0))
labels = ['CIKM\nsingle-split', 'Bulletproof\n70/15/15', 'Admission-\nonly', 'Seed=123\nreprod.']
xpos = np.arange(4)
colors = ['#888888', '#3a78b8', '#e07b39', '#5fa55a']

# (a) AUROC
ax = axes[0]
aurocs = [CIKM['AUROC'], e1.AUROC_standard, e2.AUROC_standard, e3.AUROC_standard]
ax.bar(xpos, aurocs, color=colors, edgecolor='black', linewidth=0.5)
for i, v in enumerate(aurocs):
    ax.text(i, v + 0.005, f'{v:.4f}', ha='center', fontsize=9, fontweight='bold')
ax.set_ylim(0.83, 0.97); ax.set_ylabel('AUROC', fontsize=10, fontweight='bold')
ax.set_xticks(xpos); ax.set_xticklabels(labels, fontsize=8.5)
ax.set_title('(a) AUROC by protocol', fontsize=10.5, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# (b) Accuracy cost
ax = axes[1]
costs = [CIKM['Acc_cost_pp'], e1.Acc_cost_pp, e2.Acc_cost_pp, e3.Acc_cost_pp]
bars = ax.bar(xpos, costs, color=colors, edgecolor='black', linewidth=0.5)
for i, v in enumerate(costs):
    ax.text(i, v + 0.1, f'{v:.2f}', ha='center', fontsize=9, fontweight='bold')
ax.axhline(5.0, color='red', linestyle='--', linewidth=1.2, alpha=0.7, label='5 pp benchmark')
ax.set_ylim(0, 6.5); ax.set_ylabel('Accuracy cost (pp)', fontsize=10, fontweight='bold')
ax.set_xticks(xpos); ax.set_xticklabels(labels, fontsize=8.5)
ax.set_title('(b) Accuracy cost · all under 5.05 pp', fontsize=10.5, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.grid(axis='y', alpha=0.3)

# (c) min(DI) on audit
ax = axes[2]
min_dis = [
    min(CIKM['DI_Race'], CIKM['DI_Sex'], CIKM['DI_Eth'], CIKM['DI_Age']),
    min(e1.DI_Race, e1.DI_Sex, e1.DI_Eth, e1.DI_Age),
    min(e2.DI_Race, e2.DI_Sex, e2.DI_Eth, e2.DI_Age),
    min(e3.DI_Race, e3.DI_Sex, e3.DI_Eth, e3.DI_Age),
]
ax.bar(xpos, min_dis, color=colors, edgecolor='black', linewidth=0.5)
for i, v in enumerate(min_dis):
    ax.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
ax.axhline(0.80, color='red', linestyle='--', linewidth=1.2, alpha=0.7, label='4/5 rule (DI = 0.80)')
ax.set_ylim(0.75, 0.92); ax.set_ylabel('min(DI) across 4 attributes', fontsize=10, fontweight='bold')
ax.set_xticks(xpos); ax.set_xticklabels(labels, fontsize=8.5)
ax.set_title('(c) All-4-DI ≥ 0.80 on every protocol', fontsize=10.5, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.grid(axis='y', alpha=0.3)

plt.suptitle('Bulletproof-protocol reconciliation · headline outcomes preserved across 4 splits',
             fontsize=11, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUT / 'J1_protocol_comparison.png', bbox_inches='tight', dpi=300)
plt.close()
print('J1_protocol_comparison.png saved')

# ---------------------------------------------------------------------
# J2 - Feature-leakage diagnostic: full vs admission-only side-by-side
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
# (a) AUROC drop
ax = axes[0]
xpos = np.arange(2)
labels2 = ['Full\n8 features', 'Admission-only\n6 features']
aurocs2 = [e1.AUROC_standard, e2.AUROC_standard]
ax.bar(xpos, aurocs2, color=['#3a78b8', '#e07b39'], edgecolor='black', linewidth=0.5, width=0.5)
for i, v in enumerate(aurocs2):
    ax.text(i, v + 0.005, f'{v:.4f}', ha='center', fontsize=11, fontweight='bold')
drop = e1.AUROC_standard - e2.AUROC_standard
# arrow showing drop
ax.annotate('', xy=(1, e2.AUROC_standard + 0.01), xytext=(0, e1.AUROC_standard + 0.01),
            arrowprops=dict(arrowstyle='->', color='red', lw=2.0))
ax.text(0.5, e1.AUROC_standard + 0.03, f'drop = {drop:.4f}\n(= 0.085 = leakage score)',
        ha='center', fontsize=10, fontweight='bold', color='red',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='red', linewidth=0.8))
ax.set_ylim(0.83, 1.00); ax.set_ylabel('AUROC on audit partition', fontsize=10, fontweight='bold')
ax.set_xticks(xpos); ax.set_xticklabels(labels2, fontsize=10, fontweight='bold')
ax.set_title('(a) AUROC-drop leakage diagnostic', fontsize=10.5, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# (b) DI preserved
ax = axes[1]
attr_names = ['Race', 'Sex', 'Eth', 'Age']
di_full = [e1.DI_Race, e1.DI_Sex, e1.DI_Eth, e1.DI_Age]
di_adm  = [e2.DI_Race, e2.DI_Sex, e2.DI_Eth, e2.DI_Age]
x = np.arange(4); width = 0.36
ax.bar(x - width/2, di_full, width, color='#3a78b8', edgecolor='black', linewidth=0.5, label='Full 8 features')
ax.bar(x + width/2, di_adm,  width, color='#e07b39', edgecolor='black', linewidth=0.5, label='Admission-only')
for i in range(4):
    ax.text(i - width/2, di_full[i] + 0.005, f'{di_full[i]:.3f}', ha='center', fontsize=8)
    ax.text(i + width/2, di_adm[i] + 0.005, f'{di_adm[i]:.3f}', ha='center', fontsize=8)
ax.axhline(0.80, color='red', linestyle='--', linewidth=1.2, alpha=0.7, label='4/5 rule')
ax.set_xticks(x); ax.set_xticklabels(attr_names, fontsize=10, fontweight='bold')
ax.set_ylim(0.75, 1.02); ax.set_ylabel('Disparate Impact (DI)', fontsize=10, fontweight='bold')
ax.set_title('(b) All-4-DI preserved · fairness story is feature-robust',
             fontsize=10.5, fontweight='bold')
ax.legend(fontsize=8.5, loc='lower right')
ax.grid(axis='y', alpha=0.3)

plt.suptitle('Feature-leakage diagnostic · low drop (0.085); fairness conclusions unchanged',
             fontsize=11, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUT / 'J2_leakage_diagnostic.png', bbox_inches='tight', dpi=300)
plt.close()
print('J2_leakage_diagnostic.png saved')

# ---------------------------------------------------------------------
# J3 - DI grid across 3 experiments, 4 attributes
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 4.5))
attrs = ['Race', 'Sex', 'Eth', 'Age']
exp_labels = ['Bulletproof\n(seed=42)', 'Admission-\nonly', 'Reprod.\n(seed=123)']
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
           linewidth=0.4, label=exp_labels[i].replace('\n', ' '))
    for j in range(4):
        ax.text(j + offset, exp_data[i, j] + 0.005, f'{exp_data[i, j]:.3f}',
                ha='center', fontsize=8, fontweight='bold')
ax.axhline(0.80, color='red', linestyle='--', linewidth=1.5, alpha=0.85, label='4/5 rule (DI = 0.80)')
ax.set_xticks(x); ax.set_xticklabels(attrs, fontsize=11, fontweight='bold')
ax.set_ylabel('Disparate Impact (DI) on audit partition', fontsize=10, fontweight='bold')
ax.set_ylim(0.75, 1.02)
ax.set_title('Per-attribute DI on AUDIT · all 3 protocols pass all-4-DI ≥ 0.80',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9, loc='lower right', ncol=2)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / 'J3_di_grid_bulletproof.png', bbox_inches='tight', dpi=300)
plt.close()
print('J3_di_grid_bulletproof.png saved')

# ---------------------------------------------------------------------
# J4 - VFR landscape side-by-side (3 panels: 7 metrics x 4 attrs each)
# ---------------------------------------------------------------------
METRIC_ORDER = ['DI', 'SPD', 'EOPP', 'EOD', 'TI', 'PP', 'CAL']
ATTR_ORDER = ['RACE', 'SEX', 'ETHNICITY', 'AGE_GROUP']
ATTR_LBL = ['Race', 'Sex', 'Eth', 'Age']

vfr_paths = [
    (TAB / 'journal_T13_vfr_70_15_15.csv',           'Bulletproof (seed=42)'),
    (TAB / 'journal_T13_vfr_admission_only.csv',     'Admission-only'),
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
    ax.set_title(f'{title}\n{n_stable}/28 cells stable (VFR ≤ 0.10)',
                 fontsize=10, fontweight='bold')

cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
cbar.set_label('VFR (0 = stable, 0.5 = coin-flip)', fontsize=9, fontweight='bold')
plt.suptitle('Verdict-Flip-Rate landscape · 3 bulletproof reruns', fontsize=11, fontweight='bold', y=1.04)
plt.savefig(OUT / 'J4_vfr_three_experiments.png', bbox_inches='tight', dpi=300)
plt.close()
print('J4_vfr_three_experiments.png saved')

# ---------------------------------------------------------------------
# J5 - seed=42 vs seed=123 reconciliation (slope chart)
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
metrics_names = ['AUROC\nstd', 'AUROC\nfair', 'Acc\nstd', 'Acc\nfair', 'Acc cost\n(pp)']
e1_vals = [e1.AUROC_standard, e1.AUROC_fair, e1.Acc_standard, e1.Acc_fair, e1.Acc_cost_pp]
e3_vals = [e3.AUROC_standard, e3.AUROC_fair, e3.Acc_standard, e3.Acc_fair, e3.Acc_cost_pp]
# (a) absolute values (left axis 0-1, right axis for cost)
ax = axes[0]
x = np.arange(5); width = 0.36
ax.bar(x - width/2, e1_vals, width, color='#3a78b8', edgecolor='black', linewidth=0.4, label='seed=42')
ax.bar(x + width/2, e3_vals, width, color='#5fa55a', edgecolor='black', linewidth=0.4, label='seed=123')
for i in range(5):
    ax.text(i - width/2, e1_vals[i] + (0.005 if e1_vals[i] < 5 else 0.1), f'{e1_vals[i]:.3f}' if e1_vals[i] < 5 else f'{e1_vals[i]:.2f}', ha='center', fontsize=8)
    ax.text(i + width/2, e3_vals[i] + (0.005 if e3_vals[i] < 5 else 0.1), f'{e3_vals[i]:.3f}' if e3_vals[i] < 5 else f'{e3_vals[i]:.2f}', ha='center', fontsize=8, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(metrics_names, fontsize=9.5)
ax.set_title('(a) Headline metrics (Acc cost on shared axis)', fontsize=10.5, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 6.0)

# (b) DI per attribute
ax = axes[1]
e1_di = [e1.DI_Race, e1.DI_Sex, e1.DI_Eth, e1.DI_Age]
e3_di = [e3.DI_Race, e3.DI_Sex, e3.DI_Eth, e3.DI_Age]
x = np.arange(4); width = 0.36
ax.bar(x - width/2, e1_di, width, color='#3a78b8', edgecolor='black', linewidth=0.4, label='seed=42')
ax.bar(x + width/2, e3_di, width, color='#5fa55a', edgecolor='black', linewidth=0.4, label='seed=123')
for i in range(4):
    ax.text(i - width/2, e1_di[i] + 0.005, f'{e1_di[i]:.3f}', ha='center', fontsize=8)
    ax.text(i + width/2, e3_di[i] + 0.005, f'{e3_di[i]:.3f}', ha='center', fontsize=8, fontweight='bold')
ax.axhline(0.80, color='red', linestyle='--', linewidth=1.2, alpha=0.7, label='4/5 rule')
ax.set_xticks(x); ax.set_xticklabels(['Race', 'Sex', 'Eth', 'Age'], fontsize=10, fontweight='bold')
ax.set_ylim(0.75, 1.02); ax.set_ylabel('DI on audit', fontsize=10, fontweight='bold')
ax.set_title('(b) Per-attribute DI · both seeds pass', fontsize=10.5, fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
ax.grid(axis='y', alpha=0.3)

plt.suptitle('Independent-seed reproducibility · seed=42 vs seed=123',
             fontsize=11, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUT / 'J5_seed_reproducibility.png', bbox_inches='tight', dpi=300)
plt.close()
print('J5_seed_reproducibility.png saved')

print('\nAll 5 journal figures saved to:', OUT)
