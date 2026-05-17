"""
F3 · single C4 panel matching the manuscript spec (§4.2.2 Figure 3).
7 metrics × 4 attributes = 28 cells, canonical XGBoost C4 only.
Cell colour = VFR ∈ [0, 0.5]. Cell label = P/F + VFR value.
"""
import pandas as pd, numpy as np, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

ROOT = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm")
TAB = ROOT / "output_final" / "tables"
FIG_OUT = ROOT / "paper_images" / "revisions"
FIG_OUT2 = ROOT / "output_final" / "figures" / "revisions"
FIG_OUT.mkdir(parents=True, exist_ok=True)
FIG_OUT2.mkdir(parents=True, exist_ok=True)

mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['savefig.dpi'] = 220

METRIC_ORDER = ['DI', 'SPD', 'EOPP', 'EOD', 'TI', 'PP', 'CAL']
ATTR_ORDER = ['RACE', 'SEX', 'ETHNICITY', 'AGE_GROUP']
ATTR_LABEL = ['Race', 'Sex', 'Eth', 'Age']

T_C4 = pd.read_csv(TAB / 'T13_axis1_vfr_config4.csv')
g = np.zeros((len(METRIC_ORDER), len(ATTR_ORDER)))
labels = np.empty((len(METRIC_ORDER), len(ATTR_ORDER)), dtype=object)
flip = 0
for i, m in enumerate(METRIC_ORDER):
    for j, a in enumerate(ATTR_ORDER):
        row = T_C4[(T_C4['metric'] == m) & (T_C4['attribute'] == a)]
        if len(row) == 0:
            g[i, j] = np.nan; labels[i, j] = ''; continue
        r = row.iloc[0]
        g[i, j] = r['vfr']
        v = 'P' if str(r['verdict_dominant']).lower() in ['fair', 'pass'] else 'F'
        labels[i, j] = f'{v}\n{r["vfr"]:.3f}'
        if r['vfr'] > 0: flip += 1

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(g, cmap='RdYlGn_r', vmin=0.0, vmax=0.5, aspect='auto')
ax.set_xticks(range(len(ATTR_LABEL))); ax.set_xticklabels(ATTR_LABEL, fontsize=14, fontweight='bold')
ax.set_yticks(range(len(METRIC_ORDER))); ax.set_yticklabels(METRIC_ORDER, fontsize=14, fontweight='bold')
for i in range(len(METRIC_ORDER)):
    for j in range(len(ATTR_ORDER)):
        txt = labels[i, j]
        if not txt: continue
        colour = 'white' if g[i, j] > 0.30 else 'black'
        ax.text(j, i, txt, ha='center', va='center', fontsize=11, fontweight='bold', color=colour)

cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
cbar.set_label('VFR (0 = stable verdict, 0.5 = coin-flip)', fontsize=12, fontweight='bold')
ax.set_title(f'F3 · Per-cell Verdict-Flip-Rate landscape (canonical XGBoost, Phase 5b · C4)\n'
             f'{flip} of 28 cells flip (VFR > 0); Race-axis cells dominate the high-instability quadrant',
             fontsize=12, fontweight='bold', pad=14)
plt.tight_layout()
for path in [FIG_OUT, FIG_OUT2]:
    plt.savefig(path / 'F3_vfr_heatmap.png', bbox_inches='tight', dpi=220)
    plt.savefig(path / 'F3_vfr_dual_heatmap.png', bbox_inches='tight', dpi=220)  # also overwrite the old dual name
plt.close()
print(f'F3 saved · {flip} flipping cells')
