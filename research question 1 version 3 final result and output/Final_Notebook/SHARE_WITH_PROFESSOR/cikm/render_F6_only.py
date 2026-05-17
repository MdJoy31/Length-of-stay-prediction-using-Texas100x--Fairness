"""
render_F6_only.py
=================
Per-model fairness/accuracy trade-off for all 12 classifiers, computed under
per-protected-attribute threshold shift on RACE (the achievable axis). The
intervention is the simple α-search variant; the canonical Phase 5b pipeline
extends this with greedy refinement and an additional AGE shift, but that is
demonstrated separately in T15 / F7 for the canonical XGBoost.

Headline finding to surface in the figure:
  - RACE-axis threshold shift achieves DI_RACE ≥ 0.80 on all 12 models
    at near-zero accuracy cost (max Δ_acc = -0.001).
  - AGE-axis DI remains <0.31 on all 12 models — the AGE base-rate gap
    (0.399) is a structural floor that per-attribute threshold shifting
    alone cannot cross. This is the model-agnostic confirmation of the
    manuscript §5.2 thesis.
"""
import pandas as pd, numpy as np, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

ROOT = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm")
TAB  = ROOT / "output_final" / "tables"
FIG_OUT = ROOT / "paper_images" / "revisions"
FIG_OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['savefig.dpi'] = 220

T = pd.read_csv(TAB / 'T_per_model_before_after.csv')
print(f"loaded T_per_model_before_after.csv  ({len(T)} models)")

# Two-panel figure: (a) RACE-axis trade-off (moves), (b) AGE-axis trade-off (doesn't move)
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
palette = plt.cm.tab20(np.linspace(0, 1, len(T)))

# ---------- Panel A: RACE-axis trade-off ----------
ax = axes[0]
for (_, r), c in zip(T.iterrows(), palette):
    x0, y0 = r['Acc_before'], r['DI_RACE_before']
    x1, y1 = r['Acc_after'],  r['DI_RACE_after']
    ax.scatter(x0, y0, s=120, facecolor='white', edgecolor=c, linewidth=2.0, marker='o', zorder=4)
    ax.scatter(x1, y1, s=180, facecolor=c, edgecolor='black', linewidth=1.0, marker='o', zorder=5)
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', color=c, alpha=0.7, lw=1.8), zorder=3)
    short = r['Model'].split()[0] if len(r['Model']) > 14 else r['Model']
    ax.text(x1, y1 + 0.025, short, fontsize=8.5, ha='center', color='black', fontweight='bold')
ax.axhline(0.80, color='red', linestyle='--', linewidth=2.0, alpha=0.7, label='Four-fifths rule (DI = 0.80)')
ax.axhline(1.00, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
ax.set_xlabel('Test-set accuracy', fontsize=12, fontweight='bold')
ax.set_ylabel('DI · Race axis (closer to 1 = fairer)', fontsize=12, fontweight='bold')
ax.set_title('(a) Race-axis threshold shift: works universally\nAll 12 models move above DI = 0.80 at near-zero accuracy cost',
             fontsize=11, fontweight='bold')
ax.grid(alpha=0.3)
ax.set_ylim(0.45, 1.05)
ax.set_xlim(0.77, 0.90)
ax.legend(loc='lower right', fontsize=10, frameon=True)

# ---------- Panel B: AGE-axis trade-off ----------
ax = axes[1]
for (_, r), c in zip(T.iterrows(), palette):
    x0, y0 = r['Acc_before'], r['DI_AGE_GROUP_before']
    x1, y1 = r['Acc_after'],  r['DI_AGE_GROUP_after']
    ax.scatter(x0, y0, s=120, facecolor='white', edgecolor=c, linewidth=2.0, marker='o', zorder=4)
    ax.scatter(x1, y1, s=180, facecolor=c, edgecolor='black', linewidth=1.0, marker='o', zorder=5)
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', color=c, alpha=0.7, lw=1.8), zorder=3)
    short = r['Model'].split()[0] if len(r['Model']) > 14 else r['Model']
    ax.text(x1, y1 + 0.012, short, fontsize=8.5, ha='center', color='black', fontweight='bold')
ax.axhline(0.80, color='red', linestyle='--', linewidth=2.0, alpha=0.7, label='Four-fifths rule (DI = 0.80)')
ax.axhline(1.00, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
# Annotate the structural barrier
ax.axhspan(0.0, 0.31, color='red', alpha=0.10, zorder=1)
ax.text(0.835, 0.155, 'Structural floor\n(Age base-rate gap = 0.399)\nrequires greedy refinement\n(canonical Phase 5b)',
         ha='center', va='center', fontsize=10, fontweight='bold', color='#8b0000',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#8b0000'))
ax.set_xlabel('Test-set accuracy', fontsize=12, fontweight='bold')
ax.set_ylabel('DI · Age axis', fontsize=12, fontweight='bold')
ax.set_title('(b) Age-axis threshold shift: structurally infeasible\nNo model crosses DI = 0.80 under per-attribute shift alone',
             fontsize=11, fontweight='bold')
ax.grid(alpha=0.3)
ax.set_ylim(0.0, 1.05)
ax.set_xlim(0.77, 0.90)
ax.legend(loc='upper right', fontsize=10, frameon=True)

plt.suptitle('F6 · Per-model trade-off across 12 classifiers: Race vs Age axis',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIG_OUT / 'F6_per_model_tradeoff.png', bbox_inches='tight', dpi=220)
plt.close()
print("F6 saved")

# Also copy to output_final/figures/revisions/
FIG_OUT2 = ROOT / "output_final" / "figures" / "revisions"
FIG_OUT2.mkdir(parents=True, exist_ok=True)
import shutil
shutil.copy(FIG_OUT / 'F6_per_model_tradeoff.png', FIG_OUT2 / 'F6_per_model_tradeoff.png')
print(f"Mirrored to {FIG_OUT2}")
