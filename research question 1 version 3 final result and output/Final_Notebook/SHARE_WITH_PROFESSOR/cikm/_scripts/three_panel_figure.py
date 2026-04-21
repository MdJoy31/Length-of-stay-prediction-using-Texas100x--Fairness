"""D5 Three-Panel Figure (standalone).

Requires results/intervention_lambda_sweep.csv (from lambda_sweep.py) and
results/intervention_per_cluster.csv (from per_cluster_intervention.py).

Produces figures/FIG06_intervention_three_panel.png and .pdf.
"""
import pandas as pd, numpy as np, sys, os
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding='utf-8')
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.makedirs('figures', exist_ok=True)

# Colour-blind safe palette (Wong 2011)
CB_BLACK = '#000000'
CB_GRAY = '#999999'
CB_ORANGE = '#E69F00'
CB_SKY = '#56B4E9'
CB_GREEN = '#009E73'
CB_YELLOW = '#F0E442'
CB_BLUE = '#0072B2'
CB_RED = '#D55E00'
CB_PURPLE = '#CC79A7'

# Load inputs
lam_path = 'results/intervention_lambda_sweep.csv'
per_path = 'results/intervention_per_cluster.csv'
# Candidate grid (optional, for Pareto panel a)
# Notebook writes to output/tables/ — find it if available
cand_candidates = ['results/intervention_candidate_grid.csv',
                   'output/tables/intervention_candidate_grid.csv',
                   'output/tables/Table6_Intervention.csv']
cand_path = next((p for p in cand_candidates if os.path.exists(p)), None)

has_lam = os.path.exists(lam_path)
has_per = os.path.exists(per_path)
print(f'[inputs] lambda_sweep={has_lam}  per_cluster={has_per}  candidate_grid={cand_path}')

fig, axes = plt.subplots(1, 3, figsize=(18 / 2.54, 6 / 2.54), dpi=300)

# --- Panel (a): Pareto frontier ---
ax = axes[0]
if cand_path and os.path.exists(cand_path):
    cand = pd.read_csv(cand_path)
    x_col = 'Accuracy' if 'Accuracy' in cand.columns else 'accuracy'
    y_col = 'Total_Fair' if 'Total_Fair' in cand.columns else 'total_fair'
    if x_col in cand.columns and y_col in cand.columns:
        ax.scatter(cand[x_col], cand[y_col], s=4, c=CB_GRAY, alpha=0.45, label='Candidates')
        # Pareto: non-dominated points (maximize acc AND fair)
        pts = list(zip(cand[x_col], cand[y_col]))
        pareto = []
        for p in sorted(pts, key=lambda v: (-v[0], -v[1])):
            if not pareto or p[1] > pareto[-1][1]:
                pareto.append(p)
        px = [p[0] for p in pareto]; py = [p[1] for p in pareto]
        ax.scatter(px, py, s=10, c=CB_ORANGE, label='Pareto front')
        ax.step(px, py, c=CB_ORANGE, where='post', lw=0.7)
        # Standard + Fair selected
        if 'Model' in cand.columns:
            std = cand[cand['Model'] == 'Standard']
            if len(std):
                ax.scatter(std[x_col].iloc[0], std[y_col].iloc[0],
                           s=40, c=CB_BLACK, marker='s', label='Standard')
        ax.axhline(28, ls='--', c=CB_GRAY, lw=0.5)
else:
    ax.text(0.5, 0.5, 'Candidate grid not available\n(run notebook Cell 34)', ha='center',
            va='center', fontsize=7, color=CB_GRAY)
ax.set_xlabel('Accuracy', fontsize=8)
ax.set_ylabel('Fair metrics (out of 28)', fontsize=8)
ax.set_title('(a) Pareto frontier', fontsize=9, loc='left')
ax.tick_params(labelsize=7)
ax.legend(fontsize=7, loc='lower left', frameon=False)

# --- Panel (b): DI before/after per attribute ---
ax = axes[1]
if has_per:
    per = pd.read_csv(per_path)
    attrs = ['di_race', 'di_sex', 'di_eth', 'di_age']
    attr_labels = ['Race', 'Sex', 'Eth.', 'Age']
    std_vals = [per[per['model'] == 'Standard'][a].median() for a in attrs]
    fair_vals = [per[per['model'] == 'Fair'][a].median() for a in attrs]
    x = np.arange(len(attrs)); w = 0.35
    ax.bar(x - w/2, std_vals, w, color=CB_GRAY, label='Standard', edgecolor='none')
    ax.bar(x + w/2, fair_vals, w, color=CB_GREEN, label='Fair', edgecolor='none')
    ax.axhline(0.80, c=CB_RED, ls='--', lw=0.7, label='DI=0.80')
    for xi, v in zip(x - w/2, std_vals):
        ax.text(xi, v + 0.01, f'{v:.2f}', ha='center', fontsize=6)
    for xi, v in zip(x + w/2, fair_vals):
        ax.text(xi, v + 0.01, f'{v:.2f}', ha='center', fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(attr_labels)
    ax.set_ylim(0, 1.15)
else:
    ax.text(0.5, 0.5, 'Per-cluster data unavailable\n(run per_cluster_intervention.py)',
            ha='center', va='center', fontsize=7, color=CB_GRAY)
ax.set_ylabel('Disparate Impact', fontsize=8)
ax.set_title('(b) DI before/after (cluster median)', fontsize=9, loc='left')
ax.tick_params(labelsize=7)
ax.legend(fontsize=7, frameon=False, loc='lower left')

# --- Panel (c): 2x2 subplots of paired cluster points for each attribute ---
if has_per:
    attrs = [('di_race', 'Race'), ('di_sex', 'Sex'), ('di_eth', 'Eth.'), ('di_age', 'Age')]
    # Split the third axes into 2x2 inset
    axes[2].remove()
    subgs = fig.add_gridspec(2, 2, left=0.69, right=0.99, top=0.95, bottom=0.13, wspace=0.35, hspace=0.45)
    for (metric, lbl), (r, c) in zip(attrs, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        ax3 = fig.add_subplot(subgs[r, c])
        per_lo_col = f'{metric}_ci_lo'
        per_hi_col = f'{metric}_ci_hi'
        # Pair by cluster
        std_by = per[per['model'] == 'Standard'].set_index('cluster_id')
        fair_by = per[per['model'] == 'Fair'].set_index('cluster_id')
        improved = 0; regressed = 0
        for cid in std_by.index:
            if cid not in fair_by.index: continue
            ys = std_by.loc[cid, metric]
            yf = fair_by.loc[cid, metric]
            colour = CB_GREEN if yf >= ys else CB_RED
            ax3.plot([0, 1], [ys, yf], c=colour, lw=0.5, alpha=0.7)
            ax3.scatter([0, 1], [ys, yf], c=[CB_GRAY, colour], s=5)
            if yf >= ys: improved += 1
            else: regressed += 1
        ax3.axhline(0.80, c=CB_RED, ls='--', lw=0.5)
        ax3.set_xticks([0, 1]); ax3.set_xticklabels(['Std', 'Fair'], fontsize=6)
        ax3.set_ylim(0, 1.15)
        ax3.set_title(f'{lbl}  ({improved}/20 improved)', fontsize=7, loc='left')
        ax3.tick_params(labelsize=6)
else:
    axes[2].text(0.5, 0.5, '(c) per-cluster trajectories\nunavailable',
                 ha='center', va='center', fontsize=7, color=CB_GRAY)
    axes[2].set_xticks([]); axes[2].set_yticks([])

plt.tight_layout()
out_png = 'figures/FIG06_intervention_three_panel.png'
out_pdf = 'figures/FIG06_intervention_three_panel.pdf'
plt.savefig(out_png, dpi=300, bbox_inches='tight')
plt.savefig(out_pdf, bbox_inches='tight')
plt.close()
print(f'[saved] {out_png}')
print(f'[saved] {out_pdf}')
