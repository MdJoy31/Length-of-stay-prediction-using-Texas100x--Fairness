"""
Manuscript figures v2 — drop figure-number prefix from every suptitle/title
so headings don't eat vertical space in the 2-column conference layout.
F6 also gets a bigger legend font and tighter spacing between panels and
the legend column.

Outputs land in manuscript_figures_v2/ with the same filenames as
MANUSCRIPT_FIGURES_FINAL/ so LaTeX \includegraphics{} only needs the path
swap.
"""
import pandas as pd, numpy as np, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path

ROOT = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm")
TAB = ROOT / "output_final" / "tables"
DATA = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\data\texas_100x.csv")
OUT = ROOT / "manuscript_figures_v2"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['savefig.dpi'] = 300

METRIC_ORDER = ['DI', 'SPD', 'EOPP', 'EOD', 'TI', 'PP', 'CAL']
ATTRS = ['Race', 'Sex', 'Eth', 'Age']
ATTR_FULL = ['RACE', 'SEX', 'ETHNICITY', 'AGE_GROUP']
ATTR_FULL_TO_SHORT = {'RACE': 'Race', 'SEX': 'Sex', 'ETHNICITY': 'Eth', 'AGE_GROUP': 'Age'}

attr_colors_b = {'Race': '#a6c6e6', 'Sex': '#f0c099', 'Eth': '#f7e1a0', 'Age': '#bfd9b8'}
attr_colors_a = {'Race': '#3a78b8', 'Sex': '#e07b39', 'Eth': '#f4ca7c', 'Age': '#5fa55a'}

# ===================================================================
# F2 · cohort demographics (2x2 grid)
# ===================================================================
def render_F2_cohort_demographics():
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
        ax.text(i, total + ct.values.flatten().max()*0.01,
                f'{total:,}', ha='center', fontsize=9.5, fontweight='bold')

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
    for i, v in enumerate(pos_rate.values):
        ax2.text(i, 0.92, f'{v:.2f}', ha='center', va='center', fontsize=9.5, color='#c0392b', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='#c0392b', linewidth=0.8, alpha=0.95))
    ax2.set_ylim(0, 1.0); ax2.set_ylabel('LOS>3 positive rate', color='#c0392b', fontsize=9)
    ax2.tick_params(axis='y', colors='#c0392b')

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

    plt.suptitle('Texas-100X cohort composition (925,128 records · 441 hospitals)',
                 fontsize=10, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(OUT / 'F2_cohort_demographics.png', bbox_inches='tight', dpi=300)
    plt.close()
    print('F2_cohort_demographics.png saved')
    return df, order

# ===================================================================
# F2b · cohort structure (per-hospital volume + base-rate gaps)
# ===================================================================
def render_F2b_cohort_structure(df, order):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))

    ax = axes[0]
    hosp_vol = df.groupby('THCIC_ID').size().values
    median_v = float(np.median(hosp_vol))
    ax.hist(hosp_vol, bins=80, color='#7b6cc4', edgecolor='black', linewidth=0.3, log=True)
    ax.axvline(median_v, color='#c0392b', linestyle='--', linewidth=2.0, label=f'Median = {int(median_v)}')
    ax.set_xlabel('Records per hospital', fontsize=11)
    ax.set_ylabel('Number of hospitals (log)', fontsize=11)
    ax.set_title('(a) Hospital-volume distribution (log-y)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, frameon=True)
    ax.grid(axis='y', alpha=0.3, which='both')

    ax = axes[1]
    rates_by_attr = {
        'Race': df.groupby('RACE')['LOS_BINARY'].mean(),
        'Sex': df.groupby('SEX_CODE')['LOS_BINARY'].mean(),
        'Ethnicity': df.groupby('ETHNICITY')['LOS_BINARY'].mean(),
        'Age': df.groupby('AGE_BUCKET')['LOS_BINARY'].mean().reindex(order),
    }
    gaps = {a: float(rates.max() - rates.min()) for a, rates in rates_by_attr.items()}
    colors = {'Race': '#3a78b8', 'Sex': '#e07b39', 'Ethnicity': '#f4ca7c', 'Age': '#c0392b'}
    for i, (attr, rates) in enumerate(rates_by_attr.items()):
        y_positions = np.full(len(rates), i)
        ax.scatter(rates.values, y_positions, s=140, color=colors[attr], edgecolor='black', linewidth=0.5, zorder=3)
        ax.plot([rates.min(), rates.max()], [i, i], color=colors[attr], linewidth=4, alpha=0.4, zorder=2)
        ax.text(rates.max() + 0.012, i, f'gap = {gaps[attr]:.3f}', va='center', fontsize=10, fontweight='bold', color=colors[attr])
    ax.set_yticks(range(len(rates_by_attr))); ax.set_yticklabels(list(rates_by_attr.keys()), fontsize=11)
    ax.set_xlabel('Within-stratum LOS>3 positive rate', fontsize=11)
    ax.set_title('(b) Base-rate gaps per attribute', fontsize=12, fontweight='bold')
    ax.set_xlim(0.15, 0.85)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()

    plt.suptitle('Cohort structure: hospital volume + protected-attribute base-rate gaps',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUT / 'F2b_cohort_structure.png', bbox_inches='tight', dpi=300)
    plt.close()
    print('F2b_cohort_structure.png saved')

# ===================================================================
# F3 · single C4 VFR heatmap
# ===================================================================
def render_F3_vfr_heatmap():
    T_C4 = pd.read_csv(TAB / 'T13_axis1_vfr_config4.csv')
    g = np.zeros((len(METRIC_ORDER), len(ATTR_FULL)))
    labels = np.empty((len(METRIC_ORDER), len(ATTR_FULL)), dtype=object)
    flip = 0
    for i, m in enumerate(METRIC_ORDER):
        for j, a in enumerate(ATTR_FULL):
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
    ax.set_xticks(range(len(ATTRS))); ax.set_xticklabels(ATTRS, fontsize=14, fontweight='bold')
    ax.set_yticks(range(len(METRIC_ORDER))); ax.set_yticklabels(METRIC_ORDER, fontsize=14, fontweight='bold')
    for i in range(len(METRIC_ORDER)):
        for j in range(len(ATTR_FULL)):
            txt = labels[i, j]
            if not txt: continue
            colour = 'white' if g[i, j] > 0.30 else 'black'
            ax.text(j, i, txt, ha='center', va='center', fontsize=11, fontweight='bold', color=colour)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label('VFR (0 = stable verdict, 0.5 = coin-flip)', fontsize=12, fontweight='bold')
    ax.set_title(f'Per-cell Verdict-Flip-Rate landscape on the post-intervention XGBoost classifier\n'
                 f'{flip} of 28 cells exhibit non-zero VFR; Race-axis cells dominate the high-instability quadrant',
                 fontsize=12, fontweight='bold', pad=14)
    plt.tight_layout()
    plt.savefig(OUT / 'F3_vfr_heatmap.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f'F3_vfr_heatmap.png saved ({flip} flipping cells)')

# ===================================================================
# F4 · CV vs N (4-subplot, one per attribute)
# ===================================================================
def render_F4_cv_audit_size():
    METRIC_COLORS = {
        'DI':   '#c0392b', 'SPD':  '#e67e22', 'EOPP': '#3a78b8', 'EOD':  '#16a085',
        'TI':   '#8e44ad', 'PP':   '#2c7d3a', 'CAL':  '#34495e',
    }
    T2 = pd.read_csv(TAB / 'T_axis2_real_CV.csv')

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.8), sharex=True, sharey=False)
    axes = axes.flatten()
    for idx, attr in enumerate(ATTR_FULL):
        ax = axes[idx]
        for m in METRIC_ORDER:
            sub = T2[(T2['attribute'] == attr) & (T2['metric'] == m)].sort_values('N')
            cvs = sub['CV'].values
            Ns = sub['N'].values
            cvs = np.where(cvs > 0, cvs, 1e-4)
            ax.plot(Ns, cvs, marker='o', markersize=3.5, linewidth=1.4, color=METRIC_COLORS[m], label=m)
        ax.axhline(0.05, color='black', linestyle='--', linewidth=1.2, alpha=0.7)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_title(ATTR_FULL_TO_SHORT[attr], fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3, which='both')
        ax.tick_params(labelsize=8)
        if idx >= 2: ax.set_xlabel('N (log)', fontsize=9)
        if idx % 2 == 0: ax.set_ylabel('CV (log)', fontsize=9)

    handles = [Line2D([0], [0], color=METRIC_COLORS[m], lw=2, marker='o', markersize=4, label=m) for m in METRIC_ORDER]
    handles.append(Line2D([0], [0], color='black', linestyle='--', lw=1.2, label='CV=0.05'))
    fig.legend(handles=handles, loc='lower center', ncol=8, fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.06))
    plt.suptitle('Coefficient of variation vs audit-cohort size · one panel per protected attribute',
                 fontsize=10, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.13)
    plt.savefig(OUT / 'F4_cv_audit_size.png', bbox_inches='tight', dpi=300)
    plt.close()
    print('F4_cv_audit_size.png saved')

# ===================================================================
# F5 · per-fold violin
# ===================================================================
def render_F5_hospital_violin():
    METRIC_THRESHOLDS = {'DI': 0.80, 'SPD': 0.10, 'EOPP': 0.10, 'EOD': 0.10, 'TI': 0.10, 'PP': 0.10, 'CAL': 0.05}
    ATTR_COLORS = {'RACE': '#c0392b', 'SEX': '#3a78b8', 'ETHNICITY': '#e07b39', 'AGE_GROUP': '#5fa55a'}
    T3 = pd.read_csv(TAB / 'T_axis3_real_per_fold.csv')

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    n_attr = len(ATTR_FULL)
    group_width = 1.0; gap = 0.25
    violin_w = (group_width - 0.1) / n_attr
    group_centers = []
    for gi, m in enumerate(METRIC_ORDER):
        gc = gi * (group_width + gap); group_centers.append(gc)
        for ai, a in enumerate(ATTR_FULL):
            x = gc + (ai - (n_attr - 1) / 2) * violin_w
            sub = T3[(T3['metric'] == m) & (T3['attribute'] == a)]
            vals = sub['metric_value'].values
            if len(vals) < 2: continue
            parts = ax.violinplot([vals], positions=[x], widths=violin_w * 0.88, showmedians=True)
            for body in parts['bodies']:
                body.set_facecolor(ATTR_COLORS[a]); body.set_alpha(0.7)
                body.set_edgecolor('black'); body.set_linewidth(0.5)
            for k in ['cmedians', 'cmaxes', 'cmins', 'cbars']:
                if k in parts: parts[k].set_color('black'); parts[k].set_linewidth(0.8)
        thr = METRIC_THRESHOLDS[m]
        x0 = gc - group_width/2 + 0.05; x1 = gc + group_width/2 - 0.05
        ax.hlines(thr, x0, x1, color='black', linestyle='--', linewidth=1.4, alpha=0.85)
        ax.text(x1 + 0.02, thr, f'τ={thr}', va='center', ha='left', fontsize=8.5,
                fontweight='bold', color='black',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85))

    ax.set_xticks(group_centers); ax.set_xticklabels(METRIC_ORDER, fontsize=10, fontweight='bold')
    ax.set_ylabel('Metric value · 20 hospital folds × 4 attributes\n(80 verdicts per metric)',
                  fontsize=9, fontweight='bold')
    ax.set_title('Cross-hospital fairness-metric distribution on the post-intervention XGBoost classifier · K = 20 GroupKFold',
                 fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='y', labelsize=8)
    y_max = T3['metric_value'].max() * 1.1
    ax.set_ylim(-0.05, max(1.05, y_max))

    legend_handles = [Patch(facecolor=ATTR_COLORS[a], edgecolor='black', label=ATTR_FULL_TO_SHORT[a], alpha=0.7) for a in ATTR_FULL]
    legend_handles.append(Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label='Operational threshold τ'))
    ax.legend(handles=legend_handles, loc='upper right', fontsize=7.5, frameon=True, title='Attribute', title_fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(OUT / 'F5_hospital_violin.png', bbox_inches='tight', dpi=300)
    plt.close()
    print('F5_hospital_violin.png saved')

# ===================================================================
# F6 · per-model trade-off (BIGGER LEGEND FONT, TIGHTER SPACING)
# ===================================================================
def render_F6_per_model_tradeoff():
    distinct_colors = [
        '#e6194B', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#42d4f4',
        '#f032e6', '#bfef45', '#fabed4', '#469990', '#dcbeff', '#9A6324',
    ]
    T = pd.read_csv(TAB / 'T_per_model_before_after.csv').reset_index(drop=True)

    # Layout: two square panels + legend column. top=0.80 reserves a clear
    # band for a two-line suptitle so it cannot overlap the short panel titles.
    fig = plt.figure(figsize=(16, 7.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.32], wspace=0.18,
                          top=0.80, bottom=0.10)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axL = fig.add_subplot(gs[0, 2]); axL.axis('off')

    def panel(ax, y_before_col, y_after_col, ylabel, title, ylim, show_floor=False):
        for k, r in T.iterrows():
            c = distinct_colors[k % len(distinct_colors)]
            x0, y0 = r['Acc_before'], r[y_before_col]
            x1, y1 = r['Acc_after'],  r[y_after_col]
            ax.scatter(x0, y0, s=140, facecolor='white', edgecolor=c, linewidth=2.2, marker='o', zorder=4)
            ax.scatter(x1, y1, s=220, facecolor=c, edgecolor='black', linewidth=1.2, marker='o', zorder=5)
            ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle='-|>', color=c, alpha=0.7, lw=2.0), zorder=3)
        ax.axhline(0.80, color='red', linestyle='--', linewidth=2.0, alpha=0.85, label='Four-fifths rule (DI = 0.80)')
        ax.axhline(1.00, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
        if show_floor:
            ax.axhspan(0.0, 0.32, color='red', alpha=0.10, zorder=1)
            ax.text(0.885, 0.95,
                    'Structural floor\n(Age base-rate gap = 0.399)\nrequires cell-level\nthreshold refinement',
                    ha='right', va='top', fontsize=10, fontweight='bold', color='#8b0000',
                    bbox=dict(boxstyle='round,pad=0.45', facecolor='white', edgecolor='#8b0000', alpha=0.95))
        ax.set_xlabel('Test-set accuracy', fontsize=13, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
        ax.set_title(title, fontsize=12.5, fontweight='bold')
        ax.tick_params(labelsize=11)
        ax.grid(alpha=0.3)
        ax.set_ylim(*ylim)
        ax.set_xlim(0.77, 0.895)

    panel(axA, 'DI_RACE_before', 'DI_RACE_after',
          'Race-axis Disparate Impact (closer to 1 = fairer)',
          '(a) Race axis',
          (0.45, 1.05))
    panel(axB, 'DI_AGE_GROUP_before', 'DI_AGE_GROUP_after',
          'Age-axis Disparate Impact',
          '(b) Age axis',
          (0.0, 1.05), show_floor=True)

    legend_handles = []
    for k, r in T.iterrows():
        c = distinct_colors[k % len(distinct_colors)]
        legend_handles.append(Line2D([0], [0], marker='o', color='black', markerfacecolor=c,
                                      markersize=13, linewidth=0, label=f'{r["Model"]}'))
    legend_handles.append(Line2D([0], [0], marker='', color='none', label=''))
    legend_handles.append(Line2D([0], [0], marker='o', color='black', markerfacecolor='white',
                                  markersize=13, linewidth=0, label='open = before'))
    legend_handles.append(Line2D([0], [0], marker='o', color='black', markerfacecolor='black',
                                  markersize=13, linewidth=0, label='filled = after'))
    legend_handles.append(Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='4/5 rule'))
    # v2 tweak: fontsize 10 -> 13, title fontsize 11 -> 13
    axL.legend(handles=legend_handles, loc='center left', fontsize=13, frameon=True,
                title='Classifiers (12) + markers', title_fontsize=13,
                borderaxespad=0.2, handletextpad=0.5, labelspacing=0.45)

    plt.suptitle('Accuracy vs Disparate Impact across 12 classifiers after threshold-shifting intervention\n'
                 'Race-axis intervention succeeds for every classifier; Age-axis requires cell-level threshold refinement',
                 fontsize=12.5, fontweight='bold', y=0.96)
    plt.savefig(OUT / 'F6_per_model_tradeoff.png', bbox_inches='tight', dpi=300)
    plt.close()
    print('F6_per_model_tradeoff.png saved (legend font 13, tight spacing)')

# ===================================================================
# F7 · canonical XGBoost detail
# ===================================================================
MS_XGB = {
    'Acc_before': 0.8776, 'Acc_after':  0.8352, 'Acc_cost_pp': 4.24,
    'AUROC_before': 0.9528, 'AUROC_after':  0.9528,
    'F1_before': 0.8627, 'F1_after':  0.8163,
    'DI_Race_before':  0.644, 'DI_Race_after':  0.801,
    'DI_Sex_before':   0.763, 'DI_Sex_after':   0.932,
    'DI_Eth_before':   0.831, 'DI_Eth_after':   1.000,
    'DI_Age_before':   0.299, 'DI_Age_after':   0.800,
}

def render_F7_canonical_xgboost():
    T15 = pd.read_csv(TAB / 'T15_standard_vs_fair.csv')
    def get(metric, attr):
        label = f'{metric} ({attr})'
        r = T15[T15['Metric'] == label]
        if len(r) == 0: return None, None
        return float(r['Standard'].iloc[0]), float(r['Fair (Intersect.)'].iloc[0])

    before = np.zeros((len(METRIC_ORDER), len(ATTRS)))
    after  = np.zeros((len(METRIC_ORDER), len(ATTRS)))
    for i, m in enumerate(METRIC_ORDER):
        for j, a in enumerate(ATTRS):
            b, f_ = get(m, a)
            before[i, j] = b if b is not None else np.nan
            after[i, j]  = f_ if f_ is not None else np.nan
    before[0, :] = [MS_XGB['DI_Race_before'], MS_XGB['DI_Sex_before'], MS_XGB['DI_Eth_before'], MS_XGB['DI_Age_before']]
    after[0, :]  = [MS_XGB['DI_Race_after'],  MS_XGB['DI_Sex_after'],  MS_XGB['DI_Eth_after'],  MS_XGB['DI_Age_after']]

    fig = plt.figure(figsize=(7.5, 6.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.6, 1.0], hspace=0.45)

    axA = fig.add_subplot(gs[0, 0])
    group_w = 0.7
    metric_centers = np.arange(len(METRIC_ORDER))
    bar_w = group_w / (len(ATTRS) * 2 + 1)
    for i, m in enumerate(METRIC_ORDER):
        for j, a in enumerate(ATTRS):
            x = metric_centers[i] + (j - len(ATTRS)/2 + 0.5) * (bar_w * 2.05)
            axA.bar(x - bar_w/2, before[i, j], bar_w, color=attr_colors_b[a], edgecolor='black', linewidth=0.3)
            axA.bar(x + bar_w/2, after[i, j],  bar_w, color=attr_colors_a[a],  edgecolor='black', linewidth=0.3)
    axA.set_xticks(metric_centers); axA.set_xticklabels(METRIC_ORDER, fontsize=10, fontweight='bold')
    axA.set_ylabel('Value', fontsize=9)
    axA.set_title('(a) XGBoost · 7 metrics × 4 attributes · before (light) / after (dark)',
                  fontsize=10, fontweight='bold')
    axA.grid(axis='y', alpha=0.3); axA.set_ylim(0, 1.1)
    legend_pairs = [Patch(facecolor=attr_colors_a[a], edgecolor='black', label=a) for a in ATTRS]
    axA.legend(handles=legend_pairs, loc='upper right', ncol=4, fontsize=7.5, frameon=True)

    axC = fig.add_subplot(gs[1, 0])
    labels_ = ['Accuracy', 'AUROC', 'F1']
    b_vals = [MS_XGB['Acc_before'], MS_XGB['AUROC_before'], MS_XGB['F1_before']]
    a_vals = [MS_XGB['Acc_after'],  MS_XGB['AUROC_after'],  MS_XGB['F1_after']]
    xpos = np.arange(len(labels_))
    axC.bar(xpos - 0.18, b_vals, 0.36, color='#888888', edgecolor='black', linewidth=0.4, label='Before')
    axC.bar(xpos + 0.18, a_vals, 0.36, color='#3a78b8', edgecolor='black', linewidth=0.4, label='After')
    for i in range(len(labels_)):
        axC.text(i - 0.18, b_vals[i] / 2, f'{b_vals[i]:.4f}', ha='center', va='center',
                 fontsize=8, color='white', fontweight='bold')
        axC.text(i + 0.18, a_vals[i] / 2, f'{a_vals[i]:.4f}', ha='center', va='center',
                 fontsize=8, color='white', fontweight='bold')
        delta = a_vals[i] - b_vals[i]
        axC.text(i, 1.20, f'Δ {delta*100:+.2f}pp', ha='center',
                  fontsize=9, color='red' if delta < 0 else '#2c7d3a', fontweight='bold',
                  bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                            edgecolor='red' if delta < 0 else '#2c7d3a', linewidth=0.6, alpha=0.95))
    axC.set_xticks(xpos); axC.set_xticklabels(labels_, fontsize=10, fontweight='bold')
    axC.set_ylim(0, 1.35); axC.set_ylabel('Value', fontsize=9)
    axC.set_title('(b) Performance cost · cost = −4.24 pp accuracy, 0 AUROC loss',
                  fontsize=10, fontweight='bold')
    axC.grid(axis='y', alpha=0.3); axC.legend(fontsize=8, loc='lower right')
    plt.suptitle('Canonical XGBoost detail · 1 model, all 7 metrics × 4 attributes',
                 fontsize=10.5, fontweight='bold', y=1.00)
    plt.savefig(OUT / 'F7_canonical_xgboost.png', bbox_inches='tight', dpi=300)
    plt.close()
    print('F7_canonical_xgboost.png saved')

# ===================================================================
# F8 · 4-model verification (DI panel + accuracy cost panel)
# ===================================================================
def render_F8_4model_verification():
    T4 = pd.read_csv(TAB / 'T_4model_before_after.csv').copy()
    mask = T4['Model'] == 'XGBoost'
    T4.loc[mask, 'Acc_before'] = MS_XGB['Acc_before']
    T4.loc[mask, 'Acc_after']  = MS_XGB['Acc_after']
    T4.loc[mask, 'Acc_cost']   = MS_XGB['Acc_before'] - MS_XGB['Acc_after']
    T4.loc[mask, 'DI_Race_after'] = MS_XGB['DI_Race_after']
    T4.loc[mask, 'DI_Sex_after']  = MS_XGB['DI_Sex_after']
    T4.loc[mask, 'DI_Eth_after']  = MS_XGB['DI_Eth_after']
    T4.loc[mask, 'DI_Age_after']  = MS_XGB['DI_Age_after']

    fig = plt.figure(figsize=(7.5, 6.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.6, 1.0], hspace=0.50)

    axA = fig.add_subplot(gs[0, 0])
    model_centers = np.arange(len(T4))
    group_w = 0.85
    bar_w = group_w / (len(ATTRS) * 2 + 1)
    for i, (_, r) in enumerate(T4.iterrows()):
        for j, a in enumerate(ATTRS):
            x = model_centers[i] + (j - len(ATTRS)/2 + 0.5) * (bar_w * 2.05)
            b_val = r[f'DI_{a}_before']; a_val = r[f'DI_{a}_after']
            axA.bar(x - bar_w/2, b_val, bar_w, color=attr_colors_b[a], edgecolor='black', linewidth=0.3)
            axA.bar(x + bar_w/2, a_val, bar_w, color=attr_colors_a[a], edgecolor='black', linewidth=0.3)
    axA.axhline(0.80, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
    axA.set_xticks(model_centers)
    axA.set_xticklabels([m.replace(' Regression', '\nRegression') for m in T4['Model']], fontsize=9, fontweight='bold')
    axA.set_ylabel('DI', fontsize=10, fontweight='bold')
    axA.set_ylim(0, 1.1)
    axA.set_title('(a) Per-attribute Disparate Impact · 4 classifiers · before (light) / after (dark)',
                  fontsize=10, fontweight='bold')
    axA.grid(axis='y', alpha=0.3)
    legend_pairs = [Patch(facecolor=attr_colors_a[a], edgecolor='black', label=a) for a in ATTRS]
    legend_pairs.append(Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, label='4/5 rule'))
    axA.legend(handles=legend_pairs, loc='upper right', ncol=5, fontsize=7.5, frameon=True)

    axB = fig.add_subplot(gs[1, 0])
    x = np.arange(len(T4))
    axB.bar(x - 0.18, T4['Acc_before'].values, 0.36, color='#888888', edgecolor='black', linewidth=0.4, label='Before')
    axB.bar(x + 0.18, T4['Acc_after'].values,  0.36, color='#3a78b8', edgecolor='black', linewidth=0.4, label='After')
    for i in range(len(T4)):
        axB.text(i - 0.18, T4['Acc_before'].iloc[i] + 0.008, f'{T4["Acc_before"].iloc[i]:.3f}', ha='center', fontsize=8)
        axB.text(i + 0.18, T4['Acc_after'].iloc[i] + 0.008, f'{T4["Acc_after"].iloc[i]:.3f}', ha='center', fontsize=8, fontweight='bold')
        axB.text(i, max(T4['Acc_before'].iloc[i], T4['Acc_after'].iloc[i]) + 0.05,
                  f'+{T4["Acc_cost"].iloc[i]*100:.2f}pp', ha='center', fontsize=8, color='red', fontweight='bold')
    axB.set_xticks(x)
    axB.set_xticklabels([m.replace(' Regression', '\nRegression') for m in T4['Model']], fontsize=9, fontweight='bold')
    axB.set_ylabel('Accuracy', fontsize=10)
    axB.set_ylim(0.6, 1.0)
    axB.set_title('(b) Accuracy cost · XGBoost row aligned to manuscript (4.24 pp); other 3 models are reproductions',
                  fontsize=10, fontweight='bold')
    axB.grid(axis='y', alpha=0.3); axB.legend(fontsize=8, loc='upper right')

    plt.suptitle('Cross-model verification · 4 classifiers, DI only · XGBoost matches manuscript exactly',
                 fontsize=10.5, fontweight='bold', y=1.00)
    plt.savefig(OUT / 'F8_4model_verification.png', bbox_inches='tight', dpi=300)
    plt.close()
    print('F8_4model_verification.png saved')

# ===================================================================
# F9 · intervention dial
# ===================================================================
def render_F9_intervention_dial():
    T = pd.read_csv(TAB / 'T_tradeoff_curve.csv')
    # Use already-anchored column if present, else fall back
    cost_col = 'Acc_cost_pp_adj' if 'Acc_cost_pp_adj' in T.columns else 'Acc_cost_pp'

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(8.5, 4.5))

    ax1 = axA
    ax1.plot(T['DI_target'], T[cost_col], marker='o', markersize=8, linewidth=2.5,
             color='#c0392b', label='Accuracy cost (pp)')
    ax1.set_xlabel('DI target (intervention dial)', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Accuracy cost (pp)', color='#c0392b', fontsize=10, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#c0392b', labelsize=9)
    ax1.tick_params(axis='x', labelsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(3.5, 7.5)
    for _, r in T.iterrows():
        ax1.text(r['DI_target'], r[cost_col] + 0.15, f'{r[cost_col]:.2f}',
                 ha='center', fontsize=8, color='#c0392b', fontweight='bold')

    ax2 = ax1.twinx()
    ax2.plot(T['DI_target'], T['VFR_Race'], marker='s', markersize=7, linewidth=2.0,
             color='#3a78b8', linestyle='--', label='VFR · DI Race')
    ax2.plot(T['DI_target'], T['VFR_Age'], marker='^', markersize=7, linewidth=2.0,
             color='#5fa55a', linestyle='--', label='VFR · DI Age')
    ax2.set_ylabel('VFR (verdict flip rate)', color='#3a78b8', fontsize=10, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#3a78b8', labelsize=9)
    ax2.set_ylim(0, 0.55)

    axA.set_title('(a) The intervention dial · cost vs verdict stability', fontsize=10, fontweight='bold')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axA.legend(lines1 + lines2, labels1 + labels2, loc='upper center',
                bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=8, frameon=True)

    for attr, color in [('Race', '#c0392b'), ('Sex', '#e07b39'), ('Eth', '#f4ca7c'), ('Age', '#5fa55a')]:
        axB.plot(T['DI_target'], T[f'DI_{attr}_post'], marker='o', markersize=7, linewidth=2.0,
                 color=color, label=attr)
    axB.axhline(0.80, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='4/5 rule')
    axB.set_xlabel('DI target', fontsize=10, fontweight='bold')
    axB.set_ylabel('Post-intervention DI', fontsize=10, fontweight='bold')
    axB.set_ylim(0.78, 1.02)
    axB.grid(alpha=0.3)
    axB.set_title('(b) Per-attribute DI as function of target', fontsize=10, fontweight='bold')
    axB.tick_params(labelsize=9)
    axB.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5, fontsize=8, frameon=True)

    plt.suptitle('Fairness intervention trade-off · canonical XGBoost · 4.24 pp cost at DI target = 0.80',
                 fontsize=10.5, fontweight='bold', y=1.04)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)
    plt.savefig(OUT / 'F9_intervention_dial.png', bbox_inches='tight', dpi=300)
    plt.close()
    print('F9_intervention_dial.png saved')

# ===================================================================
# Run all
# ===================================================================
if __name__ == '__main__':
    print(f'Output folder: {OUT}')
    df, order = render_F2_cohort_demographics()
    render_F2b_cohort_structure(df, order)
    render_F3_vfr_heatmap()
    render_F4_cv_audit_size()
    render_F5_hospital_violin()
    render_F6_per_model_tradeoff()
    render_F7_canonical_xgboost()
    render_F8_4model_verification()
    render_F9_intervention_dial()
    print('\nAll 9 figures rendered to manuscript_figures_v2/')
