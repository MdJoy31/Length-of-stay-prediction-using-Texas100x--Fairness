"""Insert Section 15 cells into the CIKM notebook."""
import json, uuid

NB_PATH = 'CIKM_2026_LOS_Fairness.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

def make_cell(cell_type, source_str):
    """Create a notebook cell dict."""
    c = {
        "cell_type": cell_type,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": source_str.split('\n')
    }
    # Add newlines back (except for last line)
    c["source"] = [line + '\n' for line in c["source"][:-1]] + [c["source"][-1]]
    if cell_type == "code":
        c["execution_count"] = None
        c["outputs"] = []
    return c

# ─── CELL 1: Markdown header ─────────────────────────────────────────────────
md_header = """---
## 15. Hospital-Subset Comprehensive Fairness Analysis

### Research Question
How do accuracy, all 7 fairness metrics, and verdict stability (VFR) change as we scale training data from 1 hospital to the full 441-hospital dataset?

This section provides:
- **11 hospital subsets**: 1, 2, 5, 10, 20, 50, 100, 150, 200, 300, and all 441 hospitals
- **All 7 fairness metrics** (DI, SPD, EOPP, EOD, TI, PP, CAL) × 4 protected attributes at each scale
- **VFR stability** for all 7 metrics × 4 subgroups via K=15 bootstrap
- **Lambda effect table**: Accuracy & all 7 fairness metrics at 10 reweighing strengths
- **Comparison line graphs**: Visual overlay of accuracy vs fairness across hospital scales and λ values"""

# ─── CELL 2: Hospital-subset comprehensive analysis ──────────────────────────
code_hospital = r'''# ──────────────────────────────────────────────────────────────
# Cell S15a · Hospital-Subset: All 7 Metrics × 4 Attributes + VFR
# ──────────────────────────────────────────────────────────────
import warnings; warnings.filterwarnings('ignore')

METRIC_KEYS_ALL = list(FairnessCalculator.THRESHOLDS.keys())
ATTRS_ALL = ['RACE', 'SEX', 'ETHNICITY', 'AGE_GROUP']
K_BOOT = 15   # bootstrap resamples for VFR
N_BOOT = 10000

unique_hosp = np.unique(hospital_ids_train)
n_total_hosp = len(unique_hosp)
hosp_scales = [1, 2, 5, 10, 20, 50, 100, 150, 200, 300, n_total_hosp]
hosp_scales = [h for h in hosp_scales if h <= n_total_hosp]

np.random.seed(RANDOM_STATE)
print(f"Hospital-Subset Comprehensive Analysis")
print(f"  Scales: {hosp_scales}")
print(f"  VFR: K={K_BOOT} bootstrap, N={N_BOOT:,}")
print(f"  Total hospitals available: {n_total_hosp}")
print()

hosp_rows = []
for n_h in hosp_scales:
    t0 = time.time()
    # Select top-N hospitals by patient count
    if n_h == n_total_hosp:
        sel = unique_hosp
    else:
        hc = pd.Series(hospital_ids_train).value_counts()
        sel = hc.nlargest(n_h).index.values

    mask = np.isin(hospital_ids_train, sel)
    X_s, y_s = X_train[mask], y_train[mask]
    if len(X_s) < 50 or len(set(y_s)) < 2:
        continue

    # Train LightGBM on subset
    mdl = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
        max_depth=8, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
    mdl.fit(X_s, y_s)
    yp = mdl.predict(X_test)
    ypr = mdl.predict_proba(X_test)[:, 1]

    row = {'N_Hospitals': n_h, 'N_Train': int(mask.sum()),
           'Accuracy': accuracy_score(y_test, yp),
           'AUC': roc_auc_score(y_test, ypr),
           'F1': f1_score(y_test, yp)}

    # All 7 fairness metrics × 4 attributes
    for attr in ATTRS_ALL:
        fc = FairnessCalculator(y_test, yp, ypr, protected_attrs[attr])
        mc, vc, _ = fc.compute_all()
        for mk in METRIC_KEYS_ALL:
            row[f'{mk}_{attr}'] = mc[mk]
            row[f'V_{mk}_{attr}'] = 1 if vc[mk] else 0
        row[f'N_Fair_{attr}'] = sum(vc.values())

    # VFR — K bootstrap resamples
    vfr_v = {f'{mk}_{attr}': [] for mk in METRIC_KEYS_ALL for attr in ATTRS_ALL}
    for k in range(K_BOOT):
        idx = np.random.choice(len(X_test), size=min(N_BOOT, len(X_test)), replace=False)
        for attr in ATTRS_ALL:
            fc_b = FairnessCalculator(y_test[idx], yp[idx], ypr[idx], protected_attrs[attr][idx])
            mc_b, vc_b, _ = fc_b.compute_all()
            for mk in METRIC_KEYS_ALL:
                vfr_v[f'{mk}_{attr}'].append(1 if vc_b[mk] else 0)

    for attr in ATTRS_ALL:
        for mk in METRIC_KEYS_ALL:
            n_fair = sum(vfr_v[f'{mk}_{attr}'])
            row[f'VFR_{mk}_{attr}'] = min(n_fair, K_BOOT - n_fair) / K_BOOT

    hosp_rows.append(row)
    elapsed = time.time() - t0
    print(f"  {n_h:>4d} hosp → N={row['N_Train']:>7,}  Acc={row['Accuracy']:.4f}  "
          f"Fair: R={row['N_Fair_RACE']}/7 S={row['N_Fair_SEX']}/7 "
          f"E={row['N_Fair_ETHNICITY']}/7 A={row['N_Fair_AGE_GROUP']}/7  ({elapsed:.1f}s)")

hosp_comp_df = pd.DataFrame(hosp_rows)
hosp_comp_df.to_csv(f'{TABLES_DIR}/Table11_Hospital_Subset_Comprehensive.csv', index=False)

# ─── Display per-attribute tables ─────────────────────────────────────────────
display(HTML("<h3>Table 11: Hospital-Subset — All 7 Metrics + VFR (K=15)</h3>"))
for attr in ATTRS_ALL:
    display(HTML(f"<h4>{attr}</h4>"))
    cols = ['N_Hospitals', 'N_Train', 'Accuracy']
    cols += [f'{mk}_{attr}' for mk in METRIC_KEYS_ALL]
    cols += [f'VFR_{mk}_{attr}' for mk in METRIC_KEYS_ALL]
    cols += [f'N_Fair_{attr}']

    ddf = hosp_comp_df[cols].copy()
    rn = {f'{mk}_{attr}': mk for mk in METRIC_KEYS_ALL}
    rn.update({f'VFR_{mk}_{attr}': f'VFR_{mk}' for mk in METRIC_KEYS_ALL})
    rn[f'N_Fair_{attr}'] = 'Fair/7'
    ddf = ddf.rename(columns=rn)

    fmt = {'N_Train': '{:,}', 'Accuracy': '{:.4f}', 'Fair/7': '{:.0f}'}
    for mk in METRIC_KEYS_ALL:
        fmt[mk] = '{:.3f}'
        fmt[f'VFR_{mk}'] = '{:.2f}'

    styled = ddf.style.format(fmt)
    styled = styled.background_gradient(subset=['DI'], cmap='RdYlGn', vmin=0, vmax=1)
    vfr_cols = [f'VFR_{mk}' for mk in METRIC_KEYS_ALL]
    styled = styled.background_gradient(subset=vfr_cols, cmap='RdYlGn_r', vmin=0, vmax=0.5)
    display(styled)

print(f"\n✓ Table 11 saved ({len(hosp_comp_df)} subsets × {len(hosp_comp_df.columns)} columns)")'''

# ─── CELL 3: Lambda effect comprehensive table ───────────────────────────────
code_lambda = r'''# ──────────────────────────────────────────────────────────────
# Cell S15b · Lambda Effect — All 7 Metrics × 4 Attributes
# ──────────────────────────────────────────────────────────────
METRIC_KEYS_ALL = list(FairnessCalculator.THRESHOLDS.keys())
ATTRS_ALL = ['RACE', 'SEX', 'ETHNICITY', 'AGE_GROUP']

# Map model_probs keys → lambda values
key_to_lambda = {'Standard': 0.0}
for lam in [0.5, 1.0, 3.0, 5.0, 10.0, 15.0, 30.0, 50.0, 100.0]:
    key_to_lambda[f'Reweigh_{lam:.0f}'] = lam

print("Lambda Effect: All 7 Fairness Metrics × 4 Attributes")
print("=" * 80)

lambda_rows = []
for key, lam_val in sorted(key_to_lambda.items(), key=lambda x: x[1]):
    if key not in model_probs:
        continue
    ypr = model_probs[key]
    yp = (ypr >= 0.5).astype(int)

    row = {'Lambda': lam_val,
           'Accuracy': accuracy_score(y_test, yp),
           'AUC': roc_auc_score(y_test, ypr),
           'F1': f1_score(y_test, yp)}

    total_fair = 0
    for attr in ATTRS_ALL:
        fc = FairnessCalculator(y_test, yp, ypr, protected_attrs[attr])
        mc, vc, _ = fc.compute_all()
        for mk in METRIC_KEYS_ALL:
            row[f'{mk}_{attr}'] = mc[mk]
        n_fair = sum(vc.values())
        row[f'N_Fair_{attr}'] = n_fair
        total_fair += n_fair

    row['Total_Fair'] = total_fair
    row['Acc_Drop'] = accuracy_score(y_test, best_y_pred) - row['Accuracy']
    lambda_rows.append(row)
    print(f"  λ={lam_val:>5.1f}  Acc={row['Accuracy']:.4f}  "
          f"Fair: R={row['N_Fair_RACE']}/7 S={row['N_Fair_SEX']}/7 "
          f"E={row['N_Fair_ETHNICITY']}/7 A={row['N_Fair_AGE_GROUP']}/7  Total={total_fair}/28")

lambda_comp_df = pd.DataFrame(lambda_rows)
lambda_comp_df.to_csv(f'{TABLES_DIR}/Table12_Lambda_Comprehensive.csv', index=False)

# ─── Display per-attribute tables ─────────────────────────────────────────────
display(HTML("<h3>Table 12: Lambda Effect — All 7 Metrics × 4 Attributes</h3>"))
for attr in ATTRS_ALL:
    display(HTML(f"<h4>{attr}</h4>"))
    cols = ['Lambda', 'Accuracy', 'AUC']
    cols += [f'{mk}_{attr}' for mk in METRIC_KEYS_ALL]
    cols += [f'N_Fair_{attr}']

    ddf = lambda_comp_df[cols].copy()
    rn = {f'{mk}_{attr}': mk for mk in METRIC_KEYS_ALL}
    rn[f'N_Fair_{attr}'] = 'Fair/7'
    ddf = ddf.rename(columns=rn)

    fmt = {'Lambda': '{:.1f}', 'Accuracy': '{:.4f}', 'AUC': '{:.4f}', 'Fair/7': '{:.0f}'}
    for mk in METRIC_KEYS_ALL:
        fmt[mk] = '{:.3f}'

    styled = ddf.style.format(fmt)
    styled = styled.background_gradient(subset=['DI'], cmap='RdYlGn', vmin=0, vmax=1)
    styled = styled.background_gradient(subset=['Accuracy'], cmap='YlGn')
    display(styled)

# ─── Summary table ───────────────────────────────────────────────────────────
display(HTML("<h4>Summary: Total Fair Metrics vs Lambda</h4>"))
sc = ['Lambda', 'Accuracy', 'Acc_Drop', 'Total_Fair'] + [f'N_Fair_{a}' for a in ATTRS_ALL]
ds = lambda_comp_df[sc].rename(columns={f'N_Fair_{a}': f'Fair_{a}' for a in ATTRS_ALL})
display(ds.style.format({
    'Lambda': '{:.1f}', 'Accuracy': '{:.4f}', 'Acc_Drop': '{:+.4f}', 'Total_Fair': '{:.0f}',
    **{f'Fair_{a}': '{:.0f}' for a in ATTRS_ALL}
}).background_gradient(subset=['Total_Fair'], cmap='YlGn'))

print(f"\n✓ Table 12 saved ({len(lambda_comp_df)} lambda values × {len(lambda_comp_df.columns)} columns)")'''

# ─── CELL 4: Comparison line graphs ──────────────────────────────────────────
code_graphs = r'''# ──────────────────────────────────────────────────────────────
# Cell S15c · Comparison Line Graphs — Hospital Subset & Lambda
# ──────────────────────────────────────────────────────────────
METRIC_KEYS_ALL = list(FairnessCalculator.THRESHOLDS.keys())
ATTRS_ALL = ['RACE', 'SEX', 'ETHNICITY', 'AGE_GROUP']
colors_fm = plt.cm.tab10(np.linspace(0, 0.7, 7))
markers_fm = ['s', 'D', '^', 'v', 'p', 'h', '*']

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE A: Hospital-Subset — Accuracy vs 7 Fairness Metrics (4 panels)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(22, 16))
for i, attr in enumerate(ATTRS_ALL):
    ax = axes[i // 2, i % 2]
    ax2 = ax.twinx()
    x = hosp_comp_df['N_Hospitals'].values

    # Accuracy (left axis, black)
    l1, = ax.plot(x, hosp_comp_df['Accuracy'], 'ko-', lw=2.5, ms=8, label='Accuracy', zorder=10)
    ax.fill_between(x, hosp_comp_df['Accuracy'], alpha=0.08, color='black')
    ax.set_ylabel('Accuracy', fontsize=11, color='black')
    ax.set_ylim(0.50, 0.90)
    ax.tick_params(axis='y', labelcolor='black')

    # 7 fairness metrics (right axis)
    lines = [l1]
    for j, mk in enumerate(METRIC_KEYS_ALL):
        ls = '-' if mk in ['DI', 'SPD', 'EOPP'] else '--'
        l, = ax2.plot(x, hosp_comp_df[f'{mk}_{attr}'], marker=markers_fm[j], ls=ls,
                     color=colors_fm[j], lw=1.5, ms=6, label=mk, alpha=0.85)
        lines.append(l)

    # Threshold reference lines
    ax2.axhline(0.80, color='red', ls=':', lw=1, alpha=0.5, label='DI≥0.80')
    ax2.axhline(0.10, color='orange', ls=':', lw=1, alpha=0.3, label='SPD/TI/PP/CAL≤0.10')
    ax2.axhline(0.20, color='blue', ls=':', lw=1, alpha=0.3, label='EOPP/EOD≤0.20')
    ax2.set_ylabel('Metric Value', fontsize=11)
    ax2.set_ylim(0, 1.05)

    ax.set_xscale('log')
    ax.set_xlabel('Number of Hospitals (log scale)')
    ax.set_title(f'{attr}', fontweight='bold', fontsize=13)
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='lower right', fontsize=7, ncol=2,
              framealpha=0.9, edgecolor='gray')
    ax.grid(alpha=0.3)

plt.suptitle('Hospital-Subset: Accuracy & 7 Fairness Metrics Across Scales',
             fontsize=15, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
save_fig('cikm_hosp_subset_acc_vs_fairness')
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE B: VFR Stability Heatmap per Hospital Subset
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(26, 8))
for i, attr in enumerate(ATTRS_ALL):
    ax = axes[i]
    vfr_data = hosp_comp_df[[f'VFR_{mk}_{attr}' for mk in METRIC_KEYS_ALL]].values
    sns.heatmap(vfr_data, ax=ax, xticklabels=METRIC_KEYS_ALL,
                yticklabels=hosp_comp_df['N_Hospitals'].values,
                cmap='RdYlGn_r', vmin=0, vmax=0.5, annot=True, fmt='.2f',
                linewidths=0.5, linecolor='white', cbar_kws={'label': 'VFR'})
    ax.set_xlabel('Metric'); ax.set_ylabel('N Hospitals')
    ax.set_title(f'{attr}', fontweight='bold', fontsize=12)

plt.suptitle('VFR Stability Across Hospital Subsets (0 = Stable, 0.5 = Unstable)',
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
save_fig('cikm_hosp_subset_vfr_heatmap')
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE C: Lambda Effect — Accuracy vs 7 Fairness Metrics (4 panels)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(22, 16))
for i, attr in enumerate(ATTRS_ALL):
    ax = axes[i // 2, i % 2]
    ax2 = ax.twinx()
    x = lambda_comp_df['Lambda'].values

    l1, = ax.plot(x, lambda_comp_df['Accuracy'], 'ko-', lw=2.5, ms=8, label='Accuracy', zorder=10)
    ax.fill_between(x, lambda_comp_df['Accuracy'], alpha=0.08, color='black')
    ax.set_ylabel('Accuracy', fontsize=11, color='black')
    ax.set_ylim(0.40, 0.90)
    ax.tick_params(axis='y', labelcolor='black')

    lines = [l1]
    for j, mk in enumerate(METRIC_KEYS_ALL):
        ls = '-' if mk in ['DI', 'SPD', 'EOPP'] else '--'
        l, = ax2.plot(x, lambda_comp_df[f'{mk}_{attr}'], marker=markers_fm[j], ls=ls,
                     color=colors_fm[j], lw=1.5, ms=6, label=mk, alpha=0.85)
        lines.append(l)

    ax2.axhline(0.80, color='red', ls=':', lw=1, alpha=0.5)
    ax2.axhline(0.10, color='orange', ls=':', lw=1, alpha=0.3)
    ax2.axhline(0.20, color='blue', ls=':', lw=1, alpha=0.3)
    ax2.set_ylabel('Metric Value', fontsize=11)
    ax2.set_ylim(0, 1.05)

    ax.set_xlabel('λ (Reweighing Strength)')
    ax.set_title(f'{attr}', fontweight='bold', fontsize=13)
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='center right', fontsize=7, ncol=2,
              framealpha=0.9, edgecolor='gray')
    ax.grid(alpha=0.3)

plt.suptitle('Lambda Effect: Accuracy & 7 Fairness Metrics vs Reweighing Strength',
             fontsize=15, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
save_fig('cikm_lambda_acc_vs_fairness')
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE D: Combined DI + Accuracy Summary (Hospital + Lambda side by side)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(22, 8))

# (a) Hospital-Subset DI + Accuracy
ax = axes[0]
ax2 = ax.twinx()
x1 = hosp_comp_df['N_Hospitals'].values
l0, = ax.plot(x1, hosp_comp_df['Accuracy'], 'ko-', lw=2.5, ms=8, label='Accuracy', zorder=10)
lines_a = [l0]
for i, attr in enumerate(ATTRS_ALL):
    l, = ax2.plot(x1, hosp_comp_df[f'DI_{attr}'], 'o-', color=PALETTE[i], lw=2, ms=7, label=f'DI_{attr}')
    lines_a.append(l)
ax2.axhline(0.80, color='red', ls='--', lw=1.5, alpha=0.6, label='DI=0.80')
ax.set_xscale('log'); ax.set_xlabel('Number of Hospitals (log)')
ax.set_ylabel('Accuracy'); ax.set_ylim(0.50, 0.90)
ax2.set_ylabel('Disparate Impact'); ax2.set_ylim(0, 1.05)
ax.set_title('(a) Hospital Subsets: Accuracy & DI', fontweight='bold', fontsize=13)
ax.legend(lines_a, [l.get_label() for l in lines_a], fontsize=8, loc='lower right')
ax.grid(alpha=0.3)

# (b) Lambda DI + Accuracy
ax = axes[1]
ax2 = ax.twinx()
x2 = lambda_comp_df['Lambda'].values
l0, = ax.plot(x2, lambda_comp_df['Accuracy'], 'ko-', lw=2.5, ms=8, label='Accuracy', zorder=10)
lines_b = [l0]
for i, attr in enumerate(ATTRS_ALL):
    l, = ax2.plot(x2, lambda_comp_df[f'DI_{attr}'], 'o-', color=PALETTE[i], lw=2, ms=7, label=f'DI_{attr}')
    lines_b.append(l)
ax2.axhline(0.80, color='red', ls='--', lw=1.5, alpha=0.6, label='DI=0.80')
ax.set_xlabel('λ (Reweighing Strength)')
ax.set_ylabel('Accuracy'); ax.set_ylim(0.40, 0.90)
ax2.set_ylabel('Disparate Impact'); ax2.set_ylim(0, 1.05)
ax.set_title('(b) Lambda Effect: Accuracy & DI', fontweight='bold', fontsize=13)
ax.legend(lines_b, [l.get_label() for l in lines_b], fontsize=8, loc='center right')
ax.grid(alpha=0.3)

plt.suptitle('Accuracy–Fairness Trade-off: Hospital Scale vs Reweighing Strength',
             fontsize=15, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
save_fig('cikm_combined_acc_fairness_tradeoff')
plt.show()

print("✓ 4 comparison figures saved")'''

# ─── Insert cells ────────────────────────────────────────────────────────────
new_cells = [
    make_cell("markdown", md_header),
    make_cell("code", code_hospital),
    make_cell("code", code_lambda),
    make_cell("code", code_graphs),
]

# Find insertion point: after last cell before current Section 15 display
# Cell 43 (0-indexed) = Related Work markdown (ends with references)
# Cell 44 (0-indexed) = "---\n## 15. All Generated Figures..."
INSERT_AT = 44  # Insert before current cell 44

for i, cell in enumerate(new_cells):
    nb['cells'].insert(INSERT_AT + i, cell)

# Update old Section 15 to Section 16
for cell in nb['cells'][INSERT_AT + len(new_cells):]:
    if cell['cell_type'] == 'markdown':
        src = ''.join(cell['source'])
        if '## 15. All Generated' in src:
            cell['source'] = [s.replace('## 15.', '## 16.') for s in cell['source']]
            break

# Update display code cell comment
for cell in nb['cells'][INSERT_AT + len(new_cells):]:
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if '15. Display All Generated' in src:
            cell['source'] = [s.replace('15. Display All', '16. Display All') for s in cell['source']]
            break

# Update table of contents in Cell 1
for cell in nb['cells'][:3]:
    src = ''.join(cell['source'])
    if '| 14 |' in src and '| 15 |' not in src:
        # Doesn't have section 15 in TOC yet
        pass
    if '| 14 |' in src:
        new_source = []
        for line in cell['source']:
            new_source.append(line)
            if '| 14 |' in line and 'Related Work' in line:
                new_source.append('| 15 | Hospital-Subset Comprehensive Fairness Analysis |\n')
                new_source.append('| 16 | All Generated Figures & Results Tables |\n')
            elif '| 15 |' in line:
                continue  # skip old section 15 references
        cell['source'] = new_source
        break

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Done! Inserted {len(new_cells)} cells at position {INSERT_AT}")
print(f"Total cells: {len(nb['cells'])}")
