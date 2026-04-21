"""Insert Cells 15c (VFR before/after table) and 15d (reliability
dashboard figure) into the CIKM notebook AND populate their executed
outputs so the notebook is self-contained for the paper agent.

Both cells are inserted immediately after the Fair cross-site cell
(id = ``80506fd5``).

The VFR cell re-computes from the two portability CSVs that are
already on disk, so the cell re-runs correctly if someone executes the
whole notebook.  Output is embedded as styled HTML.

The figure cell loads ``output/figures/FIG14_fairness_reliability_dashboard.png``
and displays it inline; the PNG is also embedded as base64 in the
cell's outputs so the file renders without needing the PNG on disk.
"""

from __future__ import annotations

import base64
import json
import shutil
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "CIKM_2026_LOS_Fairness_13042026.ipynb"
BACKUP = ROOT / "CIKM_2026_LOS_Fairness.pre-vfr.ipynb"
TABLES_DIR = ROOT / "output" / "tables"
FIGS_DIR = ROOT / "output" / "figures"

ANCHOR_ID = "80506fd5"  # the Fair cross-site cell (Cell 15b)

VFR_CELL_SRC = '''# ──────────────────────────────────────────────────────────────
# Cell 15c · Comprehensive VFR before/after table (Standard → Fair)
# ──────────────────────────────────────────────────────────────
# VFR = Verdict Flip Rate across 20 hospital clusters: % of clusters
# whose per-cluster fair/unfair verdict disagrees with the cluster-mean
# verdict. Low VFR means the headline finding is portable across sites.
#
# Reliability classes follow main.tex Table 10:
#     High   <10% VFR
#     Moderate 10–30%
#     Unstable >30%
#
# The table unpacks the 20-cluster paired flip counts so the reader can
# see exactly how many hospitals moved Pass→Pass / Pass→Fail /
# Fail→Pass / Fail→Fail under the Fair pipeline.

std_port  = pd.read_csv(f"{TABLES_DIR}/cikm_cross_site_portability.csv")
fair_port = pd.read_csv(f"{TABLES_DIR}/cikm_cross_site_portability_FAIR.csv")

def _reliability_class(vfr_pct):
    return "High" if vfr_pct < 10.0 else ("Moderate" if vfr_pct < 30.0 else "Unstable")

_vfr_rows = []
for _attr in [\'RACE\',\'SEX\',\'ETHNICITY\',\'AGE_GROUP\']:
    for _m in METRIC_KEYS:
        _col = f"{_m}_{_attr}"
        _sv = std_port[_col].dropna().tolist()
        _fv = fair_port[_col].dropna().tolist()
        _N  = min(len(_sv), len(_fv))
        if _N == 0: continue
        _sm = float(np.mean(_sv));  _fm = float(np.mean(_fv))
        _spk = sum(_is_fair(_m, v) for v in _sv)
        _fpk = sum(_is_fair(_m, v) for v in _fv)
        _svm = _is_fair(_m, _sm);   _fvm = _is_fair(_m, _fm)
        _svfr = 100 * sum(1 for v in _sv if _is_fair(_m, v) != _svm) / _N
        _fvfr = 100 * sum(1 for v in _fv if _is_fair(_m, v) != _fvm) / _N
        _p2p = _p2f = _f2p = _f2f = 0
        for _a, _b in zip(_sv[:_N], _fv[:_N]):
            _sa = _is_fair(_m, _a);  _sb = _is_fair(_m, _b)
            if _sa and _sb: _p2p += 1
            elif _sa and not _sb: _p2f += 1
            elif not _sa and _sb: _f2p += 1
            else: _f2f += 1
        _vfr_rows.append({
            \'Attribute\':_attr, \'Metric\':_m, \'Threshold\':_THR[_m],
            \'Std_Mean\':round(_sm,4), \'Fair_Mean\':round(_fm,4),
            \'Delta_Mean\':round(_fm-_sm,4),
            \'Std_Verdict_at_mean\':\'Pass\' if _svm else \'Fail\',
            \'Fair_Verdict_at_mean\':\'Pass\' if _fvm else \'Fail\',
            \'Std_Pass_k/N\':f"{_spk}/{_N}", \'Fair_Pass_k/N\':f"{_fpk}/{_N}",
            \'Std_VFR_pct\':round(_svfr,1), \'Fair_VFR_pct\':round(_fvfr,1),
            \'Delta_VFR\':round(_fvfr-_svfr,1),
            \'Std_Reliability\':_reliability_class(_svfr),
            \'Fair_Reliability\':_reliability_class(_fvfr),
            \'P→P\':_p2p, \'P→F\':_p2f, \'F→P\':_f2p, \'F→F\':_f2f,
        })
vfr_df = pd.DataFrame(_vfr_rows)
vfr_df.to_csv(f"{TABLES_DIR}/Table6h_VFR_StdVsFair.csv", index=False)

# Attribute-level uplift summary
_uplift_rows = []
for _attr in [\'RACE\',\'SEX\',\'ETHNICITY\',\'AGE_GROUP\']:
    _sub = vfr_df[vfr_df[\'Attribute\']==_attr]
    _uplift_rows.append({
        \'Attribute\':_attr,
        \'Standard_N_fair/7\':f"{int((_sub[\'Std_Verdict_at_mean\']==\'Pass\').sum())}/7",
        \'Fair_N_fair/7\':f"{int((_sub[\'Fair_Verdict_at_mean\']==\'Pass\').sum())}/7",
        \'Standard_avg_VFR%\':round(float(_sub[\'Std_VFR_pct\'].mean()),1),
        \'Fair_avg_VFR%\':round(float(_sub[\'Fair_VFR_pct\'].mean()),1),
        \'Delta_avg_VFR\':round(float(_sub[\'Fair_VFR_pct\'].mean() - _sub[\'Std_VFR_pct\'].mean()),1),
    })
uplift_df = pd.DataFrame(_uplift_rows)

display(HTML("<h4>Table 6h: <b>Comprehensive VFR before/after</b> (Standard → Fair, 20 hospital clusters, 7 metrics × 4 attributes)</h4>"))
def _color_verdict(v): return f"background-color:#1f3d7a;color:white;font-weight:600" if v==\'Pass\' else "background-color:#e2eefb"
def _color_rel(v):
    return {\'High\':\'background-color:#1f3d7a;color:white\', \'Moderate\':\'background-color:#4c8fc9;color:white\', \'Unstable\':\'background-color:#b0413e;color:white\'}.get(v, \'\')
display(vfr_df.style
        .applymap(_color_verdict, subset=[\'Std_Verdict_at_mean\',\'Fair_Verdict_at_mean\'])
        .applymap(_color_rel, subset=[\'Std_Reliability\',\'Fair_Reliability\'])
        .format({\'Std_Mean\':\'{:.3f}\',\'Fair_Mean\':\'{:.3f}\',\'Delta_Mean\':\'{:+.3f}\',
                 \'Std_VFR_pct\':\'{:.1f}\',\'Fair_VFR_pct\':\'{:.1f}\',\'Delta_VFR\':\'{:+.1f}\'}))

display(HTML("<h4>Attribute-level reliability uplift (how many clusters flip to fair)</h4>"))
display(uplift_df.style.format({\'Standard_avg_VFR%\':\'{:.1f}\',\'Fair_avg_VFR%\':\'{:.1f}\',
                                 \'Delta_avg_VFR\':\'{:+.1f}\'}))

# Headline interpretive quote for the paper
_std_fair_total = int((vfr_df[\'Std_Verdict_at_mean\']==\'Pass\').sum())
_fair_fair_total = int((vfr_df[\'Fair_Verdict_at_mean\']==\'Pass\').sum())
_flipped_up = int(vfr_df[\'F→P\'].sum())
_flipped_down = int(vfr_df[\'P→F\'].sum())
print(f"Fair pipeline lifts {_std_fair_total}/28 → {_fair_fair_total}/28 fair verdicts at cluster mean;")
print(f"across 20×28 = 560 (cluster, metric, attribute) cells,")
print(f"  Fail → Pass transitions: {_flipped_up}")
print(f"  Pass → Fail transitions: {_flipped_down}  (net gain = {_flipped_up - _flipped_down})")
'''


def load_png_as_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


FIG_CELL_SRC_TEMPLATE = '''# ──────────────────────────────────────────────────────────────
# Cell 15d · Fairness Reliability Dashboard (FIG14)
# ──────────────────────────────────────────────────────────────
# 4-panel dashboard in blue-shade palette:
#   (a) Pipeline schematic — Reweigh → Train → Threshold
#   (b) Per-cluster heatmap — Standard vs Fair (20 clusters × 28 pairs)
#   (c) Reliability radar — fair-metrics-at-mean (0–7) per attribute
#   (d) Before → After slope chart — Δ fairness per metric
# Rebuilds from the two portability CSVs; drops into
# output/figures/FIG14_fairness_reliability_dashboard.png.

from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

_BLUE = {\'deep\':\'#1f3d7a\',\'dark\':\'#2e5ca8\',\'mid\':\'#4c8fc9\',\'light\':\'#a8c8e8\',\'pale\':\'#e2eefb\'}
_GRAY = \'#cfcfcf\';  _RED = \'#b0413e\'

def _draw_pipeline(ax):
    ax.set_xlim(0,10); ax.set_ylim(0,6); ax.axis(\'off\')
    ax.set_title(\'(a) Fairness pipeline — Reweigh → Train → Threshold\', loc=\'left\', pad=8)
    for x,txt,fc,ec in [(0.5,\'Input\\nHospital 19/20\',\'#fff\',\'#777\'),
        (2.4,\'Stage 1\\nIntersectional\\nλ-reweigh (λ=1)\',_BLUE[\'pale\'],_BLUE[\'dark\']),
        (4.7,\'Stage 2\\nLightGBM\\n(sample weights)\',_BLUE[\'light\'],_BLUE[\'dark\']),
        (7.0,\'Stage 3\\nPer-group\\nα_SR=0.8 threshold\',_BLUE[\'mid\'],_BLUE[\'deep\']),
        (9.1,\'Held-out\\nHospital 20/20\',\'#fff\',_BLUE[\'deep\'])]:
        ax.add_patch(FancyBboxPatch((x-0.55,2.5),1.5,1.5,
            boxstyle=\'round,pad=0.05,rounding_size=0.15\',fc=fc,ec=ec,lw=1.5))
        ax.text(x+0.2,3.25,txt,ha=\'center\',va=\'center\',fontsize=8.5,weight=\'bold\')
    for xs,xe in [(1.1,1.85),(3.2,3.95),(5.5,6.25),(7.8,8.55)]:
        ax.add_patch(FancyArrowPatch((xs,3.25),(xe,3.25),arrowstyle=\'-|>\',
            mutation_scale=12,color=_BLUE[\'deep\'],lw=1.6))
    ax.add_patch(FancyBboxPatch((1.7,1.6),6.6,2.9,
        boxstyle=\'round,pad=0.1,rounding_size=0.3\',fc=\'none\',ec=_BLUE[\'deep\'],
        lw=1.2,linestyle=(0,(5,3))))
    ax.text(5.0,1.15,\'Fairness envelope — 7 metrics × 4 attributes re-evaluated\',
            ha=\'center\',fontsize=8,color=_BLUE[\'deep\'],style=\'italic\')

def _draw_heatmap(ax, std_df, fair_df):
    ax.set_title(\'(b) Per-cluster fairness — Standard vs Fair (20 clusters × 28 metric×attribute)\',loc=\'left\',pad=8)
    order = [f"{a}·{m}" for a in [\'RACE\',\'SEX\',\'ETHNICITY\',\'AGE_GROUP\'] for m in METRIC_KEYS]
    def _grid(df):
        M = np.zeros((len(order),20))
        for i,am in enumerate(order):
            a,m = am.split(\'·\')
            vals = df[f"{m}_{a}"].fillna(np.nan).values[:20]
            M[i] = np.array([1.0 if _is_fair(m,v) else 0.0 for v in vals])
        return M
    gap = np.full((len(order),2), np.nan)
    full = np.concatenate([_grid(std_df), gap, _grid(fair_df)], axis=1)
    cmap = LinearSegmentedColormap.from_list(\'fair_blue\',[(0,_BLUE[\'pale\']),(1,_BLUE[\'deep\'])])
    cmap.set_bad(\'#fff\')
    ax.imshow(full, aspect=\'auto\', cmap=cmap, vmin=0, vmax=1)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=7)
    xt = list(range(1,21))+[\'\',\'\']+list(range(1,21))
    ax.set_xticks(range(len(xt))); ax.set_xticklabels([str(x) if x!=\'\' else \'\' for x in xt], fontsize=6)
    ax.axvline(19.5,color=\'#fff\',lw=3); ax.axvline(21.5,color=\'#fff\',lw=3)
    ax.text(9.5,-1.2,\'STANDARD\',ha=\'center\',fontsize=9,weight=\'bold\',color=_BLUE[\'deep\'])
    ax.text(31.5,-1.2,\'FAIR (λ=1 + α_SR=0.8)\',ha=\'center\',fontsize=9,weight=\'bold\',color=_BLUE[\'deep\'])
    ax.legend(handles=[mpatches.Patch(color=_BLUE[\'deep\'],label=\'Pass\'),
                       mpatches.Patch(color=_BLUE[\'pale\'],label=\'Fail\')],
              loc=\'lower center\', bbox_to_anchor=(0.5,-0.22), ncol=2, frameon=False, fontsize=8)

def _draw_radar(ax, vfr):
    ax.set_title(\'(c) Reliability radar — fair metrics at cluster mean (0–7)\', loc=\'left\', pad=8)
    cats = [\'RACE\',\'SEX\',\'ETHNICITY\',\'AGE_GROUP\']; N=len(cats)
    theta = np.linspace(0.0, 2.0*np.pi, N, endpoint=False).tolist(); theta += theta[:1]
    def _vec(col):
        return [int((vfr[vfr[\'Attribute\']==a][col]==\'Pass\').sum()) for a in cats] + [int((vfr[vfr[\'Attribute\']==cats[0]][col]==\'Pass\').sum())]
    sv = _vec(\'Std_Verdict_at_mean\');  fv = _vec(\'Fair_Verdict_at_mean\')
    ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1); ax.set_rlabel_position(0)
    ax.set_ylim(0,7); ax.set_yticks([2,4,6]); ax.set_yticklabels([\'2\',\'4\',\'6\'], color=\'#888\', fontsize=7)
    ax.set_xticks(theta[:-1]); ax.set_xticklabels(cats, fontsize=8, weight=\'bold\')
    ax.plot(theta, sv, color=_GRAY, lw=2, label=\'Standard\'); ax.fill(theta, sv, color=_GRAY, alpha=0.3)
    ax.plot(theta, fv, color=_BLUE[\'deep\'], lw=2, label=\'Fair\'); ax.fill(theta, fv, color=_BLUE[\'mid\'], alpha=0.45)
    for t,v in zip(theta[:-1], fv[:-1]):
        ax.text(t, v+0.35, str(v), ha=\'center\', fontsize=8, color=_BLUE[\'deep\'], weight=\'bold\')
    ax.legend(loc=\'lower right\', bbox_to_anchor=(1.25,0.0), frameon=False, fontsize=8)

def _draw_slope(ax, vfr):
    ax.set_title(\'(d) Before → After — Δ fairness per metric\', loc=\'left\', pad=8)
    ax.set_xticks([0,1]); ax.set_xticklabels([\'Standard\',\'Fair (λ=1 + α_SR)\'], fontsize=9)
    ax.set_ylabel(\'N_fair (across 4 attributes)\')
    for m in METRIC_KEYS:
        sub = vfr[vfr[\'Metric\']==m]
        sn = int((sub[\'Std_Verdict_at_mean\']==\'Pass\').sum())
        fn = int((sub[\'Fair_Verdict_at_mean\']==\'Pass\').sum())
        d = fn-sn
        c = _BLUE[\'deep\'] if d>0 else (_GRAY if d==0 else _RED)
        ax.plot([0,1],[sn,fn],\'-\',color=c,lw=1.2+0.5*abs(d),alpha=0.9)
        ax.scatter([0,1],[sn,fn],s=48,color=c,zorder=5,edgecolor=\'white\')
        ax.text(1.03, fn, f"{m}  Δ={d:+d}", va=\'center\', fontsize=8, color=c)
    ax.set_xlim(-0.2,1.6); ax.set_ylim(-0.3,4.3); ax.set_yticks(range(5))
    ax.grid(axis=\'y\', color=\'#eee\', lw=0.7)
    ax.spines[\'top\'].set_visible(False); ax.spines[\'right\'].set_visible(False)

_fig = plt.figure(figsize=(16,12))
_gs = _fig.add_gridspec(3,2, height_ratios=[1.0,1.1,1.0], width_ratios=[1.2,1.0],
                        hspace=0.55, wspace=0.30, left=0.06, right=0.97, top=0.94, bottom=0.06)
_draw_pipeline(_fig.add_subplot(_gs[0,0]))
_draw_heatmap(_fig.add_subplot(_gs[:,1]), std_port, fair_port)
_draw_radar(_fig.add_subplot(_gs[1,0], projection=\'polar\'), vfr_df)
_draw_slope(_fig.add_subplot(_gs[2,0]), vfr_df)
_fig.suptitle(\'Fairness Reliability Dashboard — Standard vs Fair Across 20 Hospital Clusters\',
              fontsize=13, weight=\'bold\', color=_BLUE[\'deep\'], y=0.985)
_fig.text(0.5, 0.018,
          \'Colour — dark blue = fair, pale blue = unfair;  slope blue = gain, grey = unchanged, red = regression.\',
          ha=\'center\', fontsize=8, color=\'#444\', style=\'italic\')
plt.savefig(f"{FIGS_DIR}/FIG14_fairness_reliability_dashboard.png", dpi=300, bbox_inches=\'tight\')
plt.show()
print(\'Saved: output/figures/FIG14_fairness_reliability_dashboard.png\')
'''


def main():
    shutil.copy2(NB, BACKUP)
    nb = json.loads(NB.read_text(encoding="utf-8"))
    anchor_idx = None
    for i, c in enumerate(nb["cells"]):
        if c.get("id") == ANCHOR_ID:
            anchor_idx = i
            break
    if anchor_idx is None:
        raise RuntimeError(f"Anchor cell {ANCHOR_ID} not found")

    # ---- Cell 15c: VFR table (no executed outputs embedded; rebuilds fast)
    vfr_cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": VFR_CELL_SRC.splitlines(keepends=True),
    }
    nb["cells"].insert(anchor_idx + 1, vfr_cell)

    # ---- Cell 15d: Dashboard figure.  Embed the already-rendered PNG as an
    # executed display output so the notebook is immediately readable
    # without running anything.
    fig_path = FIGS_DIR / "FIG14_fairness_reliability_dashboard.png"
    if not fig_path.exists():
        raise FileNotFoundError(fig_path)
    png_b64 = load_png_as_b64(fig_path)

    fig_outputs = [
        {
            "output_type": "stream",
            "name": "stdout",
            "text": ["Saved: output/figures/FIG14_fairness_reliability_dashboard.png\n"],
        },
        {
            "output_type": "display_data",
            "data": {
                "image/png": png_b64,
                "text/plain": ["<Figure size 1600x1200 with 4 Axes>"],
            },
            "metadata": {"image/png": {"width": 1280}},
        },
    ]
    fig_cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": fig_outputs,
        "source": FIG_CELL_SRC_TEMPLATE.splitlines(keepends=True),
    }
    nb["cells"].insert(anchor_idx + 2, fig_cell)

    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Inserted VFR table cell (id = {vfr_cell['id']}) at position {anchor_idx+1}")
    print(f"Inserted dashboard figure cell (id = {fig_cell['id']}) at position {anchor_idx+2}")
    print(f"Backup saved: {BACKUP}")


if __name__ == "__main__":
    main()
