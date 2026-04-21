"""Build the big Hospital-Sweep × VFR comparison table and 5 fancy
blue-shade diagrams, plus a fixed Fleiss-style concordance heatmap.

Inputs on disk:
- output/tables/cikm_cross_site_portability.csv       (Standard, 20 folds)
- output/tables/cikm_cross_site_portability_FAIR.csv  (Fair,     20 folds)
- output/tables/cikm_cross_hospital_scale.csv         (Standard sweep, 8 levels)
- output/tables/cikm_cross_hospital_scale_FAIR.csv    (Fair sweep,     8 levels)

Outputs (all blue-shade palette, 300 dpi):
- Tables:
    Table6i_HospitalSweep_StdVsFair.csv
    Table6j_PerCluster_StdVsFair.csv
    Table6k_Subset_Stability_Summary.csv
    Table6l_Metric_Concordance.csv           (Gwet AC1 replacing Fleiss)
- Figures:
    FIG15_subset_concordance_grid.png        (20 clusters × 28 metric-attr, diff-colored)
    FIG16_hospital_scaling_dual.png          (N_hosp vs Acc + N_fair, Std vs Fair)
    FIG17_vfr_distribution_violin.png
    FIG18_reliability_sankey.png
    FIG19_metric_coflip_network.png
    FIG20_concordance_heatmap.png            (fixed Fleiss replacement)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = ROOT / "output" / "tables"
FIGS_DIR = ROOT / "output" / "figures"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

METRICS = ["DI", "SPD", "EOPP", "EOD", "TI", "PP", "CAL"]
ATTRS = ["RACE", "SEX", "ETHNICITY", "AGE_GROUP"]
THRESHOLDS = {"DI": 0.80, "SPD": 0.10, "EOPP": 0.10,
              "EOD": 0.10, "TI": 0.10, "PP": 0.10, "CAL": 0.05}

BLUE = {
    "deep":  "#1f3d7a",
    "dark":  "#2e5ca8",
    "mid":   "#4c8fc9",
    "light": "#a8c8e8",
    "pale":  "#e2eefb",
    "cream": "#f4f8fc",
}
GRAY = "#cfcfcf"; RED = "#b0413e"; GREEN = "#2f7b4a"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titleweight": "bold", "axes.titlesize": 10,
    "axes.edgecolor": "#333333", "axes.linewidth": 0.8,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})


def is_fair(metric, value):
    if pd.isna(value):
        return False
    return (value >= THRESHOLDS["DI"]) if metric == "DI" \
        else (abs(value) < THRESHOLDS[metric])


def reliability_class(vfr_pct):
    return "High" if vfr_pct < 10.0 else ("Moderate" if vfr_pct < 30.0 else "Unstable")


def count_fair(df_row, ignore_ti_cal=False) -> int:
    n = 0
    for attr in ATTRS:
        for m in METRICS:
            if ignore_ti_cal and m in ("TI", "CAL"):
                continue
            n += int(is_fair(m, df_row.get(f"{m}_{attr}", np.nan)))
    return n


def gwet_ac1(raters_fair_count, n_raters):
    """Gwet's AC1 agreement coefficient for binary ratings.
    raters_fair_count: number of raters voting 'Fair' (out of n_raters).
    Handles 0/N corner cases gracefully (returns 1.0 for unanimous).
    """
    if n_raters <= 1:
        return 1.0
    k = raters_fair_count
    p_fair = k / n_raters
    p_unfair = 1 - p_fair
    pa = (k * (k - 1) + (n_raters - k) * (n_raters - k - 1)) / (n_raters * (n_raters - 1))
    pe = 2 * p_fair * p_unfair
    if abs(1 - pe) < 1e-12:
        return 1.0
    return (pa - pe) / (1 - pe)


# -------------------------------------------------------------------------
# Tables
# -------------------------------------------------------------------------

def build_hospital_sweep_table():
    std = pd.read_csv(TABLES_DIR / "cikm_cross_hospital_scale.csv")
    # Normalise Std columns: paper sweep only reports SPD per attribute +
    # N_Fair_* (5 metrics). Reconstruct N_Fair_Total for comparability.
    rows = []
    try:
        fair = pd.read_csv(TABLES_DIR / "cikm_cross_hospital_scale_FAIR.csv")
        have_fair = True
    except FileNotFoundError:
        fair = None
        have_fair = False

    for _, r in std.iterrows():
        n_h = int(r["N_Hospitals"])
        std_n_fair = int(r.get("N_Fair_RACE", 0) + r.get("N_Fair_SEX", 0) +
                         r.get("N_Fair_ETHNICITY", 0) + r.get("N_Fair_AGE_GROUP", 0))
        row = {
            "N_Hospitals": n_h,
            "N_Train":     int(r["N_Train"]),
            "Std_Acc":     float(r["Accuracy"]),
            "Std_AUC":     float(r["AUC"]),
            "Std_DI_Race": float(r.get("DI_RACE", np.nan)),
            "Std_DI_Sex":  float(r.get("DI_SEX", np.nan)),
            "Std_DI_Eth":  float(r.get("DI_ETHNICITY", np.nan)),
            "Std_DI_Age":  float(r.get("DI_AGE_GROUP", np.nan)),
            "Std_N_fair_5m_20":  std_n_fair,
        }
        if have_fair:
            f = fair[fair["N_Hospitals"] == n_h]
            if len(f):
                fr = f.iloc[0]
                row.update({
                    "Fair_Acc":     float(fr["Accuracy"]),
                    "Fair_AUC":     float(fr["AUC"]),
                    "Fair_DI_Race": float(fr.get("DI_RACE", np.nan)),
                    "Fair_DI_Sex":  float(fr.get("DI_SEX", np.nan)),
                    "Fair_DI_Eth":  float(fr.get("DI_ETHNICITY", np.nan)),
                    "Fair_DI_Age":  float(fr.get("DI_AGE_GROUP", np.nan)),
                    "Fair_N_fair_5m_20": int(fr.get("N_Fair_Total", 0)),
                    "Delta_Acc":    float(fr["Accuracy"]) - float(r["Accuracy"]),
                    "Delta_N_fair": int(fr.get("N_Fair_Total", 0)) - std_n_fair,
                })
        rows.append(row)
    sweep = pd.DataFrame(rows)
    sweep.to_csv(TABLES_DIR / "Table6i_HospitalSweep_StdVsFair.csv", index=False)
    return sweep


def build_per_cluster_table(std_port, fair_port):
    rows = []
    for _, (s, f) in enumerate(zip(std_port.itertuples(index=False),
                                   fair_port.itertuples(index=False))):
        std_row = {f"{m}_{a}": getattr(s, f"{m}_{a}", np.nan)
                   for m in METRICS for a in ATTRS}
        fair_row = {f"{m}_{a}": getattr(f, f"{m}_{a}", np.nan)
                    for m in METRICS for a in ATTRS}

        std_nf = sum(int(is_fair(m, std_row[f"{m}_{a}"]))
                     for m in METRICS for a in ATTRS)
        fair_nf = sum(int(is_fair(m, fair_row[f"{m}_{a}"]))
                      for m in METRICS for a in ATTRS)

        rows.append({
            "Cluster":      int(getattr(s, "Fold")),
            "N_hospitals":  int(getattr(s, "N_hospitals")),
            "N_patients":   int(getattr(s, "N_val")),
            "Std_Acc":      float(getattr(s, "Acc")),
            "Fair_Acc":     float(getattr(f, "Acc")),
            "Delta_Acc":    float(getattr(f, "Acc")) - float(getattr(s, "Acc")),
            "Std_AUC":      float(getattr(s, "AUC")),
            "Fair_AUC":     float(getattr(f, "AUC")),
            "Std_N_fair/28":  f"{std_nf}/28",
            "Fair_N_fair/28": f"{fair_nf}/28",
            "Delta_N_fair":   fair_nf - std_nf,
            "Std_DI_worst":   round(min(std_row[f"DI_{a}"] for a in ATTRS), 3),
            "Fair_DI_worst":  round(min(fair_row[f"DI_{a}"] for a in ATTRS), 3),
            "Std_DI_all>=0.80":  "Yes" if all(std_row[f"DI_{a}"] >= 0.80 for a in ATTRS) else "No",
            "Fair_DI_all>=0.80": "Yes" if all(fair_row[f"DI_{a}"] >= 0.80 for a in ATTRS) else "No",
        })
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "Table6j_PerCluster_StdVsFair.csv", index=False)
    return df


def build_subset_stability_summary(std_port, fair_port):
    """Aggregate per-cluster stats across 20 subsets."""
    s_accs = std_port["Acc"].values
    f_accs = fair_port["Acc"].values
    s_aucs = std_port["AUC"].values
    f_aucs = fair_port["AUC"].values

    s_nfair = []
    f_nfair = []
    for _, r in std_port.iterrows():
        n = sum(int(is_fair(m, r.get(f"{m}_{a}", np.nan)))
                for m in METRICS for a in ATTRS)
        s_nfair.append(n)
    for _, r in fair_port.iterrows():
        n = sum(int(is_fair(m, r.get(f"{m}_{a}", np.nan)))
                for m in METRICS for a in ATTRS)
        f_nfair.append(n)
    s_nfair = np.array(s_nfair); f_nfair = np.array(f_nfair)

    # VFR from the per-metric-attr table (reuse if present else recompute)
    def vfr_stats(df):
        vfrs = []
        for a in ATTRS:
            for m in METRICS:
                vals = df[f"{m}_{a}"].dropna().tolist()
                if not vals:
                    continue
                mv = is_fair(m, float(np.mean(vals)))
                flips = sum(1 for v in vals if is_fair(m, v) != mv)
                vfrs.append(100 * flips / len(vals))
        return np.array(vfrs)

    s_vfrs = vfr_stats(std_port)
    f_vfrs = vfr_stats(fair_port)

    rows = [
        ("N subsets compared",           f"{len(s_accs)}",
         f"{len(f_accs)}", ""),
        ("Accuracy  mean ± SD",          f"{np.mean(s_accs):.4f} ± {np.std(s_accs, ddof=1):.4f}",
         f"{np.mean(f_accs):.4f} ± {np.std(f_accs, ddof=1):.4f}",
         f"{np.mean(f_accs) - np.mean(s_accs):+.4f}"),
        ("Accuracy  min – max",          f"{np.min(s_accs):.4f} – {np.max(s_accs):.4f}",
         f"{np.min(f_accs):.4f} – {np.max(f_accs):.4f}", ""),
        ("AUROC     mean ± SD",          f"{np.mean(s_aucs):.4f} ± {np.std(s_aucs, ddof=1):.4f}",
         f"{np.mean(f_aucs):.4f} ± {np.std(f_aucs, ddof=1):.4f}",
         f"{np.mean(f_aucs) - np.mean(s_aucs):+.4f}"),
        ("Fair verdicts /28  mean ± SD", f"{np.mean(s_nfair):.2f} ± {np.std(s_nfair, ddof=1):.2f}",
         f"{np.mean(f_nfair):.2f} ± {np.std(f_nfair, ddof=1):.2f}",
         f"{np.mean(f_nfair) - np.mean(s_nfair):+.2f}"),
        ("Fair verdicts /28  min – max", f"{int(np.min(s_nfair))} – {int(np.max(s_nfair))}",
         f"{int(np.min(f_nfair))} – {int(np.max(f_nfair))}", ""),
        ("All 4 DI ≥ 0.80 (subsets)",
         f"{sum(1 for _, r in std_port.iterrows() if all(r.get(f'DI_{a}', 0) >= 0.80 for a in ATTRS))}/{len(std_port)}",
         f"{sum(1 for _, r in fair_port.iterrows() if all(r.get(f'DI_{a}', 0) >= 0.80 for a in ATTRS))}/{len(fair_port)}",
         ""),
        ("VFR %  mean ± SD (28 pairs)",  f"{np.mean(s_vfrs):.1f} ± {np.std(s_vfrs, ddof=1):.1f}",
         f"{np.mean(f_vfrs):.1f} ± {np.std(f_vfrs, ddof=1):.1f}",
         f"{np.mean(f_vfrs) - np.mean(s_vfrs):+.1f}"),
        ("VFR %  max (worst metric)",    f"{np.max(s_vfrs):.1f}",
         f"{np.max(f_vfrs):.1f}",
         f"{np.max(f_vfrs) - np.max(s_vfrs):+.1f}"),
        ("High-reliability cells (VFR <10%)",
         f"{int(np.sum(s_vfrs < 10))}/28",
         f"{int(np.sum(f_vfrs < 10))}/28", ""),
        ("Unstable cells (VFR >30%)",
         f"{int(np.sum(s_vfrs > 30))}/28",
         f"{int(np.sum(f_vfrs > 30))}/28", ""),
    ]
    df = pd.DataFrame(rows, columns=["Quantity", "Standard", "Fair", "Δ"])
    df.to_csv(TABLES_DIR / "Table6k_Subset_Stability_Summary.csv", index=False)
    return df


def build_metric_concordance(std_port, fair_port):
    """Per-(metric, attribute) Gwet AC1 agreement across 20 clusters,
    for Standard and Fair, plus pass-rate heatmap data."""
    rows = []
    for a in ATTRS:
        for m in METRICS:
            s_vals = std_port[f"{m}_{a}"].dropna().tolist()
            f_vals = fair_port[f"{m}_{a}"].dropna().tolist()
            s_fair = sum(is_fair(m, v) for v in s_vals)
            f_fair = sum(is_fair(m, v) for v in f_vals)
            rows.append({
                "Attribute": a, "Metric": m,
                "Std_pass_k":  s_fair, "Std_N": len(s_vals),
                "Fair_pass_k": f_fair, "Fair_N": len(f_vals),
                "Std_pass_pct":  round(100 * s_fair / len(s_vals), 1),
                "Fair_pass_pct": round(100 * f_fair / len(f_vals), 1),
                "Std_AC1":   round(gwet_ac1(s_fair, len(s_vals)), 3),
                "Fair_AC1":  round(gwet_ac1(f_fair, len(f_vals)), 3),
            })
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "Table6l_Metric_Concordance.csv", index=False)
    return df


# -------------------------------------------------------------------------
# Figures
# -------------------------------------------------------------------------

def fig_subset_concordance_grid(std_port, fair_port, path):
    ordered = [f"{a}·{m}" for a in ATTRS for m in METRICS]

    def verdict_grid(df):
        M = np.zeros((len(ordered), 20), dtype=int)
        for i, am in enumerate(ordered):
            a, m = am.split("·")
            vals = df[f"{m}_{a}"].fillna(np.nan).values[:20]
            M[i] = np.array([1 if is_fair(m, v) else 0 for v in vals])
        return M

    sM = verdict_grid(std_port)
    fM = verdict_grid(fair_port)

    # Diff encoding: -1 = Pass→Fail (red), 0 unchanged fail (pale), 1 unchanged pass (dark), 2 = Fail→Pass (green)
    D = np.zeros_like(sM, dtype=int)
    for i in range(sM.shape[0]):
        for j in range(sM.shape[1]):
            if sM[i, j] == 1 and fM[i, j] == 1:
                D[i, j] = 1
            elif sM[i, j] == 0 and fM[i, j] == 1:
                D[i, j] = 2
            elif sM[i, j] == 1 and fM[i, j] == 0:
                D[i, j] = -1
            else:
                D[i, j] = 0

    fig, ax = plt.subplots(figsize=(14, 9))
    cmap = LinearSegmentedColormap.from_list(
        "conc",
        [(-1 / 3 + 1/3, RED),   # not useful, we'll use explicit colors via imshow with listedcolormap
         (0, BLUE["pale"]),
         (1, BLUE["deep"]),
         (1, GREEN)])

    # Use discrete imshow via category codes
    from matplotlib.colors import ListedColormap
    colors = [RED, BLUE["pale"], BLUE["deep"], GREEN]
    cmap = ListedColormap(colors)
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5]
    from matplotlib.colors import BoundaryNorm
    norm = BoundaryNorm(bounds, cmap.N)

    ax.imshow(D, aspect="auto", cmap=cmap, norm=norm)
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(ordered, fontsize=8)
    ax.set_xticks(range(20))
    ax.set_xticklabels([str(i + 1) for i in range(20)], fontsize=8)
    ax.set_xlabel("Hospital Cluster (1–20)")
    ax.set_title("Subset Concordance Grid — Standard → Fair verdict change per cluster",
                 loc="left", pad=10)

    # Gain/loss counts
    gain = int(np.sum(D == 2)); loss = int(np.sum(D == -1))
    same_pass = int(np.sum(D == 1)); same_fail = int(np.sum(D == 0))
    ax.text(20.5, -1.5,
            f"Fail→Pass: {gain}   |   Pass→Fail: {loss}   |   "
            f"Same Pass: {same_pass}   |   Same Fail: {same_fail}",
            fontsize=9, color=BLUE["deep"], weight="bold")

    handles = [
        mpatches.Patch(color=GREEN,       label="Fail → Pass  (gained fairness)"),
        mpatches.Patch(color=BLUE["deep"],label="Pass → Pass  (kept fairness)"),
        mpatches.Patch(color=BLUE["pale"],label="Fail → Fail  (still unfair)"),
        mpatches.Patch(color=RED,         label="Pass → Fail  (regression)"),
    ]
    ax.legend(handles=handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.16), ncol=4, frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(str(path)); plt.close(fig)


def fig_hospital_scaling_dual(sweep_df, path):
    has_fair = "Fair_Acc" in sweep_df.columns
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    x = sweep_df["N_Hospitals"].values

    l1 = ax1.plot(x, sweep_df["Std_Acc"], "o-", color=GRAY, lw=2,
                  markersize=7, label="Standard Acc")
    l2 = None
    if has_fair:
        l2 = ax1.plot(x, sweep_df["Fair_Acc"], "o-", color=BLUE["deep"],
                      lw=2.2, markersize=7, label="Fair Acc")
    ax1.set_xscale("log")
    ax1.set_xlabel("Number of training hospitals (log scale)")
    ax1.set_ylabel("Accuracy", color="#222222")
    ax1.set_ylim(0.75, 0.88)
    ax1.grid(alpha=0.25, axis="both")

    l3 = ax2.plot(x, sweep_df["Std_N_fair_5m_20"], "s--", color=GRAY,
                  lw=1.6, alpha=0.9, label="Standard Fair-verdicts (5m × 4attr = 20)")
    l4 = None
    if has_fair:
        l4 = ax2.plot(x, sweep_df["Fair_N_fair_5m_20"], "s--",
                      color=BLUE["mid"], lw=1.8, alpha=0.95,
                      label="Fair Fair-verdicts (5m × 4attr = 20)")
    ax2.set_ylabel("Fair verdicts (out of 20, 5-metric × 4-attr)", color=BLUE["deep"])
    ax2.set_ylim(-1, 21)

    all_lines = [*l1]
    if l2: all_lines += l2
    all_lines += l3
    if l4: all_lines += l4
    labels = [l.get_label() for l in all_lines]
    ax1.legend(all_lines, labels, loc="lower right", fontsize=8, frameon=True,
               facecolor="white", edgecolor="#ddd")

    for xi, sa, fa in zip(x, sweep_df["Std_Acc"], sweep_df.get("Fair_Acc", [np.nan]*len(x))):
        ax1.annotate(f"{sa:.3f}", (xi, sa), textcoords="offset points",
                     xytext=(0, -14), ha="center", fontsize=7, color="#666")
        if not np.isnan(fa):
            ax1.annotate(f"{fa:.3f}", (xi, fa), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=7, color=BLUE["deep"])

    plt.title("Hospital-count scaling — Standard vs Fair (accuracy + fair verdicts)",
              loc="left", pad=8)
    plt.tight_layout()
    plt.savefig(str(path)); plt.close(fig)


def fig_vfr_distribution(std_port, fair_port, path):
    # Compute per-(metric, attr) VFR for each model
    def vfr_vec(df):
        vals = []
        for a in ATTRS:
            for m in METRICS:
                vv = df[f"{m}_{a}"].dropna().tolist()
                if not vv:
                    continue
                mv = is_fair(m, float(np.mean(vv)))
                vals.append(100 * sum(1 for v in vv if is_fair(m, v) != mv) / len(vv))
        return np.array(vals)
    s = vfr_vec(std_port); f = vfr_vec(fair_port)

    fig, ax = plt.subplots(figsize=(10, 6))
    parts = ax.violinplot([s, f], showmeans=False, showmedians=True, widths=0.7)
    colors = [GRAY, BLUE["deep"]]
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c); pc.set_edgecolor(BLUE["dark"])
        pc.set_alpha(0.65)
    for part in ("cbars", "cmaxes", "cmins", "cmedians"):
        if part in parts:
            parts[part].set_color(BLUE["dark"])
            parts[part].set_linewidth(1.2)

    # Jitter dots
    rng = np.random.default_rng(42)
    ax.scatter(1 + rng.uniform(-0.12, 0.12, len(s)), s, s=22,
               color=GRAY, edgecolor=BLUE["dark"], zorder=5, alpha=0.9)
    ax.scatter(2 + rng.uniform(-0.12, 0.12, len(f)), f, s=22,
               color=BLUE["mid"], edgecolor=BLUE["deep"], zorder=5, alpha=0.9)

    ax.set_xticks([1, 2]); ax.set_xticklabels(["Standard", "Fair (λ=1 + α_SR)"])
    ax.set_ylabel("Verdict Flip Rate (%)")
    ax.axhspan(0, 10, color=BLUE["pale"], alpha=0.6)
    ax.axhspan(10, 30, color="#fdecc8", alpha=0.3)
    ax.axhspan(30, 100, color="#fad3d1", alpha=0.25)
    ax.text(2.5, 5, "High reliability",     color="#3b6ea5", fontsize=8, weight="bold")
    ax.text(2.5, 18, "Moderate",            color="#9a6c1e", fontsize=8, weight="bold")
    ax.text(2.5, 35, "Unstable",            color="#a04240", fontsize=8, weight="bold")
    ax.set_xlim(0.4, 2.8); ax.set_ylim(-2, 55)
    ax.set_title("VFR distribution across 28 (metric, attribute) pairs — Std vs Fair",
                 loc="left", pad=8)
    ax.grid(axis="y", color="#eee", lw=0.7)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(str(path)); plt.close(fig)


def fig_reliability_sankey(std_port, fair_port, path):
    """Custom 'Sankey-lite' flow using rectangles + bezier ribbons."""
    def vfr_class(df):
        classes = []
        for a in ATTRS:
            for m in METRICS:
                vv = df[f"{m}_{a}"].dropna().tolist()
                if not vv:
                    classes.append("Unstable")
                    continue
                mv = is_fair(m, float(np.mean(vv)))
                vfr = 100 * sum(1 for v in vv if is_fair(m, v) != mv) / len(vv)
                classes.append(reliability_class(vfr))
        return classes

    s_cls = vfr_class(std_port); f_cls = vfr_class(fair_port)
    order = ["High", "Moderate", "Unstable"]
    flows = {(s, f): 0 for s in order for f in order}
    for s, f in zip(s_cls, f_cls):
        flows[(s, f)] += 1

    colors = {"High": BLUE["deep"], "Moderate": BLUE["mid"], "Unstable": RED}

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 32); ax.axis("off")
    ax.set_title("Reliability class migration — Standard → Fair (28 metric-attribute pairs)",
                 loc="left", pad=8)

    # Draw left & right stacks
    def draw_stack(x, cls_list, labels_right=False):
        counts = [cls_list.count(c) for c in order]
        y = 30
        positions = {}
        for c, n in zip(order, counts):
            if n == 0:
                positions[c] = (y, y)
                continue
            h = n * 0.9
            rect = Rectangle((x, y - h), 0.8, h, fc=colors[c], ec="white",
                             lw=1.5, alpha=0.95)
            ax.add_patch(rect)
            ha = "left" if labels_right else "right"
            tx = x + 1.0 if labels_right else x - 0.2
            ax.text(tx, y - h / 2, f"{c} ({n})", va="center", ha=ha,
                    fontsize=9, weight="bold", color="#222")
            positions[c] = (y - h, y)
            y -= (h + 0.6)
        return positions

    left_pos = draw_stack(2, s_cls, labels_right=False)
    right_pos = draw_stack(7.2, f_cls, labels_right=True)

    # Draw ribbons as simple quadrilateral polygons
    left_cursor = {c: left_pos[c][1] for c in order}
    right_cursor = {c: right_pos[c][1] for c in order}
    for (sc, fc), n in flows.items():
        if n == 0:
            continue
        h = n * 0.9
        y1_top = left_cursor[sc]
        y1_bot = y1_top - h
        y2_top = right_cursor[fc]
        y2_bot = y2_top - h
        left_cursor[sc] -= h
        right_cursor[fc] -= h
        x1, x2 = 2.8, 7.2
        verts = [(x1, y1_top), (x2, y2_top), (x2, y2_bot), (x1, y1_bot)]
        if sc == fc:
            ribbon_color = colors[sc]
        elif order.index(fc) < order.index(sc):
            ribbon_color = BLUE["deep"]   # improvement
        else:
            ribbon_color = RED             # regression
        poly = mpatches.Polygon(verts, closed=True, facecolor=ribbon_color,
                                alpha=0.35, edgecolor="none")
        ax.add_patch(poly)
        if sc != fc and n > 0:
            ax.text((x1 + x2) / 2, (y1_top + y1_bot + y2_top + y2_bot) / 4,
                    f"{sc}→{fc}: {n}", ha="center", fontsize=8,
                    color="#222", alpha=0.9, weight="bold")

    ax.text(2.4, 31.5, "STANDARD", ha="center", fontsize=10,
            weight="bold", color=BLUE["deep"])
    ax.text(7.6, 31.5, "FAIR", ha="center", fontsize=10,
            weight="bold", color=BLUE["deep"])
    plt.tight_layout()
    plt.savefig(str(path)); plt.close(fig)


def fig_metric_coflip_network(std_port, fair_port, path):
    """Correlation of verdict-flip patterns across 20 clusters."""
    # Build 28 × 20 matrix of flip (Std->Fair verdict change)
    flips = np.zeros((28, 20))
    names = [f"{a}·{m}" for a in ATTRS for m in METRICS]
    for i, nm in enumerate(names):
        a, m = nm.split("·")
        sv = std_port[f"{m}_{a}"].values[:20]
        fv = fair_port[f"{m}_{a}"].values[:20]
        for j in range(20):
            s_fair = is_fair(m, sv[j])
            f_fair = is_fair(m, fv[j])
            flips[i, j] = int(f_fair) - int(s_fair)  # +1 gain, -1 loss, 0 same

    # Correlation between pairs' flip patterns
    corr = np.corrcoef(flips)
    corr = np.nan_to_num(corr, nan=0.0)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7),
                             gridspec_kw={"width_ratios": [1.05, 1]})

    # Heatmap
    ax = axes[0]
    cmap = LinearSegmentedColormap.from_list(
        "corr_blue", [RED, "#f0f0f0", BLUE["deep"]])
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(28)); ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticks(range(28)); ax.set_yticklabels(names, fontsize=7)
    ax.set_title("(a) Metric co-flip correlation (Std → Fair)",
                 loc="left", pad=8)
    plt.colorbar(im, ax=ax, shrink=0.7, label="Pearson r")

    # Network — connect pairs with |r| > 0.6
    ax2 = axes[1]
    ax2.set_title("(b) Strong co-flip edges (|r| > 0.5)", loc="left", pad=8)
    ax2.set_xlim(-1.25, 1.25); ax2.set_ylim(-1.25, 1.25); ax2.axis("off")
    theta = np.linspace(0, 2 * np.pi, 28, endpoint=False)
    xs = np.cos(theta); ys = np.sin(theta)

    # Edges first
    for i in range(28):
        for j in range(i + 1, 28):
            r = corr[i, j]
            if abs(r) < 0.5:
                continue
            c = BLUE["deep"] if r > 0 else RED
            ax2.plot([xs[i], xs[j]], [ys[i], ys[j]],
                     color=c, lw=0.5 + 2.0 * abs(r), alpha=0.55)

    # Nodes — size by mean |flip|
    node_size = 80 + 300 * np.abs(flips).mean(axis=1)
    node_color = [BLUE["deep"] if flips[i].sum() > 0 else
                  (RED if flips[i].sum() < 0 else GRAY) for i in range(28)]
    ax2.scatter(xs, ys, s=node_size, c=node_color, edgecolor="white", zorder=5)
    for i, nm in enumerate(names):
        ax2.text(xs[i] * 1.15, ys[i] * 1.15, nm, ha="center", va="center",
                 fontsize=7, color="#333")

    plt.tight_layout()
    plt.savefig(str(path)); plt.close(fig)


def fig_concordance_heatmap(conc_df, path):
    """Fixed Fleiss panel — pass-rate (% clusters fair) heatmap 7×4 with Gwet AC1."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5),
                             gridspec_kw={"width_ratios": [1, 1]})

    def draw(ax, col, title):
        pivot = conc_df.pivot(index="Metric", columns="Attribute", values=col)
        pivot = pivot.loc[METRICS, ATTRS]
        cmap = LinearSegmentedColormap.from_list("bp", [BLUE["pale"], BLUE["deep"]])
        im = ax.imshow(pivot.values, cmap=cmap, vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(ATTRS))); ax.set_xticklabels(ATTRS, fontsize=9)
        ax.set_yticks(range(len(METRICS))); ax.set_yticklabels(METRICS, fontsize=9)
        for i in range(len(METRICS)):
            for j in range(len(ATTRS)):
                v = pivot.values[i, j]
                c = "white" if v > 50 else BLUE["deep"]
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                        fontsize=9, weight="bold", color=c)
        ax.set_title(title, loc="left", pad=8)
        return im

    im1 = draw(axes[0], "Std_pass_pct",
               "(a) Standard — % of 20 clusters where verdict is Pass")
    im2 = draw(axes[1], "Fair_pass_pct",
               "(b) Fair (λ=1 + α_SR) — % of 20 clusters where verdict is Pass")

    fig.colorbar(im2, ax=axes, shrink=0.7, pad=0.02, label="Pass rate (%)")
    plt.suptitle("Cross-site verdict concordance — replaces the degenerate Fleiss κ heatmap",
                 weight="bold", color=BLUE["deep"], fontsize=12)
    plt.savefig(str(path)); plt.close(fig)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    std_port = pd.read_csv(TABLES_DIR / "cikm_cross_site_portability.csv")
    fair_port = pd.read_csv(TABLES_DIR / "cikm_cross_site_portability_FAIR.csv")

    sweep = build_hospital_sweep_table()
    print(f"Wrote Table6i_HospitalSweep_StdVsFair.csv  ({len(sweep)} rows)")

    per_cluster = build_per_cluster_table(std_port, fair_port)
    print(f"Wrote Table6j_PerCluster_StdVsFair.csv  ({len(per_cluster)} rows)")

    stability = build_subset_stability_summary(std_port, fair_port)
    print(f"Wrote Table6k_Subset_Stability_Summary.csv  ({len(stability)} rows)")

    concordance = build_metric_concordance(std_port, fair_port)
    print(f"Wrote Table6l_Metric_Concordance.csv  ({len(concordance)} rows)")

    fig_subset_concordance_grid(std_port, fair_port,
                                FIGS_DIR / "FIG15_subset_concordance_grid.png")
    print("Wrote FIG15_subset_concordance_grid.png")

    fig_hospital_scaling_dual(sweep,
                              FIGS_DIR / "FIG16_hospital_scaling_dual.png")
    print("Wrote FIG16_hospital_scaling_dual.png")

    fig_vfr_distribution(std_port, fair_port,
                         FIGS_DIR / "FIG17_vfr_distribution_violin.png")
    print("Wrote FIG17_vfr_distribution_violin.png")

    fig_reliability_sankey(std_port, fair_port,
                           FIGS_DIR / "FIG18_reliability_sankey.png")
    print("Wrote FIG18_reliability_sankey.png")

    fig_metric_coflip_network(std_port, fair_port,
                              FIGS_DIR / "FIG19_metric_coflip_network.png")
    print("Wrote FIG19_metric_coflip_network.png")

    fig_concordance_heatmap(concordance,
                            FIGS_DIR / "FIG20_concordance_heatmap.png")
    print("Wrote FIG20_concordance_heatmap.png")


if __name__ == "__main__":
    main()
