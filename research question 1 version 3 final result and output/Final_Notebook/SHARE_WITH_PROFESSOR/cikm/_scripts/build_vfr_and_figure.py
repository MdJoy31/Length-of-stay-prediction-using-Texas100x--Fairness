"""Build the VFR before/after table and the blue-shade reliability
dashboard figure for the paper.

Inputs (already on disk):
- output/tables/cikm_cross_site_portability.csv        (Standard, 20 folds)
- output/tables/cikm_cross_site_portability_FAIR.csv   (Fair, 20 folds)

Outputs (written into the notebook's artefact folders):
- output/tables/Table6h_VFR_StdVsFair.csv              (28 rows × 16 cols)
- output/tables/Table6h_VFR_StdVsFair.md               (paper-ready markdown)
- output/figures/FIG14_fairness_reliability_dashboard.png  (300 dpi, 4 panels)

Design notes
------------
VFR here is the Verdict Flip Rate across 20 hospital clusters: the
fraction of clusters whose per-cluster verdict disagrees with the
"majority verdict" (or with the cluster-mean verdict, whichever is the
binary decision the reader would make on the pooled data).  We report
both the mean-verdict-anchored VFR (`VFR_vs_mean`) and the raw
pass-rate instability `min(Pass_pct, 1-Pass_pct)`.

Reliability classes use the thresholds the paper already adopts for
Table 10: High (<10%), Moderate (10-30%), Unstable (>30%).

The dashboard uses a colourblind-safe single-hue blue palette anchored
at `#1f3d7a` (deep) → `#4c8fc9` (mid) → `#a8c8e8` (light).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

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
}
GRAY = "#cfcfcf"
RED = "#b0413e"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titleweight": "bold",
    "axes.titlesize": 10,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "axes.labelcolor": "#222222",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def is_fair(metric: str, value: float) -> bool:
    if pd.isna(value):
        return False
    return (value >= THRESHOLDS["DI"]) if metric == "DI" \
        else (abs(value) < THRESHOLDS[metric])


def reliability_class(vfr_pct: float) -> str:
    if vfr_pct < 10.0:
        return "High"
    if vfr_pct < 30.0:
        return "Moderate"
    return "Unstable"


def build_vfr_table(std_df: pd.DataFrame, fair_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for attr in ATTRS:
        for m in METRICS:
            col = f"{m}_{attr}"
            std_vals = std_df[col].dropna().to_list()
            fair_vals = fair_df[col].dropna().to_list()
            N = min(len(std_vals), len(fair_vals))
            if N == 0:
                continue

            std_mean = float(np.mean(std_vals))
            fair_mean = float(np.mean(fair_vals))
            std_pass_k = sum(is_fair(m, v) for v in std_vals)
            fair_pass_k = sum(is_fair(m, v) for v in fair_vals)
            std_pass_pct = 100 * std_pass_k / N
            fair_pass_pct = 100 * fair_pass_k / N

            std_mean_verdict = is_fair(m, std_mean)
            fair_mean_verdict = is_fair(m, fair_mean)

            # VFR vs cluster-mean verdict: % clusters disagreeing with the
            # mean-level binary verdict (the headline claim in the paper).
            std_flips = sum(1 for v in std_vals if is_fair(m, v) != std_mean_verdict)
            fair_flips = sum(1 for v in fair_vals if is_fair(m, v) != fair_mean_verdict)
            std_vfr = 100 * std_flips / N
            fair_vfr = 100 * fair_flips / N

            # Paired flip breakdown: of the 20 clusters, how many moved
            # Pass->Pass / Pass->Fail / Fail->Pass / Fail->Fail.
            p2p = p2f = f2p = f2f = 0
            for sv, fv in zip(std_vals[:N], fair_vals[:N]):
                s_fair = is_fair(m, sv)
                f_fair = is_fair(m, fv)
                if s_fair and f_fair: p2p += 1
                elif s_fair and not f_fair: p2f += 1
                elif not s_fair and f_fair: f2p += 1
                else: f2f += 1

            row = {
                "Attribute": attr,
                "Metric": m,
                "Threshold": THRESHOLDS[m],
                "Std_Mean": round(std_mean, 4),
                "Fair_Mean": round(fair_mean, 4),
                "Delta_Mean": round(fair_mean - std_mean, 4),
                "Std_Verdict_at_mean": "Pass" if std_mean_verdict else "Fail",
                "Fair_Verdict_at_mean": "Pass" if fair_mean_verdict else "Fail",
                "Std_Pass_k_over_N": f"{std_pass_k}/{N}",
                "Fair_Pass_k_over_N": f"{fair_pass_k}/{N}",
                "Std_Pass_pct": round(std_pass_pct, 1),
                "Fair_Pass_pct": round(fair_pass_pct, 1),
                "Std_VFR_pct": round(std_vfr, 1),
                "Fair_VFR_pct": round(fair_vfr, 1),
                "Delta_VFR": round(fair_vfr - std_vfr, 1),
                "Std_Reliability": reliability_class(std_vfr),
                "Fair_Reliability": reliability_class(fair_vfr),
                "Clusters_P2P": p2p,
                "Clusters_P2F": p2f,
                "Clusters_F2P": f2p,
                "Clusters_F2F": f2f,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def write_markdown(vfr: pd.DataFrame, path: Path) -> None:
    lines = []
    lines.append("# Table 6h — VFR Before/After (Standard → Fair, 20 Hospital Clusters)\n")
    lines.append(
        "The **Verdict Flip Rate (VFR)** is the fraction of 20 hospital clusters "
        "where the per-cluster fair/unfair verdict disagrees with the cluster-mean "
        "verdict. Low VFR means the headline pass/fail result is portable across "
        "sites; high VFR means the verdict is a coin-flip in a new hospital.\n"
    )
    lines.append("\nReliability classes: **High** < 10% VFR; **Moderate** 10–30%; "
                 "**Unstable** > 30%. Thresholds match main.tex Table 10.\n")

    cols = [
        "Attribute", "Metric", "Std_Mean", "Fair_Mean", "Delta_Mean",
        "Std_Verdict_at_mean", "Fair_Verdict_at_mean",
        "Std_Pass_k_over_N", "Fair_Pass_k_over_N",
        "Std_VFR_pct", "Fair_VFR_pct", "Delta_VFR",
        "Std_Reliability", "Fair_Reliability",
        "Clusters_P2P", "Clusters_P2F", "Clusters_F2P", "Clusters_F2F",
    ]
    sub = vfr[cols].copy()
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body_lines = []
    for _, r in sub.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float) and not pd.isna(v):
                cells.append(f"{v:.3f}" if c in ("Std_Mean", "Fair_Mean", "Delta_Mean") else f"{v}")
            else:
                cells.append(str(v))
        body_lines.append("| " + " | ".join(cells) + " |")
    lines.append("\n" + "\n".join([header, sep, *body_lines]))

    # Attribute summary
    lines.append("\n## Attribute Summary — Reliability Uplift\n")
    summary_rows = []
    for attr in ATTRS:
        asub = vfr[vfr["Attribute"] == attr]
        std_n = int((asub["Std_Verdict_at_mean"] == "Pass").sum())
        fair_n = int((asub["Fair_Verdict_at_mean"] == "Pass").sum())
        std_avg_vfr = float(asub["Std_VFR_pct"].mean())
        fair_avg_vfr = float(asub["Fair_VFR_pct"].mean())
        summary_rows.append((attr, std_n, fair_n, std_avg_vfr, fair_avg_vfr,
                             fair_avg_vfr - std_avg_vfr))
    h = "| Attribute | Std N_fair/7 | Fair N_fair/7 | Std avg VFR | Fair avg VFR | Δ avg VFR |"
    s = "| --- | --- | --- | --- | --- | --- |"
    b = ["| " + " | ".join([attr, f"{sn}/7", f"{fn}/7",
                            f"{sv:.1f}%", f"{fv:.1f}%", f"{dv:+.1f} pp"])
         + " |"
         for attr, sn, fn, sv, fv, dv in summary_rows]
    lines.append("\n" + "\n".join([h, s, *b]))

    lines.append(
        "\n### Reading\n"
        "Cells with **Fair_Verdict_at_mean = Pass AND Std_Verdict_at_mean = Fail** "
        "are the new fair verdicts earned by the intervention. Cells with "
        "**Fair_VFR < Std_VFR** indicate that the Fair pipeline does not just "
        "raise the mean — it also stabilises the verdict across hospitals, "
        "converting unstable verdicts into portable ones. This combination is "
        "how the paper should motivate the ‘reliability-through-intervention’ "
        "claim.\n"
    )
    path.write_text("\n".join(lines), encoding="utf-8")


# -------------------------------------------------------------------------
# Figure
# -------------------------------------------------------------------------


def draw_panel_pipeline(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("(a) Fairness pipeline — Reweigh → Train → Threshold",
                 loc="left", pad=8)

    stages = [
        (0.5, "Input\nHospital 19/20",              "#ffffff", "#777777"),
        (2.4, "Stage 1\nIntersectional\nλ-reweigh (λ=1)",  BLUE["pale"],  BLUE["dark"]),
        (4.7, "Stage 2\nLightGBM\n(sample weights)",       BLUE["light"], BLUE["dark"]),
        (7.0, "Stage 3\nPer-group\nα_SR=0.8 threshold",    BLUE["mid"],   BLUE["deep"]),
        (9.1, "Held-out\nHospital 20/20",           "#ffffff", BLUE["deep"]),
    ]
    for x, text, fc, ec in stages:
        box = FancyBboxPatch((x - 0.55, 2.5), 1.5, 1.5,
                             boxstyle="round,pad=0.05,rounding_size=0.15",
                             fc=fc, ec=ec, lw=1.5)
        ax.add_patch(box)
        ax.text(x + 0.2, 3.25, text, ha="center", va="center",
                fontsize=8.5, color="#111", weight="bold")

    for xs, xe in [(1.1, 1.85), (3.2, 3.95), (5.5, 6.25), (7.8, 8.55)]:
        arr = FancyArrowPatch((xs, 3.25), (xe, 3.25),
                              arrowstyle="-|>", mutation_scale=12,
                              color=BLUE["deep"], lw=1.6)
        ax.add_patch(arr)

    # Fairness envelope
    env = FancyBboxPatch((1.7, 1.6), 6.6, 2.9,
                         boxstyle="round,pad=0.1,rounding_size=0.3",
                         fc="none", ec=BLUE["deep"],
                         lw=1.2, linestyle=(0, (5, 3)))
    ax.add_patch(env)
    ax.text(5.0, 1.15, "Fairness envelope — 7 metrics × 4 attributes re-evaluated",
            ha="center", fontsize=8, color=BLUE["deep"], style="italic")


def draw_panel_heatmap(ax, std_df, fair_df):
    ax.set_title("(b) Per-cluster fairness — Standard vs Fair "
                 "(20 hospital clusters × 28 metric×attribute)",
                 loc="left", pad=8)

    ordered = [f"{a}·{m}" for a in ATTRS for m in METRICS]

    def grid(df):
        M = np.zeros((len(ordered), 20), dtype=float)
        for i, am in enumerate(ordered):
            attr, m = am.split("·")
            col = f"{m}_{attr}"
            vals = df[col].fillna(np.nan).values[:20]
            M[i] = np.array([1.0 if is_fair(m, v) else 0.0 for v in vals])
        return M

    std_M = grid(std_df)
    fair_M = grid(fair_df)
    gap = np.full((len(ordered), 2), np.nan)
    full = np.concatenate([std_M, gap, fair_M], axis=1)

    cmap = LinearSegmentedColormap.from_list(
        "fair_blue", [(0, BLUE["pale"]), (1, BLUE["deep"])])
    cmap.set_bad("#ffffff")
    im = ax.imshow(full, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(ordered, fontsize=7)
    xt = list(range(1, 21)) + ["", ""] + list(range(1, 21))
    ax.set_xticks(range(len(xt)))
    ax.set_xticklabels([str(x) if x != "" else "" for x in xt], fontsize=6)
    ax.axvline(x=19.5, color="#ffffff", lw=3)
    ax.axvline(x=21.5, color="#ffffff", lw=3)
    ax.text(9.5, -1.2, "STANDARD", ha="center", fontsize=9, weight="bold",
            color=BLUE["deep"])
    ax.text(31.5, -1.2, "FAIR (λ=1 + α_SR=0.8)", ha="center", fontsize=9,
            weight="bold", color=BLUE["deep"])

    legend = [
        mpatches.Patch(color=BLUE["deep"], label="Pass"),
        mpatches.Patch(color=BLUE["pale"], label="Fail"),
    ]
    ax.legend(handles=legend, loc="lower center",
              bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False, fontsize=8)

    # Pass-count annotations
    std_pass = int(std_M.sum())
    fair_pass = int(fair_M.sum())
    ax.text(9.5, len(ordered) + 0.3,
            f"{std_pass}/{len(ordered)*20} cells pass",
            ha="center", fontsize=8, color=BLUE["deep"], weight="bold")
    ax.text(31.5, len(ordered) + 0.3,
            f"{fair_pass}/{len(ordered)*20} cells pass  (+{fair_pass - std_pass})",
            ha="center", fontsize=8, color=BLUE["deep"], weight="bold")


def draw_panel_radar(ax, vfr):
    ax.set_title("(c) Reliability radar — fair metrics at cluster mean (0–7)",
                 loc="left", pad=8)
    cats = ATTRS
    N = len(cats)
    theta = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False).tolist()
    theta += theta[:1]

    def vec(verdict_col):
        out = []
        for attr in cats:
            sub = vfr[vfr["Attribute"] == attr]
            out.append(int((sub[verdict_col] == "Pass").sum()))
        out += out[:1]
        return out

    std_vec = vec("Std_Verdict_at_mean")
    fair_vec = vec("Fair_Verdict_at_mean")

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_ylim(0, 7)
    ax.set_yticks([2, 4, 6])
    ax.set_yticklabels(["2", "4", "6"], color="#888", fontsize=7)
    ax.set_xticks(theta[:-1])
    ax.set_xticklabels(cats, fontsize=8, weight="bold")

    ax.plot(theta, std_vec, color=GRAY, lw=2, label="Standard")
    ax.fill(theta, std_vec, color=GRAY, alpha=0.3)
    ax.plot(theta, fair_vec, color=BLUE["deep"], lw=2, label="Fair")
    ax.fill(theta, fair_vec, color=BLUE["mid"], alpha=0.45)

    for t, v in zip(theta[:-1], fair_vec[:-1]):
        ax.text(t, v + 0.35, str(v), ha="center", fontsize=8,
                color=BLUE["deep"], weight="bold")
    ax.legend(loc="lower right", bbox_to_anchor=(1.25, 0.0),
              frameon=False, fontsize=8)


def draw_panel_slope(ax, vfr):
    ax.set_title("(d) Before → After — Δ fairness & Δ VFR per metric",
                 loc="left", pad=8)
    metrics = METRICS
    xs = [0, 1]
    ax.set_xticks(xs)
    ax.set_xticklabels(["Standard", "Fair (λ=1 + α_SR)"], fontsize=9)
    ax.set_ylabel("N_fair metrics (across 4 attributes)")

    for m in metrics:
        sub = vfr[vfr["Metric"] == m]
        std_n = int((sub["Std_Verdict_at_mean"] == "Pass").sum())
        fair_n = int((sub["Fair_Verdict_at_mean"] == "Pass").sum())
        delta = fair_n - std_n
        colour = BLUE["deep"] if delta > 0 else (GRAY if delta == 0 else RED)
        lw = 1.2 + 0.5 * abs(delta)
        ax.plot(xs, [std_n, fair_n], "-", color=colour, lw=lw, alpha=0.9)
        ax.scatter(xs, [std_n, fair_n], s=48, color=colour, zorder=5, edgecolor="white")
        ax.text(1.03, fair_n, f"{m}  Δ={delta:+d}",
                va="center", fontsize=8, color=colour)

    ax.set_xlim(-0.2, 1.6)
    ax.set_ylim(-0.3, 4.3)
    ax.set_yticks(range(5))
    ax.grid(axis="y", color="#eee", lw=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def build_figure(vfr, std_df, fair_df, path):
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[1.0, 1.1, 1.0],
        width_ratios=[1.2, 1.0],
        hspace=0.55, wspace=0.30,
        left=0.06, right=0.97, top=0.94, bottom=0.06,
    )
    ax_pipe = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[:, 1])
    ax_radar = fig.add_subplot(gs[1, 0], projection="polar")
    ax_slope = fig.add_subplot(gs[2, 0])

    draw_panel_pipeline(ax_pipe)
    draw_panel_heatmap(ax_heat, std_df, fair_df)
    draw_panel_radar(ax_radar, vfr)
    draw_panel_slope(ax_slope, vfr)

    fig.suptitle("Fairness Reliability Dashboard — Standard vs Fair Across 20 Hospital Clusters",
                 fontsize=13, weight="bold", color=BLUE["deep"], y=0.985)
    fig.text(0.5, 0.018,
             "Colour — dark blue = fair, pale blue = unfair;  slope blue = gain, grey = unchanged, red = regression.",
             ha="center", fontsize=8, color="#444", style="italic")
    plt.savefig(path)
    plt.close(fig)


def main():
    std_df = pd.read_csv(TABLES_DIR / "cikm_cross_site_portability.csv")
    fair_df = pd.read_csv(TABLES_DIR / "cikm_cross_site_portability_FAIR.csv")

    vfr = build_vfr_table(std_df, fair_df)
    vfr.to_csv(TABLES_DIR / "Table6h_VFR_StdVsFair.csv", index=False)
    write_markdown(vfr, TABLES_DIR / "Table6h_VFR_StdVsFair.md")
    print(f"Wrote {TABLES_DIR / 'Table6h_VFR_StdVsFair.csv'}  ({len(vfr)} rows)")

    build_figure(vfr, std_df, fair_df,
                 FIGS_DIR / "FIG14_fairness_reliability_dashboard.png")
    print(f"Wrote {FIGS_DIR / 'FIG14_fairness_reliability_dashboard.png'}")

    # Quick printed summary for the user
    print("\nAttribute-level reliability uplift:")
    for attr in ATTRS:
        sub = vfr[vfr["Attribute"] == attr]
        std_n = int((sub["Std_Verdict_at_mean"] == "Pass").sum())
        fair_n = int((sub["Fair_Verdict_at_mean"] == "Pass").sum())
        std_vfr = float(sub["Std_VFR_pct"].mean())
        fair_vfr = float(sub["Fair_VFR_pct"].mean())
        print(f"  {attr:10s}  N_fair: {std_n}/7 -> {fair_n}/7   "
              f"avg VFR: {std_vfr:.1f}% -> {fair_vfr:.1f}% "
              f"({fair_vfr - std_vfr:+.1f} pp)")


if __name__ == "__main__":
    main()
