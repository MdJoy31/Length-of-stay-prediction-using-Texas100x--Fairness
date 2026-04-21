"""Insert 10 new cells into the CIKM notebook (Cells 15e .. 15n) that
display the big hospital-sweep × VFR table, per-cluster comparison,
subset-stability summary, metric concordance table, and the 6 new
figures (FIG15..20).  Each cell is given pre-rendered HTML / stdout /
PNG outputs so the notebook is viewable on GitHub without running a
kernel.

Cells are inserted immediately after the existing Fairness Reliability
Dashboard cell (id = 9766f764).
"""

from __future__ import annotations

import base64
import html
import json
import shutil
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "CIKM_2026_LOS_Fairness_13042026.ipynb"
BACKUP = ROOT / "CIKM_2026_LOS_Fairness.pre-big-cells.ipynb"
TABLES_DIR = ROOT / "output" / "tables"
FIGS_DIR = ROOT / "output" / "figures"

ANCHOR_ID = "9766f764"  # FIG14 dashboard cell

STYLE = """
<style>
.big-tab { border-collapse: collapse; font-family: Arial, sans-serif;
           font-size: 11px; margin: 6px 0 16px 0; }
.big-tab th { background: #1f3d7a; color: #fff; padding: 5px 8px;
              text-align: center; border: 1px solid #4c8fc9; font-weight: 600; }
.big-tab td { padding: 4px 7px; border: 1px solid #dbe7f5;
              text-align: center; color: #222; }
.big-tab tr:nth-child(even) td { background: #f4f8fc; }
.big-attr { background: #1f3d7a !important; color: #fff !important; font-weight: 600; }
.big-pos { color: #1f3d7a; font-weight: 700; }
.big-neg { color: #b0413e; font-weight: 700; }
.big-pass { background: #1f3d7a; color: #fff; font-weight: 600; }
.big-fail { background: #e2eefb; color: #1f3d7a; }
.big-rel-high { background: #1f3d7a; color: #fff; font-weight: 600; }
.big-rel-mod  { background: #4c8fc9; color: #fff; font-weight: 600; }
.big-rel-uns  { background: #b0413e; color: #fff; font-weight: 600; }
.big-quantity { background: #f4f8fc; color: #1f3d7a; font-weight: 600; text-align: left; padding-left: 12px; }
</style>
"""


def tbl_html(df: pd.DataFrame, formatters: dict[str, str] | None = None,
             highlight_pass: tuple[str, ...] = (),
             attr_col: str | None = None,
             row_label_col: str | None = None,
             delta_cols: tuple[str, ...] = ()) -> str:
    formatters = formatters or {}
    cols = list(df.columns)
    lines = ['<table class="big-tab">']
    lines.append("<thead><tr>"
                 + "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)
                 + "</tr></thead><tbody>")
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if pd.isna(v):
                cells.append("<td></td>")
                continue
            if c == attr_col:
                cells.append(f'<td class="big-attr">{html.escape(str(v))}</td>')
            elif c == row_label_col:
                cells.append(f'<td class="big-quantity">{html.escape(str(v))}</td>')
            elif c in highlight_pass:
                cls = "big-pass" if v in ("Pass", "Yes") else "big-fail"
                cells.append(f'<td class="{cls}">{html.escape(str(v))}</td>')
            elif isinstance(v, (int, float, np.integer, np.floating)):
                # Coerce ints-that-read-as-floats when format spec is {:d} or {:+d}
                if c in formatters and ("d}" in formatters[c]):
                    try:
                        v = int(round(float(v)))
                    except (ValueError, TypeError):
                        pass
                if c in delta_cols:
                    try:
                        fv = float(v)
                        cls = "big-pos" if fv > 0 else ("big-neg" if fv < 0 else "")
                        fmt = formatters.get(c, "{:+.3f}")
                        cells.append(f'<td class="{cls}">{fmt.format(v)}</td>')
                        continue
                    except (ValueError, TypeError):
                        pass
                if c in formatters:
                    cells.append(f'<td>{formatters[c].format(v)}</td>')
                else:
                    cells.append(f'<td>{v}</td>')
            else:
                cells.append(f'<td>{html.escape(str(v))}</td>')
        lines.append("<tr>" + "".join(cells) + "</tr>")
    lines.append("</tbody></table>")
    return STYLE + "\n".join(lines)


def png_display_output(path: Path, width=1280):
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "output_type": "display_data",
        "data": {"image/png": b64,
                 "text/plain": [f"<{path.name}>"]},
        "metadata": {"image/png": {"width": width}},
    }


def html_output(h: str):
    return {"output_type": "display_data",
            "data": {"text/html": [h], "text/plain": ["<styled HTML>"]},
            "metadata": {}}


def stdout_output(text):
    return {"output_type": "stream", "name": "stdout", "text": [text]}


def build_all():
    # --- Table 6i: Hospital sweep Std vs Fair ------------------------
    sweep = pd.read_csv(TABLES_DIR / "Table6i_HospitalSweep_StdVsFair.csv")
    sweep_html = (
        "<h4>Table 6i: <b>Hospital-count sweep</b> — Standard vs Fair at 8 training scales "
        "(1–439 hospitals)</h4>"
        + tbl_html(
            sweep,
            formatters={
                "N_Hospitals": "{}", "N_Train": "{:,}",
                "Std_Acc": "{:.4f}", "Std_AUC": "{:.4f}",
                "Fair_Acc": "{:.4f}", "Fair_AUC": "{:.4f}",
                "Std_DI_Race": "{:.3f}", "Std_DI_Sex": "{:.3f}",
                "Std_DI_Eth": "{:.3f}", "Std_DI_Age": "{:.3f}",
                "Fair_DI_Race": "{:.3f}", "Fair_DI_Sex": "{:.3f}",
                "Fair_DI_Eth": "{:.3f}", "Fair_DI_Age": "{:.3f}",
                "Std_N_fair_5m_20": "{}", "Fair_N_fair_5m_20": "{}",
                "Delta_Acc": "{:+.4f}", "Delta_N_fair": "{:+d}",
            },
            delta_cols=("Delta_Acc", "Delta_N_fair"),
        )
    )
    sweep_cell_src = (
        "# ──────────────────────────────────────────────────────────────\n"
        "# Cell 15e · Table 6i — Hospital-count sweep (Standard vs Fair)\n"
        "# ──────────────────────────────────────────────────────────────\n"
        "# Trains the Fair pipeline at 1 / 2 / 3 / 5 / 10 / 50 / 100 / 439\n"
        "# hospitals and pairs the results with the existing Standard sweep.\n"
        "# Shows that Fair's fair-verdict count stays ~13-15/20 even at tiny\n"
        "# N_hospitals while Standard only reaches 14/20 at very high N.\n"
        "sweep6i = pd.read_csv(f'{TABLES_DIR}/Table6i_HospitalSweep_StdVsFair.csv')\n"
        "display(HTML('<h4>Table 6i: <b>Hospital-count sweep</b> — Standard vs Fair at 8 training scales</h4>'))\n"
        "display(sweep6i.style.format({\n"
        "    'N_Train':'{:,}','Std_Acc':'{:.4f}','Std_AUC':'{:.4f}','Fair_Acc':'{:.4f}','Fair_AUC':'{:.4f}',\n"
        "    'Std_DI_Race':'{:.3f}','Std_DI_Sex':'{:.3f}','Std_DI_Eth':'{:.3f}','Std_DI_Age':'{:.3f}',\n"
        "    'Fair_DI_Race':'{:.3f}','Fair_DI_Sex':'{:.3f}','Fair_DI_Eth':'{:.3f}','Fair_DI_Age':'{:.3f}',\n"
        "    'Delta_Acc':'{:+.4f}','Delta_N_fair':'{:+d}'}))\n"
    )

    # --- Table 6j: Per-cluster Std vs Fair ----------------------------
    per_cluster = pd.read_csv(TABLES_DIR / "Table6j_PerCluster_StdVsFair.csv")
    per_cluster_html = (
        "<h4>Table 6j: <b>Per-cluster comparison</b> — Standard vs Fair on 20 GroupKFold hospital clusters</h4>"
        "<p>One row per held-out cluster. <code>Fair_DI_all>=0.80</code> marks clusters where the Fair pipeline satisfies the four-fifths rule on all four protected attributes.</p>"
        + tbl_html(
            per_cluster,
            formatters={
                "Cluster": "{}", "N_hospitals": "{}", "N_patients": "{:,}",
                "Std_Acc": "{:.4f}", "Fair_Acc": "{:.4f}", "Delta_Acc": "{:+.4f}",
                "Std_AUC": "{:.4f}", "Fair_AUC": "{:.4f}",
                "Std_DI_worst": "{:.3f}", "Fair_DI_worst": "{:.3f}",
                "Delta_N_fair": "{:+d}",
            },
            highlight_pass=("Std_DI_all>=0.80", "Fair_DI_all>=0.80"),
            delta_cols=("Delta_Acc", "Delta_N_fair"),
        )
    )
    per_cluster_cell_src = (
        "# ──────────────────────────────────────────────────────────────\n"
        "# Cell 15f · Table 6j — Per-cluster Standard vs Fair (20 clusters)\n"
        "# ──────────────────────────────────────────────────────────────\n"
        "# For every held-out hospital cluster, reports Std_Acc, Fair_Acc,\n"
        "# delta accuracy, fair-verdict counts, and whether all 4 DIs ≥ 0.80\n"
        "# in that site. Use this to see which hospital clusters resist the\n"
        "# Fair intervention vs which become fully fair under it.\n"
        "pc6j = pd.read_csv(f'{TABLES_DIR}/Table6j_PerCluster_StdVsFair.csv')\n"
        "display(HTML('<h4>Table 6j: <b>Per-cluster comparison</b> — Standard vs Fair (20 hospital clusters)</h4>'))\n"
        "display(pc6j.style.format({\n"
        "    'N_patients':'{:,}','Std_Acc':'{:.4f}','Fair_Acc':'{:.4f}','Delta_Acc':'{:+.4f}',\n"
        "    'Std_AUC':'{:.4f}','Fair_AUC':'{:.4f}','Std_DI_worst':'{:.3f}','Fair_DI_worst':'{:.3f}',\n"
        "    'Delta_N_fair':'{:+d}'}))\n"
    )

    # --- Table 6k: Subset stability summary --------------------------
    stab = pd.read_csv(TABLES_DIR / "Table6k_Subset_Stability_Summary.csv")
    stab_html = (
        "<h4>Table 6k: <b>Subset-stability summary</b> — aggregate across 20 subsets "
        "(hospital clusters as subsets)</h4>"
        "<p>Compares accuracy, AUROC, fair-verdict count, all-DI-pass rate, and VFR "
        "before and after the Fair intervention.</p>"
        + tbl_html(
            stab,
            row_label_col="Quantity",
            formatters={},
        )
    )
    stab_cell_src = (
        "# ──────────────────────────────────────────────────────────────\n"
        "# Cell 15g · Table 6k — Subset stability summary (20 subsets)\n"
        "# ──────────────────────────────────────────────────────────────\n"
        "# One-page reliability uplift: mean ± SD accuracy, AUROC, fair\n"
        "# verdicts, all-DI-pass rate, and VFR stats before / after the\n"
        "# Fair pipeline. Headline lines:\n"
        "#   All 4 DI ≥ 0.80  subsets:   0/20 → 12/20\n"
        "#   Unstable cells (VFR > 30%): 5/28 → 2/28\n"
        "stab6k = pd.read_csv(f'{TABLES_DIR}/Table6k_Subset_Stability_Summary.csv')\n"
        "display(HTML('<h4>Table 6k: <b>Subset-stability summary</b> — aggregate across 20 subsets</h4>'))\n"
        "display(stab6k)\n"
    )

    # --- Table 6l: Metric concordance (Gwet AC1) ----------------------
    conc = pd.read_csv(TABLES_DIR / "Table6l_Metric_Concordance.csv")
    conc_html = (
        "<h4>Table 6l: <b>Per-metric cross-site concordance</b> — pass-rate and "
        "Gwet AC1 (replaces degenerate Fleiss κ)</h4>"
        "<p>Gwet AC1 handles the 0%/100% pass-rate corner cases that make Fleiss κ "
        "collapse to ±0.11 / 1.0. Higher AC1 means the 20 clusters agree more on "
        "the metric's verdict.</p>"
        + tbl_html(
            conc,
            attr_col="Attribute",
            formatters={
                "Std_pass_pct": "{:.1f}", "Fair_pass_pct": "{:.1f}",
                "Std_AC1": "{:.3f}", "Fair_AC1": "{:.3f}",
                "Std_pass_k": "{}", "Std_N": "{}",
                "Fair_pass_k": "{}", "Fair_N": "{}",
            },
        )
    )
    conc_cell_src = (
        "# ──────────────────────────────────────────────────────────────\n"
        "# Cell 15h · Table 6l — Per-metric concordance (Gwet AC1)\n"
        "# ──────────────────────────────────────────────────────────────\n"
        "# Replaces the earlier Fleiss κ heatmap (which collapsed to ±0.11\n"
        "# or 1.0 whenever a cell was unanimously fair/unfair). Gwet AC1\n"
        "# stays well-defined at 0% and 100% pass rates.\n"
        "def _gwet_ac1(k, n):\n"
        "    if n <= 1: return 1.0\n"
        "    p = k/n; q = 1-p\n"
        "    pa = (k*(k-1) + (n-k)*(n-k-1)) / (n*(n-1))\n"
        "    pe = 2*p*q\n"
        "    return 1.0 if abs(1-pe) < 1e-12 else (pa - pe)/(1 - pe)\n"
        "conc6l = pd.read_csv(f'{TABLES_DIR}/Table6l_Metric_Concordance.csv')\n"
        "display(HTML('<h4>Table 6l: <b>Per-metric cross-site concordance</b> (Gwet AC1 replaces Fleiss κ)</h4>'))\n"
        "display(conc6l.style.format({\n"
        "    'Std_pass_pct':'{:.1f}','Fair_pass_pct':'{:.1f}','Std_AC1':'{:.3f}','Fair_AC1':'{:.3f}'}))\n"
    )

    # --- Figures ------------------------------------------------------
    fig_specs = [
        ("15i", "FIG15_subset_concordance_grid",
         "Subset concordance grid — verdict change per cluster × 28 metric-attribute pairs",
         "Green = Fail→Pass (gained fairness), Navy = Pass→Pass (kept fairness), "
         "Pale = Fail→Fail (still unfair), Red = Pass→Fail (regression). "
         "121 Fail→Pass vs 62 Pass→Fail = net +59 gains across 560 cells."),
        ("15j", "FIG16_hospital_scaling_dual",
         "Hospital-count scaling — Standard vs Fair (accuracy + fair-verdict count)",
         "Fair's fair-verdict count plateaus at ~13–15/20 from just 1 hospital "
         "onward; Standard requires 50+ hospitals to match."),
        ("15k", "FIG17_vfr_distribution_violin",
         "VFR distribution across 28 (metric, attribute) pairs — Std vs Fair",
         "Green band = High reliability (VFR < 10%), amber = Moderate, red = Unstable. "
         "Fair compresses the distribution into the High band."),
        ("15l", "FIG18_reliability_sankey",
         "Reliability class migration — Standard → Fair (28 metric-attribute pairs)",
         "14 → 17 High-reliability cells, 8 → 9 Moderate, 6 → 2 Unstable. "
         "6 Unstable cells move up to High (UnstableHigh)."),
        ("15m", "FIG19_metric_coflip_network",
         "Metric co-flip correlation + network (which (metric, attr) pairs move together)",
         "Blue edges = positive correlation (same clusters gain fairness); "
         "red edges = inverse correlation. Strong clusters indicate which metrics "
         "share underlying reliability drivers."),
        ("15n", "FIG20_concordance_heatmap",
         "Cross-site verdict concordance — pass-rate heatmap (replaces Fleiss κ)",
         "7 metrics × 4 attributes. Each cell is the % of 20 clusters where the "
         "verdict is Pass. Fair saturates the heatmap near 100% for SEX and "
         "ETHNICITY and lifts AGE_GROUP DI/SPD from 0% to 80%."),
    ]

    figure_cells = []
    for tag, fname, title, caption in fig_specs:
        fig_path = FIGS_DIR / f"{fname}.png"
        src = (
            f"# ──────────────────────────────────────────────────────────────\n"
            f"# Cell {tag} · {title}\n"
            f"# ──────────────────────────────────────────────────────────────\n"
            f"# {caption}\n"
            f"display(HTML('<h4>{title}</h4>'))\n"
            f"display(Image(f'{{FIGS_DIR}}/{fname}.png'))\n"
        )
        outputs = [
            html_output(f"<h4>{title}</h4>"
                        f"<p style='font-size:10px;color:#555'>{caption}</p>"),
            png_display_output(fig_path),
        ]
        figure_cells.append((tag, src, outputs))

    return {
        "sweep": (sweep_cell_src, [html_output(sweep_html)]),
        "per_cluster": (per_cluster_cell_src, [html_output(per_cluster_html)]),
        "stab": (stab_cell_src, [html_output(stab_html)]),
        "conc": (conc_cell_src, [html_output(conc_html)]),
        "figures": figure_cells,
    }


def main():
    shutil.copy2(NB, BACKUP)
    nb = json.loads(NB.read_text(encoding="utf-8"))

    anchor = None
    for i, c in enumerate(nb["cells"]):
        if c.get("id") == ANCHOR_ID:
            anchor = i
            break
    if anchor is None:
        raise RuntimeError(f"Anchor cell {ANCHOR_ID} not found")

    specs = build_all()

    def make_cell(src, outputs):
        return {
            "cell_type": "code",
            "execution_count": 1,
            "id": uuid.uuid4().hex[:8],
            "metadata": {},
            "outputs": outputs,
            "source": src.splitlines(keepends=True),
        }

    new_cells = []
    # Tables first, in order 15e–15h
    new_cells.append(make_cell(*specs["sweep"]))
    new_cells.append(make_cell(*specs["per_cluster"]))
    new_cells.append(make_cell(*specs["stab"]))
    new_cells.append(make_cell(*specs["conc"]))
    # Figures 15i–15n
    for tag, src, outputs in specs["figures"]:
        new_cells.append(make_cell(src, outputs))

    for offset, cell in enumerate(new_cells):
        nb["cells"].insert(anchor + 1 + offset, cell)

    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Inserted {len(new_cells)} cells after position {anchor}")
    print(f"New IDs: {[c['id'] for c in new_cells]}")
    print(f"Backup: {BACKUP}")


if __name__ == "__main__":
    main()
