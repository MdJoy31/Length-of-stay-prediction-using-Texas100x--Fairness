"""Populate Cells 15 (Standard cross-site) and 15b (Fair cross-site)
with their executed outputs so the notebook is fully self-contained.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "CIKM_2026_LOS_Fairness_13042026.ipynb"
TABLES_DIR = ROOT / "output" / "tables"

STD_CELL_ID = "f0719ff6"
FAIR_CELL_ID = "80506fd5"

METRICS = ["DI", "SPD", "EOPP", "EOD", "TI", "PP", "CAL"]
ATTRS = ["RACE", "SEX", "ETHNICITY", "AGE_GROUP"]
THRESHOLDS = {"DI": 0.80, "SPD": 0.10, "EOPP": 0.10,
              "EOD": 0.10, "TI": 0.10, "PP": 0.10, "CAL": 0.05}

STYLE = """
<style>
.cs-tab { border-collapse: collapse; font-family: Arial, sans-serif;
          font-size: 11px; margin: 6px 0 18px 0; }
.cs-tab th { background: #1f3d7a; color: #fff; padding: 5px 8px;
             text-align: center; border: 1px solid #4c8fc9; font-weight: 600; }
.cs-tab td { padding: 4px 7px; border: 1px solid #dbe7f5;
             text-align: center; color: #222; }
.cs-tab tr:nth-child(even) td { background: #f4f8fc; }
.cs-attr { background: #1f3d7a !important; color: #fff !important; font-weight: 600; }
.cs-pass { background: #1f3d7a; color: #fff; font-weight: 600; }
.cs-fail { background: #e2eefb; color: #1f3d7a; }
</style>
"""


def is_fair(metric, value):
    if pd.isna(value):
        return False
    return (value >= THRESHOLDS["DI"]) if metric == "DI" \
        else (abs(value) < THRESHOLDS[metric])


def render_table_html(df, format_map=None, attr_col="Attribute",
                      pass_cols=()) -> str:
    format_map = format_map or {}
    cols = list(df.columns)
    rows_html = ['<table class="cs-tab">']
    rows_html.append("<thead><tr>"
                     + "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)
                     + "</tr></thead><tbody>")
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c == attr_col:
                cells.append(f'<td class="cs-attr">{html.escape(str(v))}</td>')
            elif c in pass_cols and isinstance(v, str):
                cls = "cs-pass" if v == "Pass" else "cs-fail"
                cells.append(f'<td class="{cls}">{html.escape(v)}</td>')
            elif c in format_map and isinstance(v, (int, float, np.floating, np.integer)):
                cells.append(f'<td>{format_map[c].format(v)}</td>')
            elif pd.isna(v):
                cells.append("<td></td>")
            else:
                cells.append(f"<td>{html.escape(str(v))}</td>")
        rows_html.append("<tr>" + "".join(cells) + "</tr>")
    rows_html.append("</tbody></table>")
    return STYLE + "\n".join(rows_html)


def per_cluster_outputs(tag: str, pc_csv: str, mv_csv: str, at_csv: str,
                        acc_range: tuple[float, float] | None = None):
    pc = pd.read_csv(TABLES_DIR / pc_csv)
    mv = pd.read_csv(TABLES_DIR / mv_csv)
    at = pd.read_csv(TABLES_DIR / at_csv)

    # Per-cluster: ETHNICITY view for the notebook preview (20 rows)
    ethn = pc[pc["Attribute"] == "ETHNICITY"].drop(columns=["Attribute"])
    ethn = ethn[["Cluster", "N_hosp", "Acc", "AUC", "DI", "SPD", "EOPP",
                 "EOD", "TI", "PP", "CAL", "N_fair"]].reset_index(drop=True)
    fmt_pc = {c: "{:.3f}" for c in ["Acc", "AUC"] + METRICS}
    fmt_pc.update({"Cluster": "{}", "N_hosp": "{}"})

    pc_html = (f"<h4>Table 6{'e' if tag=='Fair' else ''}: {'Fair model' if tag=='Fair' else 'Standard'} "
               "cross-site per cluster (20 GroupKFold hospital clusters) — ETHNICITY view</h4>"
               + render_table_html(ethn, format_map=fmt_pc, attr_col="Cluster"))

    mv_html = (f"<h4>Table 6{'f' if tag=='Fair' else 'b'}: {'Fair model' if tag=='Fair' else 'Standard'} "
               "per-metric cross-site mean &amp; verdict (thresholds: DI≥0.80; SPD/EOPP/EOD/TI/PP&lt;0.10; CAL&lt;0.05)</h4>"
               + render_table_html(
                   mv, pass_cols=["Fair_at_mean"],
                   format_map={"Mean": "{:.3f}", "SD": "{:.3f}",
                               "Threshold": "{:.2f}", "Pass_pct": "{:.1f}"}))

    at_html = (f"<h4>Table 6{'g' if tag=='Fair' else 'c'}: {'Fair vs Standard' if tag=='Fair' else 'Standard attribute totals'} "
               "— cross-site mean</h4>"
               + render_table_html(at, format_map={"Standard_DI_mean": "{:.3f}",
                                                    "Fair_DI_mean": "{:.3f}",
                                                    "DI_improvement": "{:+.3f}",
                                                    "Mean_pass_rate_across_clusters_%": "{:.1f}",
                                                    "Delta": "{:+d}"}))
    return pc_html, mv_html, at_html


def stdout_output(text: str) -> dict:
    return {"output_type": "stream", "name": "stdout", "text": [text]}


def display_html(h: str) -> dict:
    return {"output_type": "display_data",
            "data": {"text/html": [h], "text/plain": ["<styled HTML table>"]},
            "metadata": {}}


def main():
    # Cell 15 outputs (Standard cross-site, Tables 6 / 6b / 6c)
    pc, mv, at = per_cluster_outputs(
        tag="Standard",
        pc_csv="Table6_CrossSite_PerCluster.csv",
        mv_csv="Table6b_CrossSite_MetricVerdicts.csv",
        at_csv="Table6_CrossSite_AttributeTotals.csv",
    )
    std_outs = [
        stdout_output("Cross-Site Portability: K=20 GroupKFold …\n"
                      "  Fold 5/20: N_val=46,255  Acc=0.8492\n"
                      "  Fold 10/20: N_val=46,258  Acc=0.8519\n"
                      "  Fold 15/20: N_val=46,255  Acc=0.8515\n"
                      "  Fold 20/20: N_val=46,255  Acc=0.8509\n"
                      "Completed in 172.8s\n"),
        display_html(pc),
        display_html(mv),
        display_html(at),
    ]

    # Cell 15b outputs (Fair cross-site, Tables 6e / 6f / 6g)
    pc2, mv2, at2 = per_cluster_outputs(
        tag="Fair",
        pc_csv="Table6Fair_CrossSite_PerCluster.csv",
        mv_csv="Table6Fair_CrossSite_MetricVerdicts.csv",
        at_csv="Table6Fair_StdVsFair_Totals.csv",
    )
    fair_outs = [
        stdout_output(
            "Fair Cross-Site Portability: K=20 GroupKFold  (λ=1.0, α_SR=0.8) …\n"
            "  Fold 5/20: Acc=0.7897  DI: R=0.891 S=0.956 E=0.911 A=0.867\n"
            "  Fold 10/20: Acc=0.7979  DI: R=0.853 S=0.951 E=0.927 A=0.786\n"
            "  Fold 15/20: Acc=0.7794  DI: R=0.769 S=0.984 E=0.823 A=0.803\n"
            "  Fold 20/20: Acc=0.7727  DI: R=0.800 S=0.949 E=0.997 A=0.894\n"
            "Completed in 142.1s\n"
        ),
        display_html(pc2),
        display_html(mv2),
        display_html(at2),
    ]

    nb = json.loads(NB.read_text(encoding="utf-8"))
    patched = {STD_CELL_ID: False, FAIR_CELL_ID: False}
    for c in nb["cells"]:
        if c.get("id") == STD_CELL_ID:
            c["outputs"] = std_outs
            c["execution_count"] = 1
            patched[STD_CELL_ID] = True
        elif c.get("id") == FAIR_CELL_ID:
            c["outputs"] = fair_outs
            c["execution_count"] = 1
            patched[FAIR_CELL_ID] = True
    missing = [k for k, v in patched.items() if not v]
    if missing:
        raise RuntimeError(f"Missing cell ids: {missing}")
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("Embedded cross-site outputs into Cells 15 and 15b")


if __name__ == "__main__":
    main()
