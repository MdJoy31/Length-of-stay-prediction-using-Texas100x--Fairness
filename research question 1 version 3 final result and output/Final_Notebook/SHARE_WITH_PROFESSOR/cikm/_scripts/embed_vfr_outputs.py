"""Embed the executed outputs (styled HTML tables + a summary stdout
block) into Cell 15c of the CIKM notebook so readers see the full
VFR before/after analysis without having to run anything.
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

VFR_CELL_ID = "04828feb"

METRICS = ["DI", "SPD", "EOPP", "EOD", "TI", "PP", "CAL"]
ATTRS = ["RACE", "SEX", "ETHNICITY", "AGE_GROUP"]
THRESHOLDS = {"DI": 0.80, "SPD": 0.10, "EOPP": 0.10,
              "EOD": 0.10, "TI": 0.10, "PP": 0.10, "CAL": 0.05}


def is_fair(metric, value):
    if pd.isna(value):
        return False
    return (value >= THRESHOLDS["DI"]) if metric == "DI" \
        else (abs(value) < THRESHOLDS[metric])


def reliability_class(vfr_pct):
    return "High" if vfr_pct < 10.0 else ("Moderate" if vfr_pct < 30.0 else "Unstable")


def compute_vfr():
    std = pd.read_csv(TABLES_DIR / "cikm_cross_site_portability.csv")
    fair = pd.read_csv(TABLES_DIR / "cikm_cross_site_portability_FAIR.csv")
    rows = []
    for attr in ATTRS:
        for m in METRICS:
            col = f"{m}_{attr}"
            sv = std[col].dropna().tolist()
            fv = fair[col].dropna().tolist()
            N = min(len(sv), len(fv))
            if N == 0:
                continue
            sm = float(np.mean(sv))
            fm = float(np.mean(fv))
            spk = sum(is_fair(m, v) for v in sv)
            fpk = sum(is_fair(m, v) for v in fv)
            svm = is_fair(m, sm)
            fvm = is_fair(m, fm)
            svfr = 100 * sum(1 for v in sv if is_fair(m, v) != svm) / N
            fvfr = 100 * sum(1 for v in fv if is_fair(m, v) != fvm) / N
            p2p = p2f = f2p = f2f = 0
            for a, b in zip(sv[:N], fv[:N]):
                sa = is_fair(m, a)
                sb = is_fair(m, b)
                if sa and sb: p2p += 1
                elif sa and not sb: p2f += 1
                elif not sa and sb: f2p += 1
                else: f2f += 1
            rows.append({
                "Attribute": attr, "Metric": m, "Threshold": THRESHOLDS[m],
                "Std_Mean": round(sm, 4), "Fair_Mean": round(fm, 4),
                "Delta_Mean": round(fm - sm, 4),
                "Std_Verdict_at_mean": "Pass" if svm else "Fail",
                "Fair_Verdict_at_mean": "Pass" if fvm else "Fail",
                "Std_Pass_k/N": f"{spk}/{N}", "Fair_Pass_k/N": f"{fpk}/{N}",
                "Std_VFR_pct": round(svfr, 1), "Fair_VFR_pct": round(fvfr, 1),
                "Delta_VFR": round(fvfr - svfr, 1),
                "Std_Reliability": reliability_class(svfr),
                "Fair_Reliability": reliability_class(fvfr),
                "P->P": p2p, "P->F": p2f, "F->P": f2p, "F->F": f2f,
            })
    return pd.DataFrame(rows)


def render_vfr_html(vfr):
    # Produce a hand-crafted HTML with inline styles so it renders
    # consistently in JupyterLab and on GitHub.
    head = """
<style>
.vfr-tab { border-collapse: collapse; font-family: Arial, sans-serif;
           font-size: 11px; margin: 6px 0 18px 0; }
.vfr-tab th { background: #1f3d7a; color: #fff; padding: 5px 8px;
              text-align: center; border: 1px solid #4c8fc9; font-weight: 600; }
.vfr-tab td { padding: 4px 7px; border: 1px solid #dbe7f5;
              text-align: center; color: #222; }
.vfr-tab tr:nth-child(even) td { background: #f4f8fc; }
.vfr-attr { background: #1f3d7a !important; color: #fff !important;
            font-weight: 600; }
.vfr-pass { background: #1f3d7a; color: #fff; font-weight: 600; }
.vfr-fail { background: #e2eefb; color: #1f3d7a; }
.vfr-rel-high { background: #1f3d7a; color: #fff; font-weight: 600; }
.vfr-rel-mod  { background: #4c8fc9; color: #fff; font-weight: 600; }
.vfr-rel-uns  { background: #b0413e; color: #fff; font-weight: 600; }
.vfr-delta-pos { color: #1f3d7a; font-weight: 700; }
.vfr-delta-neg { color: #b0413e; font-weight: 700; }
</style>
"""

    def verdict_class(v):
        return "vfr-pass" if v == "Pass" else "vfr-fail"

    def rel_class(v):
        return {"High": "vfr-rel-high", "Moderate": "vfr-rel-mod",
                "Unstable": "vfr-rel-uns"}.get(v, "")

    cols = [
        "Attribute", "Metric", "Threshold",
        "Std_Mean", "Fair_Mean", "Delta_Mean",
        "Std_Verdict_at_mean", "Fair_Verdict_at_mean",
        "Std_Pass_k/N", "Fair_Pass_k/N",
        "Std_VFR_pct", "Fair_VFR_pct", "Delta_VFR",
        "Std_Reliability", "Fair_Reliability",
        "P->P", "P->F", "F->P", "F->F",
    ]
    header_labels = [c.replace("_", " ").replace("/N", "/N") for c in cols]

    rows_html = []
    rows_html.append('<table class="vfr-tab">')
    rows_html.append("<thead><tr>"
                     + "".join(f"<th>{html.escape(c)}</th>" for c in header_labels)
                     + "</tr></thead><tbody>")
    for _, r in vfr.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c == "Attribute":
                cells.append(f'<td class="vfr-attr">{html.escape(str(v))}</td>')
            elif c in ("Std_Verdict_at_mean", "Fair_Verdict_at_mean"):
                cls = verdict_class(v)
                cells.append(f'<td class="{cls}">{html.escape(str(v))}</td>')
            elif c in ("Std_Reliability", "Fair_Reliability"):
                cls = rel_class(v)
                cells.append(f'<td class="{cls}">{html.escape(str(v))}</td>')
            elif c in ("Std_Mean", "Fair_Mean"):
                cells.append(f"<td>{float(v):.3f}</td>")
            elif c == "Delta_Mean":
                cls = "vfr-delta-pos" if float(v) > 0 else ("vfr-delta-neg" if float(v) < 0 else "")
                cells.append(f'<td class="{cls}">{float(v):+.3f}</td>')
            elif c == "Delta_VFR":
                cls = "vfr-delta-neg" if float(v) > 0 else ("vfr-delta-pos" if float(v) < 0 else "")
                # Here negative Δ VFR means better reliability -> style blue
                cells.append(f'<td class="{cls}">{float(v):+.1f}</td>')
            elif c in ("Std_VFR_pct", "Fair_VFR_pct"):
                cells.append(f"<td>{float(v):.1f}</td>")
            elif c == "Threshold":
                cells.append(f"<td>{float(v):.2f}</td>")
            else:
                cells.append(f"<td>{html.escape(str(v))}</td>")
        rows_html.append("<tr>" + "".join(cells) + "</tr>")
    rows_html.append("</tbody></table>")
    return head + "\n".join(rows_html)


def render_uplift_html(vfr):
    rows = []
    for attr in ATTRS:
        sub = vfr[vfr["Attribute"] == attr]
        std_n = int((sub["Std_Verdict_at_mean"] == "Pass").sum())
        fair_n = int((sub["Fair_Verdict_at_mean"] == "Pass").sum())
        std_vfr = float(sub["Std_VFR_pct"].mean())
        fair_vfr = float(sub["Fair_VFR_pct"].mean())
        rows.append((attr, std_n, fair_n, std_vfr, fair_vfr, fair_vfr - std_vfr))
    html_rows = ["<table class='vfr-tab'>"]
    html_rows.append("<thead><tr>"
                     "<th>Attribute</th><th>Std N_fair/7</th><th>Fair N_fair/7</th>"
                     "<th>Std avg VFR %</th><th>Fair avg VFR %</th>"
                     "<th>Δ avg VFR (pp)</th></tr></thead><tbody>")
    for attr, sn, fn, sv, fv, dv in rows:
        dcls = "vfr-delta-pos" if dv < 0 else ("vfr-delta-neg" if dv > 0 else "")
        html_rows.append(
            f'<tr><td class="vfr-attr">{attr}</td>'
            f'<td>{sn}/7</td><td>{fn}/7</td>'
            f'<td>{sv:.1f}</td><td>{fv:.1f}</td>'
            f'<td class="{dcls}">{dv:+.1f}</td></tr>'
        )
    html_rows.append("</tbody></table>")
    return "\n".join(html_rows)


def build_outputs(vfr):
    vfr_html = "<h4>Table 6h: <b>Comprehensive VFR before/after</b> " \
               "(Standard → Fair, 20 hospital clusters, 7 metrics × 4 attributes)</h4>" \
               + render_vfr_html(vfr)
    uplift_html = "<h4>Attribute-level reliability uplift (cluster-mean verdict + VFR)</h4>" \
                  + render_uplift_html(vfr)

    std_fair_total = int((vfr["Std_Verdict_at_mean"] == "Pass").sum())
    fair_fair_total = int((vfr["Fair_Verdict_at_mean"] == "Pass").sum())
    flipped_up = int(vfr["F->P"].sum())
    flipped_down = int(vfr["P->F"].sum())
    summary_text = (
        f"Fair pipeline lifts {std_fair_total}/28 → {fair_fair_total}/28 fair verdicts "
        f"at cluster mean;\n"
        f"across 20×28 = 560 (cluster, metric, attribute) cells,\n"
        f"  Fail → Pass transitions: {flipped_up}\n"
        f"  Pass → Fail transitions: {flipped_down}  "
        f"(net gain = {flipped_up - flipped_down})\n"
    )

    return [
        {"output_type": "display_data",
         "data": {"text/html": [vfr_html],
                  "text/plain": ["<Table 6h styled HTML>"]},
         "metadata": {}},
        {"output_type": "display_data",
         "data": {"text/html": [uplift_html],
                  "text/plain": ["<Attribute uplift styled HTML>"]},
         "metadata": {}},
        {"output_type": "stream", "name": "stdout",
         "text": [summary_text]},
    ]


def main():
    vfr = compute_vfr()
    vfr.to_csv(TABLES_DIR / "Table6h_VFR_StdVsFair.csv", index=False)
    outs = build_outputs(vfr)
    nb = json.loads(NB.read_text(encoding="utf-8"))
    patched = False
    for c in nb["cells"]:
        if c.get("id") == VFR_CELL_ID:
            c["outputs"] = outs
            c["execution_count"] = 1
            patched = True
            break
    if not patched:
        raise RuntimeError(f"Cell {VFR_CELL_ID} not found")
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Embedded VFR outputs into cell {VFR_CELL_ID}")


if __name__ == "__main__":
    main()
