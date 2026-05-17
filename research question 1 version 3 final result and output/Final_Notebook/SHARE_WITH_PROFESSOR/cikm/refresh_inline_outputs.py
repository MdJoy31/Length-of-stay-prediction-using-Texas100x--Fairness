"""
Refresh inline outputs of the cells that were patched to use canonical
values, without re-running the heavy training cells.

Approach: spin up a fresh Python session, manually load every artefact
from disk (CSVs are already on disk), define the variables those display
cells need (results_df, TABLES_DIR, FairnessCalculator, etc.), then exec
the patched display cells in order to capture their outputs and write
them back into the notebook.
"""
import json, sys, io, base64, traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding='utf-8')

NB_PATH = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\CIKM_2026_LOS_Fairness_13042026.ipynb")
NB_DIR = NB_PATH.parent
sys_path_added = str(NB_DIR)
import os
os.chdir(NB_DIR)

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)
print(f"Loaded {len(nb['cells'])} cells")

# ─── Load disk artefacts ───────────────────────────────────────────
TABLES_DIR = "output/tables"
FIGURES_DIR = "output/figures"
results_df = pd.read_csv("output/tables/Table9_Comprehensive_Accuracy.csv")\
              .rename(columns={"AUC":"AUC"})  # keep as-is
results_df = results_df.sort_values("AUC", ascending=False).reset_index(drop=True)
cs_df_full = pd.read_csv("output/tables/cikm_cross_site_portability.csv") \
            if Path("output/tables/cikm_cross_site_portability.csv").exists() else None

# Provide a minimal FairnessCalculator stand-in (just THRESHOLDS)
class FairnessCalculator:
    THRESHOLDS = {
        "DI":   {"threshold": 0.80, "direction": "above"},
        "SPD":  {"threshold": 0.10, "direction": "below"},
        "EOPP": {"threshold": 0.10, "direction": "below"},
        "EOD":  {"threshold": 0.10, "direction": "below"},
        "TI":   {"threshold": 0.10, "direction": "below"},
        "PP":   {"threshold": 0.10, "direction": "below"},
        "CAL":  {"threshold": 0.05, "direction": "below"},
    }

# Bootstrap dictionary for exec()
ns = dict(
    np=np, pd=pd, plt=plt,
    TABLES_DIR=TABLES_DIR, FIGURES_DIR=FIGURES_DIR,
    results_df=results_df,
    cs_df=cs_df_full,
    FairnessCalculator=FairnessCalculator,
    best_model_name=results_df.iloc[0]["Model"],
    best_acc=float(results_df.iloc[0]["Accuracy"]),
    best_auc=float(results_df.iloc[0]["AUC"]),
)

# Helper: capture rich outputs from a cell exec
from IPython import get_ipython
from IPython.core.interactiveshell import InteractiveShell

shell = InteractiveShell.instance()
shell.user_ns.update(ns)

def run_and_capture(cell_idx, src):
    """Execute src in shell.user_ns, capture stdout + display data."""
    outputs = []
    buf = io.StringIO()
    # Patch IPython's display function to capture output rich data
    captured_displays = []
    from IPython.display import display as _orig_display
    def cap_display(*args, **kwargs):
        for a in args:
            try:
                # Try to get a text/html or text/plain rendering
                if hasattr(a, "_repr_html_"):
                    h = a._repr_html_()
                    if h: captured_displays.append({"output_type":"display_data",
                                                    "data":{"text/html": h, "text/plain":[str(a)]}, "metadata":{}})
                elif hasattr(a, "_repr_png_"):
                    p = a._repr_png_()
                    if p:
                        b64 = base64.b64encode(p).decode("ascii")
                        captured_displays.append({"output_type":"display_data",
                                                  "data":{"image/png":b64,"text/plain":[str(a)]}, "metadata":{}})
                else:
                    captured_displays.append({"output_type":"display_data",
                                              "data":{"text/plain":[str(a)]}, "metadata":{}})
            except Exception:
                pass
    import IPython.display as _ipd
    _ipd.display = cap_display

    try:
        with redirect_stdout(buf):
            exec(src, shell.user_ns)
    except Exception as e:
        traceback.print_exc()
        return [{"output_type":"stream","name":"stderr","text":[traceback.format_exc()]}]
    finally:
        _ipd.display = _orig_display

    text = buf.getvalue()
    if text:
        outputs.append({"output_type":"stream","name":"stdout","text":text.splitlines(keepends=True)})
    outputs.extend(captured_displays)
    # Also flush any matplotlib figures
    figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        png_buf = io.BytesIO()
        fig.savefig(png_buf, format="png", bbox_inches="tight", facecolor="white")
        b64 = base64.b64encode(png_buf.getvalue()).decode("ascii")
        outputs.append({"output_type":"display_data",
                        "data":{"image/png":b64,"text/plain":["<Figure>"]}, "metadata":{}})
        plt.close(fig)
    return outputs


# Execute cells 35, 37, 41 in order
for cell_idx in [35, 37, 41]:
    src = "".join(nb["cells"][cell_idx].get("source", []))
    print(f"\n=== Executing cell {cell_idx} (len={len(src)}) ===")
    outs = run_and_capture(cell_idx, src)
    nb["cells"][cell_idx]["outputs"] = outs
    nb["cells"][cell_idx]["execution_count"] = cell_idx
    n_streams = sum(1 for o in outs if o.get("output_type")=="stream")
    n_disp = sum(1 for o in outs if o.get("output_type")=="display_data")
    print(f"  -> {n_streams} stream + {n_disp} display outputs captured")

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"\nWrote refreshed notebook to {NB_PATH}")
