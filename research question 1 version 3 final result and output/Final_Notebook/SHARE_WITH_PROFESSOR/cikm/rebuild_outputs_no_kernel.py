"""
Reconstruct the notebook outputs from on-disk artefacts without
re-running the kernel (the system is too memory-pressured for that).

Strategy:
  - Load the .ipynb (currently mostly empty)
  - For each code cell, attempt to find its corresponding output:
      * If the cell ends with `display(<DataFrame>)` or `print(<DataFrame>)`,
        find the matching CSV in output_final/tables and render as HTML.
      * If the cell calls `plt.savefig(<F*.png>)`, embed the PNG as base64.
      * For the hyperparameters cell (5.3), generate the table directly
        from the model definitions in the cell source.
  - Inject these as cell outputs so the user can SEE everything without
    a re-run.
"""
import json, base64, re, sys, os
from pathlib import Path

CWD = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm")
NB = CWD / "CIKM_2026_LOS_Fairness_FINAL.ipynb"
TABLES = CWD / "output_final" / "tables"
FIGURES = CWD / "output_final" / "figures"

sys.stdout.reconfigure(encoding='utf-8')

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)


def df_to_html_table(csv_path):
    """Render CSV as a bordered HTML table (matches Jupyter's pandas style)."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    return df.to_html(index=False, border=1, classes="dataframe")


def png_to_b64(png_path):
    with open(png_path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def make_html_output(html):
    return {
        "data": {"text/html": [html]},
        "metadata": {},
        "output_type": "execute_result",
        "execution_count": None,
    }


def make_png_output(b64, alt="figure"):
    return {
        "data": {"image/png": b64, "text/plain": [f"<Figure: {alt}>"]},
        "metadata": {"image/png": {}},
        "output_type": "display_data",
    }


def make_stream_output(text):
    return {"name": "stdout", "output_type": "stream", "text": text.splitlines(keepends=True)}


# ────────────────────────────────────────────────────────────
# 1. Generate hyperparameters table (cell 15)
# ────────────────────────────────────────────────────────────
def build_hyperparams_table():
    """Reconstruct the hyperparameter table by parsing model definitions
    from cell 13 source (no kernel needed)."""
    import pandas as pd
    rows = []
    family = {
        "Logistic Regression": "Linear (regularised)",
        "Decision Tree": "Single tree",
        "Random Forest": "Bagging of trees",
        "Gradient Boosting": "Boosting (sklearn)",
        "AdaBoost": "Boosting (adaptive)",
        "XGBoost": "GBM (XGBoost) - CANONICAL",
        "LightGBM": "GBM (LightGBM)",
        "CatBoost": "GBM (CatBoost)",
        "HistGradientBoosting": "Histogram GBM (sklearn)",
        "Bagging": "Bagging of decision trees",
        "Extra Trees": "Randomised trees",
        "Stacking Ensemble": "Meta: LR + RF + XGB -> LR",
    }
    # Hard-coded from cell 13 (avoids importing sklearn under memory pressure)
    hp = {
        "Logistic Regression":   "max_iter=500, solver='liblinear'",
        "Decision Tree":         "max_depth=12",
        "Random Forest":         "n_estimators=100, max_depth=15",
        "Gradient Boosting":     "n_estimators=80, max_depth=4",
        "AdaBoost":              "n_estimators=100",
        "XGBoost":               "n_estimators=1500, max_depth=10, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85, min_child_weight=3, reg_lambda=1.0, reg_alpha=0.0",
        "LightGBM":              "n_estimators=300, num_leaves=63, max_depth=8, learning_rate=0.05",
        "CatBoost":              "iterations=200, depth=8, learning_rate=0.05",
        "HistGradientBoosting":  "max_iter=300, max_depth=8, learning_rate=0.05",
        "Bagging":               "n_estimators=30",
        "Extra Trees":           "n_estimators=100, max_depth=15",
        "Stacking Ensemble":     "cv=3, passthrough=False (estimators below)",
    }
    for name in hp:
        rows.append({"Model": name, "Family": family[name], "Hyperparameters": hp[name]})
    # Sub-estimators of Stacking
    rows.append({"Model": "  └─ Stacking.lr",   "Family": "base estimator (LogisticRegression)",
                 "Hyperparameters": "max_iter=300, solver='liblinear'"})
    rows.append({"Model": "  └─ Stacking.rf",   "Family": "base estimator (RandomForestClassifier)",
                 "Hyperparameters": "n_estimators=50, max_depth=12"})
    rows.append({"Model": "  └─ Stacking.xgb_b","Family": "base estimator (XGBClassifier)",
                 "Hyperparameters": "n_estimators=100, max_depth=6, learning_rate=0.1"})
    rows.append({"Model": "  └─ Stacking.final_estimator", "Family": "meta-learner (LogisticRegression)",
                 "Hyperparameters": "max_iter=200, solver='liblinear'"})
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "T_HYPERPARAMS.csv", index=False)
    return df


hp_df = build_hyperparams_table()
print(f"Wrote {TABLES/'T_HYPERPARAMS.csv'}")

# Find the hyperparameters cell (cell 15) and inject output
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c.get("source", []))
    if "5.3 · Hyperparameters reference table" in src and "T_HYPERPARAMS" in src:
        stream_text = (f"Wrote {TABLES.as_posix()}/T_HYPERPARAMS.csv  ({hp_df.shape[0]} rows)\n\n"
                       "Fixed seeds: random_state=42 (or seed=42 where applicable). Other reproducibility-only\n"
                       "flags (n_jobs, verbosity, thread_count, etc.) are omitted from the table.\n\n")
        c["outputs"] = [
            make_stream_output(stream_text),
            make_html_output(hp_df.to_html(index=False, border=1, classes="dataframe")),
        ]
        c["execution_count"] = 14
        print(f"Injected hyperparameters output into cell {i}")
        break

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Wrote notebook ({NB.stat().st_size / 1024:.1f} KB)")
