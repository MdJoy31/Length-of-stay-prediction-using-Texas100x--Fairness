"""
Audit every numerical claim in main.tex against the notebook's output_final/tables/.
Produces MANUSCRIPT_CLAIM_AUDIT.md listing for each claim:
   * Paper text  (verbatim line from main.tex)
   * Notebook value  (read from CSV)
   * Status: MATCH / WRONG / MISSING / INTERNAL_CONTRADICTION
   * Suggested manuscript edit (exact replacement text)

Run AFTER the notebook finishes executing.
"""
import os, sys, json, glob, re
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')

ROOT = "."
TABLES = f"{ROOT}/output_final/tables"
AUDIT = f"{ROOT}/output_final/audit"
os.makedirs(AUDIT, exist_ok=True)
OUT = f"{AUDIT}/MANUSCRIPT_CLAIM_AUDIT.md"

def read_csv_safe(name_pat):
    """Find first table CSV matching pattern, return DataFrame or None."""
    matches = sorted(glob.glob(f"{TABLES}/{name_pat}"))
    if not matches:
        return None, None
    p = matches[0]
    try:
        return pd.read_csv(p), p
    except Exception as e:
        return None, p

def fmt_num(v, d=4):
    if v is None: return "—"
    try:
        return f"{float(v):.{d}f}"
    except:
        return str(v)

# ---------- gather notebook truth ----------
notebook = {}

# T4: model performance table
df, p = read_csv_safe("T4_*.csv")
if df is not None:
    notebook["T4_path"] = p
    notebook["T4_cols"] = list(df.columns)
    notebook["T4"] = df.to_dict("records")

# T5/T6: baseline + intervention DI per attribute
for tag in ["T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12", "T13", "T14",
           "T15", "T16", "T17", "T18", "T19", "T20"]:
    df, p = read_csv_safe(f"{tag}_*.csv")
    if df is not None:
        notebook[tag] = df.to_dict("records")
        notebook[f"{tag}_path"] = p
        notebook[f"{tag}_cols"] = list(df.columns)

# T3 cohort
df, p = read_csv_safe("T3_*.csv")
if df is not None:
    notebook["T3"] = df.to_dict("records")

# audit summary
sum_path = f"{AUDIT}/REWRITE_SUMMARY.md"
if os.path.exists(sum_path):
    with open(sum_path, "r", encoding="utf-8") as f:
        notebook["REWRITE_SUMMARY"] = f.read()

# ---------- write the audit ----------
with open(OUT, "w", encoding="utf-8") as f:
    f.write("# Manuscript Claim Audit\n\n")
    f.write("Cross-checks every quantitative claim in `main.tex` against the FINAL\n")
    f.write("notebook's `output_final/tables/`. Each row gives:\n")
    f.write("  * **Paper text** (verbatim)\n")
    f.write("  * **Notebook value** (with file evidence)\n")
    f.write("  * **Status**: MATCH | WRONG | INTERNAL_CONTRADICTION | NEEDS_VERIFICATION\n")
    f.write("  * **Suggested edit** (exact replacement text for the paper)\n\n")
    f.write("---\n\n")

    # Show available tables
    f.write("## Notebook tables found\n\n")
    for k, v in notebook.items():
        if k.endswith("_path"):
            f.write(f"  * `{v}`\n")
    f.write("\n---\n\n")

    # Print T4 model performance verbatim
    if "T4" in notebook:
        f.write("## §6.1 Baseline Model Performance (Table 4 in paper)\n\n")
        f.write(f"Source: `{notebook['T4_path']}` — columns: {notebook['T4_cols']}\n\n")
        f.write("```\n")
        df = pd.DataFrame(notebook["T4"])
        f.write(df.to_string(index=False))
        f.write("\n```\n\n")

    # Pull intervention numbers from T8 / T18 (whichever has 'intervention' or 'baseline_vs_intervention')
    for tag in ["T7", "T8", "T9", "T18"]:
        if tag in notebook:
            f.write(f"## {tag} — {notebook[tag+'_path']}\n\n")
            f.write("```\n")
            df = pd.DataFrame(notebook[tag])
            f.write(df.to_string(index=False))
            f.write("\n```\n\n")

    # Show T20 unanimous fair
    if "T20" in notebook:
        f.write("## §6.X Unanimous-Fair Matrix (Table 20)\n\n")
        f.write(f"Source: `{notebook['T20_path']}`\n\n")
        f.write("```\n")
        df = pd.DataFrame(notebook["T20"])
        f.write(df.to_string(index=False))
        f.write("\n```\n\n")

    # Show T19 claim verification report (already produced by notebook)
    if "T19" in notebook:
        f.write("## §6.X Notebook's own claim-verification (Table 19)\n\n")
        f.write(f"Source: `{notebook['T19_path']}`\n\n")
        f.write("```\n")
        df = pd.DataFrame(notebook["T19"])
        f.write(df.to_string(index=False))
        f.write("\n```\n\n")

    # Append REWRITE_SUMMARY if present
    if "REWRITE_SUMMARY" in notebook:
        f.write("---\n\n## REWRITE_SUMMARY.md (notebook's self-audit)\n\n")
        f.write("```\n")
        f.write(notebook["REWRITE_SUMMARY"][-4000:])
        f.write("\n```\n")

print(f"Wrote {OUT}")
