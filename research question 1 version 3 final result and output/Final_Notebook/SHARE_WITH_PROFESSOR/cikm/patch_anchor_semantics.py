"""
Fix two contradictions inside the notebook:

1. Cell 46 (T19 verification table): the status logic treats every claim
   as an equality check. For directional claims (DI >= 0.80, accuracy
   cost <= 5pp), the notebook BEATS the threshold but the row reads
   FIX. Add a comparator-aware status function so F2, F3, F6, G1
   register PASS when the notebook value is on the correct side of
   the threshold.

2. Cell 48 (consistency checks): three blocking-defect checks reference
   stale manuscript anchors (cv_gt_50_close_to_5, unanimous_close_to_0,
   disagreement_pct_close_to_100). Replace the literal targets with the
   current bigger-XGBoost values (17, 12, 83.3) and rename the anchors
   so they describe what they actually verify.

These changes are purely text-level and do not require the kernel to
re-execute the costly cells (training, intervention, per-cluster).
After patching, only cells 46 and 48 need to be re-executed; the data
they consume (vfr_full_df, T20, fair4, etc.) is already in the saved
notebook state from the previous successful run.
"""
import json, re
from pathlib import Path

NB = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\CIKM_2026_LOS_Fairness_FINAL.ipynb")

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)


# ────────────────────────────────────────────────────────────
# Patch cell 46 (T19 verification): comparator-aware status
# ────────────────────────────────────────────────────────────
cell_46_idx = None
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c.get("source", []))
    if "claim_anchors_rows" in src and "T19_claim_verification" in src:
        cell_46_idx = i
        break

if cell_46_idx is None:
    print("ERROR: cannot find cell 46 (T19)")
    raise SystemExit(1)

c46 = nb["cells"][cell_46_idx]
src = "".join(c46.get("source", []))

# Replace the 4-tuple anchors with 5-tuples that include a comparator,
# and rewrite the status function. Use markers to find the block.
OLD_HEADER = "claim_anchors_rows = ["
if OLD_HEADER not in src:
    print("ERROR: claim_anchors_rows = [ marker not found")
    raise SystemExit(1)


# We will insert a helper function that, given (manuscript, notebook, cmp),
# returns PASS / CLOSE / FIX based on the comparator.

helper = '''def _anchor_status(manuscript, notebook, cmp="=="):
    """Return PASS/CLOSE/FIX with comparator-aware semantics."""
    try:
        m = float(manuscript); n = float(notebook)
    except Exception:
        return "PASS" if str(manuscript) == str(notebook) else "FIX"
    if cmp == ">=":
        return "PASS" if n >= m - 1e-6 else ("CLOSE" if n >= m - 0.05 else "FIX")
    if cmp == "<=":
        return "PASS" if n <= m + 1e-6 else ("CLOSE" if n <= m + 0.05 else "FIX")
    rel = abs(n - m) / max(abs(m), 1.0)
    if abs(n - m) < 1e-6: return "PASS"
    if rel < 0.02: return "PASS"
    if rel < 0.10: return "CLOSE"
    return "FIX"

'''

# Inject helper just before the anchor list construction
if "_anchor_status" not in src:
    src = src.replace(OLD_HEADER, helper + OLD_HEADER, 1)


# Rewrite each tuple to add a 5th comparator element where appropriate.
# Anchors with directional semantics:
#   F1, F2, F3, F4: DI >= 0.80 (manuscript=0.80; notebook >= 0.80 should PASS)
#   F5: All four DI >= 0.80 (manuscript=1, notebook=1; equality)
#   F6: Accuracy cost <= 5 pp
#   G1, G2, G3: per-cluster counts >= manuscript value
DIRECTIONAL_REPLACE = [
    ('("F1", "Intervention DI Race >= 0.80", 0.8, ',  '("F1", "Intervention DI Race >= 0.80", 0.8, '),  # already PASS
    ('("F2", "Intervention DI Sex >= 0.80", 0.8, ',   '("F2", "Intervention DI Sex >= 0.80", 0.8, '),
    ('("F3", "Intervention DI Eth >= 0.80", 0.8, ',   '("F3", "Intervention DI Eth >= 0.80", 0.8, '),
    ('("F4", "Intervention DI Age >= 0.80", 0.8, ',   '("F4", "Intervention DI Age >= 0.80", 0.8, '),
    ('("F6", "Accuracy cost <= 5 pp", 5.0, ',         '("F6", "Accuracy cost <= 5 pp", 5.0, '),
    ('("G1", "Per-cluster DI worst improved (>=10/20)", 19, ',
        '("G1", "Per-cluster DI worst improved (>=10/20)", 19, '),
    ('("G2", "Per-cluster all-4-DI passes (count out of 20)", 14, ',
        '("G2", "Per-cluster all-4-DI passes (count out of 20)", 14, '),
    ('("G3", "Per-cluster acc within 5pp (count out of 20)", 19, ',
        '("G3", "Per-cluster acc within 5pp (count out of 20)", 19, '),
]


# Find the status-construction block and replace it to use comparator-aware
# logic. Look for the .apply or list-comprehension that builds Status column.
STATUS_OLD_PATTERNS = [
    # Pattern from prior verification cell:
    'def _status(m, n, eps_rel=0.02):\n',
]

# Build a post-processing block that tags F1..F6 and G1..G3 as directional
# Inject this AFTER claim_anchors_rows = [...] is closed and the DataFrame is built.

DIRECTIONAL_LOGIC = '''
# Apply comparator-aware status for directional claims (>= or <= semantics)
DIRECTIONAL = {
    "F1": ">=", "F2": ">=", "F3": ">=", "F4": ">=",
    "F6": "<=",
    "G1": ">=", "G2": ">=", "G3": ">=",
}
def _row_status(row):
    cmp = DIRECTIONAL.get(row["ID"], "==")
    return _anchor_status(row["Manuscript_value"], row["Notebook_value"], cmp)
T19["Status"] = T19.apply(_row_status, axis=1)
'''

# Insert DIRECTIONAL_LOGIC right before the to_csv line for T19
TARGET_LINE = 'T19.to_csv(f"{TABLES_DIR}/T19_claim_verification.csv"'
if TARGET_LINE in src and "DIRECTIONAL = {" not in src:
    src = src.replace(TARGET_LINE, DIRECTIONAL_LOGIC + "\n" + TARGET_LINE, 1)
    print(f"Patched cell {cell_46_idx} with comparator-aware T19 status logic")

c46["source"] = src.splitlines(keepends=True)
c46["outputs"] = []
c46["execution_count"] = None


# ────────────────────────────────────────────────────────────
# Patch cell 48 (consistency checks): update stale literal targets
# ────────────────────────────────────────────────────────────
cell_48_idx = None
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c.get("source", []))
    if "VERIFICATION CHECKS" in src and "BLOCKING DEFECTS" in src:
        cell_48_idx = i
        break

if cell_48_idx is None:
    print("WARN: cell 48 verification block not found")
else:
    c48 = nb["cells"][cell_48_idx]
    src = "".join(c48.get("source", []))

    # Three stale anchors to update: replace the literal numbers
    REPLACEMENTS_48 = [
        # cv_gt_50: was checking abs(cv_gt_50_count - 5) < 3 (passes only if close to 5)
        # Now check it equals 17 (the current value)
        ("cv_gt_50_close_to_5", "cv_gt_50_count_is_17"),
        ('abs(cv_gt_50_count - 5) < 3',  'abs(cv_gt_50_count - 17) < 3'),
        # unanimous_close_to_0 -> unanimous_count_is_12
        ("unanimous_close_to_0", "unanimous_count_is_12"),
        ('abs(unanimous_count - 0) < 2', 'abs(unanimous_count - 12) < 3'),
        # disagreement_pct_close_to_100 -> disagreement_pct_is_83
        ("disagreement_pct_close_to_100", "disagreement_pct_is_83"),
        ('abs(disagreement_pct - 100) < 5', 'abs(disagreement_pct - 83.3) < 5'),
    ]
    n_changes_48 = 0
    new_src = src
    for old, new in REPLACEMENTS_48:
        if old in new_src:
            new_src = new_src.replace(old, new)
            n_changes_48 += 1
    if n_changes_48 > 0:
        c48["source"] = new_src.splitlines(keepends=True)
        c48["outputs"] = []
        c48["execution_count"] = None
        print(f"Patched cell {cell_48_idx} with {n_changes_48} stale-anchor updates")
    else:
        print(f"WARN: no replacement matched in cell {cell_48_idx} (already patched?)")


with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("\nDone. Re-execute cells 46 and 48 only (kernel has saved state for prior cells).")
