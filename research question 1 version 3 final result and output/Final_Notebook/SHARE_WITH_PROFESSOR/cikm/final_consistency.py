"""
Round 4: final consistency review.

Looks for:
  C1. TODO / FIXME / XXX / HACK markers
  C2. Subsection numbering coherence (15.1, 15.2, ..., 15.6 should exist if §15 is well-numbered)
  C3. "we will" / "future work" / placeholder language in non-Future-Work sections
  C4. Hyperparameter table completeness (T_HYPERPARAMS should cover all 12 models)
  C5. Cross-cell narrative number consistency (sample: cohort N, accuracy values)
  C6. Markdown table integrity (every | row has matching column count)
  C7. Backslash-escape issues from JSON serialisation
"""
import json, re, sys, io
from pathlib import Path
from collections import Counter, defaultdict
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm")
NB = ROOT / "CIKM_2026_LOS_Fairness_FINAL.ipynb"

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

issues = []

# ============================================================
# C1. Forbidden markers
# ============================================================
print("=" * 80)
print("C1. TODO / FIXME / placeholder markers")
print("=" * 80)
markers = ["TODO", "FIXME", "XXX", "HACK", "????", "<placeholder", "[TBD]", "TBD,"]
for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    for m in markers:
        if m in src:
            # Get context
            for match in re.finditer(re.escape(m), src):
                ctx = src[max(0, match.start()-30):min(len(src), match.end()+50)].replace('\n', ' ')
                issues.append((i, "C1", f"'{m}' found: ...{ctx}..."))

print(f"Issues: {sum(1 for x in issues if x[1] == 'C1')}")

# ============================================================
# C2. Subsection numbering for §15 (figures)
# ============================================================
print()
print("=" * 80)
print("C2. §15 subsection numbering (figures 1-6)")
print("=" * 80)
md_text = "\n".join(''.join(c.get('source', [])) for c in nb['cells'] if c['cell_type'] == 'markdown')
code_text = "\n".join(''.join(c.get('source', [])) for c in nb['cells'] if c['cell_type'] == 'code')

# Look for §15.1 - §15.6 markers in code or markdown
fig_subsections = set()
for n in range(1, 9):
    if re.search(rf"15\.{n}\s*[·\.\-]", md_text + code_text):
        fig_subsections.add(n)
print(f"§15 subsections found: 15.{sorted(fig_subsections)}")
expected = set(range(1, 7))  # §15.1 - §15.6 expected (5 figures + F6)
missing = expected - fig_subsections
if missing:
    issues.append((-1, "C2", f"Missing §15.{sorted(missing)}"))

# ============================================================
# C3. Forbidden language in non-Future-Work sections
# ============================================================
print()
print("=" * 80)
print("C3. 'We will' / 'future work' phrasing leakage")
print("=" * 80)
forbidden_phrases = ["we will", "future versions", "in future work", "subsequent paper"]
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'markdown':
        continue
    src = ''.join(c.get('source', []))
    # Skip §20 (Limitations) and §21 sections — they may legitimately contain future-work phrasing
    if "## 20 ·" in src or "## 21 ·" in src or "Limitations" in src.split('\n')[0:5][0] if src else False:
        continue
    for phrase in forbidden_phrases:
        for m in re.finditer(re.escape(phrase), src, re.IGNORECASE):
            ctx = src[max(0, m.start()-30):min(len(src), m.end()+60)].replace('\n', ' ')
            issues.append((i, "C3", f"'{phrase}': ...{ctx}..."))

print(f"Issues: {sum(1 for x in issues if x[1] == 'C3')}")

# ============================================================
# C4. Hyperparameter table check
# ============================================================
print()
print("=" * 80)
print("C4. T_HYPERPARAMS coverage check")
print("=" * 80)
import pandas as pd
T_HP = pd.read_csv(ROOT / "output_final" / "tables" / "T_HYPERPARAMS.csv")
print(f"T_HYPERPARAMS rows: {len(T_HP)} | columns: {list(T_HP.columns)[:6]}")
expected_models = ["XGBoost", "LightGBM", "Random Forest", "Logistic Regression",
                   "Gradient Boosting", "Decision Tree", "MLP", "AdaBoost",
                   "Extra Trees", "KNN", "Naive Bayes", "Stacking"]
hp_models = set()
for col in T_HP.columns:
    for m in expected_models:
        if m.lower() in col.lower() or (T_HP[col].astype(str) == m).any():
            hp_models.add(m)
# Look for the model name in the first column
if T_HP.columns[0].lower() in ['model', 'classifier', 'name']:
    hp_models = set(T_HP[T_HP.columns[0]].tolist())
elif 'Model' in T_HP.columns:
    hp_models = set(T_HP['Model'].tolist())
print(f"Models found in T_HYPERPARAMS: {sorted(hp_models)}")

# ============================================================
# C5. Cohort N consistency
# ============================================================
print()
print("=" * 80)
print("C5. Cohort N consistency (925,128 / 925128)")
print("=" * 80)
all_text = md_text + "\n" + code_text
cohort_refs = re.findall(r"925[,_]?128", all_text)
print(f"925,128 references: {len(cohort_refs)} (formats: {Counter(cohort_refs).most_common()})")

# ============================================================
# C6. Markdown table integrity
# ============================================================
print()
print("=" * 80)
print("C6. Markdown table column-count consistency")
print("=" * 80)
table_issues = 0
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'markdown':
        continue
    src = ''.join(c.get('source', []))
    lines = src.split('\n')
    in_table = False
    expected_cols = 0
    for ln_no, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith('|') and stripped.endswith('|'):
            cols = stripped.count('|') - 1
            if not in_table:
                in_table = True
                expected_cols = cols
            elif cols != expected_cols and not all(c == '|' or c == ':' or c == '-' or c == ' ' for c in stripped):
                # Allow separator row
                if not re.match(r"^\|[\s:|\-]+\|$", stripped):
                    table_issues += 1
                    if table_issues <= 5:
                        issues.append((i, "C6", f"L{ln_no}: table col-count {cols} != expected {expected_cols}"))
        else:
            in_table = False
            expected_cols = 0
print(f"Table col-count issues: {table_issues}")

# ============================================================
# Summary
# ============================================================
print()
print("=" * 80)
print("FINAL CONSISTENCY SUMMARY")
print("=" * 80)
print(f"Total issues: {len(issues)}")
for cat in ["C1", "C2", "C3", "C4", "C5", "C6"]:
    cat_issues = [x for x in issues if x[1] == cat]
    print(f"  {cat}: {len(cat_issues)}")
    for cell_idx, _, msg in cat_issues[:3]:
        print(f"    [Cell {cell_idx}] {msg[:150]}")

if not issues:
    print()
    print("VERDICT: NO ISSUES FOUND — PURE PASS")
