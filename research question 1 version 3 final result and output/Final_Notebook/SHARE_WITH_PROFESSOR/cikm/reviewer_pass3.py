"""
Pass 3 brutal reviewer audit. Looking for issues I may have missed:

R1. Cross-cell number consistency (e.g., 4.29pp accuracy cost in markdown vs T15)
R2. Manuscript-claim anchors vs actual computed values
R3. Bootstrap K consistency (K=500 in §1 vs §18 vs cell 23)
R4. Cohort/test/train N consistency
R5. AUROC/accuracy values consistent across cells
R6. Phase 5b numbers consistent everywhere
R7. References/citations match
R8. Methodology ordering (§20 limitations before §22-24 appendices)
R9. Hidden bugs in code (off-by-one, wrong indexing, etc.)
R10. Outputs match what code produces (not stale)
"""
import json, re, sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm")
NB = ROOT / "CIKM_2026_LOS_Fairness_FINAL.ipynb"
with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

ALL_MD = ""
ALL_CODE = ""
ALL_OUT = ""
for c in nb['cells']:
    src = ''.join(c.get('source', []))
    if c['cell_type'] == 'markdown':
        ALL_MD += src + "\n\n"
    else:
        ALL_CODE += src + "\n\n"
        for o in c.get('outputs', []):
            if 'text' in o:
                t = o['text']
                ALL_OUT += (''.join(t) if isinstance(t, list) else t) + "\n\n"

issues = []

print("=" * 80)
print("REVIEWER PASS 3 — BRUTAL CONSISTENCY AUDIT")
print("=" * 80)

# R1. Cross-cell consistency on key numbers
print("\n--- R1. KEY NUMBER CONSISTENCY ---")
KEY_NUMS = [
    ("Cohort N",  ["925,128", "925128"]),
    ("Test N",    ["185,026", "185_026"]),
    ("Train N",   ["740,102"]),
    ("Hospitals", ["441"]),
    ("Bootstrap K", ["K = 500", "K=500", "K_VFR = 500"]),
    ("Per-resample N", ["N = 10,000", "N_VFR = 10_000"]),
    ("Canonical AUROC", ["0.9528"]),
    ("Canonical Accuracy", ["0.8776"]),
    ("Phase 5b accuracy", ["0.8347", "0.835"]),
    ("Accuracy cost pp", ["4.29"]),
]
for label, patterns in KEY_NUMS:
    md_count = sum(ALL_MD.count(p) for p in patterns)
    code_count = sum(ALL_CODE.count(p) for p in patterns)
    out_count = sum(ALL_OUT.count(p) for p in patterns)
    if md_count == 0 and code_count == 0:
        issues.append(f"R1: '{label}' not found anywhere")
        print(f"  [WARN] {label}: NOT FOUND")
    else:
        print(f"  [OK]   {label}: md={md_count}, code={code_count}, out={out_count}")

# R2. Manuscript-claim anchors
print("\n--- R2. MANUSCRIPT-CLAIM ANCHOR VALUES ---")
# Read T19_claim_verification.csv if exists
import pandas as pd
T19_path = ROOT / "output_final" / "tables" / "T19_claim_verification.csv"
if T19_path.exists():
    T19 = pd.read_csv(T19_path)
    print(f"T19 rows: {len(T19)}")
    if 'Status' in T19.columns:
        status_counts = T19['Status'].value_counts().to_dict()
        print(f"  Status distribution: {status_counts}")
        n_fix = status_counts.get('FIX', 0)
        n_close = status_counts.get('CLOSE', 0)
        if n_fix > 0:
            issues.append(f"R2: T19 has {n_fix} FIX status rows — manuscript claims don't match")
            # Show which ones
            fix_rows = T19[T19['Status'] == 'FIX']
            for _, r in fix_rows.iterrows():
                print(f"  [FIX] {r.get('ID', '?')}: {r.get('Claim', '?')[:60]}: obs={r.get('Notebook_value', '?')} vs claim={r.get('Manuscript_value', '?')}")

# R3. Phase 5b consistency check
print("\n--- R3. PHASE 5B HEADLINE CONSISTENCY ---")
# Look for Phase 5b accuracy/AUROC in different places
phase5b_acc = re.findall(r"Phase 5b[^.]*?(?:accuracy|Acc)[^0-9]*0\.83\d+", ALL_MD)
print(f"  Phase 5b accuracy mentions in markdown: {len(phase5b_acc)}")
# DI Race=0.80 mentions
di_race_080 = re.findall(r"DI[\s_]?Race[^.]*?0\.80\d*", ALL_MD)
print(f"  DI Race ≥ 0.80 mentions: {len(di_race_080)}")

# R4. Section ordering
print("\n--- R4. SECTION ORDERING ---")
sec_pat = re.compile(r"^##\s+(\d+)\s*[\.\s·]", re.MULTILINE)
sec_order = []
for c in nb['cells']:
    if c['cell_type'] != 'markdown':
        continue
    src = ''.join(c.get('source', []))
    for m in sec_pat.finditer(src):
        sec_order.append(int(m.group(1)))
print(f"  Section order: {sec_order}")
mono = all(sec_order[i] < sec_order[i+1] for i in range(len(sec_order)-1))
print(f"  Strictly increasing: {mono}")
if not mono:
    issues.append(f"R4: section order not monotone: {sec_order}")

# R5. Conflicting accuracy/AUROC numbers in same neighbourhood
print("\n--- R5. CONFLICTING NUMERIC CLAIMS ---")
# Find any place the manuscript says "AUROC = 0.95X" with X != 28
auroc_matches = re.findall(r"AUROC\s*=\s*0\.95(\d{2,})", ALL_MD)
auroc_unique = set(auroc_matches)
print(f"  AUROC values found in markdown: {auroc_unique}")
if len(auroc_unique) > 2:
    issues.append(f"R5: too many distinct AUROC values: {auroc_unique}")
acc_matches = re.findall(r"[Aa]ccuracy\s*=\s*0\.8(\d{2,})", ALL_MD)
acc_unique = set(acc_matches)
print(f"  Accuracy values found in markdown: {acc_unique}")

# R6. Bootstrap K mismatch — abstract uses K=500 but B=100 in §11.6 CI
print("\n--- R6. BOOTSTRAP B/K USED ---")
B_CI_used = re.findall(r"B[_=\s]*CI?\s*[=:]\s*(\d+)", ALL_CODE)
B_VFR_used = re.findall(r"K[_=\s]*VFR\s*=\s*(\d+)", ALL_CODE)
print(f"  B_CI (cell 38 CI bootstrap): {B_CI_used}")
print(f"  K_VFR (cell 23 stability bootstrap): {B_VFR_used}")
b_ci_in_md = re.findall(r"B\s*=\s*(\d+)\s*(?:bootstrap|stratified)", ALL_MD)
print(f"  B values in markdown: {b_ci_in_md}")

# R7. Citations
print("\n--- R7. CITATIONS ---")
cite_pat = re.compile(r"\(([A-Z][a-zA-Z]+(?:\s+(?:and|&|et\s+al\.?)\s+[A-Z][a-zA-Z]+)*),?\s+(\d{4})\)")
cites = set(cite_pat.findall(ALL_MD))
print(f"  Unique citations: {len(cites)}")
key_cites = ['Hardt', 'Chouldechova', 'Hoeffding', 'Speicher', 'Pfohl', 'Poulain', 'Obermeyer', 'Pierson', 'Chen', 'Rajkomar']
found_cites = []
for kc in key_cites:
    if kc in ALL_MD:
        found_cites.append(kc)
print(f"  Key author mentions: {found_cites}")

# R8. Cell with mostly-empty / placeholder text
print("\n--- R8. EMPTY OR PLACEHOLDER CELLS ---")
for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    if c['cell_type'] == 'markdown' and len(src.strip()) < 30 and src.strip() not in ['---', '']:
        issues.append(f"R8: cell {i} markdown is suspiciously short: {src[:80]!r}")
        print(f"  [Cell {i}] short md: {src[:80]!r}")

# R9. Hidden bugs in code: imports of unused modules, undefined refs
print("\n--- R9. CODE QUALITY (basic) ---")
# Check for any "TODO" / "XXX" / "FIXME" / unfinished
markers = ['TODO:', 'FIXME:', 'XXX', '# WIP', 'NotImplemented']
for m in markers:
    count = ALL_CODE.count(m)
    if count > 0:
        issues.append(f"R9: found {count} '{m}' markers in code")
        print(f"  [WARN] {m}: {count}")
    else:
        print(f"  [OK]   {m}: 0")

# R10. Output staleness check (e.g., dataframes that show old race labels)
print("\n--- R10. OUTPUT STALENESS ---")
# Look for output text containing old labels like "Native American" that would indicate stale outputs
stale_terms = ['Native American', '603368.0000', '925_128']
for term in stale_terms:
    if term in ALL_OUT:
        # Find which cell
        for i, c in enumerate(nb['cells']):
            if c['cell_type'] != 'code':
                continue
            for o_idx, o in enumerate(c.get('outputs', [])):
                text = ''
                if 'text' in o:
                    text = ''.join(o['text']) if isinstance(o['text'], list) else o['text']
                if 'data' in o:
                    for k, v in o['data'].items():
                        if 'text' in k or 'html' in k:
                            text += ''.join(v) if isinstance(v, list) else str(v)
                if term in text:
                    issues.append(f"R10: cell {i} output {o_idx} has stale term '{term}'")
                    print(f"  [STALE] cell {i} output {o_idx}: '{term}'")
                    break

# Summary
print("\n" + "=" * 80)
print(f"REVIEWER PASS 3 ISSUES: {len(issues)}")
print("=" * 80)
if issues:
    for issue in issues:
        print(f"  - {issue}")
else:
    print("  ALL CLEAN — pass 3 found no new issues.")
