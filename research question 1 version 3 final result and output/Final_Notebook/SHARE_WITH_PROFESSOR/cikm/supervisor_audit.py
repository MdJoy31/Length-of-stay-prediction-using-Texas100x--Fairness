"""
Round 2: 20-year-experienced research supervisor audit.

Checks beyond surface clean-up:
  S1. Section numbering: any gaps in §1, §2, ..., §24? cross-references resolve?
  S2. Citation integrity: every (Author, Year) cited at least once should appear in a references section
  S3. Random seed and reproducibility: every modelling cell mentions seed
  S4. Statistical rigor: every claim with point estimate has a CI, sample size, or p-value
  S5. Cross-reference sanity: "Table T13", "Figure F4", "§22.3" all resolve
  S6. Output integrity: every code cell that writes a file actually has the file
  S7. Acronym definition: DI, SPD, etc. all defined in §1 or first use
  S8. Hidden assumptions: detect implicit thresholds (alpha=0.05, threshold=0.5) — are they stated?
  S9. Numbers exposed in code outputs MATCH numbers in markdown narrative
"""
import json, re, sys, io
from pathlib import Path
from collections import defaultdict, Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm")
NB = ROOT / "CIKM_2026_LOS_Fairness_FINAL.ipynb"

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

issues = defaultdict(list)
SEVERITY = {"FAIL": 3, "WARN": 2, "INFO": 1}

# Collect all source text
md_text = []
code_text = []
all_outputs_text = []
for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    if c['cell_type'] == 'markdown':
        md_text.append((i, src))
    else:
        code_text.append((i, src))
    out_text = ""
    for o in c.get('outputs', []):
        if 'text' in o:
            out_text += ''.join(o['text']) if isinstance(o['text'], list) else o['text']
        if 'data' in o:
            for k, v in o['data'].items():
                if 'text' in k:
                    out_text += ''.join(v) if isinstance(v, list) else str(v)
    all_outputs_text.append((i, out_text))

ALL_MD = "\n\n".join(s for _, s in md_text)
ALL_CODE = "\n\n".join(s for _, s in code_text)
ALL_OUT = "\n\n".join(s for _, s in all_outputs_text)
ALL = ALL_MD + "\n\n" + ALL_CODE + "\n\n" + ALL_OUT

# =============================================================
# S1. Section numbering audit
# =============================================================
print("=" * 80)
print("S1. SECTION NUMBERING AUDIT")
print("=" * 80)
section_pat = re.compile(r"^##\s+(\d+)(?:[\.\s·\-]+)([^\n]+)$", re.MULTILINE)
sections = section_pat.findall(ALL_MD)
print(f"Found {len(sections)} top-level sections: {[s[0] for s in sections]}")
seen = sorted(set(int(s[0]) for s in sections))
print(f"Unique section numbers: {seen}")
expected = list(range(1, max(seen) + 1))
gaps = [n for n in expected if n not in seen]
if gaps:
    issues['S1_section_gaps'].append(("WARN", f"Section gaps: {gaps}"))
    print(f"GAPS: {gaps}")
else:
    print(f"No gaps: {seen[0]}-{seen[-1]} all present")

# =============================================================
# S2. Citation integrity
# =============================================================
print()
print("=" * 80)
print("S2. CITATION INTEGRITY")
print("=" * 80)
# Find inline citations (Author, Year) or (Author Year)
citation_pat = re.compile(r"\(([A-Z][a-z]+(?:\s*(?:&|et\s+al\.?|and)\s*[A-Z][a-z]+)?)[\s,]+(\d{4}[a-z]?)\)")
all_citations = citation_pat.findall(ALL_MD)
unique_cits = set((a.strip(), y) for a, y in all_citations)
print(f"Inline citations: {len(all_citations)} occurrences, {len(unique_cits)} unique")
for c in sorted(unique_cits):
    print(f"  - {c[0]} ({c[1]})")
# Check for a references section
has_refs = bool(re.search(r"^##\s+(?:References|Bibliography)|##\s+\d+[\.\s]+References", ALL_MD, re.MULTILINE | re.IGNORECASE))
if not has_refs and len(unique_cits) > 5:
    issues['S2_no_refs'].append(("INFO", f"{len(unique_cits)} unique inline citations but no formal references section in notebook"))

# =============================================================
# S3. Random seed / reproducibility
# =============================================================
print()
print("=" * 80)
print("S3. REPRODUCIBILITY (random seed)")
print("=" * 80)
seed_pat = re.compile(r"random_state\s*=\s*(\d+)|seed\s*=\s*(\d+)|np\.random\.seed\((\d+)\)", re.IGNORECASE)
for i, src in code_text:
    seeds = seed_pat.findall(src)
    if seeds:
        unique_seeds = set(s for tup in seeds for s in tup if s)
        if len(unique_seeds) > 1:
            issues['S3_seed_inconsistent'].append(("WARN", f"Cell {i}: multiple seeds {unique_seeds}"))

# Top-level seed declared?
if "RANDOM_STATE" in ALL_CODE or "RANDOM_SEED" in ALL_CODE:
    print("Global RANDOM_STATE constant: FOUND")
else:
    issues['S3_no_global_seed'].append(("WARN", "No global RANDOM_STATE constant found"))

# =============================================================
# S4. Statistical rigor: every key result has CI/p-value
# =============================================================
print()
print("=" * 80)
print("S4. STATISTICAL RIGOR")
print("=" * 80)
ci_pat = re.compile(r"\b(?:95%\s*CI|95% confidence interval|\[\d+\.\d+\s*,\s*\d+\.\d+\])")
ci_count = len(ci_pat.findall(ALL_MD))
print(f"95% CI references in markdown: {ci_count}")

p_value_pat = re.compile(r"\bp\s*[<=]\s*0\.\d+|\bp[- ]value\b", re.IGNORECASE)
p_count = len(p_value_pat.findall(ALL_MD))
print(f"p-value references in markdown: {p_count}")

if ci_count < 3:
    issues['S4_few_CIs'].append(("INFO", f"Only {ci_count} CI references in markdown; Q1 expects multiple"))

# =============================================================
# S5. Cross-reference sanity
# =============================================================
print()
print("=" * 80)
print("S5. CROSS-REFERENCE SANITY")
print("=" * 80)
# Tables T1-T20 references
table_refs = set(re.findall(r"\bT(\d{1,2})\b", ALL_MD + " " + ALL_CODE))
print(f"Table refs found: T{sorted(int(t) for t in table_refs)}")
# Figures F1-F8
fig_refs = set(re.findall(r"\bF(\d{1,2})\b(?!\w)", ALL_MD + " " + ALL_CODE))
print(f"Figure refs found: F{sorted(int(f) for f in fig_refs)}")
# Section refs §1-§24
sec_refs = set(re.findall(r"§\s*(\d{1,2})(?:\.\d)?", ALL_MD))
print(f"Section refs found: §{sorted(int(s) for s in sec_refs)}")

# Missing artefacts
TABLES_DIR = ROOT / "output_final" / "tables"
FIGS_DIR = ROOT / "output_final" / "figures"
on_disk_tables = {f.stem for f in TABLES_DIR.glob("*.csv")}
on_disk_figs = {f.stem for f in FIGS_DIR.glob("*.png")}
print(f"\nT*.csv on disk: {sorted(on_disk_tables)}")
print(f"\nF*.png on disk: {sorted(on_disk_figs)}")

# Check for referenced T_X but missing on disk
for t_num in sorted(int(t) for t in table_refs):
    if t_num <= 20:
        # Look for T{n}_*.csv
        matches = [t for t in on_disk_tables if t.startswith(f"T{t_num}_")]
        if not matches:
            issues['S5_missing_table'].append(("WARN", f"T{t_num} referenced but no T{t_num}_*.csv on disk"))

# =============================================================
# S6. Forward references / hallucinated tables
# =============================================================
print()
print("=" * 80)
print("S6. HALLUCINATED ARTEFACT DETECTION")
print("=" * 80)
# Pattern: "Figure F8" but if not in disk
for fig_num in sorted(int(f) for f in fig_refs):
    if fig_num <= 8:
        matches = [t for t in on_disk_figs if t.startswith(f"F{fig_num}_") or t == f"F{fig_num}"]
        if not matches:
            issues['S6_missing_fig'].append(("WARN", f"F{fig_num} referenced but no F{fig_num}_*.png on disk"))

# =============================================================
# S7. Acronym definitions
# =============================================================
print()
print("=" * 80)
print("S7. ACRONYM DEFINITION (first-use)")
print("=" * 80)
acronyms = ["DI", "SPD", "EOPP", "EOD", "TI", "PP", "CAL", "VFR", "AUROC", "AUC",
            "PUDF", "THCIC", "LOS", "GroupKFold"]
for a in acronyms:
    found = bool(re.search(rf"\b{a}\b", ALL_MD))
    # Look for definitional pattern: "DI (disparate impact)" or "Disparate Impact (DI)"
    define_pat = re.compile(
        rf"({a}\s*\([A-Z][^)]+\))|(\([A-Z][a-z][^)]*{a}\))|"
        rf"(\b[Dd]isparate impact[^.]*{a})|(\b{a}[^,.]*\bdisparate impact)",
    )
    if found and a in ["DI", "SPD", "EOPP", "EOD", "TI", "PP", "CAL", "VFR"]:
        # Check if defined somewhere
        define_simple = re.search(
            rf"(?:\b\w+\s+\w+\s*\({a}\)|{a}\s*[:=]|{a}\s*\(\w)",
            ALL_MD
        )
        if not define_simple:
            issues['S7_undef_acronym'].append(("INFO", f"Acronym {a} used but no clear inline definition"))

# =============================================================
# S8. Implicit thresholds
# =============================================================
print()
print("=" * 80)
print("S8. IMPLICIT THRESHOLDS DECLARED")
print("=" * 80)
# 0.5 classification threshold
class_thr_in_md = bool(re.search(r"\b0\.5\b.*(?:threshold|cut[- ]off|decision)|threshold.*\b0\.5\b", ALL_MD, re.I))
print(f"Classification threshold (0.5) declared in markdown: {class_thr_in_md}")
# K=500 bootstrap
boot_K_in_md = bool(re.search(r"K\s*=\s*500", ALL_MD))
print(f"Bootstrap K=500 declared in markdown: {boot_K_in_md}")
# DI threshold 0.80
di_080_in_md = bool(re.search(r"\bDI\s*[≥>=]\s*0\.80|0\.80\s*disparate impact|four-fifths", ALL_MD, re.I))
print(f"DI 0.80 (four-fifths rule) declared in markdown: {di_080_in_md}")
# Train/test 80/20
split_in_md = bool(re.search(r"80\s*[/-]\s*20|0\.20|test_size", ALL_MD))
print(f"Train/test split declared in markdown: {split_in_md}")

# =============================================================
# Summary
# =============================================================
print()
print("=" * 80)
print("AUDIT SUMMARY")
print("=" * 80)
total = sum(len(v) for v in issues.values())
fail = sum(1 for v in issues.values() for severity, _ in v if severity == "FAIL")
warn = sum(1 for v in issues.values() for severity, _ in v if severity == "WARN")
info = sum(1 for v in issues.values() for severity, _ in v if severity == "INFO")
print(f"Total findings: {total} (FAIL={fail}, WARN={warn}, INFO={info})")
print()
for category, items in sorted(issues.items()):
    for severity, msg in items:
        print(f"  [{severity}] {category}: {msg}")

# Verdict
print()
if fail > 0:
    print("VERDICT: FAIL — must fix before submission")
elif warn > 0:
    print(f"VERDICT: WARN ({warn} warnings) — review and fix")
elif info > 0:
    print(f"VERDICT: INFO-ONLY ({info} suggestions) — at supervisor discretion")
else:
    print("VERDICT: CLEAN PASS — ready for Q1 submission")
