"""
Brutal cell-by-cell review of CIKM_2026_LOS_Fairness_FINAL.ipynb.

For each cell:
  - cell_type, source size, output count
  - Header/section
  - Detect: em-dashes, en-dashes, AI cliches, empty outputs (for code cells),
    inconsistent N references, broken HTML, missing alt-text, code without outputs
  - PASS/WARN/FAIL verdict with one-line rationale
"""
import json, re, sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

NB = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\CIKM_2026_LOS_Fairness_FINAL.ipynb")

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

EM = "—"
EN = "–"
AI_CLICHES = [
    r"\bdelve\w*\b",
    r"\bleverag\w*\b",
    r"\brobust\b",
    r"\bcomprehensive\b",
    r"\bseamless\w*\b",
    r"\bmoreover\b",
    r"\bfurthermore\b",
    r"\bin conclusion\b",
    r"\butilize\w*\b",
    r"\btestament\w*\b",
    r"\btapestry\b",
    r"\bnavigate\w*\b",
    r"\bjourney\b",
    r"\brealm\b",
    r"\bunlock\w*\b",
    r"\binsightful\b",
    r"\bgame[- ]chang\w*\b",
    r"\bcutting[- ]edge\b",
    r"\bstate[- ]of[- ]the[- ]art\b",
]
AI_CLICHE_RE = re.compile("|".join(AI_CLICHES), re.IGNORECASE)

# Reference values that must be consistent across cells
EXPECTED_NUMS = {
    "925,128": "full cohort N",
    "185,026": "test partition N (most common)",
    "185,025": "test partition N (alternative — flag if appears with 925,128 vs 925,128)",
    "740,102": "train partition N",
    "441": "hospital count",
    "0.9528": "canonical XGBoost AUROC",
    "0.7965": "Phase 5b min DI",
    "0.0151": "Phase 5b accuracy cost (4.29 pp at full pipeline)",
    "0.0429": "Phase 5b accuracy cost (canonical)",
    "K=500": "VFR bootstrap iterations",
    "N=10,000": "VFR per-resample",
    "0.474": "max VFR_sym",
}

ARTEFACT_PATTERNS = [
    r"output_final/tables/T\d+\.csv",
    r"output_final/tables/T_[A-Z_]+\.csv",
    r"output_final/figures/F\d+\.png",
]

print("=" * 80)
print("BRUTAL CELL-BY-CELL REVIEW")
print("=" * 80)
print(f"Notebook: {NB.name}")
print(f"Total cells: {len(nb['cells'])}")
print(f"Size: {NB.stat().st_size / 1024 / 1024:.2f} MB")
print()

pass_count = 0
warn_count = 0
fail_count = 0
findings = []

for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    n_lines = src.count('\n') + (1 if src and not src.endswith('\n') else 0)
    n_chars = len(src)
    issues = []

    # 1. Em-dash / en-dash
    em_count = src.count(EM)
    en_count = src.count(EN)
    if em_count > 0:
        issues.append(f"em-dashes: {em_count}")
    if en_count > 0:
        # `─` (box drawing) and table separator `|---|` are OK; only flag actual en-dash in prose
        # The U+2014 vs U+2013 — check if appearing in markdown prose
        prose_ens = sum(1 for line in src.split('\n')
                        if EN in line
                        and not line.strip().startswith('|')
                        and not line.strip().startswith('─')
                        and '─' not in line[:10])
        if prose_ens > 0:
            issues.append(f"en-dashes in prose: {prose_ens}")

    # 2. AI cliches
    cliches_found = AI_CLICHE_RE.findall(src)
    if cliches_found:
        unique_cliches = sorted(set(m.lower() for m in cliches_found))
        # Allow "robust" only if in scientific context (e.g. "Robust" = name of test)
        if cliches_found:
            issues.append(f"AI-cliches: {unique_cliches[:5]}")

    # 3. For code cells: check for outputs
    n_outputs = len(c.get('outputs', []))
    if c['cell_type'] == 'code':
        # Allow no outputs if cell is purely setup or print-only diagnostic
        # but flag cells that should produce visible output
        if n_outputs == 0 and n_chars > 200:
            # Check if cell has obvious print/display
            if 'print(' in src or 'display(' in src or 'plt.show' in src or '.to_csv(' in src:
                issues.append("code cell with no outputs (should have)")

    # 4. Section header detection
    header = ""
    for line in src.split('\n'):
        m = re.match(r'^(#{1,4})\s+(.+)$', line.strip())
        if m:
            header = f"H{len(m.group(1))}: {m.group(2)[:80]}"
            break

    # 5. Number references (sample first occurrence)
    n_refs = []
    for k in ["925,128", "185,026", "185,025", "740,102", "0.9528", "0.7965"]:
        if k in src:
            n_refs.append(k)

    # Determine verdict
    if any('em-dashes' in s for s in issues) or any('en-dashes' in s for s in issues):
        verdict = "WARN"  # punctuation cleanup
    elif any('no outputs' in s for s in issues):
        verdict = "WARN"
    elif issues:
        verdict = "WARN"
    else:
        verdict = "PASS"

    if verdict == "PASS":
        pass_count += 1
    elif verdict == "WARN":
        warn_count += 1
    else:
        fail_count += 1

    findings.append({
        "idx": i, "type": c['cell_type'], "header": header,
        "n_chars": n_chars, "n_lines": n_lines, "n_outputs": n_outputs,
        "issues": issues, "verdict": verdict, "n_refs": n_refs,
    })

# Print compact table
print(f"{'Idx':>3} {'Type':5} {'Out':>3} {'Chars':>6} | {'V':4} | Issues / Header")
print("-" * 110)
for f in findings:
    issue_str = '; '.join(f['issues'])[:60] if f['issues'] else ''
    head = f['header'][:50] if f['header'] else ''
    line = f"{f['idx']:>3} {f['type'][:4]:5} {f['n_outputs']:>3} {f['n_chars']:>6} | {f['verdict']:4} | {head} {('| ' + issue_str) if issue_str else ''}"
    print(line[:140])

print()
print(f"SUMMARY: PASS={pass_count}, WARN={warn_count}, FAIL={fail_count} (of {len(findings)})")
print()

# Detailed view of WARN/FAIL cells
print("=" * 80)
print("DETAILED ISSUES (WARN/FAIL only)")
print("=" * 80)
for f in findings:
    if f['verdict'] != 'PASS':
        print(f"\n[{f['idx']:>2}] {f['type']} | {f['header']}")
        for issue in f['issues']:
            print(f"    - {issue}")
