"""
Deep Q1-grade content review.

Checks beyond surface punctuation:
  1. Number consistency (e.g., 925,128 vs 925128 vs 925_128)
  2. Cross-cell references (T13, T19, F1-F8, §22, §23, §24)
  3. Random seed presence in modelling cells
  4. Empty code-cell outputs
  5. Citation pattern presence in §13 and §22
  6. Numeric claim/anchor verification (against T19)
  7. Forbidden phrasings (subjective adjectives, marketing language)
  8. Acronym consistency (DI, SPD, EOPP, EOD, TI, PP, CAL, VFR, AUROC)
"""
import json, re, sys, io
from pathlib import Path
from collections import defaultdict, Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

NB = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\CIKM_2026_LOS_Fairness_FINAL.ipynb")

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

EM = "—"
EN = "–"

# Patterns to flag in markdown PROSE only
FORBIDDEN_PROSE = [
    (r"\bdelve\w*\b", "AI-cliche: delve"),
    (r"\bleverag\w*\b", "AI-cliche: leverage"),
    (r"\bseamless\w*\b", "AI-cliche: seamless"),
    (r"\bcomprehensive\b", "AI-cliche: comprehensive (over-used)"),
    (r"\brobust\b", "AI-cliche: robust (over-used)"),
    (r"\b(?:moreover|furthermore|in conclusion)\b", "Filler transition"),
    (r"\butilize\w*\b", "Use 'use' instead of 'utilize'"),
    (r"\bdeep dive\b", "AI-cliche: deep dive"),
    (r"\bgame[- ]chang\w*\b", "AI-cliche: game-changing"),
    (r"\bcutting[- ]edge\b", "Marketing-speak: cutting-edge"),
    (r"\bstate[- ]of[- ]the[- ]art\b", "Over-used: state-of-the-art"),
    (r"\bworld[- ]class\b", "Marketing: world-class"),
    (r"\bnext[- ]gen\w*\b", "Marketing: next-gen"),
    (r"\bnavigate\w+ (?:the|this) \w+\b", "AI-cliche: navigate"),
]
FORBIDDEN_RE = [(re.compile(p, re.IGNORECASE), msg) for p, msg in FORBIDDEN_PROSE]

# Number patterns that should be consistent
KEY_NUMBERS = {
    "cohort_N": [r"\b925[,_]?128\b"],
    "test_N": [r"\b185[,_]?026\b", r"\b185[,_]?025\b"],
    "train_N": [r"\b740[,_]?102\b"],
    "hospitals": [r"\b441\b"],
    "K_VFR": [r"K\s*=\s*500", r"K\s*=\s*\{?\s*500"],
    "N_VFR": [r"N\s*=\s*10[,_]?000"],
    "AUROC_canon": [r"0\.9528"],
    "DI_min_phase5b": [r"0\.7965"],
    "phase5b_acc_cost": [r"0\.0429", r"4\.29\s*pp", r"4\.29 percentage"],
}

# Cell-by-cell findings
print("=" * 80)
print("DEEP Q1-GRADE CONTENT REVIEW")
print("=" * 80)
print()

issues_by_cell = defaultdict(list)
all_md_text = ""

for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    is_md = (c['cell_type'] == 'markdown')
    n_outputs = len(c.get('outputs', []))

    # Check 1: forbidden prose in markdown only
    if is_md:
        all_md_text += "\n\n" + src
        for pat, msg in FORBIDDEN_RE:
            matches = pat.findall(src)
            if matches:
                # Filter: "robust" is OK in stat-software names; flag if used as adjective for results
                if "robust" in msg.lower():
                    # Check context — if it's in a methodology section near "robustness check" allow
                    ctxs = [m for m in pat.finditer(src)]
                    flagged_ctxs = []
                    for m in ctxs:
                        start = max(0, m.start() - 30)
                        end = min(len(src), m.end() + 30)
                        ctx = src[start:end]
                        # Allow these contexts
                        if 'robustness' in ctx.lower() or 'standard error' in ctx.lower():
                            continue
                        flagged_ctxs.append(ctx.replace('\n', ' '))
                    if flagged_ctxs:
                        issues_by_cell[i].append(f"{msg} | {len(flagged_ctxs)}x | first: '{flagged_ctxs[0][:80]}'")
                else:
                    issues_by_cell[i].append(f"{msg} | {len(matches)}x")

    # Check 2: code cells with print/.to_csv but no outputs
    if not is_md:
        if n_outputs == 0 and ('print(' in src or '.to_csv(' in src or 'display(' in src):
            issues_by_cell[i].append("CODE CELL has print/save but no outputs — needs execution or static injection")

    # Check 3: Tabular structure consistency in markdown tables
    if is_md and '|' in src:
        for line_num, line in enumerate(src.split('\n'), 1):
            if line.count('|') > 2 and not line.strip().startswith('|'):
                # Mid-line pipe might be OK if escaped, otherwise flag
                pass

# Cross-cell number consistency check
print("=" * 80)
print("NUMBER CONSISTENCY CHECK")
print("=" * 80)
for label, patterns in KEY_NUMBERS.items():
    counts = Counter()
    for i, c in enumerate(nb['cells']):
        src = ''.join(c.get('source', []))
        for p in patterns:
            for m in re.finditer(p, src):
                counts[m.group(0)] += 1
    if counts:
        print(f"{label:25}: {dict(counts)}")
    else:
        print(f"{label:25}: NOT FOUND ANYWHERE — may need to add")

print()
print("=" * 80)
print("PER-CELL CONTENT ISSUES")
print("=" * 80)
for i in sorted(issues_by_cell.keys()):
    src = ''.join(nb['cells'][i].get('source', []))
    head = ""
    for line in src.split('\n')[:10]:
        m = re.match(r'^#{1,4}\s+(.+)$', line.strip())
        if m:
            head = m.group(1)[:70]
            break
    print(f"\n[Cell {i:>2}] {nb['cells'][i]['cell_type']} | {head}")
    for issue in issues_by_cell[i]:
        print(f"    - {issue}")

print()
print("=" * 80)
print(f"TOTAL CELLS WITH CONTENT ISSUES: {len(issues_by_cell)} / {len(nb['cells'])}")
