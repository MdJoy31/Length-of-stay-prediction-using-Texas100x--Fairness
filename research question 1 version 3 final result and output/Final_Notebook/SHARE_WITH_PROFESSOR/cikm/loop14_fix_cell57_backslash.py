"""
Loop 14: fix cell 57's residual `\\n` (2-backslash) bug in line 31.

Cell 57 L31 source code:
  print(f"\\nUnanimous-fair (model, attr) combos: {n_combos_total - n_combos_unfair}/{n_combos_total}")

The `\\n` (2 backslashes + n) is interpreted by Python as literal backslash + n,
producing `\n` text in the output. Replace with `\n` (1 backslash + n) which is
the newline-escape sequence.

Also clean up the captured output 2 of cell 57, which has the literal `\n` text.
"""
import json, sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

NB = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\CIKM_2026_LOS_Fairness_FINAL.ipynb")

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Fix cell 57 source line 31
src = ''.join(nb['cells'][57].get('source', []))

# In Python source code (in memory after json.load), the bug pattern is:
# `print(f"\\nUnanimous-fair` (with 2 actual backslashes between f" and n)
# In Python script literal: target = 'print(f"' + 2*'\' + 'nUnanimous-fair'
target_bug = 'print(f"' + chr(92) + chr(92) + 'nUnanimous-fair (model, attr) combos: {n_combos_total - n_combos_unfair}/{n_combos_total}")'
target_fix = 'print(f"' + chr(92) + 'nUnanimous-fair (model, attr) combos: {n_combos_total - n_combos_unfair}/{n_combos_total}")'

if target_bug in src:
    src_new = src.replace(target_bug, target_fix)
    nb['cells'][57]['source'] = src_new.splitlines(keepends=True)
    print("Cell 57 L31: fixed 2-backslash-n bug")
else:
    print("Cell 57: bug pattern not found (may already be fixed)")
    # Show actual content
    for ln_no, line in enumerate(src.split('\n'), 1):
        if 'Unanimous-fair' in line and 'print' in line:
            print(f'  L{ln_no} repr: {repr(line[:200])}')

# Clean up cell 57 output 2 stream text (replace literal \n with actual newlines)
for o_idx, o in enumerate(nb['cells'][57].get('outputs', [])):
    text = ''
    if 'text' in o:
        t = o['text']
        text = ''.join(t) if isinstance(t, list) else t
    if '\\n' in text:
        new_text = text.replace('\\n', '\n')
        if isinstance(o.get('text'), list):
            nb['cells'][57]['outputs'][o_idx]['text'] = new_text.splitlines(keepends=True)
        else:
            nb['cells'][57]['outputs'][o_idx]['text'] = new_text
        print(f"Cell 57 output {o_idx}: cleaned literal \\n artefacts")

# Save
with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"\nFinal notebook: {NB.stat().st_size / 1024 / 1024:.2f} MB")
