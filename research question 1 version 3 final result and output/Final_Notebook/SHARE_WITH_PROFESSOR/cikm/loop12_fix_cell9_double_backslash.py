"""
Loop 12: fix the literal `\\n` double-backslash bug in cell 9 print
statements.

Cell 9 source has:
  L87:  print("\\nVerifying T3 against manuscript Table 3 ...")
  L101: print("\\n".join(mismatches))

These have 2 backslashes + n in the source code, which Python interprets
as literal backslash + n character. When printed they show as `\n` text
in stream output instead of newlines.

Replace `\\n` (2 backslashes + n, 3 chars) with `\n` (1 backslash + n, 2 chars)
which Python interprets as newline escape.

Also update the cell 9 outputs (the prior captured stream output that
contains the literal '\n' characters) so the notebook displays cleanly.
"""
import json, sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

NB = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\CIKM_2026_LOS_Fairness_FINAL.ipynb")

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Fix cell 9 source
src = ''.join(nb['cells'][9].get('source', []))
print(f"Cell 9 source backslashes before: {src.count(chr(92))}")

# Match and replace
# Pattern in source code: print("\\nVerifying  =>  print("\nVerifying  (1 backslash)
old_a = 'print("\\\\nVerifying T3 against manuscript Table 3 (image-extracted, after RACE re-mapping) ...")'
new_a = 'print("\\nVerifying T3 against manuscript Table 3 (image-extracted, after RACE re-mapping) ...")'
old_b = 'print("\\\\n".join(mismatches))'
new_b = 'print("\\n".join(mismatches))'

n_a = src.count(old_a)
n_b = src.count(old_b)
print(f"Pattern A occurrences: {n_a}")
print(f"Pattern B occurrences: {n_b}")

src = src.replace(old_a, new_a)
src = src.replace(old_b, new_b)
print(f"Cell 9 source backslashes after: {src.count(chr(92))}")

nb['cells'][9]['source'] = src.splitlines(keepends=True)

# Now re-format the cell 9 outputs to remove the literal `\n` text artefacts
# from previously captured run outputs.
for o_idx, o in enumerate(nb['cells'][9].get('outputs', [])):
    text = ''
    if 'text' in o:
        t = o['text']
        text = ''.join(t) if isinstance(t, list) else t
    if '\\n' in text:
        # Replace literal `\n` strings with actual newlines for clean display
        new_text = text.replace('\\n', '\n')
        if isinstance(o['text'], list):
            nb['cells'][9]['outputs'][o_idx]['text'] = new_text.splitlines(keepends=True)
        else:
            nb['cells'][9]['outputs'][o_idx]['text'] = new_text
        print(f"Cell 9 output {o_idx}: cleaned literal \\n artefacts")

# Save
with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"\nFinal notebook: {NB.stat().st_size / 1024 / 1024:.2f} MB")
