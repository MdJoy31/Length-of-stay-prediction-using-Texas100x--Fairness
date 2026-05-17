"""
Dump full content of each markdown cell + first 60 lines of each code
cell + first 800 chars of each output, so a reviewer can read the
substance directly. Output goes to reviewer_dump.txt.
"""
import json, sys, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm")
NB = ROOT / "CIKM_2026_LOS_Fairness_FINAL.ipynb"
OUT = ROOT / "reviewer_dump.txt"

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

with open(OUT, "w", encoding="utf-8") as out:
    for i, c in enumerate(nb['cells']):
        out.write(f"\n{'=' * 80}\n")
        out.write(f"CELL {i:>2} | {c['cell_type'].upper()}\n")
        out.write(f"{'=' * 80}\n")
        src = ''.join(c.get('source', []))
        if c['cell_type'] == 'markdown':
            out.write(src)
            out.write("\n")
        else:
            lines = src.split('\n')
            for ln in lines[:80]:
                out.write(ln + "\n")
            if len(lines) > 80:
                out.write(f"... ({len(lines) - 80} more code lines truncated) ...\n")
            # Outputs
            for o_idx, o in enumerate(c.get('outputs', [])):
                txt = ''
                if 'text' in o:
                    t = o['text']
                    txt = ''.join(t) if isinstance(t, list) else t
                if 'data' in o:
                    for k, v in o['data'].items():
                        if 'plain' in k or k == 'text/plain':
                            t = v
                            txt = ''.join(t) if isinstance(t, list) else str(t)
                if txt:
                    out.write(f"\n--- output {o_idx} ({o.get('output_type', '?')}) ---\n")
                    out.write(txt[:1200])
                    if len(txt) > 1200:
                        out.write(f"\n...({len(txt)-1200} more chars truncated)...\n")

print(f"Dumped {len(nb['cells'])} cells to {OUT}")
print(f"Output size: {OUT.stat().st_size / 1024:.0f} KB")
