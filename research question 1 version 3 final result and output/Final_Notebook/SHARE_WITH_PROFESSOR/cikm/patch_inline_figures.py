"""
Patch the FINAL notebook so every matplotlib figure is embedded INLINE
in the .ipynb (so the user sees plots when opening the notebook), not
just written to output_final/figures/.

Two changes:
 (1) Insert  %matplotlib inline   at top of the first code cell.
 (2) After every  plt.savefig(...)  line, insert  plt.show()  so the
     figure actually renders (and embeds as image/png).

Idempotent: safe to run multiple times.
"""
import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

NB = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\CIKM_2026_LOS_Fairness_FINAL.ipynb")

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

n_show_added = 0
n_inline_added = 0
first_code_cell_idx = None

for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    if first_code_cell_idx is None:
        first_code_cell_idx = i

    src_lines = c.get("source", [])
    if isinstance(src_lines, str):
        src_lines = src_lines.splitlines(keepends=True)

    # (1) Add %matplotlib inline to the first code cell if not present
    if i == first_code_cell_idx:
        full = "".join(src_lines)
        if "%matplotlib inline" not in full:
            src_lines = ["%matplotlib inline\n", "%config InlineBackend.figure_format = 'retina'\n"] + src_lines
            n_inline_added += 1

    # (2) After each plt.savefig(...) closing, insert plt.show()
    new_lines = []
    in_savefig = False
    paren_depth = 0
    for ln in src_lines:
        new_lines.append(ln)
        stripped = ln.lstrip()
        # detect savefig start
        if not in_savefig and "plt.savefig(" in ln:
            in_savefig = True
            paren_depth = ln.count("(") - ln.count(")")
            if paren_depth == 0:
                # single-line savefig
                in_savefig = False
                indent = ln[:len(ln) - len(stripped)]
                # only insert if next existing line is not already plt.show
                # we'll dedupe in a second pass
                new_lines.append(f"{indent}plt.show()\n")
                n_show_added += 1
            continue
        if in_savefig:
            paren_depth += ln.count("(") - ln.count(")")
            if paren_depth <= 0:
                in_savefig = False
                indent = ln[:len(ln) - len(stripped)] if stripped else ""
                # Use 0-indent if last line was just a ")"
                # Actually use indent from the savefig opening line — keep simple: align with last line
                indent = "                "  # 16-space default; cells use 0 normally
                # safer: match indent of the opening savefig line
                # We'll fix by scanning back for the savefig line indent
                k = len(new_lines) - 1
                while k >= 0 and "plt.savefig(" not in new_lines[k]:
                    k -= 1
                if k >= 0:
                    open_ln = new_lines[k]
                    indent = open_ln[:len(open_ln) - len(open_ln.lstrip())]
                new_lines.append(f"{indent}plt.show()\n")
                n_show_added += 1

    # Dedup: drop a plt.show() that is immediately followed by another plt.show()
    deduped = []
    for ln in new_lines:
        if (deduped and "plt.show()" in deduped[-1]
                and "plt.show()" in ln):
            continue
        deduped.append(ln)

    c["source"] = deduped
    # clear any stale outputs so re-run produces fresh embedded figures
    c["outputs"] = []
    c["execution_count"] = None

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Inserted '%matplotlib inline' magic: {n_inline_added}")
print(f"Inserted plt.show() calls: {n_show_added}")
print(f"Wrote patched notebook ({len(nb['cells'])} cells)")
