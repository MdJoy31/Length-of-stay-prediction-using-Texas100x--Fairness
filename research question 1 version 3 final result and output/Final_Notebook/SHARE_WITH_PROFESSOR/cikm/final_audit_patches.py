"""
Apply the 13 patches from the final consistency audit.
Hits both CIKM and Journal notebooks where appropriate.
"""
import json
from pathlib import Path

ROOT = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm")
CIKM = ROOT / "CIKM_2026_LOS_Fairness_FINAL.ipynb"
JNL  = ROOT / "full_journal_paper" / "Journal_LOS_Fairness_FULL.ipynb"

# Patches applied to BOTH notebooks (string replacements)
SHARED_PATCHES = [
    # §35 lead-in summary table (CIKM 132 / Journal 155)
    ('| §34.1 bulletproof 70/15/15 patient split, seed=42 | 0.9418 | 5.81 pp | **True** | 22/28 |',
     '| §34.1 bulletproof 70/15/15 patient split, seed=42 | 0.9418 | 4.98 pp | **True** | 21/28 |'),
    ('| §34.2 admission-only (drop TOTAL_CHARGES, PAT_STATUS), seed=42 | 0.8567 | 5.02 pp | **True** | 23/28 |',
     '| §34.2 admission-only (drop TOTAL_CHARGES, PAT_STATUS), seed=42 | 0.8567 | 5.02 pp | **True** | 24/28 |'),
    ('| §34.3 reproducibility, seed=123 | 0.9426 | 5.04 pp | **True** | 22/28 |',
     '| §34.3 reproducibility, seed=123 | 0.9426 | 5.04 pp | **True** | 21/28 |'),
    ('bulletproof range (5.8–6.3 pp)', 'bulletproof range (4.98–5.04 pp)'),
    # The CIKM accuracy-cost headline (4.24 pp) is on the optimistic side of the bulletproof range
    ('CIKM accuracy-cost headline (4.24 pp) is on the optimistic side of the bulletproof range (5.8–6.3 pp)',
     'CIKM accuracy-cost headline (4.24 pp) is below the bulletproof range (4.98–5.04 pp)'),
    # "21–23" -> "21–24" bullet
    ('VFR stability is split-robust (21–23 cells stable across all four reruns)',
     'VFR stability is split-robust (21–24 cells stable across all four reruns)'),
    # §35.7 Fix 6 trace-table "17 / 336" survivor
    ('"146 / 336 cells reverse at least once; 17 / 336 reach VFR > 0.10"',
     '"146 / 336 cells reverse at least once; 77 / 336 reach VFR > 0.10"'),
    # §33.8 wording-corrections table "17 / 336 = 5.1 %"
    ('the practically-significant subset (VFR > 0.10) is 17 / 336 = 5.1 %',
     'the practically-significant subset (VFR > 0.10) is 77 / 336 = 22.9 %'),
    # "23 / 28 VFR-stable cells" wherever it appears for admission-only
    ('all-four-DI ≥ 0.80 and retained 23 / 28 VFR-stable cells',
     'all-four-DI ≥ 0.80 and retained 24 / 28 VFR-stable cells'),
    ('and 23 / 28 VFR-stable cells', 'and 24 / 28 VFR-stable cells'),
    ('AUROC = 0.8567 with all-four-DI ≥ 0.80 and 23 / 28 VFR-stable cells',
     'AUROC = 0.8567 with all-four-DI ≥ 0.80 and 24 / 28 VFR-stable cells'),
    # reviewer-prep language survivors
    ('§33.8 table is drop-in', '§33.8 table is ready for direct insertion'),
    ('copy-paste-ready into the LaTeX manuscript', 'ready for direct inclusion in the LaTeX manuscript'),
    ('**copy-paste-ready** into the LaTeX manuscript', 'ready for direct inclusion in the LaTeX manuscript'),
    # §17.3 header
    ('Recommended abstract sentences (drop-in for the manuscript)',
     'Recommended abstract sentences (ready for the manuscript)'),
    # §35.7 Fix 3, 4, 5 still using "drop-in"
    ('#### Recommended paragraphs (add', '#### Recommended paragraphs (add'),
    # The §35.6 manifest still has "ready for manuscript" which is acceptable
]

# Journal-only fix: delete the stale "Hospital-disjoint 309/66/66" markdown
# cell at index 138 (it duplicates §34.3 and mis-labels seed=123 as hospital-disjoint)
JOURNAL_DELETE_NEEDLES = [
    '### 34.3 · Hospital-disjoint 309 / 66 / 66 split (external validity)',
]

# Journal-only: restore missing references in §34.6 intro (cell 144)
# Detect by the broken phrase and replace with the full sentence
JOURNAL_PATCHES = [
    ('Each figure is rendered directly from  and the per-experiment VFR tables',
     'Each figure is rendered directly from `output/tables/journal_summary.csv` and the per-experiment VFR tables'),
    ('Source generator: . PNGs at 300 dpi in .',
     'Source generator: `scripts/render_journal_figures.py`. PNGs at 300 dpi in `full_journal_paper/journal_figures/`.'),
]

def apply_patches(nb_path, label, patches, delete_needles=None, extra_patches=None):
    nb = json.loads(nb_path.read_text(encoding='utf-8'))

    # Apply string patches to markdown cells
    n_patches = 0
    for cell in nb['cells']:
        src = cell['source']
        if isinstance(src, list): src = ''.join(src)
        orig = src
        for old, new in patches:
            if old in src and old != new:
                src = src.replace(old, new)
                n_patches += 1
        if extra_patches:
            for old, new in extra_patches:
                if old in src:
                    src = src.replace(old, new)
                    n_patches += 1
        if src != orig:
            cell['source'] = src.splitlines(keepends=True)

    # Delete cells matching delete_needles (journal only)
    if delete_needles:
        new_cells = []
        n_deleted = 0
        for c in nb['cells']:
            src = c['source']
            if isinstance(src, list): src = ''.join(src)
            should_delete = any(needle in src for needle in delete_needles)
            if should_delete and c['cell_type'] == 'markdown':
                # Only delete if it's a SHORT markdown cell (not the closing analysis)
                if len(src) < 1500:
                    n_deleted += 1
                    continue
            new_cells.append(c)
        nb['cells'] = new_cells
        if n_deleted:
            print(f"  {label}: deleted {n_deleted} stale cells")

    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f"{label}: {n_patches} patches applied, total cells now {len(nb['cells'])}")

apply_patches(CIKM, 'CIKM', SHARED_PATCHES)
apply_patches(JNL, 'Journal', SHARED_PATCHES,
              delete_needles=JOURNAL_DELETE_NEEDLES,
              extra_patches=JOURNAL_PATCHES)
