"""
Replace informal terms with academic wording in both notebooks.

Two categories:
  A) "bulletproof" -> "leakage-controlled" / "stratified" depending on context.
  B) "CIKM" used as a descriptor for the single-split protocol -> "single-split
     (in-sample tuning)". CIKM is a venue name, not a methodology; using it
     mid-text as if it were is sloppy.

Other minor informal phrases are also cleaned up.
"""
import json
from pathlib import Path

ROOT = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm")
CIKM = ROOT / "CIKM_2026_LOS_Fairness_FINAL.ipynb"
JNL  = ROOT / "full_journal_paper" / "Journal_LOS_Fairness_FULL.ipynb"

# Order matters: longer / more-specific patterns first to avoid partial overlaps.
PATCHES = [
    # ----- bulletproof phrasings -----
    # In compound phrases first
    ('bulletproof three-way 70/15/15 train/val/audit rerun',
     'leakage-controlled three-way 70/15/15 train/validation/audit rerun'),
    ('bulletproof three-way 70 / 15 / 15 train / val / audit rerun',
     'leakage-controlled three-way 70 / 15 / 15 train / validation / audit rerun'),
    ('bulletproof three-way 70/15/15 split',
     'leakage-controlled three-way 70/15/15 split'),
    ('bulletproof three-way 70 / 15 / 15 split',
     'leakage-controlled three-way 70 / 15 / 15 split'),
    ('bulletproof three-way 70 / 15 / 15',
     'leakage-controlled three-way 70 / 15 / 15'),
    ('bulletproof three-way',           'leakage-controlled three-way'),
    ('three-way bulletproof split',     'leakage-controlled three-way split'),
    ('bulletproof rerun',               'independent-audit rerun'),
    ('bulletproof reruns',              'independent-audit reruns'),
    ('Bulletproof rerun',               'Independent-audit rerun'),
    ('Bulletproof reruns',              'Independent-audit reruns'),
    ('bulletproof protocol',            'leakage-controlled protocol'),
    ('Bulletproof protocol',            'Leakage-controlled protocol'),
    ('bulletproof-protocol',            'leakage-controlled protocol'),
    ('Bulletproof-protocol',            'Leakage-controlled protocol'),
    ('bulletproof split',               'leakage-controlled split'),
    ('Bulletproof split',               'Leakage-controlled split'),
    ('bulletproof 70/15/15',            'stratified 70/15/15'),
    ('Bulletproof 70/15/15',            'Stratified 70/15/15'),
    ('bulletproof 70 / 15 / 15',        'stratified 70 / 15 / 15'),
    ('Bulletproof 70 / 15 / 15',        'Stratified 70 / 15 / 15'),
    ('the bulletproof audit partition', 'the held-out audit partition'),
    ('on the bulletproof audit',        'on the held-out audit'),
    ('bulletproof patient-stratified',  'leakage-controlled patient-stratified'),
    ('Bulletproof patient',             'Leakage-controlled patient'),
    ('journal-rerun bulletproof',       'journal-rerun leakage-controlled'),
    ('Journal-rerun bulletproof',       'Journal-rerun leakage-controlled'),
    ('bulletproof', 'leakage-controlled'),
    ('Bulletproof', 'Leakage-controlled'),
    ('BULLETPROOF', 'LEAKAGE-CONTROLLED'),

    # ----- "CIKM" used as a descriptor for the single-split methodology -----
    # NOT the venue references (we keep "CIKM 2026 submission", "CIKM-style review", etc.)
    ('CIKM single-split',  'single-split (in-sample tuning)'),
    ('CIKM-submission single 80/20 split',
                          'single-split 80/20 protocol (in-sample tuning)'),
    ('CIKM canonical setting', 'single-split canonical setting'),
    ('CIKM canonical',     'single-split canonical'),
    ('CIKM-canonical',     'single-split canonical'),
    ('the CIKM accuracy-cost figure',     'the single-split accuracy-cost figure'),
    ('the CIKM accuracy-cost number',     'the single-split accuracy-cost figure'),
    ('CIKM accuracy-cost headline',       'single-split accuracy-cost headline'),
    ('CIKM headline accuracy cost',       'single-split headline accuracy cost'),
    ('the CIKM headline',  'the single-split headline'),
    ('CIKM headline',      'single-split headline'),
    ('CIKM-reported',      'single-split-reported'),
    ('CIKM-reported stable count was 21/28',
                          'single-split stable count was 21/28'),
    ('vs CIKM',            'vs single-split'),
    ('(CIKM 0.9528',       '(single-split 0.9528'),
    ('(CIKM 4.24pp',       '(single-split 4.24pp'),
    ('(CIKM 0.801)',       '(single-split 0.801)'),
    ('(CIKM 0.800)',       '(single-split 0.800)'),
    ('(CIKM True)',        '(single-split True)'),
    ('(CIKM 21/28)',       '(single-split 21/28)'),
    ('CIKM single split',  'single-split protocol'),

    # ----- other informal terms -----
    ('headline number',    'principal figure'),
    ('headline numbers',   'principal figures'),
    ('headline outcomes',  'principal outcomes'),
    ('headline figure',    'principal figure'),
    ('Headline number',    'Principal figure'),
    ('Headline numbers',   'Principal figures'),
    ('Headline outcomes',  'Principal outcomes'),
    ('Headline figure',    'Principal figure'),

    # "ammunition" still survives in one place
    ('cover-letter material', 'response document material'),

    # "selection-on-test optimism" - keep as is (standard term)

    # title sections
    ('manuscript drop-in paragraphs (CIKM 2026 final-pass review)',
     'Manuscript-revision paragraphs (final consistency pass)'),
    ('Manuscript drop-in paragraphs (CIKM 2026 final-pass review)',
     'Manuscript-revision paragraphs (final consistency pass)'),
]

for nb_path, label in [(CIKM, 'CIKM'), (JNL, 'Journal')]:
    nb = json.loads(nb_path.read_text(encoding='utf-8'))
    n = 0
    for cell in nb['cells']:
        if cell['cell_type'] != 'markdown':
            continue
        src = cell['source']
        if isinstance(src, list): src = ''.join(src)
        orig = src
        for old, new in PATCHES:
            if old in src and old != new:
                cnt = src.count(old)
                src = src.replace(old, new)
                n += cnt
        if src != orig:
            cell['source'] = src.splitlines(keepends=True)
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f"{label}: {n} academic-wording replacements applied")
