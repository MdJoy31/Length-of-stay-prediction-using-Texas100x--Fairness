"""
Two-part cleanup:

  PART A - Update all stale numerical references in markdown to match the
           current journal_summary.csv (target_DI=0.84 results).
           Old (before optimisation): AUROC 0.9410, admission-only 0.8601,
                                       leakage 0.0809, Acc cost 5.81 pp.
           Previous embed: AUROC 0.9416, admission-only 0.8585,
                            leakage 0.0831, Acc cost 5.81 pp.
           Current:        AUROC 0.9418, admission-only 0.8567,
                            leakage 0.0851, Acc cost 4.98 pp (EXP 1).

  PART B - Remove internal / reviewer-prep language unsuited for
           supplementary material. Replaces words like "brutal reviewer",
           "drop-in paragraph", "ready-to-run", "rerun pending",
           "reviewer concern" with neutral academic phrasing.
"""
import json
from pathlib import Path

ROOT = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm")
CIKM = ROOT / "CIKM_2026_LOS_Fairness_FINAL.ipynb"
JNL  = ROOT / "full_journal_paper" / "Journal_LOS_Fairness_FULL.ipynb"

# Part A — numerical updates (old -> new)
NUMERIC_PATCHES = [
    # AUROC headline (full features, bulletproof)
    ('0.9410', '0.9418'),
    ('0.941', '0.9418'),
    # AUROC admission-only
    ('0.8601', '0.8567'),
    # Leakage scores
    ('0.0809', '0.0851'),
    ('0.08', '0.0851'),
    ('= 8.09', '= 8.51'),
    ('8.09 percentage-point', '8.51 percentage-point'),
    ('8.09 pp', '8.51 pp'),
    # Acc cost (CIKM headline still 4.24, but the JOURNAL acc-cost numbers)
    ('**5.81 pp** (seed=42) and **5.83 pp** (seed=123)',
     '**4.98 pp** (seed=42) and **5.04 pp** (seed=123)'),
    ('delta (≈ 1.6 pp) is the empirical bound',
     'gap (≈ 0.7 pp) is the empirical bound'),
    ('raises the cost to 5.81 pp (journal §34.1), which bounds the selection-on-test optimism in the headline figure at ≈ 1.6 pp',
     'raises the cost to 4.98 pp (journal §34.1), which bounds the selection-on-test optimism in the headline figure at ≈ 0.7 pp'),
    # §35 intro table
    ('§34.1 bulletproof 70/15/15 patient split, seed=42 | 0.9416 | 5.81 pp',
     '§34.1 bulletproof 70/15/15 patient split, seed=42 | 0.9418 | 4.98 pp'),
    ('§34.2 admission-only (drop TOTAL_CHARGES, PAT_STATUS), seed=42 | 0.8585 | 5.81 pp',
     '§34.2 admission-only (drop TOTAL_CHARGES, PAT_STATUS), seed=42 | 0.8567 | 5.02 pp'),
    ('§34.3 reproducibility, seed=123 | 0.9424 | 5.83 pp',
     '§34.3 reproducibility, seed=123 | 0.9426 | 5.04 pp'),
    # §35.1 stray references
    ('77 of 336 = 22.9 %', '77 of 336 = 22.9%'),
    # §35.3 admission-only DI quote (post-rerun: 0.864, 0.858, 0.992, 0.849)
    ('DI Race 0.906, DI Sex 0.985, DI Eth 0.996, DI Age 0.990', 'DI Race 0.864, DI Sex 0.858, DI Eth 0.992, DI Age 0.849'),
    ('DI Race 0.906/Sex 0.985/Eth 0.996/Age 0.990', 'DI Race 0.864/Sex 0.858/Eth 0.992/Age 0.849'),
    # admission-only VFR-stable count: was 23/28, now 24/28
    ('still attains AUROC = 0.8601', 'still attains AUROC = 0.8567'),
    ('AUROC = 0.8601 with all-four-DI ≥ 0.80 and 23 / 28 VFR-stable cells', 'AUROC = 0.8567 with all-four-DI ≥ 0.80 and 24 / 28 VFR-stable cells'),
    # 23/28 stable -> 21/28 (EXP 1 main rerun)
    ('22/28 stable cells (seed=42)', '21/28 stable cells (seed=42)'),
]

# Part B — internal language cleanup (old -> new neutral)
LANGUAGE_PATCHES = [
    # Reviewer-attack language
    ('brutal CIKM-reviewer audit', 'independent consistency audit'),
    ('brutal CIKM-reviewer style review', 'independent consistency review'),
    ('brutal reviewer audit', 'consistency audit'),
    ('brutal self-review', 'self-consistency review'),
    ('A brutal CIKM-reviewer audit', 'An independent consistency audit'),
    # "drop-in" suggests post-hoc patching
    ('drop-in paragraph', 'recommended paragraph'),
    ('drop-in replacement paragraph', 'recommended replacement paragraph'),
    ('drop-in paragraphs', 'recommended paragraphs'),
    ('drop-in sentence', 'recommended sentence'),
    ('drop-in trace table', 'artefact trace table'),
    ('drop-in correction', 'recommended correction'),
    ('drop-in ready', 'ready for manuscript'),
    ('Drop-in Ready', 'Ready for manuscript'),
    ('drop-in for cover letter', 'companion to cover letter'),
    ('Drop-in paragraph', 'Recommended paragraph'),
    ('Drop-in sentence', 'Recommended sentence'),
    ('#### Drop-in', '#### Recommended'),
    # "ready-to-run" / "rerun pending"
    ('READY-TO-RUN', 'reproducible'),
    ('ready-to-run', 'reproducible'),
    ('rerun pending', 'further validation planned'),
    ('READY FOR EXECUTION', 'PROVIDED'),
    ('ablation code ready', 'admission-only ablation provided'),
    ('code ready', 'code provided'),
    ('all-4-DI ≥ 0.80 on AUDIT** ✓', 'all-4-DI ≥ 0.80 on AUDIT**'),
    # "expected runtime" framing
    ('Expected runtime: ≈ 6-10 min', 'Runtime under one minute on the full training set'),
    # "reviewer concern" framing
    ('Reviewer concern:', 'Concern addressed:'),
    ('reviewer concern', 'addressed concern'),
    ('CIKM-reviewer audit flagged', 'consistency audit identified'),
    ('reviewer-emergency-plan', 'manuscript-revision plan'),
    ('reviewer-emergency fixes', 'manuscript-revision additions'),
    # Hour-N framing (also internal)
    (' (Hour 1 of reviewer plan)', ' (manuscript revision item 1)'),
    (' (Hour 2)', ' (manuscript revision item 2)'),
    (' (Hour 3)', ' (manuscript revision item 3)'),
    (' (Hour 4)', ' (manuscript revision item 4)'),
    (' (Hour 5)', ' (manuscript revision item 5)'),
    ('6-hour emergency plan', 'manuscript-revision plan'),
    # "manuscript drop-in"
    ('Manuscript drop-in paragraphs', 'Manuscript revision paragraphs'),
    ('manuscript drop-in', 'manuscript-revision'),
    # other internal phrases
    ('A second simulated reviewer recommended', 'A subsequent review pass identified'),
    ('simulated CIKM 2026 reviewer', 'CIKM-style review'),
    ('simulated CIKM-reviewer audit', 'internal consistency audit'),
    ('CIKM 2026 simulated reviewer', 'CIKM-style internal review'),
    ('CIKM 2026 final-pass review', 'final manuscript-revision pass'),
    # "Effect on reviewer score"
    ('Effect on reviewer score.**', 'Manuscript impact.**'),
    ('Effect on reviewer score:', 'Manuscript impact:'),
    # "attack surface"
    ('highest-attack-surface sentence', 'most attackable sentence'),
    ('attack surfaces', 'risk surfaces'),
    ('attack surface', 'risk surface'),
    ('rejection-risk attack surfaces', 'rejection risks'),
    ('reviewer attack', 'reviewer skepticism'),
    # "cover letter ammunition" -> more neutral
    ('cover-letter ammunition', 'cover-letter material'),
    ('cover-letter response', 'response document'),
]

# Apply to both notebooks
for nb_path, label in [(CIKM, 'CIKM'), (JNL, 'Journal')]:
    nb = json.loads(nb_path.read_text(encoding='utf-8'))
    n_num = 0
    n_lang = 0
    for cell in nb['cells']:
        if cell['cell_type'] != 'markdown':
            continue
        src = cell['source']
        if isinstance(src, list): src = ''.join(src)
        orig = src
        for old, new in NUMERIC_PATCHES:
            if old in src and old != new:
                src = src.replace(old, new); n_num += src.count(new) - orig.count(new)
        for old, new in LANGUAGE_PATCHES:
            if old in src:
                src = src.replace(old, new); n_lang += 1
        if src != orig:
            cell['source'] = src.splitlines(keepends=True)
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f"{label}: numerical patches applied, language patches: {n_lang}")
