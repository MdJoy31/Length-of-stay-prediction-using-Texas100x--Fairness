"""
Deeper academic-wording pass.

Replaces remaining informal / colloquial / promotional / reviewer-prep
terms in both notebook markdown cells. Preserves all numbers and code.
"""
import json, re, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from pathlib import Path

ROOT = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm")
CIKM = ROOT / "CIKM_2026_LOS_Fairness_FINAL.ipynb"
JNL  = ROOT / "full_journal_paper" / "Journal_LOS_Fairness_FULL.ipynb"

# (pattern, replacement). Compiled with re.IGNORECASE except where uppercase forms differ.
PATTERNS = [
    # ----- "headline" -> "principal" or "main" -----
    (r'\bheadline\s+AUROC\b',          'main AUROC'),
    (r'\bheadline\s+accuracy[-\s]cost\b', 'main accuracy-cost'),
    (r'\bheadline\s+accuracy\b',       'main accuracy'),
    (r'\bheadline\s+claim\b',          'principal claim'),
    (r'\bheadline\s+claims\b',         'principal claims'),
    (r'\bheadline\s+DI\b',             'main DI'),
    (r'\bheadline\s+metric',           'principal metric'),
    (r'\bheadline\s+narrative\b',      'main narrative'),
    (r'\bheadline\s+configuration\b',  'principal configuration'),
    (r'\bheadline\s+admission-only',   'principal admission-only'),
    (r'\bheadline\s+performance',      'main performance'),
    (r'\bheadline\s+differences\b',    'principal differences'),
    (r'\bheadline\s+direction\b',      'principal-direction'),
    (r'\bheadline-direction\b',        'principal-direction'),
    (r'\bheadline\s+admission-time\b', 'main admission-time'),
    (r'\bheadline\s+result\b',         'main result'),
    (r'\bheadline\s+results\b',        'main results'),
    (r'\bheadline\s+figure\b',         'principal figure'),
    (r'\bheadline\s+number\b',         'principal figure'),
    (r'\bheadline\s+numbers\b',        'principal figures'),
    (r'\bheadline\s+outcomes\b',       'principal outcomes'),
    (r'\bheadline\s+reconciliation\b', 'reconciliation'),
    (r'\bheadline\b',                  'main'),
    (r'\bHeadline\b',                  'Main'),

    # ----- "honestly" / "honest" softer -----
    (r'\bhonestly\b',                  'explicitly'),
    (r'\b[Hh]onest disclosure\b',      'Explicit disclosure'),
    (r'\bhonest findings?\b',          'transparent findings'),
    (r'\bhonest reframing\b',          'transparent reframing'),
    (r'\bhonest framing\b',            'transparent framing'),
    (r'\bhonest journal-version claim\b', 'journal-version claim'),
    (r'\bhonest admission-time\b',     'admission-time'),
    (r'\bhonest reporting\b',          'transparent reporting'),
    (r'\bhonest answer\b',             'precise answer'),
    (r'\b\(honest\)\b',                ''),
    (r'\bhonestly\.',                  'explicitly.'),

    # ----- conversational / casual openings -----
    (r'\bbasically\b',                 ''),
    (r'\bturns out\b',                 'we observe'),
    (r'\byou guessed it\b',            ''),
    (r'\bguess what\b',                ''),
    (r'\bTBH\b',                       ''),
    (r'\btbh\b',                       ''),
    (r'\bfrankly\b',                   ''),
    (r'\bsort of\b',                   ''),
    (r'\bkind of\b',                   ''),

    # ----- AI-cliche / over-used -----
    (r'\bdelve into\b',                'examine'),
    (r'\bin conclusion\b',             'In conclusion'),     # leave acceptable
    (r'\bcomprehensive(?:ly)?\b',      'thorough'),
    (r'\bcrucial(?:ly)?\b',            'important'),
    (r'\bvital(?:ly)?\b',              'important'),
    (r'\babsolutely\b',                ''),
    (r'\bdefinitely\b',                ''),
    (r'\bof utmost\b',                 'of high'),
    (r'\bit is worth noting that\b',   ''),
    (r'\bit should be noted that\b',   ''),
    (r'\bit is important to note that\b', ''),

    # ----- hype / promo -----
    (r'\bsmashing\b',                  'effective'),
    (r'\bhuge(?:ly)?\b',                'substantial'),
    (r'\bmassive(?:ly)?\b',            'substantial'),
    (r'\bgame[-\s]chang(?:er|ing)\b',  'methodological-change'),
    (r'\bbest[-\s]in[-\s]class\b',     'highest-performing'),
    (r'\btrump card\b',                'principal contribution'),
    (r'\bsilver bullet\b',             'single solution'),
    (r'\blow-hanging fruit\b',         'most accessible improvement'),
    (r'\bnext-level\b',                'improved'),
    (r'\bsuper-'  ,                    ''),

    # ----- combat / attack metaphors -----
    (r'\bkill(?:er|ed)\b',             ''),
    (r'\bsavage\b',                    'thorough'),
    (r'\bbrutal(?:ly)?\b',             'thorough'),
    (r'\bcrush(?:ed|ing)?\b',          ''),
    (r'\bweaponi[sz]e[ds]?\b',         'apply'),
    (r'\bammunition\b',                'supporting material'),
    (r'\battack surfaces?\b',          'risk areas'),
    (r'\battackable\b',                'questionable'),
    (r'\bpunch[-\s]?list\b',           'task list'),
    (r'\bpunchy\b',                    'concise'),
    (r'\bsurgical fix(?:-up|es)?\b',   'targeted correction'),
    (r'\bsurgical fix\b',              'targeted correction'),
    (r'\bquick win\b',                 'short-term improvement'),

    # ----- reviewer-prep residue -----
    (r'\breviewer attack\b',           'reviewer skepticism'),
    (r'\breviewer-attack\b',           'reviewer-skepticism'),
    (r'\breviewer concerns?\b',        'review comments'),
    (r'\b6-hour emergency plan\b',     'manuscript-revision plan'),
    (r'\bManuscript impact\.\*\*',     'Manuscript implication.**'),
    (r'\bManuscript impact:\*\*',      'Manuscript implication:**'),

    # ----- specific phrasings -----
    (r'\bcopy-paste-ready\b',          'ready for direct inclusion'),
    (r'\bcopy-and-paste-ready\b',      'ready for direct inclusion'),

    # ----- "single highest-leverage paragraph" -----
    (r'\bsingle highest-leverage\b',   'most impactful'),

    # ----- "fix-up" / "fixups" -----
    (r'\bfix-up(s)?\b',                r'correction\1'),
    (r'\bFix-up(s)?\b',                r'Correction\1'),

    # ----- "fragile" -----
    (r'\breviewer-fragile\b',          'review-sensitive'),

    # ----- "smashing case" / verbs -----
    (r'\bsmash\b',                     ''),

    # ----- "ammunition" derivatives -----
    (r'\bsupporting ammunition\b',     'supporting material'),

    # ----- "go-to" -----
    (r'\bgo-to\b',                     'standard'),

    # ----- "battle-tested" -----
    (r'\bbattle-tested\b',             'validated'),

    # Empty repeated spaces after deletions
    (r'  +',                           ' '),
    (r' \.',                           '.'),
    (r' ,',                            ','),
]

# Compile patterns
COMPILED = [(re.compile(p), r) for p, r in PATTERNS]

for nb_path, label in [(CIKM, 'CIKM'), (JNL, 'Journal')]:
    nb = json.loads(nb_path.read_text(encoding='utf-8'))
    total_changes = 0
    for cell in nb['cells']:
        if cell['cell_type'] != 'markdown':
            continue
        src = cell['source']
        if isinstance(src, list): src = ''.join(src)
        orig = src
        for rx, repl in COMPILED:
            src, n = rx.subn(repl, src)
            total_changes += n
        if src != orig:
            cell['source'] = src.splitlines(keepends=True)
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f"{label}: {total_changes} replacements applied")
