"""
Rebuild v4 from v3 (clean source) + targeted fixes + new sections from part2.
"""

# Read original v3 (known clean)
with open('build_notebook_v3.py', 'r', encoding='utf-8') as f:
    v3 = f.read()

# Read the new sections (part2, also clean since it was freshly created)
with open('build_notebook_v4_part2.py', 'r', encoding='utf-8') as f:
    part2 = f.read()

# ─── STEP 1: Apply FairnessCalculator replacement ───
# Find the old FairnessCalculator class and replace it entirely
old_fc_start = "class FairnessCalculator:"
old_fc_end_marker = 'fc = FairnessCalculator()\nprint('

# Find indices
fc_start = v3.index(old_fc_start)
fc_end_line = v3.index(old_fc_end_marker)
# Find end of the print line
fc_end = v3.index('\n', fc_end_line + len(old_fc_end_marker)) + 1

# The new FairnessCalculator class with __init__ and instance compute_all
new_fc = '''class FairnessCalculator:
    # Compute 7 fairness metrics aligned with manuscript definitions.
    # Metrics: DI, SPD, EOPP, EOD, TI, PP, CAL
    # Supports multi-group attributes (not just binary).

    THRESHOLDS = {
        'DI':   {'threshold': 0.80, 'direction': 'gte', 'label': 'DI >= 0.80'},
        'SPD':  {'threshold': 0.10, 'direction': 'abs_lt', 'label': '|SPD| < 0.10'},
        'EOPP': {'threshold': 0.10, 'direction': 'abs_lt', 'label': '|EOPP| < 0.10'},
        'EOD':  {'threshold': 0.10, 'direction': 'lt', 'label': 'EOD < 0.10'},
        'TI':   {'threshold': 0.10, 'direction': 'lt', 'label': 'TI < 0.10'},
        'PP':   {'threshold': 0.10, 'direction': 'abs_lt', 'label': '|PP| < 0.10'},
        'CAL':  {'threshold': 0.05, 'direction': 'lt', 'label': 'CAL < 0.05'},
    }
    METRIC_NAMES = ['DI', 'SPD', 'EOPP', 'EOD', 'TI', 'PP', 'CAL']

    def __init__(self, y_true=None, y_pred=None, y_prob=None, attr=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob
        self.attr = attr

    @staticmethod
    def is_fair(metric_name, value):
        t = FairnessCalculator.THRESHOLDS[metric_name]
        if t['direction'] == 'gte': return value >= t['threshold']
        elif t['direction'] == 'lt': return value < t['threshold']
        elif t['direction'] == 'abs_lt': return abs(value) < t['threshold']
        return False

    @staticmethod
    def disparate_impact(y_pred, attr):
        groups = sorted(set(attr))
        rates = {g: y_pred[attr==g].mean() for g in groups if (attr==g).sum() > 0}
        if len(rates) < 2: return 1.0, rates
        vals = list(rates.values())
        return (min(vals)/max(vals) if max(vals) > 0 else 0), rates

    @staticmethod
    def statistical_parity_diff(y_pred, attr):
        groups = sorted(set(attr))
        rates = [y_pred[attr==g].mean() for g in groups if (attr==g).sum() > 0]
        return max(rates) - min(rates) if len(rates) >= 2 else 0

    @staticmethod
    def equal_opportunity_diff(y_true, y_pred, attr):
        groups = sorted(set(attr))
        tprs = []
        for g in groups:
            mask = (attr==g) & (y_true==1)
            if mask.sum() > 0: tprs.append(y_pred[mask].mean())
        return max(tprs) - min(tprs) if len(tprs) >= 2 else 0

    @staticmethod
    def equalised_odds_diff(y_true, y_pred, attr):
        groups = sorted(set(attr))
        tprs, fprs = [], []
        for g in groups:
            mask = attr==g
            if mask.sum() == 0: continue
            pos = y_true[mask]==1; neg = y_true[mask]==0
            if pos.sum() > 0: tprs.append(y_pred[mask][pos].mean())
            if neg.sum() > 0: fprs.append(y_pred[mask][neg].mean())
        tpr_gap = max(tprs) - min(tprs) if len(tprs) >= 2 else 0
        fpr_gap = max(fprs) - min(fprs) if len(fprs) >= 2 else 0
        return max(tpr_gap, fpr_gap)

    @staticmethod
    def theil_index(y_true, y_pred, y_prob=None):
        if y_prob is None:
            y_prob = y_pred.astype(float)
        benefits = 1.0 - np.abs(y_true.astype(float) - y_prob)
        benefits = np.clip(benefits, 1e-10, None)
        mu = benefits.mean()
        if mu <= 0: return 0.0
        ratios = benefits / mu
        ti = np.mean(ratios * np.log(ratios + 1e-10))
        return max(0, ti)

    @staticmethod
    def predictive_parity(y_true, y_pred, attr):
        groups = sorted(set(attr))
        ppvs = []
        for g in groups:
            mask = (attr==g) & (y_pred==1)
            if mask.sum() > 0: ppvs.append(y_true[mask].mean())
        return max(ppvs) - min(ppvs) if len(ppvs) >= 2 else 0

    @staticmethod
    def calibration_diff(y_true, y_prob, attr, n_bins=10):
        groups = sorted(set(attr)); max_diff = 0
        for g in groups:
            mask = attr==g
            if mask.sum() < n_bins: continue
            try:
                pt, pp = calibration_curve(y_true[mask], y_prob[mask], n_bins=n_bins)
                max_diff = max(max_diff, np.max(np.abs(pt - pp)))
            except: pass
        return max_diff

    def compute_all(self, y_true=None, y_pred=None, y_prob=None, attr=None):
        yt = y_true if y_true is not None else self.y_true
        yp = y_pred if y_pred is not None else self.y_pred
        ypb = y_prob if y_prob is not None else self.y_prob
        at = attr if attr is not None else self.attr
        di, rates = FairnessCalculator.disparate_impact(yp, at)
        spd = FairnessCalculator.statistical_parity_diff(yp, at)
        eopp = FairnessCalculator.equal_opportunity_diff(yt, yp, at)
        eod = FairnessCalculator.equalised_odds_diff(yt, yp, at)
        ti = FairnessCalculator.theil_index(yt, yp, ypb)
        pp = FairnessCalculator.predictive_parity(yt, yp, at)
        cal = FairnessCalculator.calibration_diff(yt, ypb, at)
        metrics = dict(DI=di, SPD=spd, EOPP=eopp, EOD=eod, TI=ti, PP=pp, CAL=cal)
        verdicts = {m: FairnessCalculator.is_fair(m, v) for m, v in metrics.items()}
        return metrics, verdicts, rates

fc = FairnessCalculator()
print("FairnessCalculator initialised - 7 manuscript-aligned metrics ready")
'''

content = v3[:fc_start] + new_fc + v3[fc_end:]
print(f"Step 1: Replaced FairnessCalculator class ({fc_end - fc_start} -> {len(new_fc)} chars)")

# ─── STEP 2: Update Cell 28 — compute_all usage (stores all_fairness/all_verdicts) ───
# Find old pattern and replace

old_cell28_pattern = "all_fairness = {}"
if old_cell28_pattern in content:
    # Find the cell 28 boundary
    c28_start = content.index("# Cell 28")
    # Find next cell or section marker
    c28_code_start = content.index("all_fairness = {}", c28_start)
    # Find the end of this code block (next '""")')
    c28_end = content.index('""")', c28_code_start) + 4

    new_cell28_body = '''all_fairness = {}
all_verdicts = {}
METRIC_KEYS = ['DI','SPD','EOPP','EOD','TI','PP','CAL']
B = 1000  # bootstrap iterations

for name, preds in test_predictions.items():
    y_p = preds['y_pred']; y_pb = preds['y_prob']
    all_fairness[name] = {}
    all_verdicts[name] = {}
    for attr in ['RACE','SEX','ETHNICITY','AGE_GROUP']:
        attr_vals = protected_attrs[attr]
        fc_m = FairnessCalculator(y_test, y_p, y_pb, attr_vals)
        metrics, verdicts, rates = fc_m.compute_all()
        all_fairness[name][attr] = metrics
        all_verdicts[name][attr] = verdicts

print(f"Fairness computed: {len(all_fairness)} models x 4 attributes x 7 metrics")
for name in list(all_fairness.keys())[:3]:
    f = all_fairness[name]['RACE']
    v = all_verdicts[name]['RACE']
    n_fair = sum(v.values())
    print(f"  {name}: DI={f['DI']:.3f} SPD={f['SPD']:.3f} EOPP={f['EOPP']:.3f} EOD={f['EOD']:.3f} TI={f['TI']:.3f} PP={f['PP']:.3f} CAL={f['CAL']:.3f} [{n_fair}/7 fair]")
""")'''

    content = content[:c28_code_start] + new_cell28_body + content[c28_end:]
    print("Step 2: Updated Cell 28 with all_fairness/all_verdicts + METRIC_KEYS")
else:
    print("Step 2 SKIPPED: Could not find Cell 28 pattern")

# ─── STEP 3: Find Section 9 marker and cut everything from there to end ───
# Then append part2 (which starts with Section 9)
section9_markers = ["# SECTION 9", "###\n# SECTION 9"]
cut_point = None
for marker in section9_markers:
    try:
        idx = content.index(marker)
        # Go back to the ### line before it
        line_start = content.rfind('\n', 0, idx) + 1
        if content[line_start:line_start+3] == '###':
            cut_point = line_start
        else:
            # Go back further to find the ### delimiter
            prev_line = content.rfind('\n', 0, line_start-1) + 1
            if content[prev_line:prev_line+3] == '###':
                cut_point = prev_line
            else:
                cut_point = line_start
        print(f"Found Section 9 at char {idx}, cutting at char {cut_point}")
        break
    except ValueError:
        continue

if cut_point is None:
    print("ERROR: Could not find Section 9 marker")
else:
    # Take everything before Section 9
    before_s9 = content[:cut_point]
    # Append part2 (which starts with Section 9)
    content = before_s9 + '\n' + part2
    print(f"Step 3: Cut at Section 9 and appended part2 ({len(part2)} chars)")

# ─── STEP 4: Verify syntax ───
try:
    compile(content, 'build_notebook_v4.py', 'exec')
    print("\nSyntax verification: OK!")
except SyntaxError as e:
    print(f"\nSyntax ERROR at line {e.lineno}: {e.msg}")
    lines = content.split('\n')
    for i in range(max(0, e.lineno-3), min(len(lines), e.lineno+3)):
        marker = '>>>' if i == e.lineno-1 else '   '
        print(f"  {marker} L{i+1}: {lines[i][:120]}")

# ─── Save ───
with open('build_notebook_v4.py', 'w', encoding='utf-8') as f:
    f.write(content)

lines_count = content.count('\n') + 1
print(f"\nSaved build_notebook_v4.py: {lines_count} lines")
