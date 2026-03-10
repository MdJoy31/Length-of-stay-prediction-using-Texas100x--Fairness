"""Fix FairnessCalculator: add __init__, change compute_all to instance method."""
import re

with open('build_notebook_v4.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add __init__ after METRIC_NAMES
old_init = "METRIC_NAMES = ['DI', 'SPD', 'EOPP', 'EOD', 'TI', 'PP', 'CAL']\n\n    @staticmethod\n    def is_fair"
new_init = """METRIC_NAMES = ['DI', 'SPD', 'EOPP', 'EOD', 'TI', 'PP', 'CAL']

    def __init__(self, y_true=None, y_pred=None, y_prob=None, attr=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob
        self.attr = attr

    @staticmethod
    def is_fair"""

if old_init in content:
    content = content.replace(old_init, new_init)
    print('Step 1 OK: __init__ added')
else:
    print('Step 1 FAILED: could not find METRIC_NAMES...@staticmethod pattern')

# 2. Change compute_all from @staticmethod to instance method
old_ca = "    @staticmethod\n    def compute_all(y_true, y_pred, y_prob, attr):"
new_ca = "    def compute_all(self, y_true=None, y_pred=None, y_prob=None, attr=None):"
if old_ca in content:
    content = content.replace(old_ca, new_ca)
    print('Step 2a OK: signature changed')
else:
    print('Step 2a FAILED')

# 3. Add self-delegation lines after the docstring in compute_all body
old_body = '        di, rates = FairnessCalculator.disparate_impact(y_pred, attr)\n        spd = FairnessCalculator.statistical_parity_diff(y_pred, attr)\n        eopp = FairnessCalculator.equal_opportunity_diff(y_true, y_pred, attr)\n        eod = FairnessCalculator.equalised_odds_diff(y_true, y_pred, attr)\n        ti = FairnessCalculator.theil_index(y_true, y_pred, y_prob)\n        pp = FairnessCalculator.predictive_parity(y_true, y_pred, attr)\n        cal = FairnessCalculator.calibration_diff(y_true, y_prob, attr)'
new_body = '        yt = y_true if y_true is not None else self.y_true\n        yp = y_pred if y_pred is not None else self.y_pred\n        ypb = y_prob if y_prob is not None else self.y_prob\n        at = attr if attr is not None else self.attr\n        di, rates = FairnessCalculator.disparate_impact(yp, at)\n        spd = FairnessCalculator.statistical_parity_diff(yp, at)\n        eopp = FairnessCalculator.equal_opportunity_diff(yt, yp, at)\n        eod = FairnessCalculator.equalised_odds_diff(yt, yp, at)\n        ti = FairnessCalculator.theil_index(yt, yp, ypb)\n        pp = FairnessCalculator.predictive_parity(yt, yp, at)\n        cal = FairnessCalculator.calibration_diff(yt, ypb, at)'

if old_body in content:
    content = content.replace(old_body, new_body, 1)  # only first occurrence in class def
    print('Step 2b OK: body updated with self-delegation')
else:
    print('Step 2b FAILED')

with open('build_notebook_v4.py', 'w', encoding='utf-8') as f:
    f.write(content)

lines = content.count('\n') + 1
print(f'File saved. Lines: {lines}')

# Verify
has_init = 'def __init__(self, y_true=None' in content
has_new_ca = 'def compute_all(self, y_true=None' in content
has_old_ca = '@staticmethod\n    def compute_all' in content
print(f'Has __init__: {has_init}')
print(f'Has new compute_all: {has_new_ca}')
print(f'Still has old @staticmethod compute_all: {has_old_ca}')
