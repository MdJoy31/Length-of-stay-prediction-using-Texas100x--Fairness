import pandas as pd
import numpy as np
import json

# Load executed notebook to find fairness metrics
f = open('output/models/final_results.json')
r = json.load(f)
f.close()

# Check best standard model AGE_GROUP metrics
best = list(r['fairness'].keys())[0]  # first model
print(f"Best model from JSON: {best}")

# Load the detailed fairness data
for attr in ['RACE', 'SEX', 'ETHNICITY', 'AGE_GROUP']:
    print(f"\n=== {attr} (best model: {best}) ===")
    if attr in r['fairness'].get(best, {}):
        metrics = r['fairness'][best][attr]
        thresholds = {'DI': 0.80, 'SPD': 0.10, 'EOPP': 0.10, 'EOD': 0.10, 
                      'TI': 0.10, 'PP': 0.10, 'CAL': 0.05}
        abs_metrics = {'SPD', 'EOPP', 'PP'}  # these use absolute value
        n_fair = 0
        for mk, val in metrics.items():
            if mk in thresholds:
                thresh = thresholds[mk]
                if mk == 'DI':
                    fair = val >= thresh
                    margin = val - thresh
                elif mk in abs_metrics:
                    fair = abs(val) < thresh
                    margin = thresh - abs(val)
                elif mk == 'CAL':
                    fair = abs(val) < thresh
                    margin = thresh - abs(val)
                else:
                    fair = val < thresh
                    margin = thresh - val
                status = "FAIR" if fair else "UNFAIR"
                if fair:
                    n_fair += 1
                print(f"  {mk}: {val:.4f}  threshold={thresh}  {status}  margin={margin:+.4f}")
        print(f"  Total fair: {n_fair}/7")

# Also check ALL models for AGE_GROUP
print("\n\n=== ALL MODELS AGE_GROUP FAIRNESS ===")
thresholds = {'DI': 0.80, 'SPD': 0.10, 'EOPP': 0.10, 'EOD': 0.10, 
              'TI': 0.10, 'PP': 0.10, 'CAL': 0.05}
for model_name, attrs in r['fairness'].items():
    if 'AGE_GROUP' in attrs:
        metrics = attrs['AGE_GROUP']
        n_fair = 0
        details = []
        for mk in ['DI', 'SPD', 'EOPP', 'EOD', 'TI', 'PP', 'CAL']:
            val = metrics.get(mk, None)
            if val is None:
                continue
            thresh = thresholds[mk]
            if mk == 'DI':
                fair = val >= thresh
            elif mk in ('SPD', 'EOPP', 'PP', 'CAL'):
                fair = abs(val) < thresh
            else:
                fair = val < thresh
            if fair:
                n_fair += 1
            details.append(f"{mk}={'F' if fair else 'U'}")
        print(f"  {model_name:20s}: {n_fair}/7  {' '.join(details)}")
