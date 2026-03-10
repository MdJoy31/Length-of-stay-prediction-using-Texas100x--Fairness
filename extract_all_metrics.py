"""Extract ALL key metrics from both executed notebooks for the final report."""
import json, re, os

def extract_all(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    metrics = {}
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] != 'code':
            continue
        text = ""
        for o in c.get('outputs', []):
            if o.get('output_type') == 'stream':
                text += ''.join(o.get('text', []))
            elif o.get('output_type') in ('display_data', 'execute_result'):
                data = o.get('data', {})
                if 'text/html' in data:
                    html = ''.join(data['text/html'])
                    clean = re.sub(r'<[^>]+>', ' ', html)
                    text += re.sub(r'\s+', ' ', clean).strip() + "\n"
                elif 'text/plain' in data:
                    text += ''.join(data['text/plain']) + "\n"

        # Model Performance
        if 'Best Model' in text and 'F1=' in text:
            m = re.search(r'Best Model:\s*(.+?)\s*\(F1=([\d.]+)\)', text)
            if m:
                metrics['best_model'] = m.group(1).strip()
                metrics['best_f1'] = float(m.group(2))

        # Training results per model
        for model_name in ['Logistic Regression', 'Random Forest', 'Gradient Boosting',
                           'XGBoost GPU', 'LightGBM GPU', 'PyTorch DNN']:
            if f'Training: {model_name}' in text or f'🔧 Training: {model_name}' in text:
                acc_m = re.search(r'Accuracy\s*│\s*([\d.]+)\s*│\s*([\d.]+)\s*│\s*([+\-][\d.]+)', text)
                auc_m = re.search(r'AUC-ROC\s*│\s*([\d.]+)\s*│\s*([\d.]+)\s*│\s*([+\-][\d.]+)', text)
                f1_m = re.search(r'F1-Score\s*│\s*—?\s*│\s*([\d.]+)', text)
                key = model_name.replace(' ', '_')
                if acc_m:
                    metrics[f'{key}_train_acc'] = float(acc_m.group(1))
                    metrics[f'{key}_test_acc'] = float(acc_m.group(2))
                    metrics[f'{key}_gap'] = float(acc_m.group(3))
                if auc_m:
                    metrics[f'{key}_test_auc'] = float(auc_m.group(2))
                if f1_m:
                    metrics[f'{key}_test_f1'] = float(f1_m.group(1))

        # Fair model
        if 'Fair model (optimized thresholds)' in text:
            m = re.search(r'Accuracy:\s*([\d.]+)\s*\|\s*F1:\s*([\d.]+)\s*\|\s*AUC:\s*([\d.]+)', text)
            if m:
                metrics['fair_acc'] = float(m.group(1))
                metrics['fair_f1'] = float(m.group(2))
                metrics['fair_auc'] = float(m.group(3))

        # AFCE results
        if 'AFCE' in text and 'Step 3' in text:
            # Get all alpha rows
            for line in text.split('\n'):
                m = re.match(r'\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\w)', line)
                if m:
                    a = float(m.group(1))
                    metrics[f'afce_a{a:.1f}_acc'] = float(m.group(2))
                    metrics[f'afce_a{a:.1f}_f1'] = float(m.group(3))
                    metrics[f'afce_a{a:.1f}_auc'] = float(m.group(4))
                    metrics[f'afce_a{a:.1f}_race'] = float(m.group(5))
                    metrics[f'afce_a{a:.1f}_sex'] = float(m.group(6))
                    metrics[f'afce_a{a:.1f}_eth'] = float(m.group(7))
                    metrics[f'afce_a{a:.1f}_age'] = float(m.group(8))
                    metrics[f'afce_a{a:.1f}_fair'] = m.group(9)

            if 'Best fair (3/4)' in text:
                m = re.search(r'Best fair \(3/4\):\s*a=([\d.]+)', text)
                if m:
                    metrics['afce_best_alpha'] = float(m.group(1))
                m = re.search(r'Accuracy:\s*([\d.]+)\s*\|\s*F1:\s*([\d.]+)\s*\|\s*AUC:\s*([\d.]+)', text)
                if m:
                    metrics['afce_best_acc'] = float(m.group(1))
                    metrics['afce_best_f1'] = float(m.group(2))
                    metrics['afce_best_auc'] = float(m.group(3))
                m = re.search(r'RACE DI:\s*([\d.]+)\s*\|\s*SEX DI:\s*([\d.]+)\s*\|\s*ETH DI:\s*([\d.]+)', text)
                if m:
                    metrics['afce_best_race_di'] = float(m.group(1))
                    metrics['afce_best_sex_di'] = float(m.group(2))
                    metrics['afce_best_eth_di'] = float(m.group(3))
                m = re.search(r'Fairness accuracy cost:\s*([\d.]+)%', text)
                if m:
                    metrics['afce_cost_pct'] = float(m.group(1))

        # Standard vs Fair comparison (from HTML table)
        if 'Standard Model' in text and 'Fairness-Derived' in text:
            for metric_name in ['RACE DI', 'SEX DI', 'ETHNICITY DI', 'AGE_GROUP DI',
                                'RACE WTPR', 'SEX WTPR', 'RACE EOD', 'SEX EOD']:
                m = re.search(rf'{metric_name}\s+([\d.]+)\s+([\d.]+)', text)
                if m:
                    key = metric_name.replace(' ', '_').lower()
                    metrics[f'std_{key}'] = float(m.group(1))
                    metrics[f'fair_{key}'] = float(m.group(2))

        # Paper comparison
        if 'IMPROVEMENT OVER PAPER' in text:
            m = re.search(r'F1:\s*([\d.]+)\s*vs\s*([\d.]+)', text)
            if m:
                metrics['our_f1_vs_paper'] = f"{m.group(1)} vs {m.group(2)}"

    return metrics

# Extract from Standard notebook
m = extract_all('final_notebooks/LOS_Prediction_Standard.ipynb')
print("=" * 70)
print("EXTRACTED METRICS FROM STANDARD NOTEBOOK")
print("=" * 70)
for k, v in sorted(m.items()):
    print(f"  {k}: {v}")

# Save as JSON
with open('final_notebooks/results/extracted_metrics.json', 'w') as f:
    json.dump(m, f, indent=2)
print(f"\nSaved to final_notebooks/results/extracted_metrics.json")
