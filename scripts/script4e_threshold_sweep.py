#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════
SCRIPT 6: THRESHOLD SWEEP TEST (τ = 99 steps)
═══════════════════════════════════════════════════════════════════════════════════

Author: Md Jannatul Rakib Joy
Institution: Swinburne University of Technology

PURPOSE: Test fairness metric sensitivity to classification threshold choice
ESTIMATED TIME: ~30 minutes
RUN: python script6_threshold_sweep.py

═══════════════════════════════════════════════════════════════════════════════════
"""

import pickle, json, numpy as np, pandas as pd
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

CONFIG = {
    'processed_dir': Path('./processed_data'),
    'models_dir': Path('./models'),
    'results_dir': Path('./results'),
    'figures_dir': Path('./figures'),
    'tables_dir': Path('./tables'),
    'threshold_steps': 99,
    'threshold_min': 0.01,
    'threshold_max': 0.99,
    'fairness_threshold': 0.8
}

for d in [CONFIG['results_dir'], CONFIG['figures_dir'], CONFIG['tables_dir']]:
    d.mkdir(parents=True, exist_ok=True)

class FairnessCalculator:
    def __init__(self, y_true, y_pred, y_prob, protected, attr_name):
        self.y_true, self.y_pred = np.array(y_true), np.array(y_pred)
        self.y_prob = np.array(y_prob) if y_prob is not None else None
        self.protected, self.attr_name = np.array(protected), attr_name
        self.subgroups = np.unique(protected)
    
    def _safe_div(self, a, b): return a / b if b > 0 else 0.0
    
    def compute_metrics(self):
        results = {'per_subgroup': {}, 'disparities': {}}
        for g in self.subgroups:
            mask = self.protected == g
            yt, yp = self.y_true[mask], self.y_pred[mask]
            tp = ((yp == 1) & (yt == 1)).sum()
            tn = ((yp == 0) & (yt == 0)).sum()
            fp = ((yp == 1) & (yt == 0)).sum()
            fn = ((yp == 0) & (yt == 1)).sum()
            n, actual_pos, actual_neg, pred_pos = mask.sum(), (yt==1).sum(), (yt==0).sum(), (yp==1).sum()
            
            results['per_subgroup'][str(g)] = {
                'demographic_parity': float(self._safe_div(pred_pos, n)),
                'tpr': float(self._safe_div(tp, actual_pos)),
                'fpr': float(self._safe_div(fp, actual_neg)),
                'ppv': float(self._safe_div(tp, pred_pos)),
                'accuracy': float(self._safe_div(tp + tn, n))
            }
        
        for m in ['demographic_parity', 'tpr', 'ppv']:
            vals = [results['per_subgroup'][str(g)][m] for g in self.subgroups]
            results['disparities'][f'{m}_ratio'] = float(self._safe_div(min(vals), max(vals)) if max(vals) > 0 else 1.0)
        return results

def load_data():
    print("=" * 70 + "\nSTEP 1: LOADING DATA\n" + "=" * 70)
    proc, models = CONFIG['processed_dir'], CONFIG['models_dir']
    y_test = np.load(proc / 'y_test.npy')
    idx_test = np.load(proc / 'idx_test.npy')
    with open(proc / 'protected_attributes.pkl', 'rb') as f:
        prot_data = pickle.load(f)
    y_prob = np.load(models / 'Logistic_Regression_y_prob.npy')
    print(f"✅ Loaded {len(y_test):,} samples")
    return y_test, idx_test, prot_data['protected'], prot_data['subgroups'], y_prob

def run_threshold_sweep(y_test, idx_test, protected, subgroups, y_prob):
    print("\n" + "=" * 70 + f"\nSTEP 2: THRESHOLD SWEEP (τ = {CONFIG['threshold_steps']} steps)\n" + "=" * 70)
    thresholds = np.linspace(CONFIG['threshold_min'], CONFIG['threshold_max'], CONFIG['threshold_steps'])
    
    results = {'thresholds': thresholds.tolist(), 'performance': {'accuracy': [], 'f1': []}, 'fairness': {}}
    for attr in protected.keys():
        results['fairness'][attr] = {'subgroups': {str(sg): defaultdict(list) for sg in subgroups[attr]}, 'disparities': defaultdict(list)}
    
    for tau in tqdm(thresholds, desc="Thresholds"):
        y_pred = (y_prob >= tau).astype(int)
        results['performance']['accuracy'].append(accuracy_score(y_test, y_pred))
        results['performance']['f1'].append(f1_score(y_test, y_pred))
        
        for attr_name, attr_values in protected.items():
            attr_test = attr_values[idx_test]
            calc = FairnessCalculator(y_test, y_pred, y_prob, attr_test, attr_name)
            metrics = calc.compute_metrics()
            
            for sg, m in metrics['per_subgroup'].items():
                for k, v in m.items():
                    results['fairness'][attr_name]['subgroups'][sg][k].append(v)
            for k, v in metrics['disparities'].items():
                results['fairness'][attr_name]['disparities'][k].append(v)
    
    return results

def find_optimal_thresholds(results, protected):
    print("\n" + "=" * 70 + "\nSTEP 3: FINDING OPTIMAL THRESHOLDS\n" + "=" * 70)
    thresholds = np.array(results['thresholds'])
    optimal = {}
    
    # Performance optimal
    acc = np.array(results['performance']['accuracy'])
    f1 = np.array(results['performance']['f1'])
    optimal['max_accuracy'] = {'threshold': thresholds[np.argmax(acc)], 'value': np.max(acc)}
    optimal['max_f1'] = {'threshold': thresholds[np.argmax(f1)], 'value': np.max(f1)}
    
    # Fairness optimal (per attribute)
    for attr in protected.keys():
        for metric in ['tpr_ratio', 'ppv_ratio']:
            ratios = np.array(results['fairness'][attr]['disparities'][metric])
            fair_mask = ratios >= CONFIG['fairness_threshold']
            if fair_mask.any():
                fair_thresholds = thresholds[fair_mask]
                fair_f1 = f1[fair_mask]
                best_idx = np.argmax(fair_f1)
                optimal[f'{attr}_{metric}_fair'] = {'threshold': fair_thresholds[best_idx], 'f1': fair_f1[best_idx], 'ratio': ratios[fair_mask][best_idx]}
    
    print(f"   Max Accuracy: τ={optimal['max_accuracy']['threshold']:.2f} ({optimal['max_accuracy']['value']:.3f})")
    print(f"   Max F1: τ={optimal['max_f1']['threshold']:.2f} ({optimal['max_f1']['value']:.3f})")
    return optimal

def create_visualizations(results, subgroups, optimal):
    print("\n" + "=" * 70 + "\nSTEP 4: CREATING VISUALIZATIONS\n" + "=" * 70)
    fig_dir, thresholds = CONFIG['figures_dir'], results['thresholds']
    colors = plt.cm.Set1(np.linspace(0, 1, 10))
    
    # Figure 1: TPR by subgroup across thresholds
    for attr_name, attr_data in results['fairness'].items():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        ax = axes[0]
        for i, sg in enumerate(subgroups[attr_name]):
            sg_key = str(sg)
            ax.plot(thresholds, attr_data['subgroups'][sg_key]['tpr'], label=sg, color=colors[i], linewidth=2)
        ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=2)
        ax.set_xlabel('Classification Threshold (τ)')
        ax.set_ylabel('TPR')
        ax.set_title(f'{attr_name}: TPR by Subgroup', fontweight='bold')
        ax.legend(loc='lower left')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        
        ax = axes[1]
        ax.plot(thresholds, attr_data['disparities']['tpr_ratio'], 'b-', linewidth=2, label='TPR Ratio')
        ax.axhline(y=CONFIG['fairness_threshold'], color='red', linestyle='--', label='Fair Threshold')
        ax.fill_between(thresholds, CONFIG['fairness_threshold'], 1, alpha=0.2, color='green', label='Fair Region')
        ax.set_xlabel('Classification Threshold (τ)')
        ax.set_ylabel('Fairness Ratio')
        ax.set_title(f'{attr_name}: Equal Opportunity Ratio', fontweight='bold')
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.1)
        ax.grid(alpha=0.3)
        
        plt.suptitle(f'Threshold Sweep Analysis: {attr_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(fig_dir / f'threshold_sweep_{attr_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    print(f"✅ Saved threshold sweep figures")
    
    # Figure 2: Performance-Fairness Trade-off
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, results['performance']['f1'], 'b-', linewidth=2, label='F1 Score')
    ax.plot(thresholds, results['performance']['accuracy'], 'g--', linewidth=2, label='Accuracy')
    
    attr = list(results['fairness'].keys())[0]
    ax.plot(thresholds, results['fairness'][attr]['disparities']['tpr_ratio'], 'r:', linewidth=2, label='TPR Ratio (Fairness)')
    
    ax.axhline(y=CONFIG['fairness_threshold'], color='red', linestyle='--', alpha=0.5)
    ax.axvline(x=optimal['max_f1']['threshold'], color='blue', linestyle=':', alpha=0.5, label=f"Opt F1: τ={optimal['max_f1']['threshold']:.2f}")
    
    ax.set_xlabel('Classification Threshold (τ)')
    ax.set_ylabel('Score / Ratio')
    ax.set_title('Performance-Fairness Trade-off Across Thresholds', fontweight='bold')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / 'threshold_performance_fairness_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved performance-fairness trade-off figure")

def create_tables(results, subgroups, optimal):
    print("\n" + "=" * 70 + "\nSTEP 5: CREATING TABLES\n" + "=" * 70)
    
    # Table: Metrics at key thresholds
    key_thresholds = [0.1, 0.25, 0.5, 0.75, 0.9]
    thresholds = np.array(results['thresholds'])
    
    table_data = []
    for tau in key_thresholds:
        idx = np.argmin(np.abs(thresholds - tau))
        row = {'Threshold': tau, 'Accuracy': results['performance']['accuracy'][idx], 'F1': results['performance']['f1'][idx]}
        for attr in results['fairness'].keys():
            row[f'{attr}_TPR_Ratio'] = results['fairness'][attr]['disparities']['tpr_ratio'][idx]
        table_data.append(row)
    
    pd.DataFrame(table_data).to_csv(CONFIG['tables_dir'] / 'threshold_key_values.csv', index=False)
    
    # Table: Optimal thresholds
    opt_data = [{'Criterion': k, **v} for k, v in optimal.items()]
    pd.DataFrame(opt_data).to_csv(CONFIG['tables_dir'] / 'threshold_optimal.csv', index=False)
    print(f"✅ Saved threshold tables")

def save_results(results, optimal):
    print("\n" + "=" * 70 + "\nSTEP 6: SAVING RESULTS\n" + "=" * 70)
    # Convert defaultdicts for pickling
    results_clean = {'thresholds': results['thresholds'], 'performance': results['performance'], 'fairness': {}}
    for attr, data in results['fairness'].items():
        results_clean['fairness'][attr] = {
            'subgroups': {sg: dict(m) for sg, m in data['subgroups'].items()},
            'disparities': dict(data['disparities'])
        }
    with open(CONFIG['results_dir'] / 'threshold_results.pkl', 'wb') as f:
        pickle.dump({'results': results_clean, 'optimal': optimal}, f)
    with open(CONFIG['results_dir'] / 'threshold_summary.json', 'w') as f:
        json.dump({'timestamp': datetime.now().isoformat(), 'steps': CONFIG['threshold_steps'], 'optimal': optimal}, f, indent=2, default=str)
    print(f"✅ Results saved")

def main():
    print("\n" + "📈" * 30 + "\n  SCRIPT 6: THRESHOLD SWEEP TEST\n" + "📈" * 30)
    start = datetime.now()
    
    y_test, idx_test, protected, subgroups, y_prob = load_data()
    results = run_threshold_sweep(y_test, idx_test, protected, subgroups, y_prob)
    optimal = find_optimal_thresholds(results, protected)
    create_visualizations(results, subgroups, optimal)
    create_tables(results, subgroups, optimal)
    save_results(results, optimal)
    
    print("\n" + "=" * 70 + "\n✅ THRESHOLD SWEEP COMPLETE\n" + "=" * 70)
    print(f"Time: {(datetime.now() - start).total_seconds() / 60:.1f} min")
    print("\n👉 NEXT: python script7_final_report.py")

if __name__ == "__main__":
    main()
