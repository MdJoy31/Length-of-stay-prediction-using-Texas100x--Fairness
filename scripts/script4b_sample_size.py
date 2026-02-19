#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════
SCRIPT 4b: SAMPLE SIZE SENSITIVITY TEST (NEW!)
═══════════════════════════════════════════════════════════════════════════════════

Author: Md Jannatul Rakib Joy
Institution: Swinburne University of Technology

PURPOSE: Test how fairness metrics change with different sample sizes
METHOD: Subsample at N = [10K, 50K, 100K, 500K, Full], 30 repeats each
OUTPUT: Convergence curves showing when metrics stabilize

RESEARCH QUESTION: How much data is needed for reliable fairness assessment?

TIME: ~1-2 hours

RUN: python script4b_sample_size.py

═══════════════════════════════════════════════════════════════════════════════════
"""

import pickle, json, numpy as np, pandas as pd
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

CONFIG = {
    'processed_dir': Path('./processed_data'),
    'results_dir': Path('./results'),
    'figures_dir': Path('./figures'),
    'tables_dir': Path('./tables'),
    'sample_sizes': [10000, 50000, 100000, 500000, None],  # None = full dataset
    'repeats_per_size': 30,
    'test_size': 0.2,
    'random_state': 42
}

for d in [CONFIG['results_dir'], CONFIG['figures_dir'], CONFIG['tables_dir']]:
    d.mkdir(parents=True, exist_ok=True)

class FairnessCalc:
    def __init__(self, y_true, y_pred, y_prob, protected, attr_name):
        self.y_true, self.y_pred = np.array(y_true), np.array(y_pred)
        self.y_prob = np.array(y_prob) if y_prob is not None else None
        self.protected = np.array(protected)
        self.subgroups = np.unique(protected)
    
    def _safe_div(self, a, b): return a / b if b > 0 else 0.0
    
    def compute(self):
        results = {'per_subgroup': {}, 'disparities': {}}
        for g in self.subgroups:
            mask = self.protected == g
            yt, yp = self.y_true[mask], self.y_pred[mask]
            tp = ((yp==1)&(yt==1)).sum()
            ap, pp = (yt==1).sum(), (yp==1).sum()
            
            results['per_subgroup'][str(g)] = {
                'tpr': float(self._safe_div(tp, ap)),
                'ppv': float(self._safe_div(tp, pp)),
                'demographic_parity': float(self._safe_div(pp, mask.sum()))
            }
        
        for m in ['tpr', 'ppv', 'demographic_parity']:
            vals = [results['per_subgroup'][str(g)][m] for g in self.subgroups]
            results['disparities'][f'{m}_ratio'] = float(self._safe_div(min(vals), max(vals)) if max(vals) > 0 else 1.0)
        
        return results

def load_data():
    print("=" * 70 + "\nLOADING DATA\n" + "=" * 70)
    proc = CONFIG['processed_dir']
    
    X = np.load(proc / 'X_scaled.npy')
    y = np.load(proc / 'y.npy')
    
    with open(proc / 'protected_attributes.pkl', 'rb') as f:
        prot = pickle.load(f)
    
    print(f"✅ Total samples: {len(y):,}")
    return X, y, prot['protected'], prot['subgroups']

def run_sample_size_test(X, y, protected, subgroups):
    print("\n" + "=" * 70)
    print("SAMPLE SIZE SENSITIVITY TEST")
    print("=" * 70)
    
    n_total = len(y)
    sample_sizes = [s if s else n_total for s in CONFIG['sample_sizes']]
    sample_sizes = [s for s in sample_sizes if s <= n_total]
    
    print(f"\nSample sizes to test: {sample_sizes}")
    print(f"Repeats per size: {CONFIG['repeats_per_size']}")
    
    results = {
        'sample_sizes': sample_sizes,
        'by_attribute': {}
    }
    
    for attr in protected.keys():
        results['by_attribute'][attr] = {
            'disparities': {m: {s: [] for s in sample_sizes} for m in ['tpr_ratio', 'ppv_ratio', 'demographic_parity_ratio']},
            'per_subgroup': {sg: {m: {s: [] for s in sample_sizes} for m in ['tpr', 'ppv']} for sg in subgroups[attr]}
        }
    
    np.random.seed(CONFIG['random_state'])
    
    for size in tqdm(sample_sizes, desc="Sample Sizes"):
        for r in tqdm(range(CONFIG['repeats_per_size']), desc=f"Size {size:,}", leave=False):
            # Subsample
            if size < n_total:
                idx = np.random.choice(n_total, size, replace=False)
            else:
                idx = np.arange(n_total)
            
            X_sub, y_sub = X[idx], y[idx]
            
            # Split
            X_tr, X_te, y_tr, y_te, _, idx_te = train_test_split(
                X_sub, y_sub, np.arange(len(y_sub)),
                test_size=CONFIG['test_size'],
                random_state=CONFIG['random_state'] + r,
                stratify=y_sub
            )
            
            # Train
            model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=CONFIG['random_state'])
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            y_prob = model.predict_proba(X_te)[:, 1]
            
            # Compute fairness
            for attr, vals in protected.items():
                attr_sub = vals[idx]
                attr_te = attr_sub[idx_te]
                
                calc = FairnessCalc(y_te, y_pred, y_prob, attr_te, attr)
                res = calc.compute()
                
                # Store disparities
                for m in ['tpr_ratio', 'ppv_ratio', 'demographic_parity_ratio']:
                    results['by_attribute'][attr]['disparities'][m][size].append(res['disparities'][m])
                
                # Store per-subgroup
                for sg in subgroups[attr]:
                    if sg in res['per_subgroup']:
                        for m in ['tpr', 'ppv']:
                            results['by_attribute'][attr]['per_subgroup'][sg][m][size].append(res['per_subgroup'][sg][m])
    
    return results

def compute_statistics(results):
    print("\n" + "=" * 70 + "\nCOMPUTING STATISTICS\n" + "=" * 70)
    
    stats = {'sample_sizes': results['sample_sizes'], 'by_attribute': {}}
    
    for attr, data in results['by_attribute'].items():
        stats['by_attribute'][attr] = {'disparities': {}, 'per_subgroup': {}}
        
        # Disparity stats
        for m, size_data in data['disparities'].items():
            stats['by_attribute'][attr]['disparities'][m] = {}
            for size, vals in size_data.items():
                if vals:
                    stats['by_attribute'][attr]['disparities'][m][size] = {
                        'mean': float(np.mean(vals)),
                        'std': float(np.std(vals)),
                        'ci_lower': float(np.percentile(vals, 2.5)),
                        'ci_upper': float(np.percentile(vals, 97.5))
                    }
    
    # Print convergence
    print("\n📊 Convergence Analysis (TPR Ratio):")
    for attr, data in stats['by_attribute'].items():
        print(f"\n   {attr}:")
        for size in results['sample_sizes']:
            if size in data['disparities']['tpr_ratio']:
                s = data['disparities']['tpr_ratio'][size]
                print(f"      N={size:>7,}: Mean={s['mean']:.3f}, Std={s['std']:.4f}")
    
    return stats

def create_visualizations(results, stats):
    print("\n" + "=" * 70 + "\nCREATING VISUALIZATIONS\n" + "=" * 70)
    
    fig_dir = CONFIG['figures_dir']
    sizes = results['sample_sizes']
    
    # Figure 1: Convergence Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = {'tpr_ratio': '#27ae60', 'ppv_ratio': '#3498db', 'demographic_parity_ratio': '#9b59b6'}
    
    for idx, attr in enumerate(list(stats['by_attribute'].keys())[:4]):
        ax = axes[idx]
        data = stats['by_attribute'][attr]['disparities']
        
        for m, color in colors.items():
            means = [data[m][s]['mean'] for s in sizes if s in data[m]]
            stds = [data[m][s]['std'] for s in sizes if s in data[m]]
            valid_sizes = [s for s in sizes if s in data[m]]
            
            ax.errorbar(valid_sizes, means, yerr=stds, marker='o', capsize=5, 
                       label=m.replace('_ratio', '').upper(), color=color, linewidth=2)
        
        ax.axhline(y=0.8, color='red', linestyle='--', label='Fairness Threshold', alpha=0.7)
        ax.set_xlabel('Sample Size')
        ax.set_ylabel('Fairness Ratio')
        ax.set_title(f'{attr}', fontweight='bold')
        ax.set_xscale('log')
        ax.legend(loc='lower right', fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.grid(alpha=0.3)
    
    plt.suptitle('Sample Size Effect on Fairness Metrics\n(Error bars = 1 std)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_dir / 'sample_size_convergence.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved sample_size_convergence.png")
    
    # Figure 2: Variance Reduction
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for attr in list(stats['by_attribute'].keys())[:4]:
        data = stats['by_attribute'][attr]['disparities']['tpr_ratio']
        stds = [data[s]['std'] for s in sizes if s in data]
        valid_sizes = [s for s in sizes if s in data]
        ax.plot(valid_sizes, stds, marker='o', label=attr, linewidth=2)
    
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Standard Deviation')
    ax.set_title('Variance Reduction with Sample Size', fontweight='bold')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'sample_size_variance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved sample_size_variance.png")

def create_tables(stats):
    print("\n" + "=" * 70 + "\nCREATING TABLES\n" + "=" * 70)
    
    # Table: Convergence summary
    table_data = []
    for attr, data in stats['by_attribute'].items():
        for size in stats['sample_sizes']:
            if size in data['disparities']['tpr_ratio']:
                s = data['disparities']['tpr_ratio'][size]
                table_data.append({
                    'Attribute': attr,
                    'Sample_Size': size,
                    'TPR_Ratio_Mean': s['mean'],
                    'TPR_Ratio_Std': s['std'],
                    'TPR_Ratio_CI_Lower': s['ci_lower'],
                    'TPR_Ratio_CI_Upper': s['ci_upper']
                })
    
    pd.DataFrame(table_data).to_csv(CONFIG['tables_dir'] / 'sample_size_convergence.csv', index=False)
    print("✅ Saved sample_size_convergence.csv")

def save_results(results, stats):
    print("\n" + "=" * 70 + "\nSAVING RESULTS\n" + "=" * 70)
    
    with open(CONFIG['results_dir'] / 'sample_size_raw.pkl', 'wb') as f:
        pickle.dump(results, f)
    with open(CONFIG['results_dir'] / 'sample_size_stats.pkl', 'wb') as f:
        pickle.dump(stats, f)
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'sample_sizes': stats['sample_sizes'],
        'repeats_per_size': CONFIG['repeats_per_size']
    }
    with open(CONFIG['results_dir'] / 'sample_size_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("✅ Results saved")

def main():
    print("\n" + "📊" * 25 + "\n  SCRIPT 4b: SAMPLE SIZE SENSITIVITY\n" + "📊" * 25)
    start = datetime.now()
    
    X, y, protected, subgroups = load_data()
    results = run_sample_size_test(X, y, protected, subgroups)
    stats = compute_statistics(results)
    create_visualizations(results, stats)
    create_tables(stats)
    save_results(results, stats)
    
    print(f"\n✅ COMPLETE ({(datetime.now()-start).total_seconds()/60:.1f} min)")
    print("👉 NEXT: python script4c_cross_hospital.py")

if __name__ == "__main__":
    main()
