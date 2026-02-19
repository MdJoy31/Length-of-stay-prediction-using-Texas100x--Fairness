#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════
SCRIPT 4c: CROSS-HOSPITAL VALIDATION TEST (NEW!)
═══════════════════════════════════════════════════════════════════════════════════

Author: Md Jannatul Rakib Joy
Institution: Swinburne University of Technology

PURPOSE: Test fairness metric consistency across different hospital sites
METHOD: Group 441 hospitals into K=50 folds, leave-one-fold-out validation
OUTPUT: Distribution of fairness metrics across hospitals, I² heterogeneity

RESEARCH QUESTION: Are fairness metrics generalizable across healthcare sites?

TIME: ~2 hours

RUN: python script4c_cross_hospital.py

═══════════════════════════════════════════════════════════════════════════════════
"""

import pickle, json, numpy as np, pandas as pd
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

CONFIG = {
    'processed_dir': Path('./processed_data'),
    'results_dir': Path('./results'),
    'figures_dir': Path('./figures'),
    'tables_dir': Path('./tables'),
    'hospital_folds': 50,  # K = 50 folds
    'min_samples_per_fold': 100,
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
            if mask.sum() < 10:  # Skip if too few samples
                continue
            yt, yp = self.y_true[mask], self.y_pred[mask]
            tp = ((yp==1)&(yt==1)).sum()
            ap, pp = (yt==1).sum(), (yp==1).sum()
            
            results['per_subgroup'][str(g)] = {
                'tpr': float(self._safe_div(tp, ap)),
                'ppv': float(self._safe_div(tp, pp)),
                'demographic_parity': float(self._safe_div(pp, mask.sum()))
            }
        
        if len(results['per_subgroup']) >= 2:
            for m in ['tpr', 'ppv', 'demographic_parity']:
                vals = [results['per_subgroup'][g][m] for g in results['per_subgroup']]
                results['disparities'][f'{m}_ratio'] = float(self._safe_div(min(vals), max(vals)) if max(vals) > 0 else 1.0)
        
        return results

def load_data():
    print("=" * 70 + "\nLOADING DATA\n" + "=" * 70)
    proc = CONFIG['processed_dir']
    
    X = np.load(proc / 'X_scaled.npy')
    y = np.load(proc / 'y.npy')
    
    with open(proc / 'protected_attributes.pkl', 'rb') as f:
        prot = pickle.load(f)
    
    # Load hospital IDs
    hospital_path = proc / 'hospital_ids.npy'
    if hospital_path.exists():
        hospital_ids = np.load(hospital_path)
        print(f"✅ Loaded {len(np.unique(hospital_ids))} unique hospitals")
    else:
        print("⚠️ Hospital IDs not found - creating synthetic hospital assignments")
        # Create synthetic hospital IDs if not available
        n_hospitals = 441
        hospital_ids = np.random.randint(1, n_hospitals + 1, len(y))
    
    print(f"✅ Total samples: {len(y):,}")
    return X, y, hospital_ids, prot['protected'], prot['subgroups']

def create_hospital_folds(hospital_ids, n_folds):
    """Group hospitals into folds."""
    unique_hospitals = np.unique(hospital_ids)
    np.random.seed(CONFIG['random_state'])
    np.random.shuffle(unique_hospitals)
    
    # Split hospitals into folds
    hospitals_per_fold = max(1, len(unique_hospitals) // n_folds)
    folds = []
    
    for i in range(n_folds):
        start = i * hospitals_per_fold
        if i == n_folds - 1:
            # Last fold gets remaining hospitals
            fold_hospitals = unique_hospitals[start:]
        else:
            fold_hospitals = unique_hospitals[start:start + hospitals_per_fold]
        
        if len(fold_hospitals) > 0:
            folds.append(fold_hospitals)
    
    print(f"\n✅ Created {len(folds)} hospital folds")
    print(f"   Hospitals per fold: {hospitals_per_fold}")
    
    return folds

def run_cross_hospital_validation(X, y, hospital_ids, protected, subgroups):
    print("\n" + "=" * 70)
    print(f"CROSS-HOSPITAL VALIDATION (K={CONFIG['hospital_folds']})")
    print("=" * 70)
    
    folds = create_hospital_folds(hospital_ids, CONFIG['hospital_folds'])
    
    results = {
        'n_folds': len(folds),
        'by_attribute': {}
    }
    
    for attr in protected.keys():
        results['by_attribute'][attr] = {
            'disparities': {m: [] for m in ['tpr_ratio', 'ppv_ratio', 'demographic_parity_ratio']},
            'fold_info': []
        }
    
    # Leave-one-fold-out validation
    for fold_idx, held_out_hospitals in enumerate(tqdm(folds, desc="Hospital Folds")):
        # Create train/test masks
        test_mask = np.isin(hospital_ids, held_out_hospitals)
        train_mask = ~test_mask
        
        n_test = test_mask.sum()
        n_train = train_mask.sum()
        
        if n_test < CONFIG['min_samples_per_fold'] or n_train < CONFIG['min_samples_per_fold']:
            continue
        
        # Split data
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        # Train model
        model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=CONFIG['random_state'])
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Compute fairness for each attribute
        for attr, vals in protected.items():
            attr_test = vals[test_mask]
            
            calc = FairnessCalc(y_test, y_pred, y_prob, attr_test, attr)
            res = calc.compute()
            
            if 'disparities' in res and res['disparities']:
                for m in ['tpr_ratio', 'ppv_ratio', 'demographic_parity_ratio']:
                    if m in res['disparities']:
                        results['by_attribute'][attr]['disparities'][m].append(res['disparities'][m])
                
                results['by_attribute'][attr]['fold_info'].append({
                    'fold': fold_idx,
                    'n_hospitals': len(held_out_hospitals),
                    'n_samples': n_test
                })
    
    return results

def compute_statistics(results):
    print("\n" + "=" * 70 + "\nCOMPUTING STATISTICS\n" + "=" * 70)
    
    stats = {'by_attribute': {}}
    
    for attr, data in results['by_attribute'].items():
        stats['by_attribute'][attr] = {'disparities': {}}
        
        for m, vals in data['disparities'].items():
            if vals:
                mean_val = np.mean(vals)
                stats['by_attribute'][attr]['disparities'][m] = {
                    'mean': float(mean_val),
                    'std': float(np.std(vals)),
                    'min': float(np.min(vals)),
                    'max': float(np.max(vals)),
                    'range': float(np.max(vals) - np.min(vals)),
                    'cv': float(np.std(vals) / mean_val) if mean_val > 0 else 0,
                    'n_folds': len(vals),
                    # I² heterogeneity statistic
                    'i_squared': compute_i_squared(vals)
                }
    
    # Print results
    print("\n📊 Cross-Hospital Heterogeneity (TPR Ratio):")
    for attr, data in stats['by_attribute'].items():
        if 'tpr_ratio' in data['disparities']:
            s = data['disparities']['tpr_ratio']
            print(f"\n   {attr}:")
            print(f"      Mean: {s['mean']:.3f}")
            print(f"      Std: {s['std']:.4f}")
            print(f"      Range: [{s['min']:.3f}, {s['max']:.3f}]")
            print(f"      CV: {s['cv']:.1%}")
            print(f"      I²: {s['i_squared']:.1%}")  # Heterogeneity
    
    return stats

def compute_i_squared(values):
    """Compute I² heterogeneity statistic."""
    if len(values) < 2:
        return 0.0
    
    k = len(values)
    mean_val = np.mean(values)
    
    # Q statistic (simplified)
    Q = sum((v - mean_val)**2 for v in values)
    
    # I² = (Q - (k-1)) / Q, bounded [0, 1]
    if Q > 0:
        i_sq = max(0, (Q - (k - 1)) / Q)
    else:
        i_sq = 0
    
    return i_sq

def create_visualizations(results, stats):
    print("\n" + "=" * 70 + "\nCREATING VISUALIZATIONS\n" + "=" * 70)
    
    fig_dir = CONFIG['figures_dir']
    
    # Figure 1: Box plots of fairness across hospitals
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, attr in enumerate(list(results['by_attribute'].keys())[:4]):
        ax = axes[idx]
        data = results['by_attribute'][attr]['disparities']
        
        plot_data = []
        for m in ['tpr_ratio', 'ppv_ratio', 'demographic_parity_ratio']:
            for v in data[m]:
                plot_data.append({'Metric': m.replace('_ratio', '').upper(), 'Ratio': v})
        
        if plot_data:
            df = pd.DataFrame(plot_data)
            sns.boxplot(data=df, x='Metric', y='Ratio', ax=ax, palette='Set2')
            ax.axhline(y=0.8, color='red', linestyle='--', label='Fairness Threshold')
            ax.set_title(f'{attr}', fontweight='bold')
            ax.set_ylim(0, 1.1)
            ax.legend()
    
    plt.suptitle('Cross-Hospital Fairness Heterogeneity\n(Each point = one hospital fold)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_dir / 'cross_hospital_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved cross_hospital_boxplot.png")
    
    # Figure 2: Distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, attr in enumerate(list(results['by_attribute'].keys())[:4]):
        ax = axes[idx]
        vals = results['by_attribute'][attr]['disparities']['tpr_ratio']
        
        if vals:
            ax.hist(vals, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
            ax.axvline(x=np.mean(vals), color='red', linestyle='-', linewidth=2, label=f'Mean: {np.mean(vals):.3f}')
            ax.axvline(x=0.8, color='green', linestyle='--', linewidth=2, label='Fair Threshold')
            ax.set_xlabel('TPR Ratio')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{attr}', fontweight='bold')
            ax.legend()
    
    plt.suptitle('Distribution of Equal Opportunity Ratio Across Hospitals', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_dir / 'cross_hospital_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved cross_hospital_distribution.png")

def create_tables(stats):
    print("\n" + "=" * 70 + "\nCREATING TABLES\n" + "=" * 70)
    
    table_data = []
    for attr, data in stats['by_attribute'].items():
        for m, s in data['disparities'].items():
            table_data.append({
                'Attribute': attr,
                'Metric': m.replace('_ratio', ''),
                'Mean': s['mean'],
                'Std': s['std'],
                'Min': s['min'],
                'Max': s['max'],
                'Range': s['range'],
                'CV': s['cv'],
                'I_Squared': s['i_squared'],
                'N_Folds': s['n_folds']
            })
    
    pd.DataFrame(table_data).to_csv(CONFIG['tables_dir'] / 'cross_hospital_heterogeneity.csv', index=False)
    print("✅ Saved cross_hospital_heterogeneity.csv")

def save_results(results, stats):
    print("\n" + "=" * 70 + "\nSAVING RESULTS\n" + "=" * 70)
    
    with open(CONFIG['results_dir'] / 'cross_hospital_raw.pkl', 'wb') as f:
        pickle.dump(results, f)
    with open(CONFIG['results_dir'] / 'cross_hospital_stats.pkl', 'wb') as f:
        pickle.dump(stats, f)
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'n_folds': results['n_folds'],
        'attributes': list(stats['by_attribute'].keys())
    }
    with open(CONFIG['results_dir'] / 'cross_hospital_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("✅ Results saved")

def main():
    print("\n" + "🏥" * 25 + "\n  SCRIPT 4c: CROSS-HOSPITAL VALIDATION\n" + "🏥" * 25)
    start = datetime.now()
    
    X, y, hospital_ids, protected, subgroups = load_data()
    results = run_cross_hospital_validation(X, y, hospital_ids, protected, subgroups)
    stats = compute_statistics(results)
    create_visualizations(results, stats)
    create_tables(stats)
    save_results(results, stats)
    
    print(f"\n✅ COMPLETE ({(datetime.now()-start).total_seconds()/60:.1f} min)")
    print("👉 NEXT: python script4d_seed_sensitivity.py")

if __name__ == "__main__":
    main()
