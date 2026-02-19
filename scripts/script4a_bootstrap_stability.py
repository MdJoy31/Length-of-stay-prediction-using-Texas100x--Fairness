#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════
SCRIPT 4a: BOOTSTRAP STABILITY TEST (B=1,000)
═══════════════════════════════════════════════════════════════════════════════════

PURPOSE: Quantify sampling uncertainty via bootstrap resampling
METHOD: Resample test set 1,000 times, compute 95% CIs
TIME: ~2 hours

RUN: python script4a_bootstrap_stability.py

═══════════════════════════════════════════════════════════════════════════════════
"""

import pickle, json, numpy as np, pandas as pd
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
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
    'bootstrap_iterations': 1000,
    'confidence_level': 0.95,
    'random_state': 42
}

for d in [CONFIG['results_dir'], CONFIG['figures_dir'], CONFIG['tables_dir']]:
    d.mkdir(parents=True, exist_ok=True)

class FairnessCalc:
    def __init__(self, y_true, y_pred, y_prob, protected, attr_name):
        self.y_true, self.y_pred = np.array(y_true), np.array(y_pred)
        self.y_prob = np.array(y_prob) if y_prob is not None else None
        self.protected, self.attr_name = np.array(protected), attr_name
        self.subgroups = np.unique(protected)
    
    def _safe_div(self, a, b): return a / b if b > 0 else 0.0
    
    def compute(self):
        results = {'per_subgroup': {}, 'disparities': {}}
        for g in self.subgroups:
            mask = self.protected == g
            yt, yp = self.y_true[mask], self.y_pred[mask]
            tp, tn = ((yp==1)&(yt==1)).sum(), ((yp==0)&(yt==0)).sum()
            fp, fn = ((yp==1)&(yt==0)).sum(), ((yp==0)&(yt==1)).sum()
            n, ap, an, pp = mask.sum(), (yt==1).sum(), (yt==0).sum(), (yp==1).sum()
            
            results['per_subgroup'][str(g)] = {
                'demographic_parity': float(self._safe_div(pp, n)),
                'tpr': float(self._safe_div(tp, ap)),
                'fpr': float(self._safe_div(fp, an)),
                'ppv': float(self._safe_div(tp, pp)),
                'ece': float(self._compute_ece(g))
            }
        
        for m in ['demographic_parity', 'tpr', 'ppv']:
            vals = [results['per_subgroup'][str(g)][m] for g in self.subgroups]
            results['disparities'][f'{m}_ratio'] = float(self._safe_div(min(vals), max(vals)) if max(vals) > 0 else 1.0)
        return results
    
    def _compute_ece(self, g):
        if self.y_prob is None: return 0.0
        mask = self.protected == g
        yt, yp = self.y_true[mask], self.y_prob[mask]
        if len(yp) == 0: return 0.0
        ece = 0.0
        for i in range(10):
            bm = (yp >= i/10) & (yp < (i+1)/10)
            if bm.sum() > 0:
                ece += (bm.sum()/len(yp)) * abs(yt[bm].mean() - yp[bm].mean())
        return ece

def load_data():
    print("=" * 70 + "\nLOADING DATA\n" + "=" * 70)
    proc, models = CONFIG['processed_dir'], CONFIG['models_dir']
    
    X_test = np.load(proc / 'X_test.npy')
    y_test = np.load(proc / 'y_test.npy')
    idx_test = np.load(proc / 'idx_test.npy')
    
    with open(proc / 'protected_attributes.pkl', 'rb') as f:
        prot = pickle.load(f)
    
    with open(models / 'Logistic_Regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    y_prob = np.load(models / 'Logistic_Regression_y_prob.npy')
    
    print(f"✅ Test samples: {len(y_test):,}")
    return X_test, y_test, idx_test, prot['protected'], prot['subgroups'], model, y_prob

def run_bootstrap(X_test, y_test, idx_test, protected, subgroups, model, y_prob):
    print(f"\n" + "=" * 70 + f"\nBOOTSTRAP RESAMPLING (B={CONFIG['bootstrap_iterations']:,})\n" + "=" * 70)
    
    np.random.seed(CONFIG['random_state'])
    results = {attr: {'per_subgroup': defaultdict(lambda: defaultdict(list)), 'disparities': defaultdict(list)} 
               for attr in protected.keys()}
    
    for b in tqdm(range(CONFIG['bootstrap_iterations']), desc="Bootstrap"):
        idx = np.random.choice(len(y_test), len(y_test), replace=True)
        y_b, y_pred_b, y_prob_b = y_test[idx], model.predict(X_test[idx]), y_prob[idx]
        
        for attr, vals in protected.items():
            calc = FairnessCalc(y_b, y_pred_b, y_prob_b, vals[idx_test][idx], attr)
            r = calc.compute()
            for sg, m in r['per_subgroup'].items():
                for k, v in m.items():
                    results[attr]['per_subgroup'][sg][k].append(v)
            for k, v in r['disparities'].items():
                results[attr]['disparities'][k].append(v)
    
    return results

def compute_cis(results, subgroups):
    print("\n" + "=" * 70 + "\nCOMPUTING 95% CONFIDENCE INTERVALS\n" + "=" * 70)
    
    cis = {}
    for attr, data in results.items():
        cis[attr] = {'per_subgroup': {}, 'disparities': {}}
        
        for sg in subgroups[attr]:
            sg_key = str(sg)
            cis[attr]['per_subgroup'][sg_key] = {}
            for m in ['demographic_parity', 'tpr', 'fpr', 'ppv', 'ece']:
                vals = np.array(data['per_subgroup'][sg_key][m]) if sg_key in data['per_subgroup'] else np.array([])
                if len(vals) == 0:
                    cis[attr]['per_subgroup'][sg_key][m] = {
                        'mean': 0.0, 'std': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0, 'ci_width': 0.0
                    }
                else:
                    cis[attr]['per_subgroup'][sg_key][m] = {
                        'mean': float(np.mean(vals)),
                        'std': float(np.std(vals)),
                        'ci_lower': float(np.percentile(vals, 2.5)),
                        'ci_upper': float(np.percentile(vals, 97.5)),
                        'ci_width': float(np.percentile(vals, 97.5) - np.percentile(vals, 2.5))
                    }
        
        for d in ['demographic_parity_ratio', 'tpr_ratio', 'ppv_ratio']:
            vals = np.array(data['disparities'][d]) if d in data['disparities'] else np.array([])
            if len(vals) == 0:
                cis[attr]['disparities'][d] = {
                    'mean': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0, 'ci_width': 0.0
                }
            else:
                cis[attr]['disparities'][d] = {
                    'mean': float(np.mean(vals)),
                    'ci_lower': float(np.percentile(vals, 2.5)),
                    'ci_upper': float(np.percentile(vals, 97.5)),
                    'ci_width': float(np.percentile(vals, 97.5) - np.percentile(vals, 2.5))
                }
    
    # Print results
    for attr, data in cis.items():
        print(f"\n📊 {attr}:")
        for sg in subgroups[attr]:
            sg_key = str(sg)
            ci = data['per_subgroup'][sg_key]['tpr']
            print(f"   {sg}: TPR = {ci['mean']:.3f} [{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}] (width: {ci['ci_width']:.4f})")
    
    return cis

def create_visualizations(results, cis, subgroups):
    print("\n" + "=" * 70 + "\nCREATING VISUALIZATIONS\n" + "=" * 70)
    
    # Forest plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, attr in enumerate(list(cis.keys())[:4]):
        ax = axes[idx]
        groups = [str(g) for g in subgroups[attr]]
        y_pos = np.arange(len(groups))
        
        means = [cis[attr]['per_subgroup'][g]['tpr']['mean'] for g in groups]
        errors = [[cis[attr]['per_subgroup'][g]['tpr']['mean'] - cis[attr]['per_subgroup'][g]['tpr']['ci_lower'],
                   cis[attr]['per_subgroup'][g]['tpr']['ci_upper'] - cis[attr]['per_subgroup'][g]['tpr']['mean']] 
                  for g in groups]
        
        ax.errorbar(means, y_pos, xerr=np.array(errors).T, fmt='o', capsize=6, color='#2980b9', markersize=10)
        ax.axvline(x=np.mean(means), color='red', linestyle='--', label=f'Mean: {np.mean(means):.3f}')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(groups)
        ax.set_xlabel('TPR')
        ax.set_title(f'{attr}', fontweight='bold')
        ax.set_xlim(0, 1)
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.suptitle('Bootstrap 95% CI for Equal Opportunity (TPR)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(CONFIG['figures_dir'] / 'bootstrap_ci_forest_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved bootstrap_ci_forest_plot.png")

def save_results(results, cis):
    print("\n" + "=" * 70 + "\nSAVING RESULTS\n" + "=" * 70)
    
    # Convert defaultdicts to regular dicts for pickling
    results_regular = {}
    for attr, data in results.items():
        results_regular[attr] = {
            'per_subgroup': {sg: dict(m) for sg, m in data['per_subgroup'].items()},
            'disparities': dict(data['disparities'])
        }
    
    with open(CONFIG['results_dir'] / 'bootstrap_raw.pkl', 'wb') as f:
        pickle.dump(results_regular, f)
    with open(CONFIG['results_dir'] / 'bootstrap_cis.pkl', 'wb') as f:
        pickle.dump(cis, f)
    
    with open(CONFIG['results_dir'] / 'bootstrap_summary.json', 'w') as f:
        json.dump({'timestamp': datetime.now().isoformat(), 'iterations': CONFIG['bootstrap_iterations']}, f)
    
    print("✅ Results saved")

def main():
    print("\n" + "🔄" * 25 + "\n  SCRIPT 4a: BOOTSTRAP STABILITY\n" + "🔄" * 25)
    start = datetime.now()
    
    X_test, y_test, idx_test, protected, subgroups, model, y_prob = load_data()
    results = run_bootstrap(X_test, y_test, idx_test, protected, subgroups, model, y_prob)
    cis = compute_cis(results, subgroups)
    create_visualizations(results, cis, subgroups)
    save_results(results, cis)
    
    print(f"\n✅ COMPLETE ({(datetime.now()-start).total_seconds()/60:.1f} min)")
    print("👉 NEXT: python script4b_sample_size.py")

if __name__ == "__main__":
    main()
