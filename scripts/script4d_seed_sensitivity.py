#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════
SCRIPT 5: RANDOM SEED SENSITIVITY TEST (S=50)
═══════════════════════════════════════════════════════════════════════════════════

Author: Md Jannatul Rakib Joy
Supervisors: Dr. Caslon Chua, Dr. Viet Vo
Institution: Swinburne University of Technology

PURPOSE: Test fairness metric stability across different random seeds

METHOD:
    1. For each of S=50 different random seeds:
       a. Split data with this seed
       b. Train model with this seed
       c. Compute all 5 fairness metrics
    2. Calculate mean, std, and coefficient of variation (CV)
    3. Assess metric stability across train/test splits

OUTPUT:
    - Metric distributions across seeds
    - Coefficient of Variation analysis
    - Violin plots showing variance
    - ICC (Intraclass Correlation Coefficient)

ESTIMATED TIME: ~2 hours

REQUIRES: script1, script2, script3
RUN: python script5_seed_sensitivity.py
NEXT: python script6_threshold_sweep.py

═══════════════════════════════════════════════════════════════════════════════════
"""

import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════════

CONFIG = {
    'processed_dir': Path('./processed_data'),
    'results_dir': Path('./results'),
    'figures_dir': Path('./figures'),
    'tables_dir': Path('./tables'),
    
    # SEED SENSITIVITY PARAMETERS
    'num_seeds': 50,           # S = 50 different seeds
    'test_size': 0.2,
    'base_seed': 42
}

CONFIG['results_dir'].mkdir(parents=True, exist_ok=True)
CONFIG['figures_dir'].mkdir(parents=True, exist_ok=True)
CONFIG['tables_dir'].mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════════
# FAIRNESS CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════════════

class FairnessCalculator:
    """Compute fairness metrics."""
    
    def __init__(self, y_true, y_pred, y_prob, protected, attr_name):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_prob = np.array(y_prob) if y_prob is not None else None
        self.protected = np.array(protected)
        self.attr_name = attr_name
        self.subgroups = np.unique(protected)
    
    def _safe_div(self, a, b):
        return a / b if b > 0 else 0.0
    
    def compute_metrics(self):
        results = {'per_subgroup': {}, 'disparities': {}}
        
        for g in self.subgroups:
            mask = self.protected == g
            yt, yp = self.y_true[mask], self.y_pred[mask]
            
            tp = ((yp == 1) & (yt == 1)).sum()
            tn = ((yp == 0) & (yt == 0)).sum()
            fp = ((yp == 1) & (yt == 0)).sum()
            fn = ((yp == 0) & (yt == 1)).sum()
            
            n = mask.sum()
            actual_pos = (yt == 1).sum()
            actual_neg = (yt == 0).sum()
            pred_pos = (yp == 1).sum()
            
            dp = self._safe_div(pred_pos, n)
            tpr = self._safe_div(tp, actual_pos)
            fpr = self._safe_div(fp, actual_neg)
            ppv = self._safe_div(tp, pred_pos)
            
            ece = 0.0
            if self.y_prob is not None:
                yprob = self.y_prob[mask]
                for i in range(10):
                    bm = (yprob >= i/10) & (yprob < (i+1)/10)
                    if bm.sum() > 0:
                        ece += (bm.sum()/len(yprob)) * abs(yt[bm].mean() - yprob[bm].mean())
            
            results['per_subgroup'][str(g)] = {
                'demographic_parity': float(dp),
                'tpr': float(tpr),
                'fpr': float(fpr),
                'ppv': float(ppv),
                'ece': float(ece)
            }
        
        for metric in ['demographic_parity', 'tpr', 'ppv']:
            vals = [results['per_subgroup'][str(g)][metric] for g in self.subgroups]
            ratio = self._safe_div(min(vals), max(vals)) if max(vals) > 0 else 1.0
            results['disparities'][f'{metric}_ratio'] = float(ratio)
        
        return results


# ═══════════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════════

def load_data():
    """Load preprocessed data."""
    print("=" * 70)
    print("STEP 1: LOADING DATA")
    print("=" * 70)
    
    proc = CONFIG['processed_dir']
    
    X_scaled = np.load(proc / 'X_scaled.npy')
    y = np.load(proc / 'y.npy')
    
    with open(proc / 'protected_attributes.pkl', 'rb') as f:
        prot_data = pickle.load(f)
    protected = prot_data['protected']
    subgroups = prot_data['subgroups']
    
    print(f"✅ Loaded data: {len(y):,} samples")
    print(f"✅ Protected attributes: {list(protected.keys())}")
    
    return X_scaled, y, protected, subgroups


# ═══════════════════════════════════════════════════════════════════════════════════
# SEED SENSITIVITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════════

def run_seed_sensitivity(X_scaled, y, protected, subgroups):
    """Run seed sensitivity analysis."""
    print("\n" + "=" * 70)
    print(f"STEP 2: SEED SENSITIVITY ANALYSIS (S={CONFIG['num_seeds']})")
    print("=" * 70)
    print(f"\nMethod: Train {CONFIG['num_seeds']} models with different random seeds")
    print(f"Each seed affects: train/test split + model initialization\n")
    
    # Storage
    seed_results = {}
    for attr_name in protected.keys():
        seed_results[attr_name] = {
            'per_subgroup': defaultdict(lambda: defaultdict(list)),
            'disparities': defaultdict(list)
        }
    
    performance_results = {'accuracy': [], 'auc': []}
    
    # Run for each seed
    for seed in tqdm(range(CONFIG['num_seeds']), desc="Random Seeds"):
        # Split data with this seed
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X_scaled, y, np.arange(len(y)),
            test_size=CONFIG['test_size'],
            random_state=seed,
            stratify=y
        )
        
        # Train model with this seed
        model = LogisticRegression(
            max_iter=1000, 
            class_weight='balanced', 
            random_state=seed
        )
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Store performance
        performance_results['accuracy'].append(accuracy_score(y_test, y_pred))
        performance_results['auc'].append(roc_auc_score(y_test, y_prob))
        
        # Compute fairness for each attribute
        for attr_name, attr_values in protected.items():
            attr_test = attr_values[idx_test]
            
            calc = FairnessCalculator(y_test, y_pred, y_prob, attr_test, attr_name)
            results = calc.compute_metrics()
            
            for sg, metrics in results['per_subgroup'].items():
                for metric_name, value in metrics.items():
                    seed_results[attr_name]['per_subgroup'][sg][metric_name].append(value)
            
            for disp_name, value in results['disparities'].items():
                seed_results[attr_name]['disparities'][disp_name].append(value)
    
    return seed_results, performance_results


def compute_statistics(seed_results, performance_results, subgroups):
    """Compute statistics from seed sensitivity results."""
    print("\n" + "=" * 70)
    print("STEP 3: COMPUTING STATISTICS")
    print("=" * 70)
    
    stats_results = {}
    
    # Performance stats
    print(f"\n   Model Performance Variation:")
    print(f"   {'Metric':<15} {'Mean':>10} {'Std':>10} {'CV':>10}")
    print(f"   {'─'*45}")
    
    for metric, values in performance_results.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        cv = std_val / mean_val if mean_val > 0 else 0
        print(f"   {metric.upper():<15} {mean_val:>10.4f} {std_val:>10.4f} {cv:>9.1%}")
    
    # Fairness stats
    for attr_name, attr_data in seed_results.items():
        stats_results[attr_name] = {'per_subgroup': {}, 'disparities': {}}
        
        for sg in subgroups[attr_name]:
            sg_key = str(sg)
            stats_results[attr_name]['per_subgroup'][sg_key] = {}
            
            for metric in ['demographic_parity', 'tpr', 'fpr', 'ppv', 'ece']:
                values = np.array(attr_data['per_subgroup'].get(sg_key, {}).get(metric, []))
                if len(values) == 0:
                    stats_results[attr_name]['per_subgroup'][sg_key][metric] = {
                        'mean': 0.0, 'std': 0.0, 'cv': 0.0, 'min': 0.0, 'max': 0.0, 'range': 0.0, 'iqr': 0.0
                    }
                    continue
                mean_val = np.mean(values)
                
                stats_results[attr_name]['per_subgroup'][sg_key][metric] = {
                    'mean': float(mean_val),
                    'std': float(np.std(values)),
                    'cv': float(np.std(values) / mean_val) if mean_val > 0 else 0,
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'range': float(np.max(values) - np.min(values)),
                    'iqr': float(np.percentile(values, 75) - np.percentile(values, 25))
                }
        
        for disp_name in ['demographic_parity_ratio', 'tpr_ratio', 'ppv_ratio']:
            values = np.array(attr_data['disparities'].get(disp_name, []))
            if len(values) == 0:
                stats_results[attr_name]['disparities'][disp_name] = {
                    'mean': 0.0, 'std': 0.0, 'cv': 0.0, 'min': 0.0, 'max': 0.0
                }
                continue
            mean_val = np.mean(values)
            
            stats_results[attr_name]['disparities'][disp_name] = {
                'mean': float(mean_val),
                'std': float(np.std(values)),
                'cv': float(np.std(values) / mean_val) if mean_val > 0 else 0,
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
    
    return stats_results


def display_results(stats_results, subgroups):
    """Display seed sensitivity results."""
    print("\n" + "=" * 70)
    print("STEP 4: SEED SENSITIVITY RESULTS")
    print("=" * 70)
    
    for attr_name, attr_data in stats_results.items():
        print(f"\n{'━'*70}")
        print(f"📊 ATTRIBUTE: {attr_name}")
        print(f"{'━'*70}")
        
        print(f"\n   TPR Statistics Across {CONFIG['num_seeds']} Seeds:")
        print(f"   {'Subgroup':<25} {'Mean':>8} {'Std':>8} {'CV':>8} {'Range':>10}")
        print(f"   {'─'*60}")
        
        for sg in subgroups[attr_name]:
            sg_key = str(sg)
            stats = attr_data['per_subgroup'][sg_key]['tpr']
            print(f"   {sg:<25} {stats['mean']:>8.3f} {stats['std']:>8.4f} "
                  f"{stats['cv']:>7.1%} {stats['range']:>10.4f}")
        
        print(f"\n   Disparity Ratio Stability:")
        for disp_name, stats in attr_data['disparities'].items():
            metric_name = disp_name.replace('_ratio', '').replace('_', ' ').title()
            print(f"   {metric_name:<25}: Mean={stats['mean']:.3f}, CV={stats['cv']:.1%}")


# ═══════════════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════════

def create_visualizations(seed_results, stats_results, subgroups):
    """Create seed sensitivity visualizations."""
    print("\n" + "=" * 70)
    print("STEP 5: CREATING VISUALIZATIONS")
    print("=" * 70)
    
    fig_dir = CONFIG['figures_dir']
    
    # ─────────────────────────────────────────────────────────────────────
    # FIGURE 1: Violin Plots
    # ─────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, attr_name in enumerate(list(seed_results.keys())[:4]):
        ax = axes[idx]
        
        plot_data = []
        for sg in subgroups[attr_name]:
            sg_key = str(sg)
            for val in seed_results[attr_name]['per_subgroup'].get(sg_key, {}).get('tpr', []):
                plot_data.append({'Subgroup': sg, 'TPR': val})
        
        df = pd.DataFrame(plot_data)
        
        sns.violinplot(data=df, x='Subgroup', y='TPR', ax=ax, palette='Set2', inner='box')
        ax.axhline(y=0.8, color='red', linestyle='--', linewidth=2, alpha=0.7,
                  label='Fairness Threshold')
        ax.set_title(f'{attr_name}\nTPR Distribution Across {CONFIG["num_seeds"]} Seeds', fontweight='bold')
        ax.tick_params(axis='x', rotation=30)
        ax.set_ylim(0, 1)
        ax.legend(loc='lower right')
    
    plt.suptitle('Random Seed Sensitivity: Fairness Metric Distributions',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(fig_dir / 'seed_sensitivity_violin.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved seed_sensitivity_violin.png")
    
    # ─────────────────────────────────────────────────────────────────────
    # FIGURE 2: CV Heatmap
    # ─────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 8))
    
    attrs = list(stats_results.keys())
    metrics = ['demographic_parity', 'tpr', 'fpr', 'ppv']
    
    data = []
    for attr in attrs:
        row = []
        for m in metrics:
            # Average CV across subgroups
            cvs = [stats_results[attr]['per_subgroup'][str(sg)][m]['cv'] 
                  for sg in subgroups[attr]]
            row.append(np.mean(cvs) * 100)  # As percentage
        data.append(row)
    
    df = pd.DataFrame(data, index=attrs, columns=[m.upper().replace('_', ' ') for m in metrics])
    
    sns.heatmap(df, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
               cbar_kws={'label': 'Coefficient of Variation (%)'})
    ax.set_title('Seed Sensitivity: Coefficient of Variation (%)\n(Lower = More Stable)',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'seed_sensitivity_cv_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved seed_sensitivity_cv_heatmap.png")
    
    # ─────────────────────────────────────────────────────────────────────
    # FIGURE 3: Line Plot - Metric Trajectory
    # ─────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, attr_name in enumerate(list(seed_results.keys())[:4]):
        ax = axes[idx]
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(subgroups[attr_name])))
        
        for i, sg in enumerate(subgroups[attr_name]):
            sg_key = str(sg)
            values = seed_results[attr_name]['per_subgroup'].get(sg_key, {}).get('tpr', [])
            ax.plot(range(CONFIG['num_seeds']), values, alpha=0.7, 
                   label=sg, color=colors[i], linewidth=1.5)
        
        ax.set_xlabel('Seed Index')
        ax.set_ylabel('TPR')
        ax.set_title(f'{attr_name}', fontweight='bold')
        ax.legend(loc='lower right', fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    
    plt.suptitle('TPR Trajectory Across Random Seeds',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(fig_dir / 'seed_sensitivity_trajectory.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved seed_sensitivity_trajectory.png")


# ═══════════════════════════════════════════════════════════════════════════════════
# TABLES
# ═══════════════════════════════════════════════════════════════════════════════════

def create_tables(seed_results, stats_results, subgroups):
    """Create seed sensitivity tables."""
    print("\n" + "=" * 70)
    print("STEP 6: CREATING TABLES")
    print("=" * 70)
    
    tables_dir = CONFIG['tables_dir']
    
    # Table: Statistics Summary
    table_data = []
    for attr_name, attr_data in stats_results.items():
        for sg in subgroups[attr_name]:
            sg_key = str(sg)
            for metric in ['demographic_parity', 'tpr', 'fpr', 'ppv', 'ece']:
                stats = attr_data['per_subgroup'][sg_key][metric]
                table_data.append({
                    'Attribute': attr_name,
                    'Subgroup': sg,
                    'Metric': metric.upper(),
                    'Mean': f"{stats['mean']:.4f}",
                    'Std': f"{stats['std']:.4f}",
                    'CV': f"{stats['cv']:.2%}",
                    'Min': f"{stats['min']:.4f}",
                    'Max': f"{stats['max']:.4f}",
                    'Range': f"{stats['range']:.4f}"
                })
    
    df = pd.DataFrame(table_data)
    df.to_csv(tables_dir / 'seed_sensitivity_statistics.csv', index=False)
    print(f"✅ Saved seed_sensitivity_statistics.csv")
    
    # Table: Stability Ranking by CV
    stability_data = []
    for attr_name, attr_data in stats_results.items():
        for sg in subgroups[attr_name]:
            sg_key = str(sg)
            avg_cv = np.mean([
                attr_data['per_subgroup'][sg_key][m]['cv'] 
                for m in ['demographic_parity', 'tpr', 'ppv']
            ])
            stability_data.append({
                'Attribute': attr_name,
                'Subgroup': sg,
                'Avg_CV': avg_cv
            })
    
    df = pd.DataFrame(stability_data)
    df = df.sort_values('Avg_CV')
    df['Stability_Rank'] = range(1, len(df) + 1)
    df.to_csv(tables_dir / 'seed_sensitivity_ranking.csv', index=False)
    print(f"✅ Saved seed_sensitivity_ranking.csv")


# ═══════════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════════════

def save_results(seed_results, stats_results):
    """Save seed sensitivity results."""
    print("\n" + "=" * 70)
    print("STEP 7: SAVING RESULTS")
    print("=" * 70)
    
    results_dir = CONFIG['results_dir']
    
    # Convert defaultdicts to regular dicts for pickling
    seed_results_regular = {}
    for attr, data in seed_results.items():
        seed_results_regular[attr] = {
            'per_subgroup': {sg: dict(m) for sg, m in data['per_subgroup'].items()},
            'disparities': dict(data['disparities'])
        }
    
    with open(results_dir / 'seed_raw_results.pkl', 'wb') as f:
        pickle.dump(seed_results_regular, f)
    
    with open(results_dir / 'seed_statistics.pkl', 'wb') as f:
        pickle.dump(stats_results, f)
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'num_seeds': CONFIG['num_seeds'],
        'attributes': list(stats_results.keys())
    }
    
    with open(results_dir / 'seed_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Saved seed_raw_results.pkl")
    print(f"✅ Saved seed_statistics.pkl")
    print(f"✅ Saved seed_summary.json")


# ═══════════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "🎲" * 30)
    print("\n  SCRIPT 5: RANDOM SEED SENSITIVITY TEST")
    print(f"  S = {CONFIG['num_seeds']} seeds")
    print("\n" + "🎲" * 30)
    
    start_time = datetime.now()
    
    # Load data
    X_scaled, y, protected, subgroups = load_data()
    
    # Run seed sensitivity
    seed_results, performance_results = run_seed_sensitivity(X_scaled, y, protected, subgroups)
    
    # Compute statistics
    stats_results = compute_statistics(seed_results, performance_results, subgroups)
    
    # Display results
    display_results(stats_results, subgroups)
    
    # Create visualizations
    create_visualizations(seed_results, stats_results, subgroups)
    
    # Create tables
    create_tables(seed_results, stats_results, subgroups)
    
    # Save results
    save_results(seed_results, stats_results)
    
    elapsed = (datetime.now() - start_time).total_seconds() / 60
    
    print("\n" + "=" * 70)
    print("✅ SEED SENSITIVITY TEST COMPLETE")
    print("=" * 70)
    print(f"\nTotal time: {elapsed:.1f} minutes")
    print(f"Results saved to: {CONFIG['results_dir']}/")
    print("\n👉 NEXT: python script6_threshold_sweep.py")


if __name__ == "__main__":
    main()
