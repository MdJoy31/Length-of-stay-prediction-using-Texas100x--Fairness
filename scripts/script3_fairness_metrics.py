#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════
SCRIPT 3: FAIRNESS METRICS TESTING
═══════════════════════════════════════════════════════════════════════════════════

Author: Md Jannatul Rakib Joy
Supervisors: Dr. Caslon Chua, Dr. Viet Vo
Institution: Swinburne University of Technology

PURPOSE: Compute all 5 fairness metrics for all 13 subgroups across 4 protected attributes

FAIRNESS METRICS TESTED:
    1. Demographic Parity: P(Ŷ=1|A=a) = P(Ŷ=1|A=b)
    2. Equalized Odds: P(Ŷ=1|Y=y,A=a) = P(Ŷ=1|Y=y,A=b) for all y
    3. Equal Opportunity: P(Ŷ=1|Y=1,A=a) = P(Ŷ=1|Y=1,A=b)
    4. Predictive Parity: P(Y=1|Ŷ=1,A=a) = P(Y=1|Ŷ=1,A=b)
    5. Calibration: P(Y=1|Ŷ=p,A=a) = p (Expected Calibration Error)

PROTECTED ATTRIBUTES:
    - RACE: White, Black, Hispanic, Asian, Other (5 subgroups)
    - ETHNICITY: Hispanic, Non-Hispanic (2 subgroups)
    - SEX: Male, Female (2 subgroups)
    - AGE_GROUP: Pediatric, Adult, Middle-aged, Elderly (4 subgroups)

REQUIRES: script1_data_preprocessing.py, script2_model_training.py
RUN: python script3_fairness_metrics.py
NEXT: python script4_bootstrap_stability.py

═══════════════════════════════════════════════════════════════════════════════════
"""

import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════════

CONFIG = {
    'processed_dir': Path('./processed_data'),
    'models_dir': Path('./models'),
    'results_dir': Path('./results'),
    'figures_dir': Path('./figures'),
    'tables_dir': Path('./tables'),
    'fairness_threshold': 0.8  # 80% rule
}

for d in [CONFIG['results_dir'], CONFIG['figures_dir'], CONFIG['tables_dir']]:
    d.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════════
# FAIRNESS METRICS CALCULATOR CLASS
# ═══════════════════════════════════════════════════════════════════════════════════

class FairnessMetricsCalculator:
    """
    Comprehensive Fairness Metrics Calculator.
    
    Computes all 5 standard fairness metrics for each subgroup within
    a protected attribute:
    
    1. Demographic Parity (Statistical Parity)
    2. Equalized Odds (TPR + FPR parity)
    3. Equal Opportunity (TPR parity)
    4. Predictive Parity (Precision parity)
    5. Calibration (Expected Calibration Error)
    """
    
    def __init__(self, y_true, y_pred, y_prob, protected_attr, attr_name):
        """
        Initialize calculator.
        
        Parameters:
        -----------
        y_true : array-like - Ground truth labels (0/1)
        y_pred : array-like - Predicted labels (0/1)
        y_prob : array-like - Predicted probabilities
        protected_attr : array-like - Protected attribute values
        attr_name : str - Name of protected attribute
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_prob = np.array(y_prob) if y_prob is not None else None
        self.protected = np.array(protected_attr)
        self.attr_name = attr_name
        self.subgroups = np.unique(protected_attr)
    
    def _safe_divide(self, a, b):
        """Safe division returning 0 if denominator is 0."""
        return a / b if b > 0 else 0.0
    
    def compute_confusion_matrix(self, subgroup):
        """Compute confusion matrix components for a subgroup."""
        mask = self.protected == subgroup
        yt, yp = self.y_true[mask], self.y_pred[mask]
        
        tp = int(((yp == 1) & (yt == 1)).sum())
        tn = int(((yp == 0) & (yt == 0)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())
        
        return {'n': mask.sum(), 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
                'actual_pos': (yt == 1).sum(), 'actual_neg': (yt == 0).sum(),
                'pred_pos': (yp == 1).sum(), 'pred_neg': (yp == 0).sum()}
    
    def compute_subgroup_metrics(self, subgroup):
        """
        Compute ALL 5 fairness metrics for a single subgroup.
        
        Returns dict with:
        - n_samples, base_rate
        - demographic_parity: P(Ŷ=1|A=subgroup)
        - tpr: True Positive Rate (Equal Opportunity)
        - fpr: False Positive Rate
        - tnr: True Negative Rate
        - fnr: False Negative Rate
        - ppv: Positive Predictive Value (Predictive Parity)
        - npv: Negative Predictive Value
        - ece: Expected Calibration Error
        - accuracy, f1_score
        """
        cm = self.compute_confusion_matrix(subgroup)
        
        # METRIC 1: DEMOGRAPHIC PARITY
        demographic_parity = self._safe_divide(cm['pred_pos'], cm['n'])
        
        # METRIC 2 & 3: EQUALIZED ODDS / EQUAL OPPORTUNITY
        tpr = self._safe_divide(cm['tp'], cm['actual_pos'])  # Sensitivity
        fpr = self._safe_divide(cm['fp'], cm['actual_neg'])
        tnr = self._safe_divide(cm['tn'], cm['actual_neg'])  # Specificity
        fnr = self._safe_divide(cm['fn'], cm['actual_pos'])
        
        # METRIC 4: PREDICTIVE PARITY
        ppv = self._safe_divide(cm['tp'], cm['pred_pos'])  # Precision
        npv = self._safe_divide(cm['tn'], cm['pred_neg'])
        
        # METRIC 5: CALIBRATION (ECE)
        ece = self._compute_ece(subgroup)
        
        # Additional metrics
        accuracy = self._safe_divide(cm['tp'] + cm['tn'], cm['n'])
        f1 = self._safe_divide(2 * cm['tp'], 2 * cm['tp'] + cm['fp'] + cm['fn'])
        base_rate = self._safe_divide(cm['actual_pos'], cm['n'])
        
        return {
            'n_samples': int(cm['n']),
            'base_rate': float(base_rate),
            'demographic_parity': float(demographic_parity),
            'tpr': float(tpr),
            'fpr': float(fpr),
            'tnr': float(tnr),
            'fnr': float(fnr),
            'ppv': float(ppv),
            'npv': float(npv),
            'ece': float(ece),
            'accuracy': float(accuracy),
            'f1_score': float(f1),
            'confusion_matrix': cm
        }
    
    def _compute_ece(self, subgroup, n_bins=10):
        """Compute Expected Calibration Error for a subgroup."""
        if self.y_prob is None:
            return 0.0
        
        mask = self.protected == subgroup
        yt = self.y_true[mask]
        yp = self.y_prob[mask]
        
        if len(yp) == 0:
            return 0.0
        
        ece = 0.0
        bin_edges = np.linspace(0, 1, n_bins + 1)
        
        for i in range(n_bins):
            bin_mask = (yp >= bin_edges[i]) & (yp < bin_edges[i + 1])
            if bin_mask.sum() > 0:
                bin_accuracy = yt[bin_mask].mean()
                bin_confidence = yp[bin_mask].mean()
                bin_weight = bin_mask.sum() / len(yp)
                ece += bin_weight * abs(bin_accuracy - bin_confidence)
        
        return ece
    
    def compute_all_metrics(self):
        """
        Compute fairness metrics for ALL subgroups and disparities.
        
        Returns comprehensive dict with:
        - per_subgroup: metrics for each subgroup
        - disparities: disparity analysis for each metric
        - summary: overall fairness summary
        """
        results = {
            'attribute': self.attr_name,
            'subgroups': [str(g) for g in self.subgroups],
            'n_subgroups': len(self.subgroups),
            'per_subgroup': {},
            'disparities': {}
        }
        
        # Compute metrics for each subgroup
        for g in self.subgroups:
            results['per_subgroup'][str(g)] = self.compute_subgroup_metrics(g)
        
        # Compute disparities
        metrics_config = [
            ('demographic_parity', 'Demographic Parity', 'ratio'),
            ('tpr', 'Equal Opportunity (TPR)', 'ratio'),
            ('fpr', 'FPR Parity', 'ratio'),
            ('ppv', 'Predictive Parity (Precision)', 'ratio'),
            ('ece', 'Calibration (ECE)', 'difference')
        ]
        
        for metric_key, metric_name, comparison in metrics_config:
            values = {str(g): results['per_subgroup'][str(g)][metric_key] for g in self.subgroups}
            vals_list = list(values.values())
            
            if comparison == 'difference' or metric_key == 'ece':
                diff = max(vals_list) - min(vals_list)
                ratio = None
                is_fair = diff <= 0.1
                best = min(values.keys(), key=lambda k: values[k])
                worst = max(values.keys(), key=lambda k: values[k])
            else:
                ratio = self._safe_divide(min(vals_list), max(vals_list)) if max(vals_list) > 0 else 1.0
                diff = max(vals_list) - min(vals_list)
                is_fair = ratio >= CONFIG['fairness_threshold']
                best = max(values.keys(), key=lambda k: values[k])
                worst = min(values.keys(), key=lambda k: values[k])
            
            results['disparities'][metric_key] = {
                'metric_name': metric_name,
                'values': values,
                'min': float(min(vals_list)),
                'max': float(max(vals_list)),
                'difference': float(diff),
                'ratio': float(ratio) if ratio else None,
                'best_subgroup': best,
                'worst_subgroup': worst,
                'is_fair': is_fair
            }
        
        # Equalized Odds (combined TPR + FPR)
        tpr_diff = results['disparities']['tpr']['difference']
        fpr_diff = results['disparities']['fpr']['difference']
        eo_combined = (tpr_diff + fpr_diff) / 2
        
        results['disparities']['equalized_odds'] = {
            'metric_name': 'Equalized Odds',
            'tpr_difference': tpr_diff,
            'fpr_difference': fpr_diff,
            'combined_disparity': float(eo_combined),
            'is_fair': eo_combined <= 0.1
        }
        
        # Summary
        n_fair = sum(1 for d in results['disparities'].values() if d.get('is_fair', False))
        n_total = len(results['disparities'])
        
        results['summary'] = {
            'n_fair': n_fair,
            'n_total': n_total,
            'fairness_score': float(n_fair / n_total)
        }
        
        return results


# ═══════════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════════

def load_data():
    """Load all necessary data for fairness analysis."""
    print("=" * 70)
    print("STEP 1: LOADING DATA")
    print("=" * 70)
    
    proc = CONFIG['processed_dir']
    models = CONFIG['models_dir']
    
    # Load test data
    y_test = np.load(proc / 'y_test.npy')
    idx_test = np.load(proc / 'idx_test.npy')
    
    # Load protected attributes
    with open(proc / 'protected_attributes.pkl', 'rb') as f:
        prot_data = pickle.load(f)
    
    protected = prot_data['protected']
    subgroups = prot_data['subgroups']
    
    # Load model predictions
    with open(models / 'all_predictions.pkl', 'rb') as f:
        predictions = pickle.load(f)
    
    print(f"✅ Loaded test data: {len(y_test):,} samples")
    print(f"✅ Protected attributes: {list(protected.keys())}")
    print(f"✅ Models: {list(predictions.keys())}")
    
    return y_test, idx_test, protected, subgroups, predictions


# ═══════════════════════════════════════════════════════════════════════════════════
# FAIRNESS EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════════

def evaluate_fairness(y_test, idx_test, protected, subgroups, predictions):
    """Evaluate fairness for all models and all protected attributes."""
    print("\n" + "=" * 70)
    print("STEP 2: COMPUTING FAIRNESS METRICS")
    print("=" * 70)
    
    all_results = {}
    
    for model_name, preds in predictions.items():
        print(f"\n{'─'*60}")
        print(f"📊 Model: {model_name.replace('_', ' ')}")
        print(f"{'─'*60}")
        
        y_pred = preds['y_pred']
        y_prob = preds['y_prob']
        
        model_results = {}
        
        for attr_name, attr_values in protected.items():
            attr_test = attr_values[idx_test]
            
            calculator = FairnessMetricsCalculator(
                y_true=y_test,
                y_pred=y_pred,
                y_prob=y_prob,
                protected_attr=attr_test,
                attr_name=attr_name
            )
            
            results = calculator.compute_all_metrics()
            model_results[attr_name] = results
            
            # Print summary
            n_fair = results['summary']['n_fair']
            n_total = results['summary']['n_total']
            print(f"   {attr_name}: {n_fair}/{n_total} metrics fair")
        
        all_results[model_name] = model_results
    
    return all_results


# ═══════════════════════════════════════════════════════════════════════════════════
# DETAILED RESULTS DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════════

def display_detailed_results(all_results):
    """Display detailed fairness results for each attribute."""
    print("\n" + "=" * 70)
    print("STEP 3: DETAILED FAIRNESS RESULTS")
    print("=" * 70)
    
    # Use Logistic Regression as primary model
    model = 'Logistic_Regression'
    results = all_results[model]
    
    for attr_name, attr_results in results.items():
        print(f"\n{'━'*70}")
        print(f"📊 ATTRIBUTE: {attr_name}")
        print(f"{'━'*70}")
        
        # Per-subgroup table
        print(f"\n   Per-Subgroup Metrics:")
        print(f"   {'Subgroup':<25} {'N':>8} {'Base%':>7} {'DP':>7} {'TPR':>7} {'FPR':>7} {'PPV':>7} {'ECE':>7}")
        print(f"   {'─'*80}")
        
        for sg, metrics in attr_results['per_subgroup'].items():
            print(f"   {sg:<25} {metrics['n_samples']:>8,} {metrics['base_rate']:>6.1%} "
                  f"{metrics['demographic_parity']:>7.3f} {metrics['tpr']:>7.3f} "
                  f"{metrics['fpr']:>7.3f} {metrics['ppv']:>7.3f} {metrics['ece']:>7.3f}")
        
        # Disparity summary
        print(f"\n   Disparity Analysis:")
        print(f"   {'Metric':<30} {'Ratio':>8} {'Diff':>8} {'Best':>15} {'Worst':>15} {'Fair?':>8}")
        print(f"   {'─'*85}")
        
        for metric_key, disp in attr_results['disparities'].items():
            if 'metric_name' in disp:
                ratio_str = f"{disp['ratio']:.3f}" if disp.get('ratio') else "N/A"
                diff_str = f"{disp.get('difference', disp.get('combined_disparity', 0)):.3f}"
                best = str(disp.get('best_subgroup', 'N/A'))[:15]
                worst = str(disp.get('worst_subgroup', 'N/A'))[:15]
                status = "✓ YES" if disp['is_fair'] else "✗ NO"
                
                print(f"   {disp['metric_name']:<30} {ratio_str:>8} {diff_str:>8} "
                      f"{best:>15} {worst:>15} {status:>8}")


# ═══════════════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════════

def create_visualizations(all_results, subgroups):
    """Create fairness visualizations."""
    print("\n" + "=" * 70)
    print("STEP 4: CREATING VISUALIZATIONS")
    print("=" * 70)
    
    fig_dir = CONFIG['figures_dir']
    results = all_results['Logistic_Regression']
    
    # ─────────────────────────────────────────────────────────────────────
    # FIGURE 1: Subgroup Metrics Bar Charts (for each attribute)
    # ─────────────────────────────────────────────────────────────────────
    for attr_name, attr_results in results.items():
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        groups = list(attr_results['per_subgroup'].keys())
        x = np.arange(len(groups))
        
        # Panel 1: TPR & FPR
        ax = axes[0]
        tpr = [attr_results['per_subgroup'][g]['tpr'] for g in groups]
        fpr = [attr_results['per_subgroup'][g]['fpr'] for g in groups]
        w = 0.35
        ax.bar(x - w/2, tpr, w, label='TPR (Sensitivity)', color='#27ae60', edgecolor='black')
        ax.bar(x + w/2, fpr, w, label='FPR', color='#e74c3c', edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=30, ha='right')
        ax.set_ylabel('Rate')
        ax.set_title('Equalized Odds Components\n(TPR and FPR by Subgroup)', fontweight='bold')
        ax.legend()
        ax.set_ylim(0, 1)
        ax.axhline(y=np.mean(tpr), color='#27ae60', linestyle='--', alpha=0.5)
        
        # Panel 2: PPV (Predictive Parity)
        ax = axes[1]
        ppv = [attr_results['per_subgroup'][g]['ppv'] for g in groups]
        colors = ['#3498db' if v >= max(ppv)*0.8 else '#f39c12' for v in ppv]
        ax.bar(x, ppv, color=colors, edgecolor='black')
        ax.axhline(y=max(ppv)*0.8, color='red', linestyle='--', linewidth=2, label='80% Threshold')
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=30, ha='right')
        ax.set_ylabel('Precision (PPV)')
        ax.set_title('Predictive Parity\n(Precision by Subgroup)', fontweight='bold')
        ax.legend()
        ax.set_ylim(0, 1)
        
        # Panel 3: Demographic Parity
        ax = axes[2]
        dp = [attr_results['per_subgroup'][g]['demographic_parity'] for g in groups]
        colors = ['#9b59b6' if v >= max(dp)*0.8 else '#f39c12' for v in dp]
        ax.bar(x, dp, color=colors, edgecolor='black')
        ax.axhline(y=max(dp)*0.8, color='red', linestyle='--', linewidth=2, label='80% Threshold')
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=30, ha='right')
        ax.set_ylabel('Selection Rate')
        ax.set_title('Demographic Parity\n(Positive Prediction Rate)', fontweight='bold')
        ax.legend()
        
        # Panel 4: Calibration (ECE)
        ax = axes[3]
        ece = [attr_results['per_subgroup'][g]['ece'] for g in groups]
        ax.bar(x, ece, color='#1abc9c', edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=30, ha='right')
        ax.set_ylabel('Expected Calibration Error')
        ax.set_title('Calibration Error\n(Lower is Better)', fontweight='bold')
        
        plt.suptitle(f'Fairness Metrics by {attr_name} Subgroup', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(fig_dir / f'fairness_metrics_{attr_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved fairness_metrics_{attr_name}.png")
    
    # ─────────────────────────────────────────────────────────────────────
    # FIGURE 2: Disparity Heatmap
    # ─────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 8))
    
    attrs = list(results.keys())
    metrics = ['demographic_parity', 'tpr', 'fpr', 'ppv', 'ece']
    metric_labels = ['Demographic\nParity', 'Equal\nOpportunity', 'FPR\nParity', 'Predictive\nParity', 'Calibration']
    
    data = []
    annot = []
    for attr in attrs:
        row_data = []
        row_annot = []
        for m in metrics:
            d = results[attr]['disparities'][m]
            if m == 'ece':
                val = d['difference']
                row_annot.append(f"{val:.3f}")
            else:
                val = 1 - d['ratio'] if d['ratio'] else 0.5
                row_annot.append(f"{d['ratio']:.3f}" if d['ratio'] else "N/A")
            row_data.append(val)
        data.append(row_data)
        annot.append(row_annot)
    
    cmap = sns.diverging_palette(145, 10, as_cmap=True)
    sns.heatmap(pd.DataFrame(data, index=attrs, columns=metric_labels),
               annot=np.array(annot), fmt='', cmap=cmap, center=0.2,
               vmin=0, vmax=0.5, linewidths=2, ax=ax,
               annot_kws={'fontsize': 11, 'fontweight': 'bold'})
    
    ax.set_title('Fairness Disparity Heatmap\n(Green=Fair, Red=Unfair)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_dir / 'fairness_disparity_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved fairness_disparity_heatmap.png")
    
    # ─────────────────────────────────────────────────────────────────────
    # FIGURE 3: Model Comparison - Fairness
    # ─────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    
    model_names = list(all_results.keys())
    attr_names = list(all_results[model_names[0]].keys())
    
    x = np.arange(len(model_names))
    width = 0.2
    colors = plt.cm.Set2(np.linspace(0, 1, len(attr_names)))
    
    for i, attr in enumerate(attr_names):
        ratios = []
        for model in model_names:
            r = all_results[model][attr]['disparities']['tpr']['ratio']
            ratios.append(r if r else 0)
        ax.bar(x + i*width, ratios, width, label=attr, color=colors[i], edgecolor='black')
    
    ax.axhline(y=0.8, color='red', linestyle='--', linewidth=2, label='Fairness Threshold')
    ax.set_xticks(x + width * (len(attr_names)-1) / 2)
    ax.set_xticklabels([m.replace('_', '\n') for m in model_names])
    ax.set_ylabel('Equal Opportunity Ratio')
    ax.set_title('Equal Opportunity Ratio by Model and Attribute', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'fairness_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved fairness_model_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════════════
# TABLES
# ═══════════════════════════════════════════════════════════════════════════════════

def create_tables(all_results):
    """Create fairness results tables."""
    print("\n" + "=" * 70)
    print("STEP 5: CREATING TABLES")
    print("=" * 70)
    
    tables_dir = CONFIG['tables_dir']
    results = all_results['Logistic_Regression']
    
    # Table 1: Fairness Metrics by Subgroup
    table_data = []
    for attr_name, attr_results in results.items():
        for sg, metrics in attr_results['per_subgroup'].items():
            table_data.append({
                'Attribute': attr_name,
                'Subgroup': sg,
                'N': metrics['n_samples'],
                'Base_Rate': f"{metrics['base_rate']:.3f}",
                'Demographic_Parity': f"{metrics['demographic_parity']:.3f}",
                'TPR': f"{metrics['tpr']:.3f}",
                'FPR': f"{metrics['fpr']:.3f}",
                'PPV': f"{metrics['ppv']:.3f}",
                'ECE': f"{metrics['ece']:.3f}",
                'Accuracy': f"{metrics['accuracy']:.3f}"
            })
    
    df = pd.DataFrame(table_data)
    df.to_csv(tables_dir / 'fairness_metrics_by_subgroup.csv', index=False)
    print(f"✅ Saved fairness_metrics_by_subgroup.csv")
    
    # Table 2: Disparity Summary
    disp_data = []
    for attr_name, attr_results in results.items():
        for metric_key, disp in attr_results['disparities'].items():
            if 'metric_name' in disp:
                disp_data.append({
                    'Attribute': attr_name,
                    'Metric': disp['metric_name'],
                    'Min': f"{disp.get('min', 0):.3f}" if 'min' in disp else "N/A",
                    'Max': f"{disp.get('max', 0):.3f}" if 'max' in disp else "N/A",
                    'Difference': f"{disp.get('difference', disp.get('combined_disparity', 0)):.3f}",
                    'Ratio': f"{disp['ratio']:.3f}" if disp.get('ratio') else "N/A",
                    'Best_Subgroup': disp.get('best_subgroup', 'N/A'),
                    'Worst_Subgroup': disp.get('worst_subgroup', 'N/A'),
                    'Is_Fair': 'Yes' if disp['is_fair'] else 'No'
                })
    
    df = pd.DataFrame(disp_data)
    df.to_csv(tables_dir / 'fairness_disparity_summary.csv', index=False)
    print(f"✅ Saved fairness_disparity_summary.csv")


# ═══════════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════════════

def save_results(all_results):
    """Save all fairness results."""
    print("\n" + "=" * 70)
    print("STEP 6: SAVING RESULTS")
    print("=" * 70)
    
    results_dir = CONFIG['results_dir']
    
    with open(results_dir / 'fairness_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    
    # JSON summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'models': list(all_results.keys()),
        'attributes': list(all_results['Logistic_Regression'].keys()),
        'fairness_threshold': CONFIG['fairness_threshold']
    }
    
    with open(results_dir / 'fairness_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Saved fairness_results.pkl")
    print(f"✅ Saved fairness_summary.json")


# ═══════════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "⚖️" * 30)
    print("\n  SCRIPT 3: FAIRNESS METRICS TESTING")
    print("\n" + "⚖️" * 30)
    
    # Load data
    y_test, idx_test, protected, subgroups, predictions = load_data()
    
    # Evaluate fairness
    all_results = evaluate_fairness(y_test, idx_test, protected, subgroups, predictions)
    
    # Display detailed results
    display_detailed_results(all_results)
    
    # Create visualizations
    create_visualizations(all_results, subgroups)
    
    # Create tables
    create_tables(all_results)
    
    # Save results
    save_results(all_results)
    
    print("\n" + "=" * 70)
    print("✅ FAIRNESS METRICS TESTING COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {CONFIG['results_dir']}/")
    print(f"Figures saved to: {CONFIG['figures_dir']}/")
    print(f"Tables saved to: {CONFIG['tables_dir']}/")
    print("\n👉 NEXT: python script4_bootstrap_stability.py")


if __name__ == "__main__":
    main()
