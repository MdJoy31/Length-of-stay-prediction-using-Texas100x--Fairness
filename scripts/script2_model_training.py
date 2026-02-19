#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════
SCRIPT 2: MODEL TRAINING
═══════════════════════════════════════════════════════════════════════════════════

Author: Md Jannatul Rakib Joy
Supervisors: Dr. Caslon Chua, Dr. Viet Vo
Institution: Swinburne University of Technology

PURPOSE: Train 4 ML models for LOS prediction
    - Logistic Regression
    - Random Forest
    - Gradient Boosting
    - Neural Network (MLP)

REQUIRES: Run script1_data_preprocessing.py first
RUN: python script2_model_training.py
NEXT: python script3_fairness_metrics.py

═══════════════════════════════════════════════════════════════════════════════════
"""

import pickle
import json
import numpy as np
from datetime import datetime
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve
)

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════════

CONFIG = {
    'input_dir': Path('./processed_data'),
    'output_dir': Path('./models'),
    'figures_dir': Path('./figures'),
    'random_state': 42
}

CONFIG['output_dir'].mkdir(parents=True, exist_ok=True)
CONFIG['figures_dir'].mkdir(parents=True, exist_ok=True)

# Model configurations
MODELS = {
    'Logistic_Regression': {
        'class': LogisticRegression,
        'params': {'max_iter': 1000, 'class_weight': 'balanced', 'random_state': 42, 'solver': 'lbfgs'}
    },
    'Random_Forest': {
        'class': RandomForestClassifier,
        'params': {'n_estimators': 100, 'max_depth': 10, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1}
    },
    'Gradient_Boosting': {
        'class': GradientBoostingClassifier,
        'params': {'n_estimators': 100, 'max_depth': 5, 'random_state': 42}
    },
    'Neural_Network': {
        'class': MLPClassifier,
        'params': {'hidden_layer_sizes': (64, 32), 'max_iter': 500, 'random_state': 42, 'early_stopping': True}
    }
}


def load_data():
    """Load preprocessed data."""
    print("=" * 70)
    print("STEP 1: LOADING PREPROCESSED DATA")
    print("=" * 70)
    
    inp = CONFIG['input_dir']
    
    X_train = np.load(inp / 'X_train.npy')
    X_test = np.load(inp / 'X_test.npy')
    y_train = np.load(inp / 'y_train.npy')
    y_test = np.load(inp / 'y_test.npy')
    
    print(f"✅ Loaded training data: {X_train.shape}")
    print(f"✅ Loaded testing data: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


def train_models(X_train, X_test, y_train, y_test):
    """Train all 4 ML models."""
    print("\n" + "=" * 70)
    print("STEP 2: TRAINING MODELS")
    print("=" * 70)
    
    results = {}
    
    for name, config in MODELS.items():
        print(f"\n{'─'*50}")
        print(f"🔧 Training: {name.replace('_', ' ')}")
        print(f"{'─'*50}")
        
        # Initialize and train
        model = config['class'](**config['params'])
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_prob) if y_prob is not None else None,
            'f1': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   AUC-ROC:   {metrics['auc']:.4f}" if metrics['auc'] else "   AUC-ROC:   N/A")
        print(f"   F1-Score:  {metrics['f1']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        
        results[name] = {
            'model': model,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'metrics': metrics,
            'confusion_matrix': cm
        }
    
    return results


def save_models(results, y_test):
    """Save trained models and predictions."""
    print("\n" + "=" * 70)
    print("STEP 3: SAVING MODELS")
    print("=" * 70)
    
    out = CONFIG['output_dir']
    
    # Save each model
    for name, data in results.items():
        # Save model
        with open(out / f'{name}_model.pkl', 'wb') as f:
            pickle.dump(data['model'], f)
        
        # Save predictions
        np.save(out / f'{name}_y_pred.npy', data['y_pred'])
        if data['y_prob'] is not None:
            np.save(out / f'{name}_y_prob.npy', data['y_prob'])
        
        print(f"✅ Saved {name}")
    
    # Save all results summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'models': list(results.keys()),
        'performance': {name: data['metrics'] for name, data in results.items()}
    }
    
    with open(out / 'model_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save combined predictions for fairness analysis
    predictions = {
        name: {'y_pred': data['y_pred'], 'y_prob': data['y_prob']}
        for name, data in results.items()
    }
    with open(out / 'all_predictions.pkl', 'wb') as f:
        pickle.dump(predictions, f)
    
    print(f"\n✅ All models saved to {out}/")


def create_visualizations(results, y_test):
    """Create model comparison visualizations."""
    print("\n" + "=" * 70)
    print("STEP 4: CREATING VISUALIZATIONS")
    print("=" * 70)
    
    fig_dir = CONFIG['figures_dir']
    
    # ─────────────────────────────────────────────────────────────────────
    # FIGURE: Model Performance Comparison
    # ─────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart of metrics
    ax = axes[0]
    models = list(results.keys())
    x = np.arange(len(models))
    width = 0.15
    
    metrics_to_plot = ['accuracy', 'auc', 'f1', 'precision', 'recall']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
    
    for i, metric in enumerate(metrics_to_plot):
        values = [results[m]['metrics'][metric] or 0 for m in models]
        ax.bar(x + i*width, values, width, label=metric.upper(), color=colors[i])
    
    ax.set_xticks(x + width*2)
    ax.set_xticklabels([m.replace('_', '\n') for m in models])
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison', fontweight='bold', fontsize=14)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    
    # ROC Curves
    ax = axes[1]
    for name, data in results.items():
        if data['y_prob'] is not None:
            fpr, tpr, _ = roc_curve(y_test, data['y_prob'])
            auc = data['metrics']['auc']
            ax.plot(fpr, tpr, label=f"{name.replace('_', ' ')} (AUC={auc:.3f})", linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves', fontweight='bold', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved model_performance_comparison.png")
    
    # ─────────────────────────────────────────────────────────────────────
    # FIGURE: Confusion Matrices
    # ─────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (name, data) in enumerate(results.items()):
        ax = axes[idx]
        sns.heatmap(data['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Normal', 'Extended'], yticklabels=['Normal', 'Extended'])
        ax.set_title(f"{name.replace('_', ' ')}", fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    plt.suptitle('Confusion Matrices by Model', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(fig_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved confusion_matrices.png")


def create_performance_table(results):
    """Create performance comparison table."""
    print("\n" + "=" * 70)
    print("STEP 5: PERFORMANCE SUMMARY TABLE")
    print("=" * 70)
    
    print(f"\n{'Model':<25} {'Accuracy':>10} {'AUC':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
    print("─" * 75)
    
    for name, data in results.items():
        m = data['metrics']
        auc_str = f"{m['auc']:.4f}" if m['auc'] else "N/A"
        print(f"{name.replace('_', ' '):<25} {m['accuracy']:>10.4f} {auc_str:>10} "
              f"{m['f1']:>10.4f} {m['precision']:>10.4f} {m['recall']:>10.4f}")
    
    # Save as CSV
    import pandas as pd
    table_data = []
    for name, data in results.items():
        m = data['metrics']
        table_data.append({
            'Model': name.replace('_', ' '),
            'Accuracy': m['accuracy'],
            'AUC': m['auc'],
            'F1': m['f1'],
            'Precision': m['precision'],
            'Recall': m['recall']
        })
    
    df = pd.DataFrame(table_data)
    df.to_csv(CONFIG['output_dir'] / 'model_performance_table.csv', index=False)
    print(f"\n✅ Saved model_performance_table.csv")


def main():
    print("\n" + "🤖" * 30)
    print("\n  SCRIPT 2: MODEL TRAINING")
    print("\n" + "🤖" * 30)
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Train models
    results = train_models(X_train, X_test, y_train, y_test)
    
    # Save models
    save_models(results, y_test)
    
    # Create visualizations
    create_visualizations(results, y_test)
    
    # Create summary table
    create_performance_table(results)
    
    print("\n" + "=" * 70)
    print("✅ MODEL TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nModels saved to: {CONFIG['output_dir']}/")
    print(f"Figures saved to: {CONFIG['figures_dir']}/")
    print("\n👉 NEXT: python script3_fairness_metrics.py")


if __name__ == "__main__":
    main()
