"""
run_analysis.py — Single entry script for the RQ1 fairness analysis.
Texas-100X LOS Prediction Fairness Study.

Usage:
    cd final_notebooks
    python scripts/run_analysis.py

Loads y_true, y_prob, protected attributes, and hospital_id from the
executed notebook's saved results, then runs the full analysis pipeline.

Expected outputs:
    results/analysis_report.json  — Full structured results
    results/analysis_summary.txt  — Human-readable summary
"""
import sys, os, json, time
import numpy as np
import pandas as pd

# Add scripts dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from analysis_utils import (
    compute_metrics, compute_fairness, bootstrap_ci,
    intersectional_audit, hospital_analysis,
    run_fairlearn_baseline, make_run_header, save_results
)

np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════
def load_data():
    """Load the Texas-100X dataset and prepare arrays."""
    print("Loading data...")
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'texas_100x.csv')
    df = pd.read_csv(data_path)
    print(f"  Loaded: {len(df):,} records, {len(df.columns)} columns")

    # Target: LOS > 3 days
    y = (df['LENGTH_OF_STAY'] > 3).astype(int).values

    # Protected attributes (data uses 0-indexed codes)
    race_map = {0: 'Other/Unknown', 1: 'White', 2: 'Black',
                3: 'Hispanic', 4: 'Asian/PI'}
    sex_map = {0: 'Female', 1: 'Male'}
    eth_map = {0: 'Non-Hispanic', 1: 'Hispanic'}

    def age_group(age_code):
        if age_code <= 3: return 'Paediatric'
        elif age_code <= 6: return 'Young Adult'
        elif age_code <= 10: return 'Middle-Aged'
        elif age_code <= 14: return 'Senior'
        else: return 'Elderly'

    protected = {
        'RACE': np.array([race_map.get(r, 'Other/Unknown') for r in df['RACE']]),
        'SEX': np.array([sex_map.get(s, 'Unknown') for s in df['SEX_CODE']]),
        'ETHNICITY': np.array([eth_map.get(e, 'Non-Hispanic') for e in df['ETHNICITY']]),
        'AGE_GROUP': np.array([age_group(a) for a in df['PAT_AGE']]),
    }
    hospital_ids = df['THCIC_ID'].values

    print(f"  Target: {y.mean():.1%} positive (LOS > 3 days)")
    print(f"  RACE groups: {sorted(set(protected['RACE']))}")
    print(f"  Hospitals: {len(set(hospital_ids)):,}")

    return df, y, protected, hospital_ids


# ═══════════════════════════════════════════════════════════════════════
# Main Analysis Pipeline
# ═══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("RQ1 Analysis: Fairness Metrics Reliability in Healthcare LOS Prediction")
    print("=" * 70)

    start_time = time.time()
    results = {'run_header': make_run_header()}

    # Load data
    df, y, protected, hospital_ids = load_data()

    # ── Use saved predictions from notebooks if available ──
    metrics_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'extracted_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            nb_metrics = json.load(f)
        print(f"\n  Loaded extracted notebook metrics")
        results['notebook_metrics'] = nb_metrics

    # ── For the analysis, we use the full dataset with a simple model ──
    # (The notebooks contain the full GPU-accelerated analysis;
    #  this script provides a reproducible CPU-only verification)
    print("\n── Training verification model (Logistic Regression, CPU) ──")
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Simple features for verification
    feature_cols = ['PAT_AGE', 'TOTAL_CHARGES', 'TYPE_OF_ADMISSION',
                    'SOURCE_OF_ADMISSION', 'PAT_STATUS']
    X = df[feature_cols].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Track protected attributes for test set
    _, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
    )
    prot_test = {k: v[test_idx] for k, v in protected.items()}
    hosp_test = hospital_ids[test_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    lr = LogisticRegression(C=1.0, class_weight='balanced',
                            solver='lbfgs', max_iter=500, random_state=42)
    lr.fit(X_train_s, y_train)

    y_pred = lr.predict(X_test_s)
    y_prob = lr.predict_proba(X_test_s)[:, 1]

    # ── Compute all metrics ──
    print("\n── Computing metrics ──")
    perf = compute_metrics(y_test, y_pred, y_prob)
    print(f"  Accuracy: {perf['accuracy']:.4f}")
    print(f"  F1:       {perf['f1']:.4f}")
    print(f"  AUC:      {perf['auc']:.4f}")
    print(f"  Brier:    {perf['brier']:.4f}")
    print(f"  ECE:      {perf['ece']:.4f}")
    results['verification_model'] = {'model': 'LogisticRegression', 'metrics': perf}

    # ── Fairness per attribute ──
    print("\n── Computing fairness metrics ──")
    fairness = {}
    for attr_name, attr_vals in prot_test.items():
        fm = compute_fairness(y_test, y_pred, attr_vals)
        fairness[attr_name] = fm
        print(f"  {attr_name:15s}: DI={fm['DI']:.3f}  WTPR={fm['WTPR']:.3f}  "
              f"EOD={fm['EOD']:.3f}  EqOdds={fm['EqOdds_proxy']:.3f}")
    results['fairness'] = fairness

    # ── Bootstrap CI ──
    print("\n── Bootstrap CI (B=200) ──")
    bootstrap = {}
    for attr_name in ['RACE', 'SEX']:
        ci = bootstrap_ci(y_test, y_pred, prot_test[attr_name])
        bootstrap[attr_name] = ci
        for g, vals in ci.items():
            print(f"  {attr_name}/{g}: TPR={vals['mean']:.3f} "
                  f"CI=[{vals['ci_low']:.3f}, {vals['ci_high']:.3f}] "
                  f"width={vals['ci_width']:.3f}")
    results['bootstrap_ci'] = bootstrap

    # ── Intersectional audit ──
    print("\n── Intersectional audit (RACE x SEX) ──")
    inter = intersectional_audit(
        y_test, y_pred, prot_test['RACE'], prot_test['SEX']
    )
    print(f"  {inter['n_intersections']} intersections evaluated")
    print(f"  Intersectional DI = {inter['intersectional_DI']:.3f}")
    for grp, vals in sorted(inter['intersections'].items()):
        print(f"    {grp:25s}: SR={vals['selection_rate']:.3f}  "
              f"TPR={vals['tpr']:.3f}  n={vals['n']:,}")
    results['intersectional_audit'] = inter

    # ── Hospital analysis ──
    print("\n── Hospital analysis (K=30) ──")
    hosp = hospital_analysis(
        y_test, y_pred, y_prob, hosp_test, prot_test['RACE']
    )
    print(f"  {hosp['n_hospitals']} hospitals analysed (min 200 records)")
    print(f"  DI: mean={hosp['DI_mean']:.3f} std={hosp['DI_std']:.3f} "
          f"range=[{hosp['DI_min']:.3f}, {hosp['DI_max']:.3f}]")
    print(f"  F1: mean={hosp['F1_mean']:.3f}  Acc: mean={hosp['Acc_mean']:.3f}")
    results['hospital_analysis'] = hosp

    # ── Fairlearn baseline ──
    print("\n── Fairlearn baseline ──")
    fl = run_fairlearn_baseline(y_test, y_prob, prot_test['RACE'])
    if 'error' in fl:
        print(f"  {fl['error']}")
    else:
        print(f"  ThresholdOptimizer: Acc={fl['metrics']['accuracy']:.4f}  "
              f"DI={fl['fairness']['DI']:.3f}")
    results['fairlearn_baseline'] = fl

    # ── Save results ──
    elapsed = time.time() - start_time
    results['elapsed_seconds'] = round(elapsed, 1)

    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    save_results(results, os.path.join(out_dir, 'analysis_report.json'))

    # ── Human-readable summary ──
    summary_path = os.path.join(out_dir, 'analysis_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("RQ1 ANALYSIS SUMMARY\n")
        f.write(f"Generated: {results['run_header']['timestamp']}\n")
        f.write("=" * 70 + "\n\n")

        f.write("VERIFICATION MODEL (Logistic Regression, CPU, 5 features)\n")
        f.write("-" * 50 + "\n")
        for k, v in perf.items():
            f.write(f"  {k:15s}: {v:.4f}\n")

        f.write("\nFAIRNESS METRICS\n")
        f.write("-" * 50 + "\n")
        for attr, fm in fairness.items():
            f.write(f"  {attr}:\n")
            f.write(f"    DI={fm['DI']:.3f}  WTPR={fm['WTPR']:.3f}  "
                    f"EOD={fm['EOD']:.3f}  EqOdds={fm['EqOdds_proxy']:.3f}\n")

        f.write("\nNOTEBOOK RESULTS (from executed GPU notebooks)\n")
        f.write("-" * 50 + "\n")
        if nb_metrics:
            f.write(f"  Best model: {nb_metrics.get('best_model', 'N/A')}\n")
            f.write(f"  Best F1: {nb_metrics.get('best_f1', 'N/A')}\n")
            f.write(f"  AFCE best (a={nb_metrics.get('afce_best_alpha', 'N/A')}):\n")
            f.write(f"    Acc={nb_metrics.get('afce_best_acc', 'N/A')}  "
                    f"F1={nb_metrics.get('afce_best_f1', 'N/A')}  "
                    f"AUC={nb_metrics.get('afce_best_auc', 'N/A')}\n")
            f.write(f"    RACE DI={nb_metrics.get('afce_best_race_di', 'N/A')}  "
                    f"SEX DI={nb_metrics.get('afce_best_sex_di', 'N/A')}  "
                    f"ETH DI={nb_metrics.get('afce_best_eth_di', 'N/A')}\n")

        f.write(f"\nCROSS-HOSPITAL VARIANCE\n")
        f.write(f"-" * 50 + "\n")
        f.write(f"  DI range: [{hosp['DI_min']:.3f}, {hosp['DI_max']:.3f}]\n")
        f.write(f"  DI std: {hosp['DI_std']:.3f}\n")

        f.write(f"\nINTERSECTIONAL AUDIT (RACE x SEX)\n")
        f.write(f"-" * 50 + "\n")
        f.write(f"  Intersectional DI: {inter['intersectional_DI']:.3f}\n")

        f.write(f"\nAnalysis completed in {elapsed:.1f}s\n")

    print(f"\n{'=' * 70}")
    print(f"Analysis complete in {elapsed:.1f}s")
    print(f"  Report: {os.path.join(out_dir, 'analysis_report.json')}")
    print(f"  Summary: {summary_path}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
