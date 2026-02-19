#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════
MASTER RUNNER: Texas-100X Fairness Analysis - ALL 5 STABILITY TESTS
═══════════════════════════════════════════════════════════════════════════════════

EXECUTION ORDER:
    Script 1:  Data Preprocessing                (~2 min)
    Script 2:  Model Training (4 models)         (~10 min)
    Script 3:  Fairness Metrics (5 metrics)      (~5 min)
    Script 4a: Bootstrap Stability (B=1000)      (~2 hours)
    Script 4b: Sample Size Sensitivity           (~1 hour)  ← NEW!
    Script 4c: Cross-Hospital Validation (K=50)  (~2 hours) ← NEW!
    Script 4d: Seed Sensitivity (S=50)           (~2 hours)
    Script 4e: Threshold Sweep (τ=99)            (~30 min)
    Script 5:  Final Report                      (~5 min)

TOTAL: ~8-10 hours (run overnight)

USAGE:
    python run_all.py              # Run all scripts
    python run_all.py --from 4a    # Start from script 4a

═══════════════════════════════════════════════════════════════════════════════════
"""

import subprocess
import sys
import time
from datetime import datetime

SCRIPTS = [
    ('script1_data_preprocessing.py', '1. Data Preprocessing', '~2 min'),
    ('script2_model_training.py', '2. Model Training (4 models)', '~10 min'),
    ('script3_fairness_metrics.py', '3. Fairness Metrics (5 metrics × 13 subgroups)', '~5 min'),
    ('script4a_bootstrap_stability.py', '4a. Bootstrap Stability (B=1,000)', '~2 hours'),
    ('script4b_sample_size.py', '4b. Sample Size Sensitivity (N=10K→Full)', '~1 hour'),
    ('script4c_cross_hospital.py', '4c. Cross-Hospital Validation (K=50)', '~2 hours'),
    ('script4d_seed_sensitivity.py', '4d. Seed Sensitivity (S=50)', '~2 hours'),
    ('script4e_threshold_sweep.py', '4e. Threshold Sweep (τ=99)', '~30 min'),
    ('script5_final_report.py', '5. Final Report Generation', '~5 min'),
]

def run_script(script, desc, est):
    print("\n" + "═" * 70)
    print(f"🚀 {desc}")
    print(f"   Script: {script} | Est: {est}")
    print("═" * 70 + "\n")
    
    start = time.time()
    try:
        subprocess.run([sys.executable, script], check=True)
        print(f"\n✅ COMPLETE ({(time.time()-start)/60:.1f} min)")
        return True
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        return False

def main():
    print("\n" + "🏥" * 25)
    print("\n  TEXAS-100X FAIRNESS ANALYSIS")
    print("  Complete Pipeline with 5 Stability Tests")
    print("\n" + "🏥" * 25)
    
    # Parse --from argument
    start_idx = 0
    if '--from' in sys.argv:
        from_arg = sys.argv[sys.argv.index('--from') + 1]
        for i, (script, _, _) in enumerate(SCRIPTS):
            if from_arg in script:
                start_idx = i
                break
    
    print(f"\n📋 Execution Plan:")
    for i, (s, d, t) in enumerate(SCRIPTS):
        status = "▶️" if i >= start_idx else "⏭️ skip"
        print(f"   {d:<50} {t:<12} {status}")
    
    print(f"\n⏱️  Total estimated time: 8-10 hours")
    input("\nPress ENTER to start...")
    
    overall_start = time.time()
    results = []
    
    for i, (script, desc, est) in enumerate(SCRIPTS):
        if i < start_idx:
            continue
        success = run_script(script, desc, est)
        results.append((desc, success))
        if not success:
            resp = input("\n⚠️ Continue? (y/n): ").strip().lower()
            if resp != 'y':
                break
    
    # Summary
    print("\n" + "═" * 70)
    print("SUMMARY")
    print("═" * 70)
    print(f"\nTotal time: {(time.time()-overall_start)/60:.1f} min")
    for desc, ok in results:
        print(f"   {'✅' if ok else '❌'} {desc}")
    
    print("\n📁 Output Locations:")
    print("   ./processed_data/  - Preprocessed data")
    print("   ./models/          - Trained ML models")
    print("   ./results/         - Analysis results (pkl)")
    print("   ./figures/         - Visualizations (PNG)")
    print("   ./tables/          - Tables (CSV)")
    print("   ./report/          - Final report")

if __name__ == "__main__":
    main()
