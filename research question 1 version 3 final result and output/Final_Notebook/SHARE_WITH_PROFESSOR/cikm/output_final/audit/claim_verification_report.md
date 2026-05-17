# Manuscript-Claim Verification Report

_Run: 2026-05-01T10:50:23_



| ID | Claim | Manuscript | Notebook | Status |
| --- | --- | --- | --- | --- |
| A1 | 925,128 records | 925128 | 925128 | **PASS** |
| A2 | 441 hospitals | 441 | 441 | **PASS** |
| B1 | 336 model-metric-attr combinations | 336 | 336 | **PASS** |
| B2 | Pct flipped (VFR>0) | 43.5 | 43.4524 | **PASS** |
| B3 | Max VFR (%) | 47.4 | 47.4 | **PASS** |
| B4 | VFR <= 10% practical-stability count (NEW anchor) | 259 | 259 | **PASS** |
| B5 | Perfectly-stable VFR=0 count | 190 | 190 | **PASS** |
| C1 | Cells with CV > 0.50 (NEW anchor) | 17 | 17 | **PASS** |
| C2 | Overall Fleiss kappa | 0.506 | 0.5061 | **PASS** |
| D1 | Unanimous fair count [T20 7/7 cells out of 48] | 12 | 12 | **PASS** |
| D2 | Disagreement rate (NEW anchor) | 83.3 | 83.3333 | **PASS** |
| E1 | Best AUROC | 0.953 | 0.9528 | **PASS** |
| E2 | Best Accuracy | 0.878 | 0.8776 | **PASS** |
| F1 | Intervention DI Race >= 0.80 | 0.8 | 0.8009 | **PASS** |
| F2 | Intervention DI Sex >= 0.80 | 0.8 | 0.9317 | **PASS** |
| F3 | Intervention DI Eth >= 0.80 | 0.8 | 1.0 | **PASS** |
| F4 | Intervention DI Age >= 0.80 | 0.8 | 0.8 | **PASS** |
| F5 | All four DI >= 0.80 jointly | 1 | 1 | **PASS** |
| F6 | Accuracy cost <= 5 pp | 5.0 | 4.2879 | **PASS** |
| G1 | Per-cluster DI worst improved (>=10/20) | 19 | 19 | **PASS** |
| G2 | Per-cluster all-4-DI passes (count out of 20) | 14 | 14 | **PASS** |
| G3 | Per-cluster acc within 5pp (count out of 20) | 16 | 16 | **PASS** |