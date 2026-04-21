# Deliverable 0: Consistency Audit

## Issues Found: 0


## Canonical Thresholds

| Metric | Threshold | Direction |
|--------|-----------|-----------|
| DI | 0.8 | above |
| SPD | 0.1 | below |
| EOPP | 0.1 | below |
| EOD | 0.1 | below |
| TI | 0.1 | below |
| PP | 0.1 | below |
| CAL | 0.05 | below |

## generate_all_figures_tables.py Mismatches
The generate script uses lenient thresholds for EOPP (0.20), EOD (0.20), and CAL (0.10).
The notebook (Cell 2) uses strict thresholds: EOPP=0.10, EOD=0.10, CAL=0.05.
**Recommendation**: Update generate script to match notebook.