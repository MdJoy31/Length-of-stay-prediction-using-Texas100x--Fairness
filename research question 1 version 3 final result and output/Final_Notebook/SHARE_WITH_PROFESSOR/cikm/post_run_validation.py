"""
Post-run validation: confirm the FINAL notebook produced everything it should.
Run this AFTER the notebook finishes executing to summarise the deliverables.
"""
import os, json, glob, hashlib, sys
sys.stdout.reconfigure(encoding='utf-8')

ROOT = "."
CKPT_T = sorted(glob.glob(f"{ROOT}/output_final/tables/T*.csv"))
CKPT_F = sorted(glob.glob(f"{ROOT}/output_final/figures/F*.png"))
CKPT_A = sorted(glob.glob(f"{ROOT}/output_final/audit/*"))

def line(s, length=80, char="="):
    return char * length + "\n  " + s + "\n" + char * length

print(line("FINAL NOTEBOOK — DELIVERABLE AUDIT"))
print(f"\n  T-files (output_final/tables/): {len(CKPT_T)}")
for p in CKPT_T:
    sz = os.path.getsize(p)
    print(f"    {os.path.basename(p):42s} {sz:>10,} bytes")

print(f"\n  F-files (output_final/figures/): {len(CKPT_F)}")
for p in CKPT_F:
    sz = os.path.getsize(p)
    print(f"    {os.path.basename(p):42s} {sz:>10,} bytes")

print(f"\n  Audit artefacts (output_final/audit/): {len(CKPT_A)}")
for p in CKPT_A:
    sz = os.path.getsize(p)
    print(f"    {os.path.basename(p):42s} {sz:>10,} bytes")

# Check verification status
report = f"{ROOT}/output_final/audit/REWRITE_SUMMARY.md"
if os.path.exists(report):
    print("\n" + line("REWRITE_SUMMARY.md (last 30 lines)", char="-"))
    with open(report, "r", encoding="utf-8") as f:
        text = f.read()
    print(text[-2500:])
else:
    print(f"\nMISSING: {report}")

# Required T-files
expected_T = [f"T{i}_" for i in range(3, 20)]
missing_T = [p for p in expected_T
              if not any(p in os.path.basename(f) for f in CKPT_T)]
expected_F = [f"F{i}_" for i in range(1, 6)]
missing_F = [p for p in expected_F
              if not any(p in os.path.basename(f) for f in CKPT_F)]

print("\n" + line("MANIFEST"))
print(f"  T-files: {len(expected_T) - len(missing_T)}/{len(expected_T)} present")
if missing_T: print(f"    MISSING: {missing_T}")
print(f"  F-files: {len(expected_F) - len(missing_F)}/{len(expected_F)} present")
if missing_F: print(f"    MISSING: {missing_F}")

# Final notebook size
nb_path = f"{ROOT}/CIKM_2026_LOS_Fairness_FINAL.ipynb"
if os.path.exists(nb_path):
    sz = os.path.getsize(nb_path)
    with open(nb_path) as f:
        nb = json.load(f)
    n_with = sum(1 for c in nb["cells"] if c["cell_type"]=="code" and c.get("outputs"))
    n_total = sum(1 for c in nb["cells"] if c["cell_type"]=="code")
    print(f"\n  Notebook: {sz/1024/1024:.2f} MB · {n_with}/{n_total} code cells with outputs")
