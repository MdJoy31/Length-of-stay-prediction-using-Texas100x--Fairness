"""Extract per-metric AGE details from notebook."""
import json

with open("RQ1_LOS_Fairness_Analysis.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]

# Scan ALL code cells for any text output
for i, cell in enumerate(code_cells):
    for o in cell.get("outputs", []):
        if o.get("output_type") in ["stream", "execute_result"]:
            text = "".join(o.get("text", o.get("data", {}).get("text/plain", [])))
            if text.strip() and len(text) > 30:
                lines = text.strip().split("\n")
                for line in lines:
                    ll = line.lower()
                    # Look for fairness metric lines with "fair" or metric names
                    if ("age_group" in ll or "age" in ll) and ("fair" in ll or "di=" in ll or "spd=" in ll):
                        if "executing" not in ll and "feature" not in ll:
                            print(f"[{i}] {line.strip()}")

