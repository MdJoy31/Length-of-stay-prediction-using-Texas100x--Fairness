"""Fix f-string nested quote issue in cell 104 - direct string replacement."""
import nbformat

NB_PATH = 'RQ1_LOS_Fairness_Complete.ipynb'

nb = nbformat.read(NB_PATH, as_version=4)

for i, cell in enumerate(nb.cells):
    if cell.cell_type != 'code':
        continue
    if 'Paper Table 1 Claims Verification' not in cell.source:
        continue

    lines = cell.source.split('\n')
    new_lines = []
    inserted_vars = False

    for line in lines:
        # Insert variable assignments before the problematic prints
        if not inserted_vars and "print(f'\\n1. MULTI-CRITERIA FAIRNESS EVALUATION')" in line:
            new_lines.append("metrics_str = ', '.join(fairness_metrics_used)")
            new_lines.append("attrs_str = ', '.join(protected_attributes_used)")
            inserted_vars = True

        # Fix the two problematic lines
        if "join(fairness_metrics_used)" in line and "Metrics used" in line:
            new_lines.append("print(f'   Metrics used:    {len(fairness_metrics_used)} ({metrics_str})')")
        elif "join(protected_attributes_used)" in line and "Attributes used" in line:
            new_lines.append("print(f'   Attributes used: {len(protected_attributes_used)} ({attrs_str})')")
        else:
            new_lines.append(line)

    cell.source = '\n'.join(new_lines)
    cell.outputs = []  # Clear old outputs
    print(f'Fixed cell {i+1} (0-indexed: {i})')
    break

nbformat.write(nb, NB_PATH)
print(f'Saved: {NB_PATH}')
