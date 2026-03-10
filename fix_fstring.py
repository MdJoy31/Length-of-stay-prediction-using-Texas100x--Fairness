"""Fix f-string nested quote issue in cell 104."""
import nbformat

NB_PATH = 'RQ1_LOS_Fairness_Complete.ipynb'

nb = nbformat.read(NB_PATH, as_version=4)

fixes = 0
for i, cell in enumerate(nb.cells):
    if cell.cell_type != 'code':
        continue
    src = cell.source
    if "Paper Table 1 Claims Verification" in src and "', '.join(fairness_metrics_used)" in src:
        # Fix two problematic f-strings with nested same-type quotes
        new_src = src.replace(
            "dim1_pass = len(fairness_metrics_used) >= 3 and len(protected_attributes_used) >= 2\n"
            "print(f'\\n1. MULTI-CRITERIA FAIRNESS EVALUATION')\n"
            "print(f'   Metrics used:    {len(fairness_metrics_used)} ({\\', \\'.join(fairness_metrics_used)})')\n"
            "print(f'   Attributes used: {len(protected_attributes_used)} ({\\', \\'.join(protected_attributes_used)})')\n"
            "print(f'   Status: {\"PASS ●\" if dim1_pass else \"FAIL ○\"}')",
            "dim1_pass = len(fairness_metrics_used) >= 3 and len(protected_attributes_used) >= 2\n"
            "metrics_str = ', '.join(fairness_metrics_used)\n"
            "attrs_str = ', '.join(protected_attributes_used)\n"
            "print(f'\\n1. MULTI-CRITERIA FAIRNESS EVALUATION')\n"
            "print(f'   Metrics used:    {len(fairness_metrics_used)} ({metrics_str})')\n"
            "print(f'   Attributes used: {len(protected_attributes_used)} ({attrs_str})')\n"
            "print(f'   Status: {\"PASS ●\" if dim1_pass else \"FAIL ○\"}')"
        )
        if new_src != src:
            cell.source = new_src
            fixes += 1
            print(f'Fixed cell {i} (1-indexed: {i+1})')
            # Clear old outputs
            if cell.outputs:
                cell.outputs = []
                print(f'  Cleared old outputs')
        else:
            # Try raw replacement
            print(f'  Pattern did not match. Trying raw approach...')
            old_line1 = "print(f'   Metrics used:    {len(fairness_metrics_used)} ({\", \".join(fairness_metrics_used)})')"
            old_line2 = "print(f'   Attributes used: {len(protected_attributes_used)} ({\", \".join(protected_attributes_used)})')"
            # Actually let's just look at the raw source
            for j, line in enumerate(src.split('\n')):
                if 'join(fairness_metrics_used)' in line:
                    print(f'  Line {j}: [{line}]')
                if 'join(protected_attributes_used)' in line:
                    print(f'  Line {j}: [{line}]')

print(f'\nFixes applied: {fixes}')

if fixes == 0:
    print("\nDumping cell with Table 1 for inspection:")
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code' and 'Paper Table 1 Claims Verification' in cell.source:
            # Print lines with join
            for j, line in enumerate(cell.source.split('\n')):
                if 'join(' in line or 'Metrics used' in line or 'Attributes used' in line:
                    print(f'  Line {j}: {repr(line)}')
            break

nbformat.write(nb, NB_PATH)
print(f'Saved: {NB_PATH}')
