"""Fix test_protected -> protected_attrs in notebook cells 80 and 82."""
import nbformat

NB_PATH = 'RQ1_LOS_Fairness_Complete.ipynb'

nb = nbformat.read(NB_PATH, as_version=4)

fixes_applied = 0
for i, cell in enumerate(nb.cells):
    if cell.cell_type != 'code':
        continue
    src = cell.source
    if 'test_protected' in src:
        # Cell 80 (seed perturbation): test_protected[attr] -> pd.Series(protected_attrs[attr])
        # Also need .values on boolean mask since pd.Series comparison returns Series
        new_src = src.replace(
            'groups = test_protected[attr]\n'
            '        privileged = groups.value_counts().idxmax()\n'
            '        priv_mask = (groups == privileged)\n'
            '        unpriv_mask = ~priv_mask',
            'groups = pd.Series(protected_attrs[attr])\n'
            '        privileged = groups.value_counts().idxmax()\n'
            '        priv_mask = (groups == privileged).values\n'
            '        unpriv_mask = ~priv_mask'
        )
        # Cell 82 (CV analysis): test_protected[attr].values[idx] -> protected_attrs[attr][idx]
        new_src = new_src.replace(
            'groups = test_protected[attr].values[idx]',
            'groups = protected_attrs[attr][idx]'
        )
        if new_src != src:
            cell.source = new_src
            fixes_applied += 1
            print(f'Fixed cell {i} (1-indexed: {i+1})')

        # Also clear old error outputs from this cell
        if cell.outputs:
            old_count = len(cell.outputs)
            cell.outputs = []
            print(f'  Cleared {old_count} old outputs')

print(f'\nTotal cells fixed: {fixes_applied}')

# Save
nbformat.write(nb, NB_PATH)
print(f'Saved: {NB_PATH}')
