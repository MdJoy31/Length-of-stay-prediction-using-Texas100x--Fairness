"""Execute notebook cells one at a time using subprocess-based kernel.
This avoids papermill's terminal management issues.
Uses nbconvert's ExecutePreprocessor with explicit timeout per cell.
"""
import json
import sys
import time
import subprocess
import os

NB_PATH = 'LOS_Prediction_Detailed.ipynb'

# First, try converting to script and running directly
# This avoids all Jupyter kernel issues

print("Converting notebook to script...")
with open(NB_PATH) as f:
    nb = json.load(f)

# Extract all code cells
code_cells = [(i, c) for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code']
print(f"Found {len(code_cells)} code cells")

# Create a single Python script from all code cells
script_lines = [
    "# Auto-generated from LOS_Prediction_Detailed.ipynb\n",
    "import sys, os, json, time\n",
    "os.makedirs('figures', exist_ok=True)\n",
    "os.makedirs('results', exist_ok=True)\n",
    "\n",
    "# Redirect matplotlib to Agg backend for non-interactive\n",
    "import matplotlib\n",
    "matplotlib.use('Agg')\n",
    "\n",
    "_cell_outputs = {}\n",
    "_current_cell = 0\n",
    "\n",
    "def _mark_cell(idx, desc=''):\n",
    "    global _current_cell\n",
    "    _current_cell = idx\n",
    "    print(f'\\n{\"=\"*60}')\n",
    "    print(f'CELL {idx}: {desc}')\n",
    "    print(f'{\"=\"*60}')\n",
    "    sys.stdout.flush()\n",
    "\n",
]

for idx, (cell_idx, cell) in enumerate(code_cells):
    src = ''.join(cell['source'])
    first_line = cell['source'][0].strip()[:50] if cell['source'] else ''

    script_lines.append(f"\n_mark_cell({idx+1}, {repr(first_line)})\n")
    script_lines.append(f"_cell_start = time.time()\n")
    script_lines.append(src)
    script_lines.append(f"\nprint(f'  Cell {idx+1} completed in {{time.time()-_cell_start:.1f}}s')\n")
    script_lines.append(f"sys.stdout.flush()\n")

script_content = ''.join(script_lines)

# Write the script
script_path = 'run_notebook_script.py'
with open(script_path, 'w', encoding='utf-8') as f:
    f.write(script_content)

print(f"Script written to {script_path} ({len(script_lines)} lines)")
print(f"Run with: .venv\\Scripts\\python.exe {script_path}")
