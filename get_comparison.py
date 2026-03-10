"""Extract comparison table HTML from cell 39."""
import json, re

with open("final_notebooks/LOS_Prediction_Standard.ipynb", 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        src = ''.join(c.get('source', []))
        if 'Standard Model vs Fairness' in src:
            for o in c.get('outputs', []):
                if o.get('output_type') in ('display_data', 'execute_result'):
                    data = o.get('data', {})
                    if 'text/html' in data:
                        html = ''.join(data['text/html'])
                        clean = re.sub(r'<[^>]+>', '\t', html)
                        clean = re.sub(r'\s+', ' ', clean).strip()
                        # Parse table rows
                        rows = re.findall(r'(\d+)\s+([\w\s]+?)\s+([\d.]+)\s+([\d.]+)\s+([+\-][\d.]+\s+[✅⚠️]+)', clean)
                        if rows:
                            for r in rows:
                                print(f"  {r[1].strip():<30} Std={r[2]}  Fair={r[3]}  {r[4]}")
                        else:
                            # Try simpler extraction
                            parts = clean.split()
                            for j, p in enumerate(parts):
                                if 'DI' in p or 'WTPR' in p or 'EOD' in p or 'Accuracy' in p or 'F1' in p or 'AUC' in p:
                                    context = ' '.join(parts[max(0,j-1):min(len(parts),j+5)])
                                    print(f"  {context}")
            break
