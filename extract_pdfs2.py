"""Targeted extraction of results sections and table data from each PDF."""
import re, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from pdfminer.high_level import extract_text

PAPER_DIR = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\Paper")
OUTPUT = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\pdf_extraction_details.txt")

def san(t):
    if not t: return t
    for k,v in {'\u2010':'-','\u2011':'-','\u2013':'-','\u2014':'-','\u2018':"'",'\u2019':"'",'\u201c':'"','\u201d':'"','\u00e9':'e','\u00e8':'e','\u00c6':'AE'}.items():
        t = t.replace(k,v)
    return t

pdfs = sorted(PAPER_DIR.glob("*.pdf"))

with open(OUTPUT, 'w', encoding='utf-8') as f:
    def p(s=""): f.write(san(str(s))+"\n")

    for pdf in pdfs:
        p(f"\n{'='*100}")
        p(f"FILE: {pdf.name}")
        p(f"{'='*100}")
        text = extract_text(str(pdf))
        if not text:
            p("  FAILED"); continue
        tl = text.lower()

        # Find and print Results section
        for kw in ['4. results', '5. results', '3. results', 'results and discussion', 'results\n']:
            idx = tl.find(kw)
            if idx > 1000:  # Not in abstract
                start = max(0, idx-100)
                end = min(len(text), idx+5000)
                p(f"\n--- RESULTS SECTION (found '{kw.strip()}' at {idx}) ---")
                p(san(text[start:end]))
                break

        # Find Conclusion section
        for kw in ['conclusion\n', 'conclusions\n', '5. conclusion', '6. conclusion', '7. conclusion', 'conclusion:']:
            idx = tl.find(kw)
            if idx > 1000:
                start = max(0, idx-100)
                end = min(len(text), idx+3000)
                p(f"\n--- CONCLUSION SECTION (found '{kw.strip()}' at {idx}) ---")
                p(san(text[start:end]))
                break

        # Find lines with specific decimal numbers like 0.8XX patterns (metric values)
        metric_lines = []
        for line in text.split('\n'):
            ll = line.lower()
            if re.search(r'0\.[6-9]\d+|0\.8\d+|0\.9\d+|[89]\d\.\d+%|7\d\.\d+%', ll):
                if any(w in ll for w in ['accuracy','auc','f1','precision','recall','r2','r-squared','mae','rmse','sensitivity','specificity','brier']):
                    metric_lines.append(line.strip())
        if metric_lines:
            p(f"\n--- LINES WITH METRIC VALUES ---")
            seen = set()
            for ml in metric_lines[:30]:
                s = san(ml)
                if s not in seen:
                    seen.add(s)
                    p(f"  {s}")

        p("")

    p("DONE")

print(f"Written to {OUTPUT}")
