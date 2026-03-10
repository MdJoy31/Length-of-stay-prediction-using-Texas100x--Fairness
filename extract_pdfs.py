"""Extract text and key information from all 7 PDFs in the Paper folder."""
import os
import re
import sys
from pathlib import Path

# Force UTF-8 output
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def sanitize(text):
    """Replace problematic unicode chars with ASCII equivalents."""
    if not text:
        return text
    replacements = {
        '\u2010': '-', '\u2011': '-', '\u2012': '-', '\u2013': '-', '\u2014': '-',
        '\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"',
        '\u2026': '...', '\u00e9': 'e', '\u00e8': 'e', '\u00e0': 'a',
        '\u00c6': 'AE', '\u00ae': '(R)', '\u00a9': '(c)', '\u2022': '*',
        '\u00f6': 'o', '\u00fc': 'u',
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

# Try pdfminer first (better text extraction), fallback to PyPDF2
def extract_with_pdfminer(pdf_path):
    from pdfminer.high_level import extract_text
    return extract_text(pdf_path)

def extract_with_pypdf2(pdf_path):
    from PyPDF2 import PdfReader
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_pdf(pdf_path):
    try:
        text = extract_with_pdfminer(pdf_path)
        if text and len(text.strip()) > 100:
            return text
    except Exception as e:
        print(f"  pdfminer failed: {e}")
    try:
        text = extract_with_pypdf2(pdf_path)
        if text and len(text.strip()) > 100:
            return text
    except Exception as e:
        print(f"  PyPDF2 failed: {e}")
    return ""

PAPER_DIR = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\Paper")
OUTPUT_FILE = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\pdf_extraction_results.txt")

pdfs = sorted(PAPER_DIR.glob("*.pdf"))

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    def p(s=""):
        f.write(sanitize(str(s)) + "\n")

    p(f"Found {len(pdfs)} PDFs\n")

    for pdf_path in pdfs:
        p("=" * 120)
        p(f"FILE: {pdf_path.name}")
        p("=" * 120)

        text = extract_text_from_pdf(str(pdf_path))

        if not text:
            p("  [ERROR] Could not extract text from this PDF.\n")
            continue

        p(f"  Extracted {len(text)} characters\n")

        # First 6000 chars (title, authors, abstract)
        p("--- FIRST 6000 CHARS (Title/Authors/Abstract) ---")
        p(sanitize(text[:6000]))
        p("\n--- END FIRST 6000 ---\n")

        text_lower = text.lower()

        # Metrics
        p("--- KEY METRICS SEARCH ---")
        auc_patterns = re.findall(r'(?:auc|auroc|auc[\s\-]?roc)[^\n]{0,120}', text_lower)
        if auc_patterns:
            p("AUC mentions:")
            for m in auc_patterns[:12]:
                p(f"  {sanitize(m.strip())}")

        acc_patterns = re.findall(r'accuracy[^\n]{0,120}', text_lower)
        if acc_patterns:
            p("Accuracy mentions:")
            for m in acc_patterns[:12]:
                p(f"  {sanitize(m.strip())}")

        f1_patterns = re.findall(r'f1[\s\-]?score[^\n]{0,120}', text_lower)
        if f1_patterns:
            p("F1 mentions:")
            for m in f1_patterns[:12]:
                p(f"  {sanitize(m.strip())}")

        pr_patterns = re.findall(r'(?:precision|recall|sensitivity|specificity)[^\n]{0,100}', text_lower)
        if pr_patterns:
            p("Precision/Recall/Sensitivity/Specificity mentions:")
            for m in pr_patterns[:12]:
                p(f"  {sanitize(m.strip())}")

        # Fairness
        p("\n--- FAIRNESS KEYWORDS ---")
        fair_patterns = re.findall(r'(?:fairness|bias|equit|disparit|demographic parity|equal opportunity|equalized odds)[^\n]{0,120}', text_lower)
        if fair_patterns:
            for m in fair_patterns[:12]:
                p(f"  {sanitize(m.strip())}")
        else:
            p("  No explicit fairness keywords found")

        # Dataset
        p("\n--- DATASET MENTIONS ---")
        dataset_patterns = re.findall(r'(?:mimic|eicu|dataset|data\s*set|ehr|electronic health record)[^\n]{0,120}', text_lower)
        if dataset_patterns:
            for m in dataset_patterns[:10]:
                p(f"  {sanitize(m.strip())}")

        # Models
        p("\n--- MODEL MENTIONS ---")
        model_kws = r'(?:random forest|xgboost|gradient boost|logistic regression|neural network|deep learning|lstm|gru|cnn|svm|support vector|decision tree|lightgbm|catboost|naive bayes|knn|k-nearest|ada\s*boost|elastic\s*net|lasso|ridge|histgradient|extra\s*tree)'
        model_patterns = re.findall(model_kws + r'[^\n]{0,80}', text_lower)
        if model_patterns:
            seen = set()
            for m in model_patterns[:20]:
                key = m.strip()[:40]
                if key not in seen:
                    seen.add(key)
                    p(f"  {sanitize(m.strip())}")

        # LOS / mortality
        p("\n--- LOS / MORTALITY ---")
        los_patterns = re.findall(r'(?:length of stay|los |mortality|readmission|icu stay)[^\n]{0,120}', text_lower)
        if los_patterns:
            for m in los_patterns[:10]:
                p(f"  {sanitize(m.strip())}")

        # Results/conclusion section
        p("\n--- RESULTS/CONCLUSION SECTION ---")
        for keyword in ['conclusion', 'results', 'discussion']:
            idx = text_lower.find(keyword)
            if idx != -1:
                start = max(0, idx - 50)
                end = min(len(text), idx + 3000)
                snippet = text[start:end]
                p(f"\n[Found '{keyword}' at position {idx}]")
                p(sanitize(snippet[:3000]))
                break

        # Numeric metric values
        p("\n--- NUMERIC METRIC VALUES ---")
        metric_vals = re.findall(r'(?:accuracy|auc|auroc|f1|precision|recall|sensitivity|specificity|rmse|mae|mse|r-?squared|r2)[\s:=]+(?:of\s+)?(\d+\.?\d*%?|\d+\.\d+)', text_lower)
        if metric_vals:
            p("Extracted values:")
            for v in metric_vals[:20]:
                p(f"  {v}")

        table_nums = re.findall(r'((?:accuracy|auc|auroc|f1|precision|recall)[^\n]*(?:0\.\d+|\d+\.\d+%?))', text_lower)
        if table_nums:
            p("Table-like metric lines:")
            for t in table_nums[:15]:
                p(f"  {sanitize(t.strip())}")

        p("\n\n")

    p("EXTRACTION COMPLETE")

print(f"Results written to {OUTPUT_FILE}")
print("Done!")
