"""
Convert the two HTML exports to PDF using playwright's sync API.

Sync API avoids the Windows asyncio NotImplementedError that breaks
nbconvert's --to webpdf path.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm")
PAIRS = [
    (ROOT / "CIKM_2026_LOS_Fairness_FINAL.html",
     ROOT / "CIKM_2026_LOS_Fairness_FINAL.pdf"),
    (ROOT / "full_journal_paper" / "Journal_LOS_Fairness_FULL.html",
     ROOT / "full_journal_paper" / "Journal_LOS_Fairness_FULL.pdf"),
]

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    for html_path, pdf_path in PAIRS:
        print(f"Rendering {html_path.name} -> {pdf_path.name} ...")
        # Viewport wide enough so figures + tables do not get cropped before
        # the PDF rasteriser sees them.
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        page.goto(html_path.as_uri(), wait_until="networkidle", timeout=300000)
        # Force a layout pass on every image and table cell so nothing is
        # half-loaded when we hand control to PDF.
        page.evaluate("""
            () => {
              return Promise.all(Array.from(document.images).map(img =>
                img.complete ? Promise.resolve() :
                new Promise(res => { img.onload = img.onerror = res; })
              ));
            }
        """)
        page.emulate_media(media="print")
        # Standard A4 portrait, scale down so wide figures (F6 is 16in) still
        # fit horizontally. scale=0.75 = the F6 figure becomes 12in wide on
        # the page, comfortably inside the 190mm printable width.
        page.pdf(
            path=str(pdf_path),
            format="A4",
            scale=0.75,
            print_background=True,
            margin={"top": "10mm", "bottom": "10mm", "left": "10mm", "right": "10mm"},
            prefer_css_page_size=False,
        )
        page.close()
        size_mb = pdf_path.stat().st_size / 1024 / 1024
        print(f"  saved {pdf_path.name} ({size_mb:.1f} MB)")
    browser.close()
print("Done.")
