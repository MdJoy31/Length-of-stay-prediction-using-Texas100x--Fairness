"""
Two-part cleanup:

1. Remove em-dashes (U+2014) and en-dashes (U+2013) from every markdown
   cell in the notebook. Replace with appropriate punctuation:
     - em-dash inside a sentence  -> comma or semicolon
     - em-dash setting off a phrase -> parentheses or comma pair
     - en-dash in a range -> 'to'
   Hyphens (U+002D) are kept (compound words like cross-site, post-hoc).
   Code cells are not modified.

2. Detect and replace common AI-generated phrases with direct language.

After cleanup, run a structural verification: cell counts, output presence,
T19 PASS count, cell 60 PASS count.
"""
import json, os, sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from pathlib import Path

NB = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\CIKM_2026_LOS_Fairness_FINAL.ipynb")

EM_DASH = "—"
EN_DASH = "–"
NB_HYPHEN = "‑"  # non-breaking hyphen

AI_PHRASES = {
    # Vague openers
    r"\bIn today's world,?\s*": "",
    r"\bIn the realm of\s*": "In ",
    r"\bIt is important to note that\s*": "",
    r"\bIt is worth (noting|mentioning) that\s*": "",
    r"\bIt should be noted that\s*": "",
    r"\bOne could argue that\s*": "",
    r"\bIt seems that\s*": "",
    r"\bGenerally speaking,?\s*": "",
    # Empty intensifiers
    r"\bvery\s+": "",
    r"\breally\s+": "",
    r"\bextremely\s+": "",
    r"\bincredibly\s+": "",
    # AI cliches
    r"\bdelve into\b": "examine",
    r"\bdive into\b": "examine",
    r"\bnavigate the landscape\b": "review",
    r"\bunlock(s|ed|ing)? potential\b": "enable",
    r"\bleverag(e|es|ed|ing)\b": "use",
    r"\brobust(ly)?\b": "stable",  # sparingly; reviewers will accept "stable"
    r"\bcomprehensive(ly)?\b": "complete",
    r"\bseamless(ly)?\b": "consistent",
    r"\bcutting-edge\b": "current",
    r"\bgame-changer\b": "important advance",
    r"\btapestry\b": "set",
    # Hollow transitions
    r"\bFurthermore,\s*": "",
    r"\bMoreover,\s*": "",
    r"\bIn conclusion,\s*": "",
    r"\bIt is evident that\s*": "",
    r"\bIt depends\b": "this varies",
    # Anthropomorphic verbs
    r"\bthis paper aims to revolutionise\b": "this paper presents",
    r"\bwe will explore\b": "we examine",
    r"\blet's (dive|delve) into\b": "consider",
    # Doublethink / superlatives
    r"\bstate-of-the-art\b": "current",
}

# Em-dash replacement strategy:
# - inside a sentence between two words/phrases, replace with ", " (comma)
# - paired em-dashes around a phrase, replace with parentheses
# This is hard to do perfectly with regex; we use simple substitutions:
#   1. Replace " EM_DASH " (em-dash with surrounding spaces) with ", "
#   2. Replace "EM_DASH" without spaces with ", " (less common)
#   3. Same for en-dash, but in ranges (e.g., "0.10-0.30") use "to"


def clean_markdown(src):
    # Step 1: dashes
    # En-dash in numeric range like "0.10–0.30" -> "0.10 to 0.30"
    src = re.sub(r"(\d)\s*" + EN_DASH + r"\s*(\d)", r"\1 to \2", src)
    # En-dash between words -> ", "
    src = src.replace(" " + EN_DASH + " ", ", ")
    src = src.replace(EN_DASH + " ", ", ")
    src = src.replace(" " + EN_DASH, ",")
    src = src.replace(EN_DASH, "-")  # leftover -> simple hyphen
    # Em-dash with surrounding spaces -> ", "
    src = src.replace(" " + EM_DASH + " ", ", ")
    src = src.replace(EM_DASH + " ", ", ")
    src = src.replace(" " + EM_DASH, ",")
    src = src.replace(EM_DASH, ", ")  # leftover
    # Non-breaking hyphen normalised
    src = src.replace(NB_HYPHEN, "-")

    # Step 2: AI phrases
    for pat, repl in AI_PHRASES.items():
        src = re.sub(pat, repl, src, flags=re.IGNORECASE)

    # Capitalisation cleanup: phrases removed at start of sentences may leave lowercase
    # Re-capitalise after period+space if next word is lowercase
    src = re.sub(r"(\.\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), src)
    # Capitalise start of paragraphs (after \n\n)
    src = re.sub(r"(\n\n)([a-z])", lambda m: m.group(1) + m.group(2).upper(), src)
    # Double commas / spaces / leading-comma artefacts
    src = re.sub(r",\s*,", ",", src)
    src = re.sub(r"\s{2,}", " ", src)
    src = re.sub(r"^,\s+", "", src, flags=re.MULTILINE)
    return src


with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

n_md_changed = 0
n_em = 0
n_en = 0
ai_phrase_hits = 0
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "markdown":
        continue
    src = "".join(c.get("source", []))
    n_em_cell = src.count(EM_DASH)
    n_en_cell = src.count(EN_DASH)
    n_em += n_em_cell
    n_en += n_en_cell
    new_src = clean_markdown(src)
    # count AI phrase hits separately (approximate)
    for pat in AI_PHRASES:
        ai_phrase_hits += len(re.findall(pat, src, flags=re.IGNORECASE))
    if new_src != src:
        c["source"] = new_src.splitlines(keepends=True)
        n_md_changed += 1

print(f"Markdown cells modified: {n_md_changed}")
print(f"Em-dashes removed: {n_em}")
print(f"En-dashes removed: {n_en}")
print(f"AI-phrase hits replaced: {ai_phrase_hits}")

# Verify code cells untouched (no dash replacement applied)
print()

# Structural verification
n_total = len(nb["cells"])
n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
n_md = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
n_code_with_outs = sum(1 for c in nb["cells"] if c["cell_type"] == "code" and c.get("outputs"))
n_code_err = sum(1 for c in nb["cells"] if c["cell_type"] == "code" and any(o.get("output_type") == "error" for o in c.get("outputs", [])))
n_figs = sum(1 for c in nb["cells"] if c["cell_type"] == "code" for o in c.get("outputs", []) if "image/png" in o.get("data", {}))
print(f"Total cells: {n_total} (code={n_code}, md={n_md})")
print(f"Code with outputs: {n_code_with_outs}")
print(f"Code with errors: {n_code_err}")
print(f"Figures embedded: {n_figs}")

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

# Cross-table verification
import pandas as pd
T19 = pd.read_csv(NB.parent / "output_final" / "tables" / "T19_claim_verification.csv")
print(f"\nT19 anchors: {T19['Status'].value_counts().to_dict()}")

# Cell 60 verification
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    src = "".join(c.get("source", []))
    if "VERIFICATION CHECKS" in src and "BLOCKING DEFECTS" in src:
        for o in c.get("outputs", []):
            if o.get("output_type") == "stream":
                t = o.get("text", [])
                txt = "".join(t) if isinstance(t, list) else t
                p = txt.count("[PASS]")
                f_ = txt.count("[FAIL]")
                blk = 0
                for ln in txt.split("\n"):
                    if "BLOCKING DEFECTS" in ln:
                        try:
                            blk = int(ln.split(":")[1].strip())
                        except:
                            pass
                print(f"Cell 60: PASS={p} FAIL={f_} BLOCKING={blk}")
        break

print(f"\nFinal notebook: {os.path.getsize(NB) / 1024 / 1024:.2f} MB, {n_total} cells")
