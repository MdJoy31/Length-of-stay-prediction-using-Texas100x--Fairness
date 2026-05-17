"""
Inject a 'Headline Claim Evidence' cell pair right after Section 17.13 (T15 ·
Standard vs Fair head-to-head) so the reviewer can SEE the all-four-DI-pass
evidence at the top of the intervention block, not buried in a 31-row table.
"""
import json
from pathlib import Path

NB = Path(r"d:\Research study\Research question ML\fairness_project_v2\fairness_project_v1\research question 1 version 3 final result and output\Final_Notebook\SHARE_WITH_PROFESSOR\cikm\CIKM_2026_LOS_Fairness_13042026.ipynb")
with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

def md(*text): return {"cell_type":"markdown","metadata":{},"source":list(text)}
def code(src): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":src.splitlines(keepends=True)}

HEADLINE = r'''# ════════════════════════════════════════════════════════════════════════
# HEADLINE CLAIM EVIDENCE — defending the abstract's strongest assertion
# "All four DI >= 0.80 simultaneously at <5pp accuracy cost"
# ════════════════════════════════════════════════════════════════════════
import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
from IPython.display import display, Markdown

NB_DIR = os.getcwd()
RES = os.path.join(NB_DIR, "results")
PRT = os.path.join(NB_DIR, "output", "paper_ready_figs")
os.makedirs(PRT, exist_ok=True)

sf = pd.read_csv(os.path.join(RES, "intervention_standard_vs_fair.csv"))
print("=" * 100)
print("  HEADLINE CLAIM · Standard vs Fair model · Texas-100X (n=185,026 held-out)")
print("=" * 100)

# Pull headline numbers
acc_std  = float(sf[sf["Metric"]=="Accuracy"]["Standard"].iloc[0])
acc_fair = float(sf[sf["Metric"]=="Accuracy"]["Fair (Intersect.)"].iloc[0])
acc_drop = (acc_std - acc_fair) * 100
auc_std  = float(sf[sf["Metric"]=="AUC"]["Standard"].iloc[0])
auc_fair = float(sf[sf["Metric"]=="AUC"]["Fair (Intersect.)"].iloc[0])

di_std  = {a: float(sf[sf["Metric"]==f"DI ({a})"]["Standard"].iloc[0])         for a in ["Race","Sex","Eth","Age"]}
di_fair = {a: float(sf[sf["Metric"]==f"DI ({a})"]["Fair (Intersect.)"].iloc[0]) for a in ["Race","Sex","Eth","Age"]}

# Print the headline summary in a reviewer-friendly box
print(f"\n  Predictive performance:")
print(f"    Accuracy:   Standard {acc_std:.4f}  ->  Fair {acc_fair:.4f}   (drop = {acc_drop:+.2f} pp)")
print(f"    AUROC:      Standard {auc_std:.4f}  ->  Fair {auc_fair:.4f}   (drop = {(auc_std-auc_fair)*100:+.2f} pp)")
print(f"\n  Disparate Impact (four-fifths rule, threshold = 0.80):")
all_pass = True
for a in ["Race","Sex","Eth","Age"]:
    s, f = di_std[a], di_fair[a]
    s_v = "PASS" if s >= 0.80 else "FAIL"
    f_v = "PASS" if f >= 0.80 else "FAIL"
    if f < 0.80: all_pass = False
    print(f"    {a:10s}:  Standard DI = {s:.4f} [{s_v}]  ->  Fair DI = {f:.4f} [{f_v}]   (gain {(f-s)*100:+.2f} pp)")
print(f"\n  >>> ALL FOUR DI >= 0.80 SIMULTANEOUSLY: {'YES (claim supported)' if all_pass else 'NO'}")
print(f"  >>> Accuracy cost: {acc_drop:.2f} pp ({'<=5 pp -- claim supported' if acc_drop <= 5.0 else '>5 pp -- claim NOT supported'})")
print("=" * 100)

# Show the full 31-row Table 8 styled
display(Markdown("### Table 8 · Standard vs Fair Model — Performance & All Fairness Metrics (canonical)"))
display(sf.style
        .format({"Standard":"{:.4f}","Fair (Intersect.)":"{:.4f}","Change":"{:+.4f}"})
        .set_table_styles([
            {"selector":"thead th","props":"background:#1f2937;color:white;text-align:center;padding:6px 10px;"},
            {"selector":"tbody td","props":"padding:5px 10px;text-align:center;"},
            {"selector":"tbody tr:nth-child(even)","props":"background:#f9fafb;"},
        ])
        .hide(axis="index"))
'''

HEADLINE_FIG = r'''# ─── Fancy headline-claim diagram (single page; reviewer sees this first) ──
fig = plt.figure(figsize=(16, 9), constrained_layout=True)
gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0], width_ratios=[1.4, 1, 1])

# Panel A · DI bar comparison (the centrepiece)
ax = fig.add_subplot(gs[0, :])
attrs = ["Race","Sex","Ethnicity","Age Group"]
attr_keys = ["Race","Sex","Eth","Age"]
std_vals  = [di_std[a]  for a in attr_keys]
fair_vals = [di_fair[a] for a in attr_keys]
x = np.arange(4); bw = 0.36
b1 = ax.bar(x - bw/2, std_vals,  bw, color="#94a3b8", edgecolor="black", lw=0.8, label="Standard model", alpha=0.95)
b2 = ax.bar(x + bw/2, fair_vals, bw, color="#16a34a", edgecolor="black", lw=0.8, label="Fair model (3-stage intersectional)", alpha=0.95)
ax.axhline(0.80, color="#dc2626", ls="--", lw=2)
ax.text(3.55, 0.82, "0.80 four-fifths rule (EEOC)", fontsize=10, color="#dc2626", fontweight="bold")
for xi, (s, f) in enumerate(zip(std_vals, fair_vals)):
    ax.annotate("", xy=(xi+bw/2, f), xytext=(xi-bw/2, s),
                arrowprops=dict(arrowstyle="->", color="#2563eb", lw=2.4))
    ax.text(xi, max(s,f)+0.04, f"+{(f-s)*100:.1f} pp",
            ha="center", fontsize=11, color="#2563eb", fontweight="bold")
    # Pass/fail badge below x-tick
    pass_badge = "PASS" if f >= 0.80 else "FAIL"
    color = "#16a34a" if f >= 0.80 else "#dc2626"
    ax.text(xi, -0.07, pass_badge, ha="center", fontsize=10, color="white",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=color, edgecolor="black", lw=0.6))
ax.set_xticks(x); ax.set_xticklabels(attrs, fontsize=12, fontweight="bold")
ax.set_ylabel("DI (disparate impact)", fontsize=11)
ax.set_ylim(-0.15, 1.15)
ax.set_title("(A) DI on each protected attribute — Standard vs Fair (3-stage intersectional intervention)",
             fontsize=12.5, fontweight="bold", loc="left", pad=10)
ax.legend(loc="upper left", fontsize=10.5, frameon=True)

# Panel B · Accuracy / AUROC trade-off
ax = fig.add_subplot(gs[1, 0])
ax.bar(["Standard","Fair"], [acc_std, acc_fair], color=["#94a3b8","#16a34a"],
       edgecolor="black", alpha=0.92)
for i, v in enumerate([acc_std, acc_fair]):
    ax.text(i, v+0.005, f"{v:.4f}", ha="center", fontsize=10.5, fontweight="bold")
ax.set_ylim(0.7, 0.9)
ax.set_ylabel("Accuracy")
ax.set_title(f"(B) Accuracy cost = {acc_drop:.2f} pp (<= 5 pp budget)",
             fontsize=11.5, fontweight="bold", loc="left")

# Panel C · 28-cell Δ heatmap
ax = fig.add_subplot(gs[1, 1])
metrics = ["DI","SPD","EOPP","EOD","TI","PP","CAL"]
delta = np.zeros((7, 4))
for i, m in enumerate(metrics):
    for j, a in enumerate(attr_keys):
        s = float(sf[sf["Metric"]==f"{m} ({a})"]["Standard"].iloc[0])
        f = float(sf[sf["Metric"]==f"{m} ({a})"]["Fair (Intersect.)"].iloc[0])
        delta[i, j] = f - s
mx = max(abs(delta.min()), abs(delta.max()))
im = ax.imshow(delta, cmap="RdBu_r", vmin=-mx, vmax=+mx, aspect="auto")
ax.set_xticks(range(4)); ax.set_xticklabels(["Race","Sex","Eth","Age"], fontweight="bold", fontsize=9.5)
ax.set_yticks(range(7)); ax.set_yticklabels(metrics, fontweight="bold", fontsize=9.5)
for i in range(7):
    for j in range(4):
        ax.text(j, i, f"{delta[i,j]:+.2f}", ha="center", va="center",
                fontsize=8, fontweight="bold",
                color="white" if abs(delta[i,j])>mx*0.6 else "black")
ax.set_title("(C) Δ = Fair − Standard (28 cells)", fontsize=11.5, fontweight="bold", loc="left")
ax.grid(False)

# Panel D · headline-claim status badges
ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
ax.set_xlim(0,1); ax.set_ylim(0,1)
y = 0.86
for label, val, ok in [
    ("DI Race ≥ 0.80",     f"{di_fair['Race']:.4f}",  di_fair["Race"]  >= 0.80),
    ("DI Sex ≥ 0.80",      f"{di_fair['Sex']:.4f}",   di_fair["Sex"]   >= 0.80),
    ("DI Eth ≥ 0.80",      f"{di_fair['Eth']:.4f}",   di_fair["Eth"]   >= 0.80),
    ("DI Age ≥ 0.80",      f"{di_fair['Age']:.4f}",   di_fair["Age"]   >= 0.80),
    ("All four jointly",   "Yes" if all_pass else "No", all_pass),
    ("Accuracy cost ≤ 5pp", f"{acc_drop:.2f} pp",     acc_drop <= 5.0),
]:
    color = "#16a34a" if ok else "#dc2626"
    ax.add_patch(FancyBboxPatch((0.05, y-0.07), 0.90, 0.10,
                                boxstyle="round,pad=0.02",
                                facecolor=color, edgecolor="black", lw=0.6, alpha=0.92))
    ax.text(0.10, y - 0.02, label, fontsize=10, fontweight="bold", color="white", va="center")
    ax.text(0.93, y - 0.02, val, fontsize=10, fontweight="bold", color="white", va="center", ha="right")
    ax.text(0.93, y - 0.05, "PASS" if ok else "FAIL", fontsize=8, color="white", va="center", ha="right")
    y -= 0.13
ax.set_title("(D) Headline-claim status", fontsize=11.5, fontweight="bold", loc="left")

fig.suptitle("Headline Claim · The intersectional intervention satisfies all four DI thresholds\n"
             f"simultaneously at a {acc_drop:.2f} pp accuracy cost (≤ 5 pp budget)",
             fontsize=15, fontweight="bold", y=1.02)
plt.savefig(os.path.join(PRT, "T15B_headline_claim.png"), bbox_inches="tight", facecolor="white")
print(f"  saved {os.path.join(PRT, 'T15B_headline_claim.png')}")
plt.show()
'''

# Find Section 17.13 (T15) markdown header so we can insert the headline cells right BEFORE it
target_idx = None
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    if c["cell_type"] == "markdown" and "17.13" in src and "Standard vs Fair" in src:
        target_idx = i
        break
if target_idx is None:
    # fallback: find the T15 source code
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code" and "T15 · Standard vs Fair" in "".join(c.get("source", [])):
            target_idx = i
            break

print(f"Inserting headline-claim cells before notebook cell {target_idx}")

# Drop any prior headline-claim insertion
def is_marker(cell):
    src = "".join(cell.get("source", []))
    return ("HEADLINE CLAIM EVIDENCE" in src) or ("17.12b · Headline-claim summary" in src)

nb["cells"] = [c for c in nb["cells"] if not is_marker(c)]
# Re-find target after the cleanup
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    if c["cell_type"] == "markdown" and "17.13" in src and "Standard vs Fair" in src:
        target_idx = i; break

new_cells = [
    md("### 17.12b · Headline claim · all four DI ≥ 0.80 simultaneously at < 5 pp accuracy cost\n",
       "*This is the abstract's strongest assertion; the cells below load the canonical "
       "intervention table (`results/intervention_standard_vs_fair.csv`), print the four DI "
       "passes explicitly, and visualise the evidence on a single page so a reviewer "
       "scanning the notebook cannot miss it.*\n"),
    code(HEADLINE),
    code(HEADLINE_FIG),
]
nb["cells"][target_idx:target_idx] = new_cells

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"Inserted {len(new_cells)} cells. Total cells now: {len(nb['cells'])}")
print(f"Wrote {NB}")
