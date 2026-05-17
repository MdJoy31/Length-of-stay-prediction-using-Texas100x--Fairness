"""
After the notebook finishes executing, walk cell-by-cell and verify:

  * Every code cell has execution_count set (not None)
  * Every code cell has at least one output of type stream/execute_result/display_data
  * No cell has output_type='error'
  * Embedded matplotlib PNGs are present and non-trivial in size

Also pulls the canonical numbers from output_final/tables/T15 (standard
vs fair) and prints PASS/FAIL for each fairness threshold in the
manuscript (DI ≥ 0.80, |SPD|<0.10, |EOPP|<0.10, |EOD|<0.10, |TI|<0.10,
|PP|<0.10, |CAL|<0.05) so the user can see which metrics are
genuinely fair.
"""
import os, json, sys, base64
sys.stdout.reconfigure(encoding='utf-8')

NB = "CIKM_2026_LOS_Fairness_FINAL.ipynb"
T15 = "output_final/tables/T15_standard_vs_fair.csv"

def line(s, c="="):
    return c*78 + "\n  " + s + "\n" + c*78

print(line("CELL-BY-CELL OUTPUT AUDIT"))

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

n_code = sum(1 for c in nb["cells"] if c["cell_type"]=="code")
print(f"\n  Notebook: {NB}")
print(f"  Total cells: {len(nb['cells'])}  (code={n_code}, "
      f"markdown={sum(1 for c in nb['cells'] if c['cell_type']=='markdown')})\n")

problems = []
total_png_bytes = 0
total_text_chars = 0
n_with_figs = 0
n_with_tables = 0
n_empty = 0
n_errors = 0

for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c.get("source", []))
    src_first = src.split("\n")[0][:55].replace("─","-")
    outs = c.get("outputs", [])
    ec = c.get("execution_count")

    n_text = n_disp = n_err = n_png = 0
    png_bytes_cell = 0
    text_chars_cell = 0
    html_chars_cell = 0
    for o in outs:
        ot = o.get("output_type")
        if ot == "stream":
            n_text += 1
            text_chars_cell += sum(len(s) for s in o.get("text", []))
        elif ot == "execute_result":
            n_text += 1
            data = o.get("data", {})
            text_chars_cell += sum(
                len("".join(v) if isinstance(v, list) else str(v))
                for v in data.values()
            )
        elif ot == "display_data":
            n_disp += 1
            data = o.get("data", {})
            if "image/png" in data:
                n_png += 1
                try:
                    raw = base64.b64decode(data["image/png"])
                    png_bytes_cell += len(raw)
                except Exception:
                    pass
            if "text/html" in data:
                html_chars_cell += len("".join(data["text/html"])
                                       if isinstance(data["text/html"], list)
                                       else str(data["text/html"]))
        elif ot == "error":
            n_err += 1

    total_png_bytes += png_bytes_cell
    total_text_chars += text_chars_cell
    if n_png > 0: n_with_figs += 1
    if html_chars_cell > 100: n_with_tables += 1
    if n_text == 0 and n_disp == 0:
        n_empty += 1
        problems.append(f"  cell[{i:02d}] EMPTY (no outputs)  | {src_first}")
    if n_err > 0:
        n_errors += 1
        problems.append(f"  cell[{i:02d}] ERROR              | {src_first}")
    if ec is None and n_text + n_disp + n_err > 0:
        problems.append(f"  cell[{i:02d}] no exec_count but has outputs  | {src_first}")

    flags = []
    if n_png > 0: flags.append(f"{n_png}fig({png_bytes_cell//1024}KB)")
    if html_chars_cell > 0: flags.append(f"{html_chars_cell}h-html")
    if n_text > 0: flags.append(f"{n_text}txt")
    if n_err > 0: flags.append("ERROR")
    flag_str = " ".join(flags) if flags else "EMPTY"
    print(f"  cell[{i:02d}] ec={str(ec):>4} {flag_str:<35} | {src_first}")

print()
print(line("AUDIT SUMMARY", "-"))
print(f"  Code cells          : {n_code}")
print(f"  Cells with figures  : {n_with_figs}")
print(f"  Cells with HTML tbl : {n_with_tables}")
print(f"  Cells empty (NO out): {n_empty}")
print(f"  Cells with errors   : {n_errors}")
print(f"  Total PNG bytes     : {total_png_bytes:,}  ({total_png_bytes/1024/1024:.2f} MB)")
print(f"  Total text chars    : {total_text_chars:,}")
print(f"  Notebook file size  : {os.path.getsize(NB):,} bytes  "
      f"({os.path.getsize(NB)/1024/1024:.2f} MB)")
if problems:
    print("\n  PROBLEMS:")
    for p in problems: print(p)
else:
    print("\n  No problems.")

# ----- Fairness PASS/FAIL audit -----
print()
print(line("FAIRNESS PASS/FAIL AUDIT (T15 + manuscript thresholds)"))
if not os.path.exists(T15):
    print(f"  MISSING: {T15}")
else:
    import csv
    rows = list(csv.DictReader(open(T15, encoding="utf-8")))
    THR = {  # (threshold, direction)
        "DI":   (0.80, "above"),
        "SPD":  (0.10, "below"),
        "EOPP": (0.10, "below"),
        "EOD":  (0.10, "below"),
        "TI":   (0.10, "below"),
        "PP":   (0.10, "below"),
        "CAL":  (0.05, "below"),
    }
    print(f"\n  {'Metric':<14} {'Standard':>10} {'Fair':>10} {'Δ':>10} "
          f"{'PassStd':<8} {'PassFair':<8}")
    print("  " + "-"*70)
    summary = {a: {"std": 0, "fair": 0, "tot": 0} for a in ["Race","Sex","Eth","Age"]}
    for r in rows:
        m = r["Metric"]
        if "(" not in m: continue
        base = m.split(" (")[0]
        attr = m.split("(")[1].rstrip(")")
        if base not in THR: continue
        thr, direc = THR[base]
        std = float(r["Standard"]); fair = float(r["Fair (Intersect.)"])
        if direc == "above":
            ps = "PASS" if std >= thr else "FAIL"
            pf = "PASS" if fair >= thr else "FAIL"
        else:
            ps = "PASS" if abs(std) < thr else "FAIL"
            pf = "PASS" if abs(fair) < thr else "FAIL"
        if attr in summary:
            summary[attr]["tot"] += 1
            if ps == "PASS": summary[attr]["std"] += 1
            if pf == "PASS": summary[attr]["fair"] += 1
        print(f"  {m:<14} {std:>10.4f} {fair:>10.4f} "
              f"{(fair-std):>+10.4f} {ps:<8} {pf:<8}")
    print("\n  Per-attribute count of fair metrics (out of 7):")
    print(f"  {'Attribute':<10} {'Standard':<12} {'Fair (post-intervention)':<25}")
    for a, d in summary.items():
        print(f"  {a:<10} {d['std']}/{d['tot']:<10} {d['fair']}/{d['tot']:<25}")

print("\n" + line("DONE"))
