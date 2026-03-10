"""Fix encoding issues in build_notebook_v4.py — replace mojibake with ASCII."""
with open('build_notebook_v4.py', 'rb') as f:
    raw = f.read()

# Strip BOM if present
if raw.startswith(b'\xef\xbb\xbf'):
    raw = raw[3:]
    print("Stripped BOM")

# Try to read as UTF-8 first
content = raw.decode('utf-8')

# Replace common mojibake sequences (UTF-8 bytes interpreted as Latin-1)
replacements = {
    '\u00e2\u0080\u0094': '--',   # em dash
    '\u00e2\u0080\u0093': '-',    # en dash
    '\u00e2\u0080\u0099': "'",    # right single quote
    '\u00e2\u0080\u009c': '"',    # left double quote
    '\u00e2\u0080\u009d': '"',    # right double quote
    '\u00e2\u0080\u0098': "'",    # left single quote
    '\u00e2\u0089\u00a5': '>=',   # ≥
    '\u00c3\u0097': 'x',          # ×
    '\u00e2\u009c\u0093': 'v',    # ✓
    '\u00e2\u009c\u0097': 'x',    # ✗
    '\u00e2\u009c\u0085': '[OK]', # ✅
}

# Also replace actual Unicode chars that might cause issues
unicode_replacements = {
    '\u2014': '--',   # em dash
    '\u2013': '-',    # en dash
    '\u2019': "'",    # right single quote
    '\u201c': '"',    # left double quote
    '\u201d': '"',    # right double quote
    '\u201e': '"',    # double low-9 quote
    '\u2018': "'",    # left single quote
    '\u00d7': 'x',    # ×
    '\u2265': '>=',   # ≥
    '\u2264': '<=',   # ≤
    '\u0394': 'delta_',# Δ
    '\u03ba': 'kappa',       # κ
    '\u03bb': 'lambda',      # λ
}

count = 0
for old, new in {**replacements, **unicode_replacements}.items():
    c = content.count(old)
    if c > 0:
        content = content.replace(old, new)
        count += c
        print(f"  Replaced {repr(old)} -> {repr(new)} ({c} times)")

# Also handle multi-byte garbled sequences
# â€" (3 Latin-1 chars representing em dash in garbled UTF-8)
garbled_patterns = [
    ('â€"', '--'),
    ('â€"', '-'),
    ('â€™', "'"),
    ('â€œ', '"'),
    ('â€\x9d', '"'),
    ('â€˜', "'"),
    ('â‰¥', '>='),
    ('â‰¤', '<='),
    ('Ã—', 'x'),
    ('âœ"', 'v'),
    ('âœ—', 'x'),
    ('âœ…', '[OK]'),
    ('Î"', 'delta_'),
    ('Îº', 'kappa'),
    ('Î»', 'lambda'),
    ('â€', "'"),  # catch remaining
]
for old, new in garbled_patterns:
    c = content.count(old)
    if c > 0:
        content = content.replace(old, new)
        count += c
        print(f"  Replaced garbled {repr(old)} -> {repr(new)} ({c} times)")

# Also remove any remaining non-ASCII chars in the source that aren't in strings
# (We keep them in strings since md() content can have unicode)

print(f"\nTotal replacements: {count}")

# Verify syntax
try:
    compile(content, 'build_notebook_v4.py', 'exec')
    print("Syntax OK!")
except SyntaxError as e:
    print(f"SYNTAX ERROR at line {e.lineno}: {e.msg}")
    # Show context
    lines = content.split('\n')
    for i in range(max(0,e.lineno-3), min(len(lines),e.lineno+3)):
        marker = '>>>' if i == e.lineno-1 else '   '
        print(f"  {marker} L{i+1}: {lines[i][:100]}")

with open('build_notebook_v4.py', 'w', encoding='utf-8') as f:
    f.write(content)
print(f"Saved. Lines: {content.count(chr(10))+1}")
