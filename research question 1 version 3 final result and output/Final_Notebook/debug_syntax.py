"""Debug syntax error in build_notebook_v4.py"""
with open('build_notebook_v4.py', 'r', encoding='utf-8-sig') as f:
    lines = f.readlines()

# Check triple-quote balance
count = 0
for i, line in enumerate(lines[:1575], 1):
    c = line.count('"""')
    count += c
    if c % 2 == 1 and i > 1550:
        print(f'L{i}: tq={c}, total={count}, line={line.rstrip()[:100]}')

print(f'Triple quote count at line 1574: {count} (even={count%2==0})')

# Also check for any weird characters
for i in range(1570, min(1580, len(lines))):
    line = lines[i]
    for j, ch in enumerate(line):
        if ord(ch) > 127 and ord(ch) < 256:
            print(f'L{i+1} pos {j}: suspicious char U+{ord(ch):04X} = {repr(ch)}')
