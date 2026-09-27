import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('submission.jsonl', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f'Total entries: {len(lines)}')
print()

issues = []
for i, line in enumerate(lines, 1):
    try:
        d = json.loads(line)
        tid = d.get('test_id', f'T{i:02d}')
        body = d.get('body', '')
        cta = d.get('cta', '')
        send_as = d.get('send_as', '')
        n = len(body)

        if n < 80:
            issues.append(f'{tid}: TOO SHORT ({n} chars): {body[:80]!r}')
        if n > 1500:
            # Print the problematic entry
            print(f'=== {tid} OVERSIZED ({n} chars) ===')
            print(f'Body start: {body[:300]!r}')
            print()
            issues.append(f'{tid}: TOO LONG ({n} chars)')
        if not cta:
            issues.append(f'{tid}: MISSING cta')
        if not send_as:
            issues.append(f'{tid}: MISSING send_as')
    except Exception as e:
        issues.append(f'T{i:02d}: JSON ERROR: {e}')

lengths = [len(json.loads(l).get('body', '')) for l in lines]
print(f'Length stats: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)//len(lengths)}')
print()

if issues:
    print(f'Issues found ({len(issues)}):')
    for iss in issues:
        print(f'  {iss}')
else:
    print('All 30 entries valid!')
