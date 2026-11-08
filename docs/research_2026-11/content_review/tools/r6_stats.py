import json, sys, math, collections
sys.stdout.reconfigure(encoding='utf-8')
A = 'E:/MATH-REASONING-THESIS/vi-math-smt/docs/research_2026-11/content_review/annotations/'
def wilson(k, n, z=1.96):
    if n == 0: return (0, 0, 0)
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (100 * p, 100 * max(0, c - h), 100 * min(1, c + h))
def fmt(k, n): p, lo, hi = wilson(k, n); return f'{k}/{n} = {p:.1f}% [{lo:.1f}–{hi:.1f}]'
for tag in sys.argv[1:]:
    rows = [json.loads(l) for l in open(A + tag + '.jsonl', encoding='utf-8')]
    n = len(rows)
    print(f'##### {tag} n={n}')
    print('  any issue:', fmt(sum(1 for r in rows if r['codes']), n))
    sev = collections.Counter(r['severity'] for r in rows)
    print('  usable as-is (none/minor):', fmt(sum(1 for r in rows if r['severity'] in (None, 'minor')), n))
    print('  fixable (major):', fmt(sev['major'], n), ' discard (critical):', fmt(sev['critical'], n))
    if any('origin' in r for r in rows):
        tr = sum(1 for r in rows if r['codes'] and 'translation' in r.get('origin', ''))
        so = sum(1 for r in rows if r['codes'] and 'source' in r.get('origin', ''))
        trmaj = sum(1 for r in rows if r['codes'] and 'translation' in r.get('origin', '') and r['severity'] in ('major', 'critical'))
        print('  translation-induced any:', fmt(tr, n), '| translation-induced major+critical:', fmt(trmaj, n), '| inherited-from-source:', fmt(so, n))
        mc = sum(1 for r in rows if set(r['codes']) & {'NL-MEANING', 'NL-OMIT'} and 'translation' in r.get('origin', ''))
        print('  translation NL-MEANING/NL-OMIT:', fmt(mc, n))
    cc = collections.Counter(c for r in rows for c in set(r['codes']))
    for c, k in sorted(cc.items(), key=lambda x: -x[1]):
        print(f'    {c:22s} {fmt(k, n)}')
    if 'type' in rows[0]:
        bt = collections.defaultdict(lambda: [0, 0, 0])
        for r in rows:
            t = r['type']; bt[t][0] += 1
            if r['severity'] in ('major', 'critical'): bt[t][1] += 1
            if r['severity'] == 'critical': bt[t][2] += 1
        print('  by type (n, major+crit, crit):', dict(bt))
