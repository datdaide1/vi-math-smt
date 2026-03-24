import json
import os
from collections import Counter

data = [json.loads(l) for l in open(os.path.join("..", "..", "..", "data", "finetune", "augmented", "phase3_informalized_clean.jsonl"), 'r', encoding='utf-8')]

ok = [d for d in data if d['status']=='ok']
failed = [d for d in data if d['status']=='failed']

print(f'=== FILE CLEAN ===')
print(f'Total: {len(data)}')
print(f'OK: {len(ok)} ({len(ok)/len(data)*100:.1f}%)')
print(f'Failed: {len(failed)} ({len(failed)/len(data)*100:.1f}%)')

# Rescued entries breakdown
rescued = [d for d in ok if 'rescued:' in d.get('feedback','')]
print(f'\nRescued entries: {len(rescued)}')
for d in rescued[:5]:
    gt = d["ground_truth"]
    ans = str(d["answer"])[:40]
    fb = d["feedback"]
    print(f'  gt={gt!r:20s} ans={ans!r:42s} reason={fb}')

# Subject distribution of OK
print(f'\n=== OK entries by subject ===')
for k,v in Counter(d['subject'] for d in ok).most_common():
    print(f'  {k}: {v}')

# Level distribution of OK
print(f'\n=== OK entries by level ===')
for k,v in sorted(Counter(d['level'] for d in ok).items()):
    print(f'  Level {k}: {v}')

# Mutation coverage
print(f'\n=== OK entries by mutation ===')
for k,v in sorted(Counter(d.get('mutation_strategy') for d in ok).items()):
    print(f'  mut_{k}: {v}')

# Average problem/solution length
avg_prob = sum(len(d.get('problem','')) for d in ok) / len(ok)
avg_sol = sum(len(d.get('solution','')) for d in ok) / len(ok)
print(f'\nAvg problem length: {avg_prob:.0f} chars')
print(f'Avg solution length: {avg_sol:.0f} chars')

# Empty checks
empty_prob = sum(1 for d in ok if not d.get('problem','').strip())
empty_sol = sum(1 for d in ok if not d.get('solution','').strip())
empty_ans = sum(1 for d in ok if not str(d.get('answer','')).strip())
print(f'\nEmpty problem in OK: {empty_prob}')
print(f'Empty solution in OK: {empty_sol}')
print(f'Empty answer in OK: {empty_ans}')

# Sample 3 OK entries to verify quality
print(f'\n=== SAMPLE OK ENTRIES ===')
import random
random.seed(42)
samples = random.sample(ok, 3)
for i, d in enumerate(samples):
    print(f'\n--- Sample {i+1} ---')
    print(f'Subject: {d["subject"]}, Level: {d["level"]}, Mutation: {d.get("mutation_strategy")}')
    print(f'Problem: {d["problem"][:200]}...')
    print(f'Solution: {d["solution"][:200]}...')
    print(f'Answer: {d["answer"]}')
    print(f'Ground Truth: {d["ground_truth"]}')

# Source coverage
sources = set()
phase2 = [json.loads(l) for l in open(os.path.join("..", "..", "..", "data", "finetune", "augmented", "phase2_mutated_smt.jsonl"), 'r', encoding='utf-8')]
ok_indices = set(d['index'] for d in ok)
for idx in ok_indices:
    if idx < len(phase2):
        sources.add(phase2[idx].get('source_index'))
print(f'\n=== SOURCE COVERAGE ===')
print(f'Unique original problems covered by OK entries: {len(sources)} / 6817')
print(f'Coverage: {len(sources)/6817*100:.1f}%')
