# %%
import json
import os
import glob
from collections import OrderedDict

# ======================== CONFIG ========================
# Assumes this script is run from its own directory (src/merge/)
VI_DATA_DIR   = os.path.join("..", "..", "..", "data", "translation", "MATH", "train")
SMT_DATA_DIR  = os.path.join("..", "..", "..", "data", "formalize_output", "math_cleaned")
OUTPUT_DIR    = os.path.join("..", "..", "..", "data", "finetune", "math")

SUBJECTS = [
    'algebra', 'counting_and_probability', 'geometry',
    'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus'
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}")

# %% [markdown]
# ## Step 1 — Load & Filter SMT Data
# 
# Chỉ giữ các entry có SMT code hợp lệ (pass + cannot_parse_gt). Bỏ hẳn 103 câu lỗi.

# %%
# Load all SMT formalization results, dedup, and filter
smt_data = {}  # key: (subject, index) -> smt_lib code
stats = {'total': 0, 'kept_pass': 0, 'kept_gt': 0, 'dropped': 0}

for subj in SUBJECTS:
    smt_file = os.path.join(SMT_DATA_DIR, f'{subj}.jsonl')
    if not os.path.exists(smt_file):
        print(f"[WARN] SMT file not found: {smt_file}")
        continue

    # Dedup: keep last record per index, prefer 'pass'
    records = {}
    with open(smt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                idx = rec['index']
                if idx in records and records[idx].get('status') == 'pass':
                    continue
                records[idx] = rec
            except Exception:
                pass

    for idx, rec in records.items():
        stats['total'] += 1
        status = rec.get('status', '')
        error  = str(rec.get('error', ''))
        smt_code = rec.get('smt_lib', '')

        if status == 'pass':
            smt_data[(subj, idx)] = smt_code
            stats['kept_pass'] += 1
        elif error.startswith('cannot_parse_ground_truth') and smt_code and smt_code.strip():
            smt_data[(subj, idx)] = smt_code
            stats['kept_gt'] += 1
        else:
            # empty_smt, wrong_answer, z3_error, hardcoded → DROP
            stats['dropped'] += 1

print(f"\n=== SMT Data Statistics ===")
print(f"Total records:               {stats['total']}")
print(f"Kept (pass):                 {stats['kept_pass']}")
print(f"Kept (cannot_parse_gt):      {stats['kept_gt']}")
print(f"Dropped (bad quality):       {stats['dropped']}")
print(f"Total available for merge:   {len(smt_data)}")

# %% [markdown]
# ## Step 2 — Load Vietnamese Data & Merge
# 
# Chỉ giữ entries có SMT code hợp lệ. Entries không match sẽ bị bỏ.

# %%
merged_data = []
matched_count = 0
skipped_count = 0

for subj in SUBJECTS:
    vi_file = os.path.join(VI_DATA_DIR, f'{subj}_vi.jsonl')
    if not os.path.exists(vi_file):
        print(f"[WARN] Vietnamese file not found: {vi_file}")
        continue

    subj_matched = 0
    subj_skipped = 0

    with open(vi_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue

            idx = item['index']
            key = (subj, idx)

            if key not in smt_data:
                subj_skipped += 1
                skipped_count += 1
                continue

            # Build output record:
            # index -> problem -> smt -> level -> type -> solution -> subject -> ground_truth
            record = OrderedDict([
                ('index',        idx),
                ('problem',      item.get('problem', '')),
                ('smt',          smt_data[key]),
                ('level',        item.get('level', '')),
                ('type',         item.get('type', '')),
                ('solution',     item.get('solution', '')),
                ('subject',      subj),
                ('ground_truth', item.get('ground_truth', '')),
            ])
            merged_data.append(record)
            subj_matched += 1
            matched_count += 1

    print(f"  {subj:<30s}  kept={subj_matched:<5d}  dropped={subj_skipped}")

# Sort by subject then index
merged_data.sort(key=lambda x: (x['subject'], x['index']))

print(f"\n=== Merge Results ===")
print(f"Total entries:   {matched_count}")
print(f"Dropped:         {skipped_count}")

# %% [markdown]
# ## Step 3 — Export

# %%
out_json  = os.path.join(OUTPUT_DIR, 'math_train_vietnamese_final.json')
out_jsonl = os.path.join(OUTPUT_DIR, 'math_train_vietnamese_final.jsonl')

# Export JSON (pretty)
print(f"Exporting JSON:  {out_json}")
with open(out_json, 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=2)

# Export JSONL
print(f"Exporting JSONL: {out_jsonl}")
with open(out_jsonl, 'w', encoding='utf-8') as f:
    for item in merged_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"\n✅ Done! {len(merged_data)} entries exported.")
print(f"   JSON:  {out_json}")
print(f"   JSONL: {out_jsonl}")

# %% [markdown]
# ## Step 4 — Verification

# %%
import pandas as pd

df = pd.read_json(out_jsonl, lines=True)

print(f"Total entries:  {len(df)}")
print(f"Columns:        {list(df.columns)}")
print(f"Null SMT:       {df['smt'].isna().sum()}")
print(f"Empty SMT:      {(df['smt'] == '').sum()}")
print()

print("=== Per subject ===")
print(df.groupby('subject').size().to_string())
print()

print("=== Sample entry ===")
sample = df.iloc[0].to_dict()
for k, v in sample.items():
    val = str(v)[:100] + ('...' if len(str(v)) > 100 else '')
    print(f"  {k}: {val}")
