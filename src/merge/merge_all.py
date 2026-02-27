# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from collections import OrderedDict

sns.set_theme(style='whitegrid')
# Assumes this script is run from its own directory (src/merge/)
img_dir = os.path.join("..", "..", "data", "merge_img")
os.makedirs(img_dir, exist_ok=True)

gsm8k_path = os.path.join("..", "..", "data", "finetune", "gsm8k", "gsm8k_train_vietnamese_final.jsonl")
math_path  = os.path.join("..", "..", "data", "finetune", "math", "math_train_vietnamese_final.jsonl")
output_dir = os.path.join("..", "..", "data", "finetune", "mine")
os.makedirs(output_dir, exist_ok=True)

# Load both datasets
df_gsm8k = pd.read_json(gsm8k_path, lines=True)
df_math  = pd.read_json(math_path, lines=True)

# Add source column to track origin
df_gsm8k['source'] = 'gsm8k'
df_math['source']  = 'math'

# Normalize MATH level: "Level 2" -> 2
df_math['level'] = df_math['level'].apply(lambda x: int(str(x).replace('Level ', '')) if pd.notna(x) else 0)

print(f'GSM8K: {len(df_gsm8k)} entries')
print(f'MATH:  {len(df_math)} entries')
print(f'Total: {len(df_gsm8k) + len(df_math)} entries')

# %%
# Merge and shuffle
# Ensure same column order
cols = ['index', 'problem', 'smt', 'level', 'type', 'solution', 'subject', 'ground_truth', 'source']
df_all = pd.concat([df_gsm8k[cols], df_math[cols]], ignore_index=True)

# Random shuffle — tốt cho training vì tránh bias do batch toàn cùng 1 subject
df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)

# Re-index from 0
df_all['index'] = range(len(df_all))

print(f'Merged dataset: {len(df_all)} entries (shuffled, re-indexed)')
print(f'Columns: {list(df_all.columns)}')

# %%
# === Statistics ===
print('=== Source distribution ===')
print(df_all['source'].value_counts())
print()

print('=== Level distribution ===')
print(df_all['level'].value_counts().sort_index())
print()

print('=== Subject distribution ===')
print(df_all['subject'].value_counts().sort_values(ascending=False))
print()

print('=== SMT coverage ===')
has_smt = (df_all['smt'].notna() & (df_all['smt'] != '')).sum()
print(f'With SMT: {has_smt} ({has_smt/len(df_all)*100:.1f}%)')
print(f'Without:  {len(df_all)-has_smt} ({(len(df_all)-has_smt)/len(df_all)*100:.1f}%)')

# %%
# === Charts ===
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Source pie chart
src_counts = df_all['source'].value_counts()
axes[0,0].pie(src_counts.values, labels=[f'{s.upper()}\n({v})' for s, v in zip(src_counts.index, src_counts.values)],
              autopct='%1.1f%%', colors=['#2196F3', '#FF9800'], startangle=90, textprops={'fontsize': 11})
axes[0,0].set_title('Nguồn dữ liệu', fontsize=14, fontweight='bold')

# 2. Level distribution
level_counts = df_all['level'].value_counts().sort_index()
colors_level = ['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#F44336']
bars = axes[0,1].bar(level_counts.index.astype(str), level_counts.values,
                     color=colors_level[:len(level_counts)])
axes[0,1].set_title('Phân phối Level (tổng hợp)', fontsize=14, fontweight='bold')
axes[0,1].set_xlabel('Level')
axes[0,1].set_ylabel('Số lượng')
for bar, val in zip(bars, level_counts.values):
    axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                  str(val), ha='center', fontweight='bold', fontsize=9)

# 3. Subject distribution
subj_counts = df_all['subject'].value_counts().sort_values(ascending=True)
colors_subj = sns.color_palette('Set2', len(subj_counts))
bars2 = axes[1,0].barh(subj_counts.index, subj_counts.values, color=colors_subj)
axes[1,0].set_title('Phân phối Môn học', fontsize=14, fontweight='bold')
axes[1,0].set_xlabel('Số lượng')
for bar, val in zip(bars2, subj_counts.values):
    axes[1,0].text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
                  str(val), va='center', fontweight='bold', fontsize=9)

# 4. Level by source (stacked)
ct = df_all.groupby(['level', 'source']).size().unstack(fill_value=0)
ct.plot(kind='bar', stacked=True, ax=axes[1,1], color=['#2196F3', '#FF9800'])
axes[1,1].set_title('Level theo Nguồn', fontsize=14, fontweight='bold')
axes[1,1].set_xlabel('Level')
axes[1,1].set_ylabel('Số lượng')
axes[1,1].legend(title='Source')
axes[1,1].tick_params(axis='x', rotation=0)

plt.suptitle(f'Dataset tổng hợp: {len(df_all):,} entries', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'merged_distribution.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved: {os.path.join(img_dir, "merged_distribution.png")}')

# %%
# === Export ===
# Drop 'source' column for the final training file
df_export = df_all.drop(columns=['source'])
final_cols = ['index', 'problem', 'smt', 'level', 'type', 'solution', 'subject', 'ground_truth']
df_export = df_export[final_cols]

out_json  = os.path.join(output_dir, 'train_vietnamese_final.json')
out_jsonl = os.path.join(output_dir, 'train_vietnamese_final.jsonl')

# JSON
print(f'Exporting JSON: {out_json}')
records = [OrderedDict(zip(final_cols, row)) for row in df_export.values]
with open(out_json, 'w', encoding='utf-8') as f:
    json.dump(records, f, ensure_ascii=False, indent=2, default=str)

# JSONL
print(f'Exporting JSONL: {out_jsonl}')
with open(out_jsonl, 'w', encoding='utf-8') as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')

print(f'\n✅ Done! {len(records)} entries exported.')
print(f'   JSON:  {out_json}')
print(f'   JSONL: {out_jsonl}')
