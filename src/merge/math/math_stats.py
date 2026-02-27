# %% [markdown]
# # MATH — Thống kê phân phối Level & Môn học

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style='whitegrid')
# Assumes this script is run from its own directory (src/merge/)
img_dir = os.path.join("..", "..", "..", "data", "merge_img")
os.makedirs(img_dir, exist_ok=True)

data_path = os.path.join("..", "..", "..", "data", "finetune", "math", "math_train_vietnamese_final.jsonl")
df = pd.read_json(data_path, lines=True)

print(f'Total entries: {len(df)}')
print(f'Columns: {list(df.columns)}')
print(f'Subjects: {sorted(df["subject"].unique())}')
print()
print('=== Level distribution ===')
level_counts = df['level'].value_counts().sort_index()
print(level_counts)
print()
print('=== Subject distribution ===')
subj_counts = df['subject'].value_counts().sort_values(ascending=False)
print(subj_counts)
print()
print('=== Level x Subject ===')
ct = df.groupby(['subject', 'level']).size().unstack(fill_value=0)
ct['total'] = ct.sum(axis=1)
print(ct)

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Level bar chart
colors_level = ['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#F44336']
level_counts = df['level'].value_counts().sort_index()
bars = axes[0].bar(level_counts.index.astype(str), level_counts.values,
                   color=colors_level[:len(level_counts)])
axes[0].set_title('MATH — Phân phối Level', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Level')
axes[0].set_ylabel('Số lượng')
for bar, val in zip(bars, level_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                str(val), ha='center', fontweight='bold', fontsize=9)

# Subject bar chart
subj_counts = df['subject'].value_counts().sort_values(ascending=True)
colors_subj = sns.color_palette('Set2', len(subj_counts))
bars2 = axes[1].barh(subj_counts.index, subj_counts.values, color=colors_subj)
axes[1].set_title('MATH — Phân phối Môn học', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Số lượng')
for bar, val in zip(bars2, subj_counts.values):
    axes[1].text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                str(val), va='center', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'math_level_subject_distribution.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved: {os.path.join(img_dir, "math_level_subject_distribution.png")}')

# %%
# Heatmap: Level x Subject
ct = df.groupby(['subject', 'level']).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(ct, annot=True, fmt='d', cmap='YlOrRd', ax=ax, linewidths=0.5)
ax.set_title('MATH — Heatmap Level × Môn học', fontsize=14, fontweight='bold')
ax.set_ylabel('Môn học')
ax.set_xlabel('Level')

plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'math_level_subject_heatmap.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved: {os.path.join(img_dir, "math_level_subject_heatmap.png")}')
