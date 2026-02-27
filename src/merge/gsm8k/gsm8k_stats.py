# %% [markdown]
# # GSM8K — Thống kê phân phối Level

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style='whitegrid')
# Assumes this script is run from its own directory (src/merge/)
img_dir = os.path.join("..", "..", "..", "data", "merge_img")
os.makedirs(img_dir, exist_ok=True)

data_path = os.path.join("..", "..", "..", "data", "finetune", "gsm8k", "gsm8k_train_vietnamese_final.jsonl")
df = pd.read_json(data_path, lines=True)

print(f'Total entries: {len(df)}')
print(f'Columns: {list(df.columns)}')
print(f'Subjects: {df["subject"].unique()}')
print()
print('=== Level distribution ===')
level_counts = df['level'].value_counts().sort_index()
print(level_counts)
print()
print('=== Level % ===')
print((level_counts / len(df) * 100).round(1))

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart
colors = ['#4CAF50', '#2196F3', '#FF9800']
level_counts = df['level'].value_counts().sort_index()
bars = axes[0].bar(level_counts.index.astype(str), level_counts.values, color=colors[:len(level_counts)])
axes[0].set_title('GSM8K — Phân phối Level', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Level')
axes[0].set_ylabel('Số lượng')
for bar, val in zip(bars, level_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, str(val), ha='center', fontweight='bold')

# Pie chart
axes[1].pie(level_counts.values, labels=[f'Level {l}' for l in level_counts.index],
            autopct='%1.1f%%', colors=colors[:len(level_counts)], startangle=90)
axes[1].set_title('GSM8K — Tỷ lệ Level', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'gsm8k_level_distribution.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f'Saved: {os.path.join(img_dir, "gsm8k_level_distribution.png")}')
