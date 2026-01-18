# %%
import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Thiết lập
sns.set_theme(style="whitegrid")
img_dir = r'./output/img'
if not os.path.exists(img_dir): os.makedirs(img_dir)

# Tải và làm sạch dữ liệu
# Assumes this script is run from its own directory (src/formalize/)
raw_dir = os.path.join("..", "..", "..", "data", "formalize_output", "math_cleaned-copy")
df_all = pd.concat([pd.read_json(f, lines=True).assign(subject=os.path.basename(f).replace('.jsonl','')) for f in glob.glob(os.path.join(raw_dir, "*.jsonl"))])
df_all['status_rank'] = df_all['status'].map({'pass': 0, 'fail': 1})
df = df_all.sort_values(by=['subject', 'index', 'status_rank']).drop_duplicates(subset=['index', 'subject'], keep='first').drop(columns=['status_rank'])

# Tích hợp Level
math_data_root = os.path.join("..", "..", "..", "data", "translation", "MATH")
meta_list = []
for split in ['train']:
    for f in glob.glob(os.path.join(math_data_root, split, "*_vi.jsonl")): 
        try: meta_list.append(pd.read_json(f, lines=True)[['index', 'subject', 'level']])
        except: pass
df_meta = pd.concat(meta_list).drop_duplicates(subset=['index', 'subject'])
df = pd.merge(df, df_meta, on=['index', 'subject'], how='left')
df['level'] = df['level'].str.replace('Level ', 'L').fillna('L?')

print(f"Hoàn thành tải {len(df)} câu.")

# %% [markdown]
# ## 1. Hiệu suất theo Độ khó (Level)

# %%
plt.figure(figsize=(12, 6))
lv_stats = df.groupby('level')['status'].value_counts(normalize=True).unstack().fillna(0) * 100
lv_stats[['pass', 'fail']].rename(columns={'pass':'Đúng','fail':'Sai'}).plot(kind='bar', stacked=True, color=['#4CAF50','#F44336'], ax=plt.gca())
plt.title("Tỷ lệ Đúng/Sai theo Level")
plt.ylabel("Phần trăm (%)")
plt.savefig(os.path.join(img_dir, '01_success_by_level.png'), bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 2. Hiệu suất theo Môn học

# %%
plt.figure(figsize=(12, 6))
sj_stats = df.groupby('subject')['status'].value_counts(normalize=True).unstack().fillna(0) * 100
sj_stats = sj_stats.sort_values(by='pass', ascending=False)
sj_stats[['pass', 'fail']].rename(columns={'pass':'Đúng','fail':'Sai'}).plot(kind='bar', stacked=True, color=['#4CAF50','#F44336'], ax=plt.gca())
plt.title("Tỷ lệ Đúng/Sai theo Môn học")
plt.ylabel("Phần trăm (%)")
plt.xticks(rotation=45)
plt.savefig(os.path.join(img_dir, '02_success_by_subject.png'), bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 3. Biểu đồ Tổng hợp (Môn học - Đúng/Sai - Loại lỗi)
# 
# Biểu đồ này gộp tất cả thông tin vào một hệ tọa độ duy nhất.

# %%
import seaborn as sns

# Tạo bảng pivot gọn gàng
# Trục tung: Môn học
# Trục hoành: Level x Trạng thái (Đúng/Sai)
compact_df = df.groupby(['subject', 'level', 'status']).size().unstack(fill_value=0)

# Tính tỷ lệ phần trăm theo dòng (môn học)
compact_df_pct = compact_df.div(compact_df.sum(axis=1), axis=0) * 100

# Reshape để có Level trên trục hoành
pivot_table = df.pivot_table(index='subject', columns=['level', 'status'], aggfunc='size', fill_value=0)
# Chuẩn hóa tỷ lệ theo từng cụm (Môn học - Level)
pivot_pct = pivot_table.div(pivot_table.groupby(level=0, axis=1).sum(), axis=1) * 100

# Vẽ Heatmap gọn gàng
plt.figure(figsize=(16, 8))
sns.heatmap(pivot_pct, annot=True, fmt=".1f", cmap="YlOrRd", cbar_kws={'label': 'Tỷ lệ % trong Level'})

plt.title("HIỆU SUẤT ĐÚNG/SAI (%) THEO MÔN HỌC & LEVEL", fontsize=18, pad=20)
plt.xlabel("Level - Trạng thái (fail: Sai / pass: Đúng)", fontsize=12)
plt.ylabel("Môn học", fontsize=12)

plt.savefig(os.path.join(img_dir, '03_compact_heatmap.png'), bbox_inches='tight', dpi=300)
plt.show()
