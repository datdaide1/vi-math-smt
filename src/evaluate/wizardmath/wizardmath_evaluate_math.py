# %% [markdown]
# ### Đánh giá Mô hình WizardMath trên bộ dữ liệu MATH (Bản đầy đủ)
# 
# Notebook này phân tích chi tiết hiệu năng theo Phân môn (Subject) và Độ khó (Level), đi kèm với logic trích xuất LaTeX chuẩn và phân tích cấu trúc suy luận.

# %%
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import re
import glob

sns.set_theme(style="whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.family'] = 'sans-serif'

# Assumes this script is run from its own directory (src/evaluate/)
data_dir = os.path.join("..", "..", "..", "data", "result", "wizardMATH", "MATH")
output_dir = os.path.join("..", "..", "..", "data", "evaluate", "evaluate_result", "WizardMATH", "MATH")
img_dir = os.path.join("..", "..", "..", "data", "evaluate", "img", "WizardMATH", "MATH")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)

print("Môi trường đã sẵn sàng.")

# %% [markdown]
# ### 1. Tải và Chuẩn hóa Dữ liệu

# %%
file_paths = glob.glob(os.path.join(data_dir, "*.jsonl"))
all_data = []
for path in file_paths:
    with open(path, 'r', encoding='utf-8') as f:
        for line in f: all_data.append(json.loads(line))
df = pd.DataFrame(all_data)

df['subject'] = df['subject'].replace('precalculus_vi_v2.jsonl', 'precalculus')
subject_map = {
    'algebra': 'Algebra', 'counting_and_probability': 'Counting & Probability', 
    'geometry': 'Geometry', 'intermediate_algebra': 'Intermidiate Algebra',
    'number_theory': 'Number Theory', 'prealgebra': 'Prealgebra', 'precalculus': 'Precalculus'
}
df['subject_clean'] = df['subject'].map(subject_map).fillna(df['subject'])

print(f"Tổng mẫu đã tải: {len(df)}")

# %% [markdown]
# ### 2. Trích xuất Đáp án Nâng cao và Chỉ số

# %%
def extract_boxed(text):
    text = str(text)
    start = text.rfind("\\boxed{")
    if start == -1: return "NF"
    c_start = start + len("\\boxed{")
    open_b = 1
    for i in range(c_start, len(text)):
        if text[i] == '{': open_b += 1
        elif text[i] == '}': open_b -= 1
        if open_b == 0: return text[c_start:i].strip()
    return "NF"

def extract_math(text):
    ans = extract_boxed(text)
    if ans != "NF": return ans
    match = re.search(r'# Answer\s*\n*(.*)', text, re.IGNORECASE)
    return match.group(1).strip() if match else "NF"

def normalize(t): return str(t).lower().replace("\\ ", "").replace(" ", "")
def get_metrics(text):
    text = str(text)
    words = len(text.split())
    steps = len([s for s in re.split(r'[.\n]', text) if len(s.strip()) > 5])
    return words, steps

df['ref_ans'] = df['solution'].apply(extract_boxed)
df['model_ans'] = df['model_reasoning'].apply(extract_math)
df['is_correct'] = df.apply(lambda r: normalize(r['ref_ans']) == normalize(r['model_ans']), axis=1)

df['ref_words'], df['ref_steps'] = zip(*df['solution'].apply(get_metrics))
df['model_words'], df['model_steps'] = zip(*df['model_reasoning'].apply(get_metrics))
df['word_ratio'] = df['model_words'] / df['ref_words']

print("Trích xuất và tính toán chỉ số hoàn tất.")

# %% [markdown]
# ### 3. Trực quan hoá - Hiệu năng phân môn

# %%
# 3.1 Xếp hạng Hiệu năng theo Phân môn
plt.figure(figsize=(12, 6))
subj_acc = df.groupby('subject_clean')['is_correct'].mean().sort_values() * 100
sns.barplot(x=subj_acc.index, y=subj_acc.values, palette="viridis", hue=subj_acc.index, legend=False)
plt.title('Độ chính xác theo Phân môn', fontsize=15, fontweight='bold')
plt.ylabel('Độ chính xác (%)')
plt.xticks(rotation=45)
plt.savefig(os.path.join(img_dir, 'math_accuracy_subjects.png'), dpi=300)
plt.show()

# %% [markdown]
# ### 4. Trực quan hoá - Hiệu năng theo Độ khó

# %%
# 4.1 Ma trận Heatmap Độ chính xác (Subject x Level)
level_order = ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5']
pivot_acc = df.pivot_table(index='subject_clean', columns='level', values='is_correct', aggfunc='mean').reindex(columns=level_order) * 100
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_acc, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Độ chính xác kết quả (%)'})
plt.title('Ma trận Độ chính xác theo Phân môn và Độ khó', fontsize=16, fontweight='bold')
plt.ylabel('Phân môn')
plt.savefig(os.path.join(img_dir, 'math_heatmap_accuracy.png'), dpi=300)
plt.show()

# %%
# 4.2 Xu hướng Độ chính xác theo Level
plt.figure(figsize=(10, 6))
level_acc = df.groupby('level')['is_correct'].mean().reindex(level_order) * 100
sns.lineplot(x=level_acc.index, y=level_acc.values, marker='o', color='red', linewidth=3)
plt.title('Xu hướng Độ chính xác kết quả theo Mức độ khó', fontsize=14, fontweight='bold')
plt.ylabel('Độ chính xác (%)')
plt.savefig(os.path.join(img_dir, 'math_level_trend.png'), dpi=300)
plt.show()

# %% [markdown]
# ### 5. Phân tích Độ dài Suy luận

# %%
# 5.1 Heatmap Độ dài Suy luận Trung bình (Words)
level_order = ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5']
pivot_words = df.pivot_table(index='subject_clean', columns='level', values='model_words', aggfunc='mean').reindex(columns=level_order)
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_words, annot=True, fmt=".1f", cmap="Oranges", cbar_kws={'label': 'Độ dài Suy luận trung bình'})
plt.title('Ma trận Độ dài Suy luận trung bình theo Phân môn và Độ khó', fontsize=16, fontweight='bold')
plt.ylabel('Phân môn')
plt.savefig(os.path.join(img_dir, 'math_heatmap_words.png'), dpi=300)
plt.show()

# 5.2 Phân phối Số từ theo Level (KDE)
plt.figure(figsize=(12, 6))
sns.kdeplot(data=df, x='model_words', hue='level', palette='magma', common_norm=False)
plt.title('Phân phối Độ dài Suy luận theo Mức độ khó', fontsize=15)
plt.xlabel('Số từ')
plt.savefig(os.path.join(img_dir, 'math_dist_words_kde.png'), dpi=300)
plt.show()

# %%
### 6. Tổng hợp Kết quả và Lưu Báo cáo

# 6.1 Tính toán Độ chính xác Trung bình và Chỉ số Tổng quát
overall_accuracy = df['is_correct'].mean() * 100
avg_words_model = df['model_words'].mean()
avg_words_ref = df['ref_words'].mean()
word_ratio = avg_words_model / avg_words_ref if avg_words_ref > 0 else 0

summary_stats = {
    'total_samples': len(df),
    'overall_accuracy_percent': round(overall_accuracy, 2),
    'avg_words_model': round(avg_words_model, 2),
    'avg_words_ref': round(avg_words_ref, 2),
    'word_ratio': round(word_ratio, 2)
}

print("--- BAO CAO TONG QUAT (MATH) ---")
print(f"Tong so mau: {summary_stats['total_samples']}")
print(f"Do chinh xac trung binh: {summary_stats['overall_accuracy_percent']}% ")
print(f"So tu trung binh (Model): {summary_stats['avg_words_model']}")
print(f"So tu trung binh (Reference): {summary_stats['avg_words_ref']}")
print(f"Ty le do dai (Model/Ref): {summary_stats['word_ratio']}")

with open(os.path.join(output_dir, 'math_final_report_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary_stats, f, indent=4, ensure_ascii=False)

# 6.2 Luu chi tiet tung phan mon
subject_stats = df.groupby('subject_clean')['is_correct'].mean().to_dict()
with open(os.path.join(output_dir, 'math_subject_accuracy.json'), 'w', encoding='utf-8') as f:
    json.dump(subject_stats, f, indent=4, ensure_ascii=False)

# 6.3 Luu ket qua chi tiet
df.to_json(os.path.join(output_dir, 'math_evaluation_results.jsonl'), orient='records', lines=True, force_ascii=False)
print("Da luu tat ca bao cao vao thu muc evaluate_result.")

# %% [markdown]
# ### 6. Kết luận & Báo cáo

# %%
report = {
    "Overall_Accuracy": f"{df['is_correct'].mean()*100:.2f}%",
    "Subject_Stats": (df.groupby('subject_clean')['is_correct'].mean() * 100).to_dict(),
    "Complexity": {
        "Avg_Word_Ratio": round(df['word_ratio'].mean(), 2),
        "Avg_Model_Words": round(df['model_words'].mean(), 2)
    }
}
with open(os.path.join(output_dir, 'math_final_report_v2.json'), 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=4)
print(json.dumps(report, indent=4))
