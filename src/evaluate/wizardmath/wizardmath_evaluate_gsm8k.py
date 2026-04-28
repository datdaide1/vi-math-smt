# %%
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import re

sns.set_theme(style="whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.family'] = 'sans-serif'

# Assumes this script is run from its own directory (src/evaluate/)
data_path = os.path.join("..", "..", "..", "data", "result", "wizardMATH", "gsm8k", "results_WizardMath-7B-V1.1_gsm8k_vi.jsonl")
output_dir = os.path.join("..", "..", "..", "data", "evaluate", "evaluate_result", "WizardMATH", "GSM8K")
img_dir = os.path.join("..", "..", "..", "data", "evaluate", "img", "WizardMATH", "GSM8K")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)

print("Môi trường đã sẵn sàng.")

# %% [markdown]
# ### 1. Trích xuất và Kiểm tra Độ chính xác

# %%
def extract_answer(text):
    text = str(text)
    match = re.search(r'####\s*(-?\d+(?:\.\d+)?)', text)
    if match: return match.group(1).replace(',', '')
    numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    return numbers[-1].replace(',', '') if numbers else "NOT_FOUND"

def check_accuracy(row):
    gt = str(row['ground_truth']).strip()
    pred = extract_answer(row['model_reasoning'])
    if gt == "NOT_FOUND" or pred == "NOT_FOUND": return False
    try: return float(gt) == float(pred)
    except: return gt == pred

data = []
with open(data_path, 'r', encoding='utf-8') as f:
    for line in f: data.append(json.loads(line))
df = pd.DataFrame(data)

df['extracted_answer'] = df['model_reasoning'].apply(extract_answer)
df['is_correct'] = df.apply(check_accuracy, axis=1)

print(f"Tổng mẫu: {len(df)}")

# %% [markdown]
# ### 2. Phân tích Sai lệch Đơn vị và Tính chỉ số

# %%
def get_metrics(text):
    text = str(text)
    words = len(text.split())
    steps = len([s for s in re.split(r'[.\n]', text) if len(s.strip()) > 5])
    digits = len(re.findall(r'\d', text))
    density = digits / len(text) if len(text) > 0 else 0
    correction = any(kw in text.lower() for kw in ['actually', 'wait', 'mistake', 'nhầm', 'sửa lại'])
    return words, steps, density, correction

df['ref_words'], df['ref_steps'], _, _ = zip(*df['original_solution'].apply(get_metrics))
df['model_words'], df['model_steps'], df['numeric_density'], df['has_correction'] = zip(*df['model_reasoning'].apply(get_metrics))
df['word_ratio'] = df['model_words'] / df['ref_words']
df['step_ratio'] = df['model_steps'] / df['ref_steps']

def check_unit_error(row):
    if row['is_correct']: return False
    try:
        gt, pred = float(row['ground_truth']), float(row['extracted_answer'])
        return (pred / gt == 1000) or (gt / pred == 1000)
    except: return False
df['is_unit_mismatch'] = df.apply(check_unit_error, axis=1)

print("Tính toán chỉ số hoàn tất.")

# %% [markdown]
# ### 3. Trực quan hoá - Kết quả chung

# %%
# 3.1 Tỷ lệ chính xác
plt.figure(figsize=(7, 7))
counts = df['is_correct'].value_counts().sort_index()
plt.pie(counts, labels=['Sai', 'Đúng'], autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff'], explode=(0.05, 0))
plt.title('Tỷ lệ Độ chính xác GSM8K', fontsize=14, fontweight='bold')
plt.savefig(os.path.join(img_dir, 'gsm8k_accuracy_pie.png'), dpi=300)
plt.show()

# %% [markdown]
# ### 4. Trực quan hoá - Các Phân phối đặc trưng

# %%
# 4.1 Phân phối Số từ
plt.figure(figsize=(10, 6))
sns.histplot(df['model_words'], kde=True, color='skyblue')
plt.title('Phân phối Độ dài Suy luận (Số từ)', fontsize=14)
plt.xlabel('Số từ')
plt.ylabel('Tần suất')
plt.savefig(os.path.join(img_dir, 'gsm8k_dist_words.png'), dpi=300)
plt.show()

# %%
# 4.2 Phân phối Số bước giải
plt.figure(figsize=(10, 6))
sns.histplot(df['model_steps'], kde=True, color='salmon')
plt.title('Phân phối Số bước giải lập luận', fontsize=14)
plt.xlabel('Số bước')
plt.ylabel('Tần suất')
plt.savefig(os.path.join(img_dir, 'gsm8k_dist_steps.png'), dpi=300)
plt.show()

# %%
# 4.3 Phân phối Mật độ con số
plt.figure(figsize=(10, 6))
sns.histplot(df['numeric_density'], kde=True, color='green')
plt.title('Phân phối Mật độ số', fontsize=14)
plt.xlabel('Mật độ (ký tự số / văn bản)')
plt.ylabel('Tần suất')
plt.savefig(os.path.join(img_dir, 'gsm8k_dist_density.png'), dpi=300)
plt.show()

# %% [markdown]
# ### 5. Trực quan hoá - So sánh Model vs Reference

# %%
# 5.1 Tương quan độ dài
plt.figure(figsize=(10, 6))
sns.regplot(data=df, x='ref_words', y='model_words', scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
plt.title('Tương quan Độ dài: Model (EN) vs Reference (VI)', fontsize=14)
plt.xlabel('Số từ (Reference)')
plt.ylabel('Số từ (Model)')
plt.savefig(os.path.join(img_dir, 'gsm8k_correlation_length.png'), dpi=300)
plt.show()

# %%
# 5.2 Hiệu suất số bước (Step Ratio)
plt.figure(figsize=(10, 6))
sns.kdeplot(df['step_ratio'], fill=True, color='purple')
plt.axvline(1.0, color='blue', linestyle='--', label='Reference Level')
plt.title('Hiệu suất số bước giải (Model / Reference)', fontsize=14)
plt.xlabel('Tỷ lệ')
plt.xlim(0, 4)
plt.legend()
plt.savefig(os.path.join(img_dir, 'gsm8k_step_ratio.png'), dpi=300)
plt.show()

# %% [markdown]
# ### 6. Kết luận & Báo cáo

# %%
### 6. Tổng hợp Kết quả và Lưu Báo cáo

# 6.1 Tính toán Độ chính xác Trung bình và Chỉ số Tổng quát
overall_accuracy = df['is_correct'].mean() * 100
unit_error_rate = df['is_unit_mismatch'].mean() * 100
avg_words_model = df['model_words'].mean()
avg_words_ref = df['ref_words'].mean()
word_ratio = avg_words_model / avg_words_ref if avg_words_ref > 0 else 0

summary_stats = {
    'total_samples': len(df),
    'overall_accuracy_percent': round(overall_accuracy, 2),
    'unit_error_rate_percent': round(unit_error_rate, 2),
    'adjusted_accuracy_if_units_fixed': round(overall_accuracy + unit_error_rate, 2),
    'avg_words_model': round(avg_words_model, 2),
    'avg_words_ref': round(avg_words_ref, 2),
    'word_ratio': round(word_ratio, 2)
}

print("--- BAO CAO TONG QUAT (GSM8K) ---")
print(f"Tong so mau: {summary_stats['total_samples']}")
print(f"Do chinh xac rut trich: {summary_stats['overall_accuracy_percent']}% ")
print(f"Ty le sai don vi (x1000): {summary_stats['unit_error_rate_percent']}% ")
print(f"Do chinh xac dieu chinh: {summary_stats['adjusted_accuracy_if_units_fixed']}% ")
print(f"So tu trung binh (Model): {summary_stats['avg_words_model']}")
print(f"So tu trung binh (Reference): {summary_stats['avg_words_ref']}")

with open(os.path.join(output_dir, 'gsm8k_final_report_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary_stats, f, indent=4, ensure_ascii=False)

# 6.2 Luu ket qua chi tiet
df.to_json(os.path.join(output_dir, 'gsm8k_evaluation_results.jsonl'), orient='records', lines=True, force_ascii=False)
print("Da luu tat ca bao cao vao thu muc evaluate_result.")
