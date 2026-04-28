# %%
# Đọc dữ liệu MATH
# NOTE: get_true_gt / get_true_pred / check_correctness / reasoning_similarity
# are shared helpers defined in base_evaluate_gsm8k.py — run that first in the
# same session, or port the relevant functions over.
import glob
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assumes this script is run from its own directory (src/evaluate/)
files = glob.glob(os.path.join("..", "..", "..", "data", "result", "original-sft", "math", "*_predictions.jsonl"))
data = []
for file in files:
    subject = os.path.basename(file).split('_vi_')[0]
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            row['subject'] = subject
            data.append(row)

results = []
for row in data:
    true_gt = get_true_gt(row)
    true_pred = get_true_pred(row)
    
    is_correct = check_correctness(true_pred, true_gt)
    has_boxed = "\\boxed{" in row.get('model_reason', '')
    
    sim_score = reasoning_similarity(row.get('model_reason', ''), row.get('solution', ''))
    
    results.append({
        'index': row.get('index', 0),
        'subject': row.get('subject', 'unknown'),
        'level': row.get('level', 'unknown'),
        'true_gt': true_gt,
        'true_pred': true_pred,
        'is_correct': is_correct,
        'has_boxed': has_boxed,
        'reasoning_length': len(row.get('model_reason', '')),
        'reasoning_steps': str(row.get('model_reason', '')).count('\n') + 1,
        'reasoning_similarity': sim_score
    })

df = pd.DataFrame(results)

acc = df['is_correct'].mean() * 100
format_ok = df['has_boxed'].mean() * 100
avg_len = df['reasoning_length'].mean()
avg_steps = df['reasoning_steps'].mean()
avg_sim = df['reasoning_similarity'].mean() * 100

print("="*40)
print(f"KẾT QUẢ ĐÁNH GIÁ MATH (SFT)")
print("="*40)
print(f"Tổng số câu: {len(df)}")
print(f"Độ chính xác (Accuracy): {acc:.2f}%")
print(f"Tỷ lệ tuân thủ định dạng (Format Compliance): {format_ok:.2f}%")
print(f"Độ dài câu trả lời trung bình: {avg_len:.0f} ký tự")
print(f"Số bước suy luận trung bình: {avg_steps:.1f} bước")
print(f"Độ tương đồng lý luận (Reasoning Similarity): {avg_sim:.2f}%")

print("\nChi tiết độ chính xác theo chủ đề:")
subject_acc = df.groupby('subject')['is_correct'].agg(['mean', 'count'])
subject_acc['mean'] = subject_acc['mean'] * 100
print(subject_acc)

print("\nChi tiết độ chính xác theo độ khó (Level):")
level_acc = df.groupby('level')['is_correct'].agg(['mean', 'count'])
level_acc['mean'] = level_acc['mean'] * 100
# Sắp xếp level theo thứ tự tự nhiên (Level 1 -> Level 5)
level_order = ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]
existing_levels = [l for l in level_order if l in level_acc.index]
other_levels = [l for l in level_acc.index if l not in level_order]
level_acc = level_acc.reindex(existing_levels + other_levels)
print(level_acc)


# %%
save_dir = os.path.join("..", "..", "..", "data", "evaluate", "img", "orriginal-sft", "math")
os.makedirs(save_dir, exist_ok=True)

plt.rcParams['font.family'] = 'sans-serif'

# 1. Bar chart Độ chính xác tổng thể
plt.figure(figsize=(8, 6))
ax = sns.barplot(x=['Đúng', 'Sai'], y=[df['is_correct'].sum(), (~df['is_correct']).sum()], palette='Set2')
plt.title('Độ chính xác dự đoán - MATH', fontsize=14)
plt.ylabel('Số lượng câu', fontsize=12)
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=11, color='black', xytext=(0, 5), textcoords='offset points')
plt.savefig(os.path.join(save_dir, 'math_accuracy_overall_bar.png'), dpi=300, bbox_inches='tight')
plt.show()

# 2. Bar chart Độ chính xác theo chủ đề
plt.figure(figsize=(12, 6))
ax = sns.barplot(x=subject_acc.index, y=subject_acc['mean'], palette='pastel')
plt.title('Độ chính xác theo chủ đề (Bar chart) - MATH', fontsize=14)
plt.xlabel('Chủ đề', fontsize=12)
plt.ylabel('Độ chính xác (%)', fontsize=12)
plt.xticks(rotation=45, ha='right')
for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'math_accuracy_subject_bar.png'), dpi=300)
plt.show()

# 3. Line chart Độ chính xác 7 dạng toán trên cùng 1 biểu đồ
plt.figure(figsize=(10, 6))
ax = sns.lineplot(data=subject_acc.reset_index(), x='subject', y='mean', marker='o', color='purple', linewidth=2.5, markersize=10)
plt.title('Độ chính xác theo 7 chủ đề (Line chart) - MATH', fontsize=14)
plt.xlabel('Chủ đề', fontsize=12)
plt.ylabel('Độ chính xác (%)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'math_accuracy_subject_line.png'), dpi=300)
plt.show()

# 4. Pie chart Tuân thủ định dạng
plt.figure(figsize=(8, 6))
plt.pie([df['has_boxed'].sum(), (~df['has_boxed']).sum()], labels=['Có \\boxed{}', 'Không có \\boxed{}'], 
        autopct='%1.1f%%', colors=['#4CAF50', '#F44336'], startangle=90, textprops={'fontsize': 12})
plt.title('Tỷ lệ tuân thủ định dạng \\boxed{} - MATH', fontsize=14)
plt.savefig(os.path.join(save_dir, 'math_format_compliance_pie.png'), dpi=300, bbox_inches='tight')
plt.show()

# 5. Histogram độ dài câu trả lời
plt.figure(figsize=(10, 6))
sns.histplot(df['reasoning_length'], bins=30, kde=True, color='darkorange')
plt.title('Phân bố độ dài câu trả lời (ký tự) - MATH', fontsize=14)
plt.xlabel('Độ dài (ký tự)', fontsize=12)
plt.ylabel('Số lượng câu', fontsize=12)
plt.savefig(os.path.join(save_dir, 'math_reasoning_length_hist.png'), dpi=300, bbox_inches='tight')
plt.show()

# 6. Histogram Độ tương đồng lý luận
plt.figure(figsize=(10, 6))
sns.histplot(df['reasoning_similarity'], bins=30, kde=True, color='crimson')
plt.title('Phân bố độ tương đồng lý luận so với đáp án gốc - MATH', fontsize=14)
plt.xlabel('Độ tương đồng (Jaccard/SequenceMatcher ratio)', fontsize=12)
plt.ylabel('Số lượng câu', fontsize=12)
plt.savefig(os.path.join(save_dir, 'math_reasoning_similarity_hist.png'), dpi=300, bbox_inches='tight')
plt.show()

# 7. Bar chart Độ chính xác theo Level
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=level_acc.index, y=level_acc['mean'], palette='crest')
plt.title('Độ chính xác theo cấp độ khó (Level) - MATH', fontsize=14)
plt.xlabel('Cấp độ khó (Level)', fontsize=12)
plt.ylabel('Độ chính xác (%)', fontsize=12)
plt.ylim(0, 100)
for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'math_accuracy_level_bar.png'), dpi=300)
plt.show()
