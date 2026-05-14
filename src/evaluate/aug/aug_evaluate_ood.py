# %%
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Cấu hình hiển thị đồ họa tiếng Việt
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid")

print("Đã nạp thư viện và cấu hình đồ họa tiếng Việt thành công!")


# %%
def check_format_compliance(text):
    if not isinstance(text, str):
        return False
    if "\\boxed" in text or "\\fbox" in text:
        return True
    indicators = [
        "đáp án", "đáp số", "kết quả là", "vậy ta có", "vậy ta được",
        "the answer", "final answer"
    ]
    text_lower = text.lower()
    for ind in indicators:
        if ind in text_lower:
            return True
    return False

def clean_and_split(text):
    if not text: return []
    # Tách từ character-by-character bulletproof (không bị escape lỗi của Jupyter JSON)
    for char in '$`\\{}()[],.:+-*/':
        text = text.replace(char, ' ')
    return [w.lower() for w in text.split() if w.strip()]

def get_rouge_recall(pred, ref):
    w1 = set(clean_and_split(pred))
    w2 = set(clean_and_split(ref))
    if not w2: return 0.0
    return len(w1 & w2) / len(w2)

def extract_answer(text):
    if not isinstance(text, str): return None
    m = re.search(r'\\boxed\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', text)
    if m: return m.group(1).strip()
    m = re.search(r'####\\s*(.+?)$', text, re.MULTILINE)
    if m: return m.group(1).strip()
    nums = re.findall(r'-?\\d+(?:\\.\\d+)?(?:/\\d+)?', text)
    if nums: return nums[-1]
    return None

def clean_latex(text: str) -> str:
    if not text: return ""
    for token in ["\\text{", "\\dfrac{", "\\tfrac{", "\\frac{", "{", "}", "\\"]:
        text = text.replace(token, "")
    return text

def extract_numbers(s: str):
    temp = ""
    for char in s:
        if char.isdigit() or char in '.-/':
            temp += char
        else:
            temp += ' '
    return [w for w in temp.split() if w.strip()]

def verify_correctness(model_ans: str, ground_truth: str) -> bool:
    if not model_ans: return False
    
    m_clean = clean_latex(str(model_ans)).strip().lower()
    g_clean = clean_latex(str(ground_truth)).strip().lower()
    
    # Loại bỏ dấu cách và ký tự phân cách cơ bản
    for char in ' $,.()[]\\_':
        m_clean = m_clean.replace(char, '')
        g_clean = g_clean.replace(char, '')
        
    if m_clean == g_clean:
        return True
        
    m_nums = extract_numbers(m_clean)
    g_nums = extract_numbers(g_clean)
    
    if m_nums and g_nums:
        try:
            if '/' in m_nums[0] and '/' in g_nums[0]:
                m_parts = m_nums[0].split('/')
                g_parts = g_nums[0].split('/')
                if float(m_parts[0])/float(m_parts[1]) == float(g_parts[0])/float(g_parts[1]):
                    return True
            else:
                if float(m_nums[0]) == float(g_nums[0]):
                    return True
        except:
            pass
            
    return False

print("Đã thiết lập bộ chuẩn hóa đáp án LaTeX và kiểm tra tính chính xác OOD thành công!")


# %%
# 1. Khớp nối dữ liệu SVAMP
print("Đang xử lý kết quả kiểm thử SVAMP...")
svamp_results = []
# Assumes this script is run from its own directory (src/evaluate/)
svamp_path = os.path.join("..", "..", "..", "data", "result", "augmented-sft-dpo", "ood", "svamp_results.jsonl")
with open(svamp_path, 'r', encoding='utf-8') as f:
    for line in f:
        row = json.loads(line)
        pred_ans = extract_answer(row.get('model_reason', ''))
        is_corr_raw = verify_correctness(pred_ans, row.get('ground_truth', ''))
        sim_score = get_rouge_recall(row.get('model_reason', ''), row.get('problem', ''))
        
        # Ngưỡng tương đồng với bài toán CRA >= 0.40
        is_correct = is_corr_raw and (sim_score >= 0.40)
        is_format_compliant = check_format_compliance(row.get('model_reason', ''))
        
        svamp_results.append({
            'id': row.get('id'),
            'problem': row.get('problem'),
            'ground_truth': row.get('ground_truth'),
            'model_answer': pred_ans,
            'is_correct': is_correct,
            'is_format_compliant': is_format_compliant,
            'reasoning_similarity': sim_score,
            'reasoning_length': len(str(row.get('model_reason', '')))
        })
svamp_df = pd.DataFrame(svamp_results)

# 2. Khớp nối dữ liệu ASDiv
print("Đang xử lý kết quả kiểm thử ASDiv...")
asdiv_results = []
asdiv_path = os.path.join("..", "..", "..", "data", "result", "augmented-sft-dpo", "ood", "asdiv_results.jsonl")
with open(asdiv_path, 'r', encoding='utf-8') as f:
    for line in f:
        row = json.loads(line)
        pred_ans = extract_answer(row.get('model_reason', ''))
        is_corr_raw = verify_correctness(pred_ans, row.get('ground_truth', ''))
        sim_score = get_rouge_recall(row.get('model_reason', ''), row.get('problem', ''))
        
        # Ngưỡng tương đồng với bài toán CRA >= 0.40
        is_correct = is_corr_raw and (sim_score >= 0.40)
        is_format_compliant = check_format_compliance(row.get('model_reason', ''))
        
        asdiv_results.append({
            'id': row.get('id'),
            'problem': row.get('problem'),
            'ground_truth': row.get('ground_truth'),
            'model_answer': pred_ans,
            'is_correct': is_correct,
            'is_format_compliant': is_format_compliant,
            'reasoning_similarity': sim_score,
            'reasoning_length': len(str(row.get('model_reason', '')))
        })
asdiv_df = pd.DataFrame(asdiv_results)

print(f"\nThành công! Đã nạp {len(svamp_df)} mẫu SVAMP và {len(asdiv_df)} mẫu ASDiv.")


# %%
svamp_acc = svamp_df['is_correct'].mean() * 100
svamp_fcr = svamp_df['is_format_compliant'].mean() * 100
svamp_sim = svamp_df['reasoning_similarity'].mean() * 100

asdiv_acc = asdiv_df['is_correct'].mean() * 100
asdiv_fcr = asdiv_df['is_format_compliant'].mean() * 100
asdiv_sim = asdiv_df['reasoning_similarity'].mean() * 100

print("="*60)
print("     KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH OOD (SVAMP & ASDiv) - CRA METRIC")
print("="*60)
print("1. BỘ TẬP DỮ LIỆU SVAMP:")
print(f"  - Số lượng mẫu thử: {len(svamp_df)}")
print(f"  - Độ chính xác (Accuracy): {svamp_acc:.2f}%")
print(f"  - Tỷ lệ tuân thủ định dạng (FCR): {svamp_fcr:.2f}%")
print(f"  - Độ tương đồng lý luận (ROUGE-1 Recall với đề): {svamp_sim:.2f}%")
print("-" * 60)
print("2. BỘ TẬP DỮ LIỆU ASDiv:")
print(f"  - Số lượng mẫu thử: {len(asdiv_df)}")
print(f"  - Độ chính xác (Accuracy): {asdiv_acc:.2f}%")
print(f"  - Tỷ lệ tuân thủ định dạng (FCR): {asdiv_fcr:.2f}%")
print(f"  - Độ tương đồng lý luận (ROUGE-1 Recall với đề): {asdiv_sim:.2f}%")
print("="*60)


# %%
fig_dir = os.path.join("..", "..", "..", "data", "result", "augmented-sft-dpo", "ood", "plots")
os.makedirs(fig_dir, exist_ok=True)

# 1. So sánh các chỉ số cốt lõi giữa SVAMP và ASDiv
metrics_data = {
    'Bộ dữ liệu': ['SVAMP', 'SVAMP', 'SVAMP', 'ASDiv', 'ASDiv', 'ASDiv'],
    'Độ đo': ['Độ chính xác', 'Tuân thủ định dạng (FCR)', 'Tương đồng lý luận', 
              'Độ chính xác', 'Tuân thủ định dạng (FCR)', 'Tương đồng lý luận'],
    'Giá trị (%)': [svamp_acc, svamp_fcr, svamp_sim, asdiv_acc, asdiv_fcr, asdiv_sim]
}
df_metrics = pd.DataFrame(metrics_data)

plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Độ đo', y='Giá trị (%)', hue='Bộ dữ liệu', data=df_metrics, palette='crest')
plt.title('So sánh các chỉ số hiệu năng OOD giữa SVAMP và ASDiv', fontsize=14, fontweight='bold', pad=15)
plt.ylabel('Tỷ lệ phần trăm (%)', fontsize=12)
plt.xlabel('Độ đo đánh giá', fontsize=12)
plt.ylim(0, 105)
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{height:.1f}%',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom',
                    xytext=(0, 5),
                    textcoords='offset points', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'so_sanh_chi_so_ood.png'), dpi=300)
plt.show()

# 2. Biểu đồ phân phối độ tương đồng lý luận (Histogram)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(svamp_df['reasoning_similarity'] * 100, bins=20, kde=True, color='teal')
plt.title('Phân phối Độ tương đồng lý luận - SVAMP', fontsize=12, fontweight='bold')
plt.xlabel('Độ tương đồng', fontsize=10)
plt.ylabel('Tần suất (Số lượng mẫu)', fontsize=10)

plt.subplot(1, 2, 2)
sns.histplot(asdiv_df['reasoning_similarity'] * 100, bins=20, kde=True, color='olive')
plt.title('Phân phối Độ tương đồng lý luận - ASDiv', fontsize=12, fontweight='bold')
plt.xlabel('Độ tương đồng', fontsize=10)
plt.ylabel('Tần suất (Số lượng mẫu)', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'phan_phoi_tuong_dong_ood.png'), dpi=300)
plt.show()

print("Đã tạo và lưu thành công các biểu đồ phân tích ngoại miền OOD vào thư mục 'plots'!")
