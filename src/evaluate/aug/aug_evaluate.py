# %%
import re
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import difflib
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from sympy import simplify

# Cấu hình hiển thị và đồ họa tiếng Việt
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("Đã khai báo toàn bộ thư viện và cấu hình đồ họa tiếng Việt thành công!")


# %%
def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx is None:
        return None
    else:
        return string[idx:right_brace_idx + 1]

def fallback_extract(text):
    if not isinstance(text, str): return None
    patterns = [
        r'Đáp án cuối cùng là:?\\s*(.*)',
        r'Vậy đáp án là:?\\s*(.*)',
        r'The answer is:?\\s*(.*)'
    ]
    for p in patterns:
        match = re.search(p, text)
        if match:
            ans = match.group(1).strip()
            return ans.rstrip('.')
    return None

def extract_answer(text):
    if not isinstance(text, str): return None
    boxed = last_boxed_only_string(text)
    if boxed is not None:
        return remove_boxed(boxed)
    return fallback_extract(text)

def remove_right_units(string):
    if "\\text{" in string:
        return string.split("\\text{")[0]
    return string

def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_b = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_b
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_b = substr[2:]
                        new_str += "{" + a + "}" + b + post_b
                    else:
                        new_str += "{" + a + "}" + b
    return new_str

def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        return "\\frac{" + str(a) + "}{\\" + str(b) + "}"
    except:
        return string

def normalize_answer(string):
    if not string:
        return ""
    string = str(string)
    string = string.replace("\\$", "")
    string = string.replace("$", "")
    string = string.replace("\\%", "")
    string = string.replace("%", "")
    string = string.replace("\\degree", "^{\\circ}")
    string = string.replace("^\\\circ", "^{\\circ}")
    string = remove_right_units(string)
    string = string.replace("\\half", "\\frac{1}{2}")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("\\{", "{")
    string = string.replace("\\}", "}")
    for c in "()[]":
        string = string.replace(c, "")
    string = string.replace(" ", "")
    string = string.replace(",", "")
    string = string.replace("\\", "")
    string = fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = fix_a_slash_b(string)
    return string.strip()

def sympy_equiv(s1, s2):
    if not s1 or not s2: return False
    try:
        transformations = (standard_transformations + (implicit_multiplication_application,))
        
        def prep(s):
            s = s.replace('\\pi', 'pi').replace('\\infty', 'oo')
            s = re.sub(r'\\root\\s*(\\d+)\\s*\\of\\s*\\{([^\\}]+)\\}', r'(((\\2)**(1/\\1)))', s)
            s = re.sub(r'\\sqrt\\[(\\d+)\\]\\{([^\\}]+)\\}', r'(((\\2)**(1/\\1)))', s)
            
            while True:
                idx = s.find('\\frac')
                if idx < 0:
                    idx = s.find('\\dfrac')
                    if idx < 0:
                        idx = s.find('\\tfrac')
                        if idx < 0: break
                
                if s[idx:idx+6] == '\\dfrac' or s[idx:idx+6] == '\\tfrac': p_len = 6
                else: p_len = 5
                
                i = idx + p_len
                while i < len(s) and s[i] == ' ': i += 1
                if i >= len(s) or s[i] != '{': break
                start1 = i
                open_b = 0
                end1 = None
                while i < len(s):
                    if s[i] == '{': open_b += 1
                    elif s[i] == '}':
                        open_b -= 1
                        if open_b == 0: end1 = i; break
                    i += 1
                if end1 is None: break
                arg1 = s[start1+1:end1]
                
                i = end1 + 1
                while i < len(s) and s[i] == ' ': i += 1
                if i >= len(s) or s[i] != '{': break
                start2 = i
                open_b = 0
                end2 = None
                while i < len(s):
                    if s[i] == '{': open_b += 1
                    elif s[i] == '}':
                        open_b -= 1
                        if open_b == 0: end2 = i; break
                    i += 1
                if end2 is None: break
                arg2 = s[start2+1:end2]
                
                s = s[:idx] + f"(({arg1})/({arg2}))" + s[end2+1:]
                
            while True:
                idx = s.find('\\sqrt')
                if idx < 0: break
                i = idx + 5
                while i < len(s) and s[i] == ' ': i += 1
                if i >= len(s) or s[i] != '{': break
                start1 = i
                open_b = 0
                end1 = None
                while i < len(s):
                    if s[i] == '{': open_b += 1
                    elif s[i] == '}':
                        open_b -= 1
                        if open_b == 0: end1 = i; break
                    i += 1
                if end1 is None: break
                arg1 = s[start1+1:end1]
                s = s[:idx] + f"sqrt({arg1})" + s[end1+1:]
                
            s = s.replace('^', '**')
            for _ in range(3):
                s = re.sub(r'\\{([a-zA-Z0-9_\\.\\s\\+\\-\\*/\\(\\)]+)\\}', r'(\\1)', s)
            return s
        
        ps1 = prep(s1)
        ps2 = prep(s2)
        e1 = parse_expr(ps1, transformations=transformations)
        e2 = parse_expr(ps2, transformations=transformations)
        return simplify(e1 - e2) == 0
    except: return False

def check_correctness(pred, gt):
    if pred is None or gt is None:
        return False
    norm_pred = normalize_answer(pred)
    norm_gt = normalize_answer(gt)
    if norm_pred == norm_gt:
        return True
    try:
        if abs(float(norm_pred) - float(norm_gt)) < 1e-5:
            return True
    except:
        pass
    return sympy_equiv(pred, gt)

def get_true_gt(row):
    sol = row.get('solution', '')
    ext_gt = extract_answer(sol)
    if ext_gt is not None:
        return ext_gt
    if '####' in sol:
        return sol.split('####')[-1].strip()
    return str(row.get('ground_truth', ''))

def get_true_pred(row):
    reason = row.get('model_reason', '')
    ext_pred = extract_answer(reason)
    if ext_pred is not None:
        return ext_pred
    return str(row.get('model_answer', ''))

print("Đã cài đặt các hàm chuẩn hóa LaTeX và kiểm tra bằng SymPy thành công!")


# %%
def check_format_compliance(text):
    if not isinstance(text, str):
        return False
    if "\\boxed" in text or "\\fbox" in text:
        return True
    # Các từ khóa chỉ thị kết luận toán học tự nhiên tiếng Việt & tiếng Anh
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
    for char in '$`\\\\{}()[],.:+-*/':
        text = text.replace(char, ' ')
    return [w.lower() for w in text.split() if w.strip()]

def reasoning_similarity(pred_reason, gt_solution):
    if not pred_reason or not gt_solution:
        return 0.0
    w1 = set(clean_and_split(pred_reason))
    w2 = set(clean_and_split(gt_solution))
    if not w2: return 0.0
    return len(w1 & w2) / len(w2)

print("Đã cài đặt các hàm tính FCR (Inclusive) và Reasoning Similarity (RSS) thành công!")


# %%
# 1. Nạp và xây dựng bảng tra cứu lời giải mẫu (Ground-truth CoT) chi tiết
print("Đang nạp các tập dữ liệu đối chứng (Ground-truth)... ")
# Assumes this script is run from its own directory (src/evaluate/)
gsm_lookup = {}
with open(os.path.join("..", "..", "..", "data", "test", "gsm8k", "gsm8k_test_vietnamese.jsonl"), 'r', encoding='utf-8') as f:
    for line in f:
        row = json.loads(line)
        gsm_lookup[" ".join(row['question'].split())] = row['answer']

math_lookup = {}
for fpath in glob.glob(os.path.join("..", "..", "..", "data", "test", "math", "*.jsonl")):
    with open(fpath, 'r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            math_lookup[" ".join(row['problem'].split())] = row['solution']

print(f"Nạp thành công {len(gsm_lookup)} lời giải mẫu GSM8K và {len(math_lookup)} lời giải mẫu MATH!")

# 2. Khớp nối và xử lý kết quả dự đoán GSM8K
print("Đang xử lý kết quả dự đoán GSM8K...")
gsm_results = []
with open(os.path.join("..", "..", "..", "data", "result", "augmented-sft-dpo", "gsm8k", "gsm8k_results.jsonl"), 'r', encoding='utf-8') as f:
    for line in f:
        row = json.loads(line)
        prob_norm = " ".join(row['problem'].split())
        ref_cot = gsm_lookup.get(prob_norm, "")
        
        row['solution'] = ref_cot
        
        true_gt = get_true_gt(row)
        true_pred = get_true_pred(row)
        
        is_correct_raw = check_correctness(true_pred, true_gt)
        sim_score = reasoning_similarity(row.get('model_reason', ''), ref_cot)
        
        # Coherent Reasoning Accuracy (acc) - GSM8K: Correct AND RSS >= 0.46
        is_correct = is_correct_raw and (sim_score >= 0.46)
        is_format_compliant = check_format_compliance(row.get('model_reason', ''))
        
        gsm_results.append({
            'id': row.get('id'),
            'problem': row.get('problem'),
            'true_gt': true_gt,
            'true_pred': true_pred,
            'is_correct': is_correct,
            'is_format_compliant': is_format_compliant,
            'reasoning_similarity': sim_score,
            'reasoning_length': len(str(row.get('model_reason', '')))
        })
gsm_df = pd.DataFrame(gsm_results)

# 3. Khớp nối và xử lý kết quả dự đoán MATH
print("Đang xử lý kết quả dự đoán MATH...")
math_results = []
with open(os.path.join("..", "..", "..", "data", "result", "augmented-sft-dpo", "math", "math_results.jsonl"), 'r', encoding='utf-8') as f:
    for line in f:
        row = json.loads(line)
        prob_norm = " ".join(row['problem'].split())
        ref_cot = math_lookup.get(prob_norm, "")
        
        row['solution'] = ref_cot
        
        true_gt = get_true_gt(row)
        true_pred = get_true_pred(row)
        
        is_correct_raw = check_correctness(true_pred, true_gt)
        sim_score = reasoning_similarity(row.get('model_reason', ''), ref_cot)
        
        # Coherent Reasoning Accuracy (acc) - MATH: Correct AND RSS >= 0.52
        is_correct = is_correct_raw and (sim_score >= 0.52)
        is_format_compliant = check_format_compliance(row.get('model_reason', ''))
        
        math_results.append({
            'id': row.get('id'),
            'problem': row.get('problem'),
            'subject': row.get('subject'),
            'level': row.get('level'),
            'true_gt': true_gt,
            'true_pred': true_pred,
            'is_correct': is_correct,
            'is_format_compliant': is_format_compliant,
            'reasoning_similarity': sim_score,
            'reasoning_length': len(str(row.get('model_reason', '')))
        })
math_df = pd.DataFrame(math_results)

print("\n--- HOÀN THÀNH SO KHỚP VÀ XỬ LÝ DỮ LIỆU CẢ 2 TẬP --- ")


# %%
gsm_acc = gsm_df['is_correct'].mean() * 100
gsm_fcr = gsm_df['is_format_compliant'].mean() * 100
gsm_sim = gsm_df['reasoning_similarity'].mean() * 100

math_acc = math_df['is_correct'].mean() * 100
math_fcr = math_df['is_format_compliant'].mean() * 100
math_sim = math_df['reasoning_similarity'].mean() * 100

print("1. BỘ TẬP DỮ LIỆU GSM8K:")
print(f"  - Số lượng mẫu thử: {len(gsm_df)}")
print(f"  - Độ chính xác acc (Accuracy): {gsm_acc:.2f}%")
print(f"  - Tỷ lệ tuân thủ định dạng (FCR): {gsm_fcr:.2f}%")
print(f"  - Độ tương đồng lý luận (RSS): {gsm_sim:.2f}%")
print("-" * 60)
print("2. BỘ TẬP DỮ LIỆU MATH:")
print(f"  - Số lượng mẫu thử: {len(math_df)}")
print(f"  - Độ chính xác acc (Accuracy): {math_acc:.2f}%")
print(f"  - Tỷ lệ tuân thủ định dạng (FCR): {math_fcr:.2f}%")
print(f"  - Độ tương đồng lý luận (RSS): {math_sim:.2f}%")
print("="*60)

# Phân tích theo Subject (Môn học) của MATH
print("\n" + "="*60)
print("  PHÂN TÍCH ĐỘ CHÍNH XÁC acc & TƯƠNG ĐỒNG THEO CHỦ ĐỀ (MATH)")
print("="*60)
subject_analysis = math_df.groupby('subject').agg(
    So_Luong=('is_correct', 'count'),
    Do_Chinh_Xac=('is_correct', lambda x: x.mean() * 100),
    Tuan_Thu_Dinh_Dang=('is_format_compliant', lambda x: x.mean() * 100),
    Tuong_Dong_Ly_Luan=('reasoning_similarity', lambda x: x.mean() * 100)
).round(2)
subject_analysis.columns = ['Số lượng', 'Độ chính xác acc (%)', 'Tuân thủ định dạng (%)', 'Tương đồng lý luận (%)']
print(subject_analysis.to_string())

# Phân tích theo Level (Độ khó) của MATH
print("\n" + "="*60)
print("  PHÂN TÍCH ĐỘ CHÍNH XÁC acc & TƯƠNG ĐỒNG THEO ĐỘ KHÓ (MATH)")
print("="*60)
level_analysis = math_df.groupby('level').agg(
    So_Luong=('is_correct', 'count'),
    Do_Chinh_Xac=('is_correct', lambda x: x.mean() * 100),
    Tuan_Thu_Dinh_Dang=('is_format_compliant', lambda x: x.mean() * 100),
    Tuong_Dong_Ly_Luan=('reasoning_similarity', lambda x: x.mean() * 100)
).round(2)
level_analysis.columns = ['Số lượng', 'Độ chính xác acc (%)', 'Tuân thủ định dạng (%)', 'Tương đồng lý luận (%)']
print(level_analysis.to_string())


# %%
fig_dir = os.path.join("..", "..", "..", "data", "result", "augmented-sft-dpo", "plots")
os.makedirs(fig_dir, exist_ok=True)

# 1. Biểu đồ cột nhóm so sánh các độ đo cốt lõi giữa 2 tập
metrics_data = {
    'Bộ dữ liệu': ['GSM8K', 'GSM8K', 'GSM8K', 'MATH', 'MATH', 'MATH'],
    'Độ đo': ['Độ chính xác acc', 'Tuân thủ định dạng (FCR)', 'Tương đồng lý luận', 
              'Độ chính xác acc', 'Tuân thủ định dạng (FCR)', 'Tương đồng lý luận'],
    'Giá trị (%)': [gsm_acc, gsm_fcr, gsm_sim, math_acc, math_fcr, math_sim]
}
df_metrics = pd.DataFrame(metrics_data)

plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Độ đo', y='Giá trị (%)', hue='Bộ dữ liệu', data=df_metrics, palette='viridis')
plt.title('So sánh các chỉ số hiệu năng acc giữa GSM8K và MATH', fontsize=14, fontweight='bold', pad=15)
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
plt.savefig(os.path.join(fig_dir, 'so_sanh_chi_so.png'), dpi=300)
plt.show()

# 2. Phân tích chi tiết theo Môn học của MATH
sub_df = subject_analysis.reset_index()
plt.figure(figsize=(12, 6))
x_indices = np.arange(len(sub_df))
width = 0.35

plt.bar(x_indices - width/2, sub_df['Độ chính xác acc (%)'], width, label='Độ chính xác acc', color='#3498db')
plt.bar(x_indices + width/2, sub_df['Tương đồng lý luận (%)'], width, label='Tương đồng lý luận (ROUGE-1)', color='#2ecc71')

plt.title('Độ chính xác acc và Độ tương đồng lý luận theo Chủ đề (MATH)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Chủ đề toán học', fontsize=12)
plt.ylabel('Tỷ lệ phần trăm (%)', fontsize=12)
plt.xticks(x_indices, sub_df['subject'], rotation=20, ha='right')
plt.ylim(0, 105)
plt.legend(frameon=True, facecolor='white', edgecolor='none')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'math_chu_de.png'), dpi=300)
plt.show()

# 3. Phân tích chi tiết theo Cấp độ khó của MATH
lvl_df = level_analysis.reset_index()
plt.figure(figsize=(10, 6))
x_indices_lvl = np.arange(len(lvl_df))

plt.bar(x_indices_lvl - width/2, lvl_df['Độ chính xác acc (%)'], width, label='Độ chính xác acc', color='#e74c3c')
plt.bar(x_indices_lvl + width/2, lvl_df['Tương đồng lý luận (%)'], width, label='Tương đồng lý luận (ROUGE-1)', color='#f1c40f')

plt.title('Độ chính xác acc và Độ tương đồng lý luận theo Cấp độ khó (MATH)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Cấp độ khó (Level)', fontsize=12)
plt.ylabel('Tỷ lệ phần trăm (%)', fontsize=12)
plt.xticks(x_indices_lvl, lvl_df['level'])
plt.ylim(0, 105)
plt.legend(frameon=True, facecolor='white', edgecolor='none')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'math_do_kho.png'), dpi=300)
plt.show()

# 4. Phân phối độ tương đồng lý luận (Histogram)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(gsm_df['reasoning_similarity'] * 100, bins=20, kde=True, color='purple')
plt.title('Phân phối Độ tương đồng lý luận - GSM8K', fontsize=12, fontweight='bold')
plt.xlabel('Độ tương đồng (ROUGE-1 %)', fontsize=10)
plt.ylabel('Tần suất (Số lượng mẫu)', fontsize=10)

plt.subplot(1, 2, 2)
sns.histplot(math_df['reasoning_similarity'] * 100, bins=20, kde=True, color='orange')
plt.title('Phân phối Độ tương đồng lý luận - MATH', fontsize=12, fontweight='bold')
plt.xlabel('Độ tương đồng (ROUGE-1 %)', fontsize=10)
plt.ylabel('Tần suất (Số lượng mẫu)', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'phan_phoi_tuong_dong.png'), dpi=300)
plt.show()

print("Đã tạo và lưu thành công toàn bộ 4 biểu đồ báo cáo tiếng Việt vào thư mục 'plots'!")
