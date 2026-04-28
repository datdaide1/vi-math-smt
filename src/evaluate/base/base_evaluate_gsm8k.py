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
        r'Đáp án cuối cùng là:?\s*(.*)',
        r'Vậy đáp án là:?\s*(.*)',
        r'The answer is:?\s*(.*)'
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
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
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
    string = string.replace("^\\circ", "^{\\circ}")
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
        from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
        from sympy import simplify
        transformations = (standard_transformations + (implicit_multiplication_application,))
        
        def prep(s):
            s = s.replace('\\pi', 'pi').replace('\\infty', 'oo')
            s = re.sub(r'\\root\s*(\d+)\s*\\of\s*\{([^\}]+)\}', r'((\\2)**(1/\\1))', s)
            s = re.sub(r'\\sqrt\[(\d+)\]\{([^\}]+)\}', r'((\\2)**(1/\\1))', s)
            
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
                s = re.sub(r'\{([a-zA-Z0-9_.\s\+\-\*/\(\)]+)\}', r'(\1)', s)
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

def reasoning_similarity(pred_reason, gt_solution):
    if not pred_reason or not gt_solution:
        return 0.0
    pred_words = str(pred_reason).split()
    gt_words = str(gt_solution).split()
    sm = difflib.SequenceMatcher(None, pred_words, gt_words)
    return sm.ratio()


# %%

# Đọc dữ liệu GSM8K — assumes this script is run from its own directory (src/evaluate/)
data = []
with open(os.path.join("..", "..", "..", "data", "result", "original-sft", "gsm8k", "gsm8k_predictions.jsonl"), 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

results = []
for row in data:
    true_gt = get_true_gt(row)
    true_pred = get_true_pred(row)
    
    is_correct = check_correctness(true_pred, true_gt)
    has_boxed = "\\boxed{" in row.get('model_reason', '')
    
    sim_score = reasoning_similarity(row.get('model_reason', ''), row.get('solution', ''))
    
    results.append({
        'index': row.get('index', 0),
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
print(f"KẾT QUẢ ĐÁNH GIÁ GSM8K (SFT)")
print("="*40)
print(f"Tổng số câu: {len(df)}")
print(f"Độ chính xác (Accuracy): {acc:.2f}%")
print(f"Tỷ lệ tuân thủ định dạng (Format Compliance): {format_ok:.2f}%")
print(f"Độ dài câu trả lời trung bình: {avg_len:.0f} ký tự")
print(f"Số bước suy luận trung bình: {avg_steps:.1f} bước")
print(f"Độ tương đồng lý luận (Reasoning Similarity): {avg_sim:.2f}%")


# %%

save_dir = os.path.join("..", "..", "..", "data", "evaluate", "img", "orriginal-sft", "gsm8k")
os.makedirs(save_dir, exist_ok=True)

plt.rcParams['font.family'] = 'sans-serif'

# 1. Bar chart Độ chính xác
plt.figure(figsize=(8, 6))
ax = sns.barplot(x=['Đúng', 'Sai'], y=[df['is_correct'].sum(), (~df['is_correct']).sum()], palette='viridis')
plt.title('Độ chính xác dự đoán - GSM8K', fontsize=14)
plt.ylabel('Số lượng câu', fontsize=12)
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=11, color='black', xytext=(0, 5), textcoords='offset points')
plt.savefig(os.path.join(save_dir, 'gsm8k_accuracy_bar.png'), dpi=300, bbox_inches='tight')
plt.show()

# 2. Pie chart Tuân thủ định dạng
plt.figure(figsize=(8, 6))
plt.pie([df['has_boxed'].sum(), (~df['has_boxed']).sum()], labels=['Có \\boxed{}', 'Không có \\boxed{}'], 
        autopct='%1.1f%%', colors=['#4CAF50', '#F44336'], startangle=90, textprops={'fontsize': 12})
plt.title('Tỷ lệ tuân thủ định dạng \\boxed{} - GSM8K', fontsize=14)
plt.savefig(os.path.join(save_dir, 'gsm8k_format_compliance_pie.png'), dpi=300, bbox_inches='tight')
plt.show()

# 3. Histogram độ dài câu trả lời
plt.figure(figsize=(10, 6))
sns.histplot(df['reasoning_length'], bins=30, kde=True, color='royalblue')
plt.title('Phân bố độ dài câu trả lời (ký tự) - GSM8K', fontsize=14)
plt.xlabel('Độ dài (ký tự)', fontsize=12)
plt.ylabel('Số lượng câu', fontsize=12)
plt.savefig(os.path.join(save_dir, 'gsm8k_reasoning_length_hist.png'), dpi=300, bbox_inches='tight')
plt.show()

# 4. Histogram Độ tương đồng lý luận
plt.figure(figsize=(10, 6))
sns.histplot(df['reasoning_similarity'], bins=30, kde=True, color='mediumseagreen')
plt.title('Phân bố độ tương đồng lý luận so với đáp án gốc - GSM8K', fontsize=14)
plt.xlabel('Độ tương đồng (Jaccard/SequenceMatcher ratio)', fontsize=12)
plt.ylabel('Số lượng câu', fontsize=12)
plt.savefig(os.path.join(save_dir, 'gsm8k_reasoning_similarity_hist.png'), dpi=300, bbox_inches='tight')
plt.show()
