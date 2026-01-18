# %% [markdown]
# # Phân tích và Xử lý câu lỗi (MATH Failures)
# 
# Notebook này tập trung vào việc soi chi tiết các câu formalize thất bại để cải thiện logic/prompt.

# %%
import pandas as pd
import json
import os

# Assumes this script is run from its own directory (src/formalize/)
fail_path = os.path.join("..", "..", "..", "data", "formalize_output", "math-fail.jsonl")
df = pd.read_json(fail_path, lines=True)

def categorize_err(e):
    if pd.isna(e) or e == '': return 'Khác'
    e = str(e).lower()
    if 'timeout' in e: return 'Timeout'
    if 'wrong_answer' in e: return 'Sai kết quả'
    if 'syntax' in e or 'parse' in e: return 'Lỗi cú pháp/LaTeX'
    if 'z3' in e or 'solver' in e: return 'Lỗi Solver'
    return 'Khác'

df['error_type'] = df['error'].apply(categorize_err)
print(f"Tổng số câu lỗi cần xử lý: {len(df)}")
display(df['error_type'].value_counts())

# %% [markdown]
# ## Soi Sample theo từng loại lỗi
# Thay đổi `target_error` để xem các mẫu khác nhau.

# %%
target_error = 'Sai kết quả' # ['Timeout', 'Sai kết quả', 'Lỗi cú pháp/LaTeX', 'Lỗi Solver']
samples = df[df['error_type'] == target_error].head(5)

for i, row in samples.iterrows():
    print(f"\n=== INDEX: {row['index']} | SUBJECT: {row['subject']} ===")
    print(f"ERROR: {row['error']}")
    print(f"GROUND TRUTH: {row['ground_truth']}")
    print(f"SMT_LIB CODE:\n{row['smt_lib']}")
    print("-"*80)
