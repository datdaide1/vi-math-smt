# %% [markdown]
# # Đánh giá kết quả Formalize GSM8K
# Notebook này thực hiện tải dữ liệu kết quả từ Z3 và so sánh với Ground Truth để đánh giá hiệu suất của mô hình.

# %%
import json, os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown

# ================= CONFIG =================
# Assumes this script is run from its own directory (src/formalize/)
INPUT_FILE  = os.path.join("..", "..", "..", "data", "generate", "gsm8k", "train.jsonl")
OUTPUT_FILE = os.path.join("..", "..", "..", "data", "formalize_output", "gsm8k", "gsm8k-smt-nvidia-FULL.jsonl")

def load_jsonl(path):
    data = []
    if not os.path.exists(path):
        return pd.DataFrame()
    with open(path, "r", encoding="utf-8") as f:
        for line in f: 
            if line.strip():
                data.append(json.loads(line))
    return pd.DataFrame(data)

df_in = load_jsonl(INPUT_FILE)
df_out = load_jsonl(OUTPUT_FILE)

print(f"[INFO] Đã tải {len(df_in)} câu hỏi gốc.")
print(f"[INFO] Đã tải {len(df_out)} kết quả từ file output (đã sort & dedup).")

# %% [markdown]
# ## 1. Thống kê tổng quan

# %%
if not df_out.empty:
    total = len(df_out)
    passed = df_out['z3_score'].sum()
    failed = total - passed
    pass_rate = (passed / total) * 100

    print("=" * 40)
    print(f"TỔNG SỐ MẪU ĐÁNH GIÁ: {total}")
    print(f"THÀNH CÔNG (PASS):    {int(passed)} ({pass_rate:.2f}%)")
    print(f"THẤT BẠI (FAIL):      {int(failed)} ({100 - pass_rate:.2f}%)")
    print("=" * 40)
else:
    print("Chưa có dữ liệu kết quả.")

# %% [markdown]
# ## 2. Phân tích lỗi

# %%
if not df_out.empty:
    if 'validation_error' not in df_out.columns:
        df_out['validation_error'] = None
        
    fail_df = df_out[df_out['z3_score'] != 1.0].copy()
    if not fail_df.empty:
        error_counts = fail_df['validation_error'].fillna("Unknown/Timeout").value_counts()
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=error_counts.values, y=error_counts.index, palette="viridis", hue=error_counts.index, legend=False)
        plt.title("Phân bổ các loại lỗi Formalization")
        plt.xlabel("Số lượng")
        plt.ylabel("Loại lỗi")
        plt.show()
        
        print("\nDANH SÁCH CÁC CÂU THẤT BẠI:")
        for _, row in fail_df.iterrows():
            print(f"- Index {row['index']}: {row['validation_error'] or 'Timeout/Unknown'}")
    else:
        print("Chúc mừng! Không có câu nào bị lỗi.")

# %% [markdown]
# ## 3. Xem chi tiết các câu thất bại
# Sử dụng bảng này để hiểu tại sao mô hình lại sai và chuẩn bị cho việc sửa lỗi bằng DeepSeek V3.1.

# %%
df_failed = df_out[df_out['z3_score'] == 0].copy()
# Merge với input để lấy ground_truth nếu cần
df_merged = pd.merge(df_failed, df_in[['index', 'ground_truth']], on='index', how='left')

# Hiển thị 5 câu lỗi đầu tiên
pd.set_option('display.max_colwidth', 500)
df_merged[['index', 'question', 'smt_lib', 'z3_error', 'ground_truth']].head(10)
