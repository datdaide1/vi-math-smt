# %% [markdown]
# # Z3 SMT-LIB Evaluator
# Đọc file `gsm8k_smt_output.json`, giải từng bài toán bằng Z3, so sánh kết quả với ground truth.

# %%
# # Cell 1: Cài đặt thư viện (chạy 1 lần)
# !pip install z3-solver

# %%
# Cell 2: Import thư viện
import json
import re
from pathlib import Path
from z3 import *

print("Import OK")

# %%
# Cell 3: Load dữ liệu — assumes this script is run from its own directory
# (src/evaluate/); gsm8k_smt_output.json lives in _scratch/ (see repo root)
JSON_PATH = Path("..") / ".." / "_scratch" / "gsm8k_smt_output.json"

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Tổng số bài: {len(data)}")
has_smt = sum(1 for d in data if d.get("smt_lib_code", "").strip())
print(f"Bài có SMT code: {has_smt}")
print(f"Bài không có SMT code: {len(data) - has_smt}")

# %%
# Cell 4: Hàm parse & giải SMT-LIB bằng Z3
def parse_and_solve_smtlib(smt_code: str):
    """
    Dùng z3.parse_smt2_string() để parse code SMT-LIB 2.
    Trả về (status, result_str):
      - status : "sat" | "unsat" | "unknown" | "error" | "skipped"
      - result : chuỗi số hoặc thông báo lỗi
    """
    if not smt_code or not smt_code.strip():
        return "skipped", "No SMT code"

    code = smt_code.strip()

    # Thu thập biến cần lấy từ (get-value (...))
    get_value_match = re.findall(r'\(get-value\s*\(([^)]+)\)\)', code)

    # Bỏ (check-sat) và (get-value ...) trước khi parse
    code_for_parse = re.sub(r'\(check-sat\)', '', code)
    code_for_parse = re.sub(r'\(get-value\s*\([^)]+\)\)', '', code_for_parse)

    try:
        assertions = parse_smt2_string(code_for_parse)
    except Exception as e:
        return "error", f"Parse error: {e}"

    solver = Solver()
    solver.add(assertions)

    result = solver.check()

    if result == unsat:
        return "unsat", "UNSAT"
    elif result == unknown:
        return "unknown", "UNKNOWN"

    # SAT → đọc model
    model = solver.model()

    # Lấy giá trị các biến trong get-value
    extracted_values = []
    for gv_block in get_value_match:
        tokens = gv_block.split()
        for tok in tokens:
            tok = tok.strip()
            if not tok:
                continue
            for decl in model.decls():
                if decl.name() == tok:
                    extracted_values.append((tok, model[decl]))
                    break

    # Fallback: lấy tất cả biến trong model
    if not extracted_values:
        for decl in model.decls():
            extracted_values.append((decl.name(), model[decl]))

    if extracted_values:
        last_val = extracted_values[-1][1]
        try:
            # 1. Thử lấy số nguyên (Int)
            result_num = str(last_val.as_long())
        except Exception:
            try:
                # 2. Lấy phân số (Real), nhưng KHÔNG ép sang float
                num = last_val.numerator_as_long()
                den = last_val.denominator_as_long()
                
                # Nếu mẫu số là 1 thì chỉ in tử số
                if den == 1:
                    result_num = str(num)
                else:
                    # Nối chuỗi để giữ nguyên dạng phân số a/b
                    result_num = f"{num}/{den}"
                    
            except Exception:
                # 3. Fallback cho các kiểu dữ liệu khác (algebraic, bool...)
                result_num = str(last_val)
                
        return "sat", str(result_num)
    else:
        return "sat", "No variables found"

print("Hàm parse_and_solve_smtlib đã sẵn sàng")

# %%
# Cell 5: Hàm chuẩn hoá & so sánh kết quả
def normalize_number(s):
    """Bỏ dấu phẩy, ký hiệu $, chuyển về int/float."""
    if s is None:
        return None
    s = str(s).strip().replace(",", "").lstrip("$")
    try:
        val = float(s)
        return int(val) if val == int(val) else val
    except ValueError:
        return s

def compare(predicted, ground_truth) -> bool:
    """So sánh predicted vs ground truth sau khi chuẩn hoá."""
    p = normalize_number(predicted)
    g = normalize_number(ground_truth)
    if p is None or g is None:
        return False
    try:
        return float(p) == float(g)
    except (ValueError, TypeError):
        return str(p) == str(g)

print("Hàm normalize & compare đã sẵn sàng")

# %%
# Cell 6: Chạy đánh giá toàn bộ dataset
results = []

print(f"{'Idx':>4}  {'✓/✗':2}  {'Status':<10}  {'Z3 Answer':<14}  {'Ground Truth':<14}  Problem")
print("-" * 85)

for i, item in enumerate(data):
    smt_code     = item.get("smt_lib", "")
    ground_truth = item.get("ground_truth", "")
    problem_short = item.get("question", "")[:55].replace("\n", " ")

    status, z3_answer = parse_and_solve_smtlib(smt_code)

    is_correct = compare(z3_answer, ground_truth) if status == "sat" else False

    results.append({
        "index":        i,
        "status":       status,
        "z3_answer":    z3_answer,
        "ground_truth": ground_truth,
        "correct":      is_correct,
    })

    mark = "✓" if is_correct else ("─" if status in ("skipped", "error") else "✗")
    print(f"{i:4d}  {mark:2}  {status:<10}  {str(z3_answer):<14}  {ground_truth:<14}  {problem_short}")

print("-" * 85)
print("Đã xử lý xong!")

# %%
# Cell 7: Tổng hợp kết quả (accuracy)
total       = len(results)
sat_items   = [r for r in results if r["status"] == "sat"]
skipped     = [r for r in results if r["status"] == "skipped"]
errors      = [r for r in results if r["status"] == "error"]
unsat_items = [r for r in results if r["status"] in ("unsat", "unknown")]
correct     = [r for r in sat_items if r["correct"]]
wrong       = [r for r in sat_items if not r["correct"]]

print("=" * 55)
print("TỔNG KẾT")
print("=" * 55)
print(f"  Tổng số bài           : {total}")
print(f"  Có SMT code → SAT     : {len(sat_items)}")
print(f"  Không có SMT code     : {len(skipped)}")
print(f"  Lỗi parse / runtime   : {len(errors)}")
print(f"  UNSAT / Unknown       : {len(unsat_items)}")
print("-" * 55)
if sat_items:
    print(f"  Đúng / Tổng SAT       : {len(correct)} / {len(sat_items)}  "
          f"({100*len(correct)/len(sat_items):.1f}%)")
print(f"  Sai (trong số SAT)    : {len(wrong)}")
print("-" * 55)
print(f"  Accuracy tổng thể     : {len(correct)} / {total}  "
      f"({100*len(correct)/total:.1f}%)")
print("=" * 55)

# %%
# Cell 8: Xem danh sách bài GIẢI SAI
print(f"Bài giải SAI ({len(wrong)} bài):")
print("-" * 75)
for r in wrong:
    item = data[r["index"]]
    prob = item.get("problem", "")[:65].replace("\n", " ")
    print(f"  [{r['index']:3d}]  z3={str(r['z3_answer']):<12}  gt={r['ground_truth']:<10}  {prob}")

# %%
# Cell 9: Xem danh sách bài LỖI PARSE
print(f"Bài lỗi parse ({len(errors)} bài):")
print("-" * 75)
for r in errors:
    item = data[r["index"]]
    prob = item.get("problem", "")[:55].replace("\n", " ")
    print(f"  [{r['index']:3d}]  gt={r['ground_truth']:<8}  {r['z3_answer'][:60]}")
    print(f"         Problem: {prob}")

# %%
# Cell 10: Lưu kết quả chi tiết ra file JSON
OUT_PATH = Path("..") / ".." / "_scratch" / "z3_evaluation_results.json"

output = []
for r in results:
    item = data[r["index"]]
    output.append({
        "index":        r["index"],
        "problem":      item.get("problem", ""),
        "smt_lib_code": item.get("smt_lib_code", ""),
        "ground_truth": r["ground_truth"],
        "z3_status":    r["status"],
        "z3_answer":    r["z3_answer"],
        "correct":      r["correct"],
    })

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"Đã lưu kết quả vào: {OUT_PATH.resolve()}")
