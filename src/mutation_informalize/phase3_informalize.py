"""
Phase 3: Informalize SMT → Vietnamese Natural Math Problems
Paper-Enhanced (Constraint-Grounded Version)

NEW:
- SMT constraint decomposition (paper-style)
- Better semantic grounding
- No change to LLM pipeline logic
"""

import asyncio
import json
import os
import re
import sys
import time
import math
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    PHASE2_MUTATED_OUTPUT,
    PHASE3_INFORMALIZED_OUTPUT,
    OUTPUT_DIR,
    INPUT_FILE
)

from llm_client import (
    AsyncLLMClient,
    load_checkpoint,
    append_checkpoint,
    print_progress,
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 🧠 NEW: SMT Decomposition (PAPER STYLE ADDITION)
# ============================================================

def decompose_smt(smt_code: str):
    """
    Convert SMT → structured constraints (lightweight paper idea)
    """

    constraints = []

    for line in smt_code.split("\n"):
        line = line.strip()

        if not line:
            continue

        # keep only assertions (core semantics)
        if line.startswith("(assert"):
            constraints.append(line)

    return constraints


# ============================================================
# XML PROMPTS (GENERATOR AND SOLVER)
# ============================================================

GENERATOR_PROMPT = """
<system>
Bạn là chuyên gia ra đề toán học bằng tiếng Việt.
Bạn rất giỏi trong việc Sửa Đổi (Modify) bài toán gốc để phù hợp với các ràng buộc SMT mới.
</system>

<user>
<THÔNG_TIN>
Môn học: {subject}
Độ khó: {level}/5
</THÔNG_TIN>

<BÀI_TOÁN_GỐC>
{original_problem}
</BÀI_TOÁN_GỐC>

<SMT_CONSTRAINTS>
{constraints}
</SMT_CONSTRAINTS>

Nhiệm vụ: Sửa đổi <BÀI_TOÁN_GỐC> sao cho khớp với <SMT_CONSTRAINTS> mới. Bạn phải tuân thủ tuyệt đối từng dòng SMT.
BẮT BUỘC PHẢI TRẢ VỀ THEO FORMAT SAU:
<CONSTRAINT_MAPPING>
[Viết tóm tắt ánh xạ các biến SMT vào đề bài cực kỳ ngắn gọn trong đúng 1 dòng duy nhất]
</CONSTRAINT_MAPPING>
<BÀI_TOÁN>
[Nội dung bài toán đã được sửa đổi dựa trên mapping ở trên. Yêu cầu giữ văn phong tự nhiên.]
</BÀI_TOÁN>

</user>
"""

SOLVER_PROMPT = """
<system>
Bạn là học sinh xuất sắc giải toán bằng tiếng Việt.
</system>

<user>
<BÀI_TOÁN>
{problem}
</BÀI_TOÁN>

Nhiệm vụ: Giải bài toán trên bằng tư duy từng bước (Chain of Thought).
BẮT BUỘC PHẢI TRẢ VỀ THEO FORMAT SAU (Không thêm bớt tag):
<LỜI_GIẢI>
[Lời giải chi tiết từng bước ở đây]
</LỜI_GIẢI>
<ĐÁP_ÁN>
\\boxed{{[Đáp án cuối cùng]}}
</ĐÁP_ÁN>
</user>
"""


# ============================================================
# PARSING (UNCHANGED)
# ============================================================

def extract_between(text, start_tag, end_tag):
    pattern = rf"{re.escape(start_tag)}(.*?){re.escape(end_tag)}"
    m = re.search(pattern, text, re.DOTALL)
    return m.group(1).strip() if m else ""


def parse_problem(response: str):
    if not response: return ""
    # Dùng rsplit để luôn lấy khối <BÀI_TOÁN> cuối cùng
    if "<BÀI_TOÁN>" in response:
        content = response.rsplit("<BÀI_TOÁN>", 1)[-1]
        if "</BÀI_TOÁN>" in content:
            content = content.split("</BÀI_TOÁN>")[0]
        return content.strip()
    # Fallback nếu model quên mở tag <BÀI_TOÁN> nhưng có </CONSTRAINT_MAPPING>
    if "</CONSTRAINT_MAPPING>" in response:
        content = response.rsplit("</CONSTRAINT_MAPPING>", 1)[-1]
        if "</BÀI_TOÁN>" in content:
            content = content.split("</BÀI_TOÁN>")[0]
        return content.strip()
    # Fallback tìm kiếm tag không gạch dưới
    if "<BÀI TOÁN>" in response:
        content = response.rsplit("<BÀI TOÁN>", 1)[-1]
        if "</BÀI TOÁN>" in content:
            content = content.split("</BÀI TOÁN>")[0]
        return content.strip()
    return ""

def parse_solution(response: str):
    solution = extract_between(response, "<LỜI_GIẢI>", "</LỜI_GIẢI>")
    if not solution and "<LỜI_GIẢI>" in response:
        solution = response.split("<LỜI_GIẢI>")[-1].split("<ĐÁP_ÁN>")[0].strip()
        
    answer_block = extract_between(response, "<ĐÁP_ÁN>", "</ĐÁP_ÁN>")
    if not answer_block and "<ĐÁP_ÁN>" in response:
        answer_block = response.split("<ĐÁP_ÁN>")[-1].strip()

    answer = ""
    if answer_block:
        boxed_match = re.search(r'\\boxed\{((?:[^{}]+|\{[^{}]*\})*)\}', answer_block)
        if boxed_match:
            answer = boxed_match.group(1).strip()
        else:
            answer = answer_block.replace("\\boxed", "").strip()
            if answer.startswith("{") and answer.endswith("}"):
                answer = answer[1:-1].strip()
    else:
        boxed_match = re.findall(r'\\boxed\{((?:[^{}]+|\{[^{}]*\})*)\}', response)
        if boxed_match:
            answer = boxed_match[-1].strip()

    return {
        "solution": solution,
        "answer": answer,
    }


# ============================================================
# VALIDATION (UNCHANGED)
# ============================================================

def normalize_answer(ans):
    if ans is None:
        return ""
    ans = str(ans).strip()
    # Nếu Z3 trả về có dấu hỏi chấm ở cuối (thể hiện xấp xỉ), loại bỏ nó
    if ans.endswith("?"):
        ans = ans[:-1].strip()
        
    # Loại bỏ các khoảng trắng mỏng trong LaTeX (\,) và (\;)
    ans = ans.replace(r"\,", "").replace(r"\;", "").replace(r"\ ", "")
    ans = ans.replace(" ", "").replace(",", "").replace("$", "")
    ans = ans.replace(r"^\circ", "").replace(r"\circ", "")
    
    # Hỗ trợ ký hiệu xấp xỉ (\approx) giống như dấu bằng
    ans = ans.replace(r"\approx", "=")
    if "=" in ans:
        ans = ans.split("=")[-1].strip()
        
    # Convert \frac{a}{b} and \dfrac{a}{b} to a/b
    ans = re.sub(r'\\d?frac\{([^{}]+)\}\{([^{}]+)\}', r'\1/\2', ans)
    return ans


def validate_generation(parsed, ground_truth):

    if not parsed["problem"]:
        return False, "missing_problem"

    if not parsed["solution"]:
        return False, "missing_solution"

    if len(parsed["solution"]) < 50:
        return False, "solution_too_short"

    # Kiểm tra xem đáp án có mang tính chất xấp xỉ không
    is_approx = False
    if ground_truth and str(ground_truth).strip().endswith("?"):
        is_approx = True
    if any(term in str(parsed["answer"]) for term in [r"\approx", "xấp xỉ", "approx"]):
        is_approx = True

    gt = normalize_answer(ground_truth)
    raw_ans = str(parsed["answer"]).strip()

    # Trích xuất các chuỗi ứng viên (candidates) từ raw answer nếu có chứa nhiều phần phân cách
    candidates = [raw_ans]
    # Phân tách theo dấu phẩy, chấm phẩy, qquad, quad, từ 'và', 'and'
    splits = re.split(r'(?:,|;|\bvà\b|\band\b|\\qquad|\\quad)', raw_ans)
    if len(splits) > 1:
        candidates.extend([s.strip() for s in splits if s.strip()])

    # Định nghĩa ngưỡng sai số cho phép: xấp xỉ thì cho phép sai số 0.005, ngược lại 1e-5
    threshold = 0.005 if is_approx else 1e-5

    # Numeric evaluation fallback
    def robust_eval(expr):
        expr = str(expr).replace(r"\\", "\\").replace("{", "(").replace("}", ")")
        expr = expr.replace(r"\pi", str(math.pi))
        expr = expr.replace(r"\sqrt", "math.sqrt")
        expr = expr.replace("^", "**")
        return eval(expr, {"math": math})

    # Hàm kiểm tra 1 chuỗi ứng viên đã chuẩn hóa
    def check_single_pred(pred):
        pred = re.sub(r'\\text\{[^}]*\}', '', pred)
        pred = re.sub(r'\(xấpxỉ\)', '', pred)
        if not pred:
            return False
        if gt == pred:
            return True
        try:
            v_gt = robust_eval(gt)
            v_pred = robust_eval(pred)
            if abs(v_gt - v_pred) < threshold or (v_gt != 0 and abs((v_gt - v_pred) / v_gt) < 0.005):
                return True
        except Exception:
            pass
        try:
            from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
            transformations = (standard_transformations + (implicit_multiplication_application,))
            def sympy_prep(s):
                s = str(s).replace('\\pi', 'pi').replace('\\infty', 'oo')
                s = re.sub(r'\\text\{[^}]*\}', '', s)
                s = s.replace('\\sqrt', 'sqrt')
                s = re.sub(r'\\d?frac\{([^{}]+)\}\{([^{}]+)\}', r'((\1)/(\2))', s)
                s = s.replace('^', '**')
                s = s.replace('{', '(').replace('}', ')')
                return s
            e1 = parse_expr(sympy_prep(gt), transformations=transformations)
            e2 = parse_expr(sympy_prep(pred), transformations=transformations)
            diff_val = abs((e1 - e2).evalf())
            if diff_val < threshold:
                return True
            v1 = abs(e1.evalf())
            if v1 != 0 and (diff_val / v1) < 0.005:
                return True
        except Exception:
            pass
        return False

    # Kiểm tra lần lượt các ứng viên
    for cand in candidates:
        norm_cand = normalize_answer(cand)
        if check_single_pred(norm_cand):
            return True, "ok"

    return False, "answer_mismatch"


# ============================================================
# REFLECTION PROMPT (UNCHANGED)
# ============================================================

REFLECTION_PROMPT = """
<system>
Bạn là chuyên gia sửa lỗi dữ liệu toán học.
</system>

<user>

<FEEDBACK>
{feedback}
</FEEDBACK>

<PREVIOUS_OUTPUT>
<![CDATA[
{previous_output}
]]>
</PREVIOUS_OUTPUT>

Sửa lại <BÀI_TOÁN> cho chặt chẽ hơn để Solver không bị giải sai. Bạn phải xem xét kỹ phản hồi lỗi để chỉnh lại ngữ từ trong Đề Bài.
BẮT BUỘC TRẢ VỀ THEO ĐÚNG FORMAT NÀY:
<CONSTRAINT_MAPPING>
[Liệt kê từng dòng constraint, và câu văn tương ứng]
</CONSTRAINT_MAPPING>
<BÀI_TOÁN>
[Nội dung bài toán mới]
</BÀI_TOÁN>

</user>
"""


# ============================================================
# AGENT LOOP (ONLY CHANGE: constraint input)
# ============================================================

async def generate_math_problem_agentic(
    llm,
    smt_code,
    subject,
    level,
    answer,
    original_problem,
    max_attempts=1,
):

    last_feedback = "Initial generation"
    previous_problem_output = ""

    constraints = decompose_smt(smt_code)

    for attempt in range(max_attempts):

        print(f"      Attempt {attempt+1}/{max_attempts}")

        # ----------------------------------------------------
        # BƯỚC 1: GENERATOR (Sinh Đề Bài)
        # ----------------------------------------------------
        if attempt == 0:
            prompt1 = GENERATOR_PROMPT.format(
                subject=subject,
                level=level,
                original_problem=original_problem,
                constraints="\n".join(constraints),
            )
        else:
            prompt1 = REFLECTION_PROMPT.format(
                feedback=last_feedback,
                previous_output=previous_problem_output,
            )

        gen_resp = await llm.call("", prompt1, temperature=0.7)

        if not gen_resp:
            last_feedback = "Generator returned empty"
            continue

        problem_text = parse_problem(gen_resp)
        if not problem_text or len(problem_text) < 10:
            last_feedback = "Lỗi: Không tìm thấy thẻ <BÀI_TOÁN> hoặc đề bài quá ngắn."
            previous_problem_output = gen_resp
            continue

        # ----------------------------------------------------
        # BƯỚC 2: SOLVER (Giải Đề Bài)
        # ----------------------------------------------------
        prompt2 = SOLVER_PROMPT.format(problem=problem_text)
        sol_resp = await llm.call("", prompt2, temperature=0.0)

        if not sol_resp:
            last_feedback = "Solver returned empty. Có thể đề bài lỗi, hãy sinh lại bài toán khác."
            previous_problem_output = gen_resp
            continue

        parsed_sol = parse_solution(sol_resp)

        parsed_combined = {
            "problem": problem_text,
            "solution": parsed_sol["solution"],
            "answer": parsed_sol["answer"],
        }

        ok, feedback = validate_generation(parsed_combined, answer)

        if ok:
            return {
                "success": True,
                "problem": problem_text,
                "solution": parsed_sol["solution"],
                "answer": parsed_sol["answer"],
                "attempts": attempt + 1,
            }

        last_feedback = f"Giải sai. Đáp án chuẩn từ Z3 là: {answer}. Đáp án của solver (bị mù SMT) tính ra là: {parsed_sol['answer']}. Thông tin: {feedback}. Hãy sửa lại câu văn trong <BÀI_TOÁN> cho chặt chẽ hơn."
        previous_problem_output = gen_resp
        
        print(f"        -> Thất bại: {feedback} (Z3: {answer} vs Solver: {parsed_sol['answer']})")

        # Fail-fast optimization: Nếu Z3 ra 0 rác do SMT tràn số ngầm, việc thử lại 3 lần là hoàn toàn vô ích.
        if answer.strip() == "0" and attempt == 0:
            print(f"        -> [Fail-Fast] Phát hiện Z3 trả về 0 rác do lỗi SMT ngầm, ngắt sớm tránh lãng phí API!")
            last_feedback = f"Z3 ra 0 rác (SMT tràn số). Ngắt sớm sau 1 attempt."
            break

    print(f"      [!] Bó tay sau {max_attempts} attempts. Lỗi cuối: {last_feedback}")
    return {
        "success": False,
        "problem": problem_text if 'problem_text' in locals() else "",
        "solution": parsed_sol["solution"] if 'parsed_sol' in locals() else "",
        "answer": parsed_sol["answer"] if 'parsed_sol' in locals() else "",
        "attempts": max_attempts,
        "feedback": last_feedback
    }


# ============================================================
# MAIN LOOP (UNCHANGED LOGIC)
# ============================================================

async def run_phase3():

    print("=" * 80)
    print("PHASE 3 (PAPER-ENHANCED)")
    print("=" * 80)

    processed = load_checkpoint(PHASE3_INFORMALIZED_OUTPUT, clean_failed=False)

    phase2_data = []

    with open(PHASE2_MUTATED_OUTPUT, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                phase2_data.append(json.loads(line))

    # Đọc dữ liệu gốc từ Phase 1 để lấy Original Problem
    original_dict = {}
    if os.path.exists(INPUT_FILE):
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        idx = rec.get("index")
                        if idx is not None:
                            original_dict[idx] = rec.get("problem", "")
                    except:
                        pass
    print(f"Loaded {len(original_dict)} original problems.")

    llm = AsyncLLMClient()

    stats = defaultdict(int)

    start_time = time.time()

    total = len(phase2_data)
    to_do_indices = [i for i in range(total) if i not in processed]
    total_to_do = len(to_do_indices)
    done_session = 0

    # Semaphore 50: cân bằng giữa throughput và ổn định kết nối
    sem = asyncio.Semaphore(50)

    async def worker(idx, item):
        nonlocal done_session
        async with sem:
            original_prob = original_dict.get(item.get("source_index"), "")
            result = await generate_math_problem_agentic(
                llm=llm,
                smt_code=item["smt_code"],
                subject=item.get("subject", "unknown"),
                level=item.get("level", 1),
                answer=item.get("z3_answer", ""),
                original_problem=original_prob,
            )

            output_item = {
                "index": idx,
                "subject": item.get("subject"),
                "level": item.get("level"),
                "smt_code": item["smt_code"],
                "ground_truth": item.get("z3_answer", ""),
                "problem": result["problem"],
                "solution": result["solution"],
                "answer": result["answer"],
                "status": "ok" if result["success"] else "failed",
                "feedback": result.get("feedback", ""),
                "attempts": result["attempts"],
                "mutation_strategy": item.get("mutation_id"),
            }

            append_checkpoint(PHASE3_INFORMALIZED_OUTPUT, idx, output_item)
            stats[output_item["status"]] += 1

            done_session += 1
            extra_info = f"[Index gốc: {idx}/{total}]"
            print_progress(done_session, total_to_do, "generating...", extra=extra_info)

    # Thu thập danh sách (idx, item) cần xử lý
    # SẮP XẾP BREADTH-FIRST: mutation_id=0 của tất cả bài trước, rồi mutation_id=1, ...
    # Đảm bảo mỗi bài gốc đều có ít nhất 1 mutation dù dừng giữa chừng
    work_items = []
    for idx, item in enumerate(phase2_data):
        if idx in processed:
            continue
        work_items.append((idx, item))
    
    # Sắp xếp theo mutation_id tăng dần (0 trước, rồi 1, 2, 3, 4)
    work_items.sort(key=lambda x: x[1].get("mutation_id", 0))
    
    # Thống kê nhanh
    from collections import Counter
    mut_counts = Counter(item.get("mutation_id", 0) for _, item in work_items)
    print(f"Thứ tự ưu tiên (Breadth-First): {dict(sorted(mut_counts.items()))}")

    BATCH_SIZE = 100
    total_batches = (len(work_items) + BATCH_SIZE - 1) // BATCH_SIZE
    BATCH_TIMEOUT = 600  # 10 phút tối đa mỗi batch

    print(f"Bắt đầu xử lý {len(work_items)} tác vụ theo {total_batches} batch (mỗi batch {BATCH_SIZE}, timeout {BATCH_TIMEOUT}s)...")

    for batch_num in range(total_batches):
        batch_start = batch_num * BATCH_SIZE
        batch = work_items[batch_start:batch_start + BATCH_SIZE]
        
        batch_tasks = [worker(idx, item) for idx, item in batch]
        
        print(f"\n--- Batch {batch_num+1}/{total_batches} ({len(batch_tasks)} tasks) ---")
        
        try:
            await asyncio.wait_for(
                asyncio.gather(*batch_tasks, return_exceptions=True),
                timeout=BATCH_TIMEOUT
            )
        except asyncio.TimeoutError:
            print(f"\n[!] Batch {batch_num+1} bị TREO quá {BATCH_TIMEOUT}s! Reset kết nối, chuyển batch tiếp...")
            # Reset toàn bộ client connections
            llm._clients.clear()
            llm.key_queue = None
            await asyncio.sleep(3)
        except KeyboardInterrupt:
            print("\nPipeline interrupted safely.")
            return

    print("\nDONE")
    print(stats)
    
    # 🧠 PAPER METRIC: Vẽ biểu đồ Consistency Rate
    plot_evaluation_metrics()


# ============================================================
# EVALUATION METRICS (PAPER: Consistency Rate)
# ============================================================

def plot_evaluation_metrics():
    print("Đang vẽ biểu đồ đánh giá (Consistency Rate)...")
    data = []
    if not os.path.exists(PHASE3_INFORMALIZED_OUTPUT):
        print("Chưa có dữ liệu để vẽ biểu đồ.")
        return
        
    with open(PHASE3_INFORMALIZED_OUTPUT, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
                
    if not data:
        return
        
    df = pd.DataFrame(data)
    
    sns.set_theme(style="whitegrid")
    
    # 1. Pie Chart: Consistency Rate Tổng Thể
    plt.figure(figsize=(8, 6))
    status_counts = df['status'].value_counts()
    
    plot_labels = []
    colors = []
    for stat in status_counts.index:
        if stat == 'ok':
            plot_labels.append('Thành công (Khớp Z3)')
            colors.append('#2ecc71')
        else:
            plot_labels.append('Thất bại (Không khớp/Lỗi)')
            colors.append('#e74c3c')
            
    plt.pie(status_counts, labels=plot_labels, autopct='%1.1f%%', colors=colors, startangle=140)
    plt.title('Tỉ Lệ Nhất Quán (Consistency Rate) Tổng Thể', fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(OUTPUT_DIR, 'consistency_rate_overall.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Bar Chart: Tỉ Lệ Thành Công Theo Môn Học
    if 'subject' in df.columns:
        plt.figure(figsize=(10, 6))
        # Compute consistency rate per subject
        subject_success = df.groupby('subject').apply(lambda x: (x['status'] == 'ok').mean() * 100).reset_index(name='Consistency Rate (%)')
        
        ax = sns.barplot(data=subject_success, x='subject', y='Consistency Rate (%)', palette='viridis')
        plt.title('Tỉ Lệ Nhất Quán (Consistency Rate) Theo Môn Học', fontsize=14, fontweight='bold')
        plt.xlabel('Môn Học', fontsize=12)
        plt.ylabel('Consistency Rate (%)', fontsize=12)
        plt.xticks(rotation=45)
        plt.ylim(0, 100)
        
        # Add labels on top of bars
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='baseline', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')
        
        plt.savefig(os.path.join(OUTPUT_DIR, 'consistency_rate_by_subject.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    print(f"Vẽ biểu đồ hoàn tất. Đã lưu tại thư mục {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(run_phase3())