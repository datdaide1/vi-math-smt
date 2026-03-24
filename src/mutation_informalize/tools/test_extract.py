import re

def extract_answer_from_text(text: str) -> str:
    m = re.search(r'\\boxed\{([^}]*(?:\{[^}]*\}[^}]*)*)\}', text)
    if m:
        return m.group(1).strip()
    m = re.search(r'Đáp án cuối cùng là:\s*(.+)', text)
    if m:
        return m.group(1).strip()
    return ""

text_9 = r'<s>[INST] Bạn là trợ lý toán học. Giải bài toán sau từng bước bằng Tiếng Việt và viết SMT-LIB code để xác minh. Đưa kết quả cuối cùng theo định dạng \boxed{kết quả}.\n\nTìm chữ số thứ $1314$ sau dấu thập phân trong biểu diễn thập phân của $\dfrac{5}{14}$. [/INST]\n## SMT-LIB\n```smt2\n(set-logic QF_LRA)\n(declare-fun a () Real)\n(declare-fun answer () Real)\n; The 1314th digit after the decimal point of 5/14 is 2.\n(assert (= a 2.0))\n(assert (= answer a))\n(check-sat)\n(get-value (answer))\n```\n\n## Lời giải\nĐầu tiên, ta tìm biểu diễn thập phân tuần hoàn của 5/14: $$ \\frac{5}{14} = \\frac{5}{5} \\cdot \\frac{5}{14} = \\frac{25}{70} = \\frac{25}{7} \\cdot \\frac{1}{10} = (3.\\overline{571428})(0.1) = 0.3\\overline{571428}. $$Chữ số thứ $1314$ sau dấu thập phân là chữ số thứ $1313$ trong khối tuần hoàn 6 chữ số 5-7-1-4-2-8. Vì $1313 \\div 6$ dư 5, nên đáp án của chúng ta là chữ số thứ 5 trong khối 6 chữ số, đó là \boxed{2}.\n\nĐáp án cuối cùng là: \boxed{2}</s>'

print(f"Extracted: '{extract_answer_from_text(text_9)}'")
