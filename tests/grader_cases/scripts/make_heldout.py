"""Emit tests/grader_cases/heldout_style.jsonl: MATH / exam-style responses written by X1 AFTER the grader was
frozen for the dev cases, labelled before the first run.  First-run verdicts are what the report counts."""
import json, sys

def G(t, v, **k):
    d = {"answer_type": t, "answer_canonical": v, "locale": k.pop("locale", "vi")}
    d.update(k)
    return d

C = []
def c(cid, pred, gold, expect, note=""):
    C.append({"id": cid, "pred": pred, "gold": gold, "expect": expect, "note": note, "source": "heldout-X1"})

c("H01", "Ta có $\\Delta = 25 - 24 = 1$ nên phương trình có hai nghiệm phân biệt.\n\n$x_1 = \\dfrac{5+1}{4} = \\dfrac{3}{2}$, $x_2 = \\dfrac{5-1}{4} = 1$.\n\nVậy tập nghiệm của phương trình là $S = \\left\\{1; \\dfrac{3}{2}\\right\\}$.",
  G("set", "1; 3/2"), True)
c("H02", "Giải hệ ta được $x = 2$, $y = -1$.\n\nVậy hệ phương trình có nghiệm duy nhất $(x; y) = (2; -1)$.",
  G("tuple", "2; -1", vars=["x", "y"]), True)
c("H03", "Giải hệ ta được $x = 2$, $y = -1$.\n\nVậy hệ phương trình có nghiệm duy nhất $(x; y) = (-1; 2)$.",
  G("tuple", "2; -1", vars=["x", "y"]), False, "swapped order with named tuple on lhs")
c("H04", "Điều kiện: $x \\ne 1$. Bất phương trình tương đương $(x-1)(x-3) < 0$.\n\nVậy tập nghiệm là $S = (1; 3)$.",
  G("interval", "(1,3)"), True)
c("H05", "Vậy tập nghiệm là $S = [1; 3]$.", G("interval", "(1,3)"), False)
c("H06", "Thể tích khối chóp là $V = \\dfrac{1}{3} \\cdot a^2 \\cdot \\dfrac{a\\sqrt{3}}{2} = \\boxed{\\dfrac{a^3\\sqrt{3}}{6}}$.",
  G("expr", "\\frac{\\sqrt{3}a^3}{6}"), True)
c("H07", "Thể tích khối chóp là $\\boxed{\\dfrac{a^3\\sqrt{3}}{2}}$.", G("expr", "\\frac{\\sqrt{3}a^3}{6}"), False)
c("H08", "Xác suất cần tìm là $P = \\dfrac{C_5^2}{C_{10}^2} = \\dfrac{10}{45} = \\dfrac{2}{9}$.",
  G("rational", "2/9"), True, "no box, no cue: last fraction")
c("H09", "Xác suất cần tìm là $P \\approx 0{,}22$.", G("rational", "2/9"), False, "rounded decimal without round_dp")
c("H10", "Xác suất cần tìm là $P \\approx 0{,}22$.", G("rational", "2/9", round_dp=2), True)
c("H11", "Số tiền lãi sau 2 năm là $100\\,000\\,000 \\cdot (1{,}06^2 - 1) = 12\\,360\\,000$ đồng.\n\nĐáp số: $12\\,360\\,000$ đồng.",
  G("int", "12360000"), True)
c("H12", "Đáp số: 12.360.000 đồng.", G("int", "12360000"), True)
c("H13", "Đáp số: 12,36 triệu đồng.", G("int", "12360000"), True, "triệu multiplier with decimal comma")
c("H14", "Giá trị lớn nhất của hàm số trên đoạn $[0; 2]$ là $\\max f(x) = f(1) = 3$.", G("int", "3"), True)
c("H15", "Giá trị lớn nhất là 3, đạt được khi $x = 1$.", G("int", "3"), True,
  "cue-less prose; the stated value is 3, x = 1 is where it is reached")
c("H16", "Phương trình có hai nghiệm $x = \\dfrac{\\pi}{6} + k2\\pi$ và $x = \\dfrac{5\\pi}{6} + k2\\pi$ $(k \\in \\mathbb{Z})$.",
  G("set", "\\frac{\\pi}{6}+2k\\pi; \\frac{5\\pi}{6}+2k\\pi"), True, "trig families; hard")
c("H17", "The answer is $\\boxed{\\frac{7}{3}}$.", G("rational", "7/3", locale="en"), True)
c("H18", "So the sum is $1,000 + 234 = \\boxed{1,234}$.", G("int", "1234", locale="en"), True)
c("H19", "So the sum is $\\boxed{1,234}$.", G("int", "1234", locale="vi"), True,
  "English writer means 1234; the vi-locale tie-break reads 1.234")
c("H20", "Tọa độ điểm $M$ là $\\boxed{M(1; -2; 3)}$.", G("tuple", "1; -2; 3"), True)
c("H21", "Vậy $m \\in (-\\infty; -1) \\cup (2; +\\infty)$.", G("interval", "(-\\infty,-1)\\cup(2,\\infty)"), True)
c("H22", "Vậy $m < -1$ hoặc $m > 2$.", G("interval", "(-\\infty,-1)\\cup(2,\\infty)"), True)
c("H23", "Vậy $m \\le -1$ hoặc $m > 2$.", G("interval", "(-\\infty,-1)\\cup(2,\\infty)"), False)
c("H24", "Đáp án: B", G("choice", "B"), True)
c("H25", "Chọn đáp án **D**.", G("choice", "B"), False)
c("H26", "Ta có $\\log_2 8 = 3$. Vậy $\\boxed{3}$.", G("int", "3"), True)
c("H27", "Đạo hàm $f'(x) = 3x^2 - 6x$. Vậy $\\boxed{f'(x) = 3x^2 - 6x}$.", G("expr", "3x^2-6x"), True)
c("H28", "Vậy $\\boxed{f'(x) = 3x^2 + 6x}$.", G("expr", "3x^2-6x"), False)
c("H29", "Nghiệm của phương trình là $x = 2$ hoặc $x = -2$.", G("set", "2; -2"), True)
c("H30", "Nghiệm của phương trình là $x = 2$.", G("set", "2; -2"), False)
c("H31", "Vậy phương trình vô nghiệm.", G("set", ""), True)
c("H32", "Vậy phương trình có nghiệm $x = 0$.", G("set", ""), False)
c("H33", "Số học sinh nam là 18 em, số học sinh nữ là 22 em.\n\nĐáp số: Nam 18 em, nữ 22 em.",
  G("tuple", "18; 22"), True, "prose tuple with names in Vietnamese words")
c("H34", "Vận tốc của ô tô là $\\boxed{60 \\text{ km/h}}$.", G("int", "60"), True)
c("H35", "Diện tích hình tròn là $\\boxed{25\\pi \\text{ cm}^2}$.", G("expr", "25\\pi"), True)
c("H36", "Diện tích hình tròn là $\\boxed{78{,}5 \\text{ cm}^2}$.", G("expr", "25\\pi"), False)
c("H37", "<think>Hmm, có thể là 4... không, tính lại: 2 + 3 = 5.</think>\nĐáp số: $\\boxed{5}$", G("int", "5"), True)
c("H38", "Giá trị của biểu thức là $\\boxed{\\sqrt{2} + 1}$.", G("radical", "1+\\sqrt{2}"), True)
c("H39", "Giá trị của biểu thức là $\\boxed{2{,}414}$.", G("radical", "1+\\sqrt{2}"), False)
c("H40", "Lãi suất là $\\boxed{6\\%}$ một năm.", G("decimal", "0.06"), True)

with open(sys.argv[1], "w", encoding="utf-8") as f:
    for x in C:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")
print(len(C), "held-out cases")
