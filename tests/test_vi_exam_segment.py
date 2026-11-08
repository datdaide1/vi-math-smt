"""Vi-ExamMath segmentation tests (X2).

Reproduces R6's X-SEG-COLLISION: the prototype (`_scratch/vi_exam/scrape_loigiaihay.py::segment`) stored
questions in a dict keyed by "Câu N", so in the 2026 three-part format (PHẦN I / II / III, numbering restarts)
later parts overwrote earlier ones (Lào Cai 2026: 22 questions on the page, 12 kept). The new segmenter keeps a
list keyed by DOM order and must keep every question and sub-question.
"""
from __future__ import annotations

import re

import pytest
from bs4 import BeautifulSoup

from vi_math_verified_aug.vi_exam.crawl import read_cache
from vi_math_verified_aug.vi_exam.page import blocks, parse_page
from vi_math_verified_aug.vi_exam.segment import (align, flat_questions, parse_questions, solution_chunks,
                                                  split_by_labels, worked_part)

EXAM_2026 = """
<div class="box-question" id="sub-question-1"><p id="question-title-idx-1">Đề bài</p>
<div class="WordSection1">
<table><tr><td>SỞ GIÁO DỤC VÀ ĐÀO TẠO X<br>ĐỀ CHÍNH THỨC</td><td>KỲ THI TUYỂN SINH VÀO LỚP 10 NĂM HỌC 2026 – 2027</td></tr></table>
<p>PHẦN I. Câu trắc nghiệm nhiều phương án lựa chọn.</p>
<p>Câu 1. \\(\\sqrt {64} \\) bằng:</p>
<p>A. \\(8\\).</p><p>B. \\(-8\\).</p><p>C. 32.</p><p>D. 64.</p>
</div>
<div class="WordSection2">
<p>Câu 2. Phương trình nào là phương trình bậc hai?</p>
<p>A. \\(x+3y=2\\). B. \\(3x^2+2x-2026=0\\). C. \\(5x-1=0\\). D. \\(x-\\sqrt x=0\\).</p>
<p>Câu 3. Hệ số góc của đường thẳng \\(y=-3x+5\\) là:</p>
<p>A. 3.</p><p>B. -3.</p><p>C. 5.</p><p>D. -5.</p>
</div>
<p>PHẦN II. Câu trắc nghiệm đúng sai.</p>
<p>Câu 1. Cho phương trình \\(x^2+7x+5=0\\) (1).</p>
<p>a) Hệ số \\(b\\) là \\(7\\).</p><p>b ) Biệt thức \\(\\Delta = 29\\).</p>
<p>c) \\(x=-7+\\sqrt{29}\\) là một nghiệm.</p><p>d) \\(x_1+x_2=-7\\).</p>
<p>Câu 2. Cho biểu thức \\(P=\\frac{1}{\\sqrt x-2}\\).</p>
<p>a) P không xác định tại \\(x=4\\).</p><p>b) Khi \\(x=9\\) thì \\(P=1\\).</p>
<p>c) P xác định khi \\(x \\ge 0\\).</p><p>d) \\(P>0\\) khi \\(x>4\\).</p>
<p>PHẦN III. Câu trắc nghiệm trả lời ngắn.</p>
<p>Câu 1. Tính giá trị của \\(A=\\sqrt{12}\\cdot\\sqrt{3}\\).</p>
<p>Câu 2: Giải phương trình \\(x^2-5x+6=0\\), nhập nghiệm lớn hơn.</p>
<p>- HẾT -</p>
Ad Tag - PC_Inread_ 1/3 bài2026_loigiaihay.com - ZMEDIA
</div>
<div class="box-question" id="sub-question-2"><p id="question-title-idx-2">Lời giải</p>
<p>PHẦN III</p>
<p>Câu 1. Tính giá trị của \\(A=\\sqrt{12}\\cdot\\sqrt{3}\\).</p><p>Cách giải:</p><p>Vậy \\(A = 6\\).</p>
<p>Câu 2: Giải phương trình \\(x^2-5x+6=0\\), nhập nghiệm lớn hơn.</p><p>Cách giải:</p><p>Đáp án: 3</p>
</div>
"""


def legacy_segment(block_text: str) -> dict:
    """Verbatim logic of the prototype's segment(): dict keyed by 'Câu N' (the R6 bug)."""
    cau_re = re.compile(r"^\s*(Câu|Bài)\s+([IVX0-9]+)\b", re.I)
    segs, cur, curkey = {}, [], None
    for ln in block_text.split("\n"):
        m = cau_re.match(ln)
        if m:
            if curkey:
                segs[curkey] = "\n".join(cur).strip()
            curkey = f"{m.group(1).title()} {m.group(2).upper()}"
            cur = [ln]
        else:
            cur.append(ln)
    if curkey:
        segs[curkey] = "\n".join(cur).strip()
    return segs


def _problem_blocks(html: str):
    soup = BeautifulSoup(html, "lxml")
    return blocks(soup.select("div.box-question")[0])


def test_legacy_dict_loses_repeated_numbering():
    text = "\n".join(b.text for b in _problem_blocks(EXAM_2026))
    legacy = legacy_segment(text)
    # 7 questions on the page (3 + 2 + 2), but only 3 distinct "Câu N" keys survive
    assert len(legacy) == 3
    assert "x^2+7x+5" not in legacy["Câu 1"]  # PHẦN II Câu 1 was overwritten by PHẦN III Câu 1


def test_dom_segmentation_keeps_every_question():
    secs = parse_questions(_problem_blocks(EXAM_2026))
    assert [s.stype for s in secs] == ["mcq", "true_false", "short_answer"]
    assert [len(s.questions) for s in secs] == [3, 2, 2]
    qs = [q for _, _, q in flat_questions(secs)]
    assert [q.number for q in qs] == [1, 2, 3, 1, 2, 1, 2]
    # MCQ options attached (incl. options inlined in one paragraph)
    assert set(qs[0].options) == set("ABCD") and set(qs[1].options) == set("ABCD")
    # true/false statements are separate sub-items linked to their parent
    assert [c.label for c in qs[3].children] == ["a", "b", "c", "d"]
    assert "x^2+7x+5" in qs[3].own_text()
    # junk after HẾT and the header table are dropped
    assert all("ZMEDIA" not in q.full_text() for q in qs)
    leaves = sum(max(1, len(q.children)) for q in qs)
    assert leaves == 3 + 8 + 2


def test_solution_alignment_with_restarted_numbers():
    soup = BeautifulSoup(EXAM_2026, "lxml")
    secs = parse_questions(blocks(soup.select("div.box-question")[0]))
    qs = [q for _, _, q in flat_questions(secs)]
    chunks, _ = solution_chunks(blocks(soup.select("div.box-question")[1]))
    al = align(qs, chunks)
    # the two solution chunks ("Câu 1", "Câu 2" of PHẦN III) must join PHẦN III, not PHẦN I
    assert [a for a, _ in al] == [None, None, None, None, None, 0, 1]


def test_nested_subquestions_and_solution_split():
    html = """<div class="box-question"><p>Bài III (2 điểm)</p>
    <p>1) Giải phương trình \\({x^4} - 7{x^2} - 18 = 0.\\)</p>
    <p>2) Cho \\((d): y = 2mx - m^2 + 1\\) và \\((P): y = x^2\\).</p>
    <p>a) Chứng minh (d) luôn cắt (P) tại hai điểm phân biệt.</p>
    <p>b) Tìm m để \\(x_1 + x_2 = 4\\).</p>
    <p>Bài IV (3 điểm) Cho tam giác ABC.</p></div>"""
    secs = parse_questions(_problem_blocks(html))
    q = secs[0].questions[0]
    assert q.number == 3 and [c.label for c in q.children] == ["1", "2"]
    assert [g.label for g in q.children[1].children] == ["a", "b"]
    assert secs[0].questions[1].number == 4
    worked = worked_part(["Bài III", "1) Giải ...", "Phương pháp:", "1) đặt t", "Cách giải:",
                          "1) Đặt t = x^2", "Vậy S = {-3; 3}", "2) a) ...", "b) Vậy m = 2"])
    parts = split_by_labels(worked, ["1", "2"])
    assert parts["1"][-1] == "Vậy S = {-3; 3}" and parts["2"][-1] == "b) Vậy m = 2"


LAO_CAI_2026 = "https://loigiaihay.com/de-thi-vao-10-mon-toan-lao-cai-2026-co-dap-an-va-loi-giai-chi-tie-a194508.html"


def test_real_page_lao_cai_2026_keeps_22_questions():
    html, _ = read_cache(LAO_CAI_2026)
    if not html:
        pytest.skip("cached page not available")
    P = parse_page(html)
    legacy = legacy_segment("\n".join(b.text for b in P["problem_blocks"]))
    secs = parse_questions(P["problem_blocks"])
    assert len(legacy) == 12                                  # R6: "live page has 22 questions, 12 kept"
    assert [len(s.questions) for s in secs] == [12, 4, 6]     # fixed: all 22 kept
    assert sum(len(q.children) for s in secs for q in s.questions) == 16  # 4 T/F questions × a–d
