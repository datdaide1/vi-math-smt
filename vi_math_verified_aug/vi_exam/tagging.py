"""Rule-based item tags: has_figure, answer_type, topic, symbolically_certifiable.

All rules are keyword/regex rules over the Vietnamese text (LaTeX kept). They are *provisional*: the QA queue
(T5 §3.1 stage 3–4) is where wrong tags get fixed. Each rule set is ordered; the first hit wins for the primary
label, and every hit is kept in ``topics``.

``symbolically_certifiable`` ∈ {yes, maybe, no} (01_TECHNICAL_DESIGN C1 / P3):
  yes   – equation / system / quadratic / Vi-ét / inequation / function or expression value, gradable answer,
          no geometry, no proof (a SymPy/Z3 program can certify the key);
  maybe – word problems (need modelling), counting / probability / statistics, extremum questions;
  no    – proofs, synthetic geometry, construction / drawing, figure-dependent, MCQ about concepts.
"""
from __future__ import annotations

import re

from .page import IMG_MARK

RE_FIG_TEXT = re.compile(
    r"(hình vẽ|như hình|hình bên|hình dưới|hình sau|(?<!\w)Hình\s*\d|biểu đồ|quan sát hình|xem hình|"
    r"đồ thị (?:dưới|bên|sau|như hình|ở hình|trong hình|đã cho)|minh hoạ|minh họa|trong hình|ở hình)", re.I)
RE_PROOF = re.compile(r"(chứng minh|chứng tỏ|\bCMR\b|giải thích vì sao)", re.I)
RE_CONSTRUCT = re.compile(r"(^|\W)(vẽ|lập bảng|biểu diễn .* trên (trục|mặt phẳng)|xác định (vị trí|tâm))", re.I)
RE_SIMPLIFY = re.compile(r"(rút gọn|thu gọn|viết .* dưới dạng)", re.I)
RE_SOLVE_EQ = re.compile(r"(giải (các |mỗi )?(phương trình|hệ phương trình|hệ|bất phương trình|pt|hpt)|"
                         r"tìm (các )?nghiệm (của|còn lại)?)", re.I)
RE_NAMED_DEF = re.compile(r"\\\(\s*([A-Z])\s*(?:\(x\))?\s*=")


def multi_target(ask: str) -> bool:
    """'Tính A = …; B = …' / 'Tính S = x1 + x2 và P = x1x2': two answers in one unit (to split in QA)."""
    names = set(RE_NAMED_DEF.findall(ask))
    return len(names) >= 2 and bool(re.search(r"tính|rút gọn|thu gọn|giá trị", ask, re.I))
RE_PARAM = re.compile(r"(tìm (tất cả )?(các )?(giá trị )?(của )?(tham số )?\\?\(?\s*[mkab]\s*\\?\)?\s*(để|sao cho|thỏa|thoả|biết|khi))|"
                      r"tìm (tất cả )?(các )?giá trị (của )?\\?\(?\s*m\b", re.I)
RE_NUMERIC_ASK = re.compile(r"(tính|tìm|hỏi|bao nhiêu|bằng bao|là bao|xác định|cho biết|có mấy|giá trị (lớn|nhỏ) nhất|"
                            r"ước tính|làm tròn|kết quả)", re.I)
RE_EXTREMUM = re.compile(r"(giá trị (lớn nhất|nhỏ nhất)|GTLN|GTNN|max|min)", re.I)
RE_INEQ_PROOF = re.compile(r"(chứng minh .*(≥|≤|\\ge|\\le|\\geqslant|\\leqslant|>|<|bất đẳng thức))", re.I)

TOPIC_RULES = [
    ("vieta", r"vi-?ét|viète|vi-et|{x_1}|x_1|x_{1}|x1\s*,\s*x2|hai nghiệm .*thỏa|hai nghiệm .*thoả"),
    ("system", r"hệ phương trình|\\left\\{|\\begin\{cases\}|\\begin\{array\}.*\\\\.*\\end\{array\}\s*\\right\."),
    ("inequation", r"bất phương trình"),
    ("quadratic", r"bậc hai|phương trình .*x\^2|{x\^2}|x\^\{?2\}?|\\Delta|biệt thức"),
    ("equation", r"phương trình|giải pt"),
    ("function", r"hàm số|đồ thị|parabol|\(P\)|\(d\)|đường thẳng .*y\s*=|y\s*=\s*.*x|tọa độ|toạ độ|hệ số góc"),
    ("expression", r"biểu thức|rút gọn|\\sqrt|căn"),
    ("probability_statistics", r"xác suất|biến cố|tần số|tần suất|ghép nhóm|thống kê|không gian mẫu|ngẫu nhiên|"
                               r"biểu đồ|số trung bình|mẫu số liệu"),
    ("solid_geometry", r"hình trụ|hình nón|hình cầu|hình chóp|lăng trụ|hình hộp|thể tích|mặt cầu|bán kính đáy"),
    ("plane_geometry", r"tam giác|đường tròn|tứ giác|tiếp tuyến|\bgóc\b|nội tiếp|ngoại tiếp|"
                       r"hình (vuông|chữ nhật|thoi|thang) \\\(\s*[A-Z]{4}|"
                       r"đường kính|dây cung|\bcung\b|trung điểm|vuông góc|song song với|\btia\b|đoạn thẳng"),
    ("word_problem", r"giải bài toán bằng cách lập|lập phương trình|lập hệ|một (người|ô tô|xe|công ty|cửa hàng|"
                     r"đội|lớp|trường|mảnh|khu|thửa)|giá (bán|niêm yết)|lãi suất|vận tốc|năng suất|km/h|đồng\b|"
                     r"sản phẩm|học sinh|mảnh (đất|vườn)|thửa ruộng|chiều (dài|rộng)"),
    ("inequality_extremum", r"bất đẳng thức|giá trị (lớn nhất|nhỏ nhất)|GTLN|GTNN"),
    ("number_theory", r"số nguyên|chia hết|số tự nhiên|số chính phương|ước|bội|nguyên tố"),
]
TOPIC_RE = [(n, re.compile(p, re.I)) for n, p in TOPIC_RULES]
ALGEBRA = {"vieta", "system", "inequation", "quadratic", "equation", "function", "expression"}


def has_figure(text: str, imgs: list[str]) -> bool:
    return bool(imgs) or IMG_MARK in text or bool(RE_FIG_TEXT.search(text))


def topics(text: str) -> list[str]:
    return [n for n, r in TOPIC_RE if r.search(text)]


def answer_type(own: str, context: str, section_type: str, n_options: int) -> str:
    """own = the sub-question's own text (the ask); context = stem + own."""
    if section_type == "true_false":
        return "true_false"
    if n_options >= 2:
        return "mcq"
    ask = own or context
    proof = bool(RE_PROOF.search(ask))
    ask_num = bool(RE_NUMERIC_ASK.search(re.sub(r"chứng minh.*", "", ask, flags=re.I)))
    if proof and not (ask_num and re.search(r"\b(và|rồi)\s+(tính|tìm)", ask, re.I)):
        return "proof"
    if proof:
        return "mixed_proof"
    if RE_CONSTRUCT.search(ask) and not RE_NUMERIC_ASK.search(re.sub(r"(^|\W)vẽ.*?(\.|;|$)", " ", ask, flags=re.I)):
        return "construction"
    if RE_SIMPLIFY.search(ask):
        return "expression"
    if RE_SOLVE_EQ.search(re.sub(r"không (cần )?giải (phương trình|hệ)", "", ask, flags=re.I)):
        return "solution_set"
    if RE_PARAM.search(ask):
        return "parameter_condition"
    if section_type == "short_answer":
        return "numeric"
    if ask_num:
        return "numeric"
    return "open"


GRADABLE = {"numeric", "expression", "solution_set", "parameter_condition"}


def certifiable(ans_type: str, tps: list[str], fig: bool) -> str:
    if ans_type in ("proof", "mixed_proof", "construction", "open", "true_false") or fig:
        return "no"
    geo = {"plane_geometry", "solid_geometry"} & set(tps)
    alg = ALGEBRA & set(tps)
    if ans_type == "mcq":
        return "maybe" if alg and not geo else "no"
    if geo and not alg:
        return "no" if "plane_geometry" in geo else "maybe"
    if "word_problem" in tps or "probability_statistics" in tps or "inequality_extremum" in tps:
        return "maybe"
    if alg:
        return "yes"
    if "number_theory" in tps:
        return "maybe"
    return "maybe"


def primary_topic(tps: list[str]) -> str:
    for pref in ("plane_geometry", "solid_geometry", "probability_statistics", "word_problem", "vieta", "system",
                 "inequation", "quadratic", "equation", "function", "inequality_extremum", "expression",
                 "number_theory"):
        if pref in tps:
            return pref
    return "other"
