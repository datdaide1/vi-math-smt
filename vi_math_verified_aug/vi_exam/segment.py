"""Segment the ordered DOM blocks of an exam page into a question tree, and align the worked solution.

Design (fixes R6 X-SEG-COLLISION, where the prototype keyed a dict by "Câu N" and the 2026 three-part format,
whose numbering restarts in PHẦN I / II / III, overwrote earlier questions):
  * Questions are a **list in document order**. Identity = (section ordinal, position in section), never the
    printed number. The printed label is kept only as metadata.
  * Boundaries come from DOM leaf blocks (``page.blocks``): a block that *starts* with a section header,
    a question header (Câu/Bài + Arabic/Roman), a sub-question marker (``a)``, ``1)``) or an MCQ option opens
    a new node; any other block is appended to the deepest open node.
  * Sub-questions become child nodes (two levels: e.g. ``2)`` -> ``a)``). The first marker style seen in a
    question is level 1; the other style nests under the last level-1 node.
  * Solutions: the solution container is cut into question chunks the same way, then aligned to the problem
    questions by a monotonic DP over text similarity (+ a bonus when printed numbers agree), so restarting
    numbers and "Bài IV" vs "Câu 4" mismatches cannot mis-join. Inside an aligned chunk the worked part
    (after "Cách giải" / "Lời giải") is split by the sub-question labels of the problem node.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

from .page import IMG_MARK, Block

ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10}

RE_SECTION = re.compile(
    r"^\s*(?:(?:PHẦN|Phần|phần)\s+(?:[IVX]{1,4}|\d|[A-C])\b|"
    r"(?:[IVX]{1,4}|[A-C])\s*[\.\-–:]\s*(?:PHẦN\s+)?(?:TRẮC\s+NGHIỆM|TỰ\s+LUẬN|Trắc\s+nghiệm|Tự\s+luận)|"
    r"(?:PHẦN\s+)?(?:TRẮC\s+NGHIỆM|TỰ\s+LUẬN)\s*(?:\(|:|$))")
RE_QHEAD = re.compile(
    r"^\s*(?:Câu|Bài|CÂU|BÀI)\s*(\d{1,2}|[IVX]{1,4})(?![\w\d])\s*"
    r"((?:\([^)]{0,40}\)\s*)*)\s*[\.:\-–)]?\s*")
RE_POINTS = re.compile(r"(\d+(?:[,\.]\d+)?)\s*(?:điểm|đ)\b", re.I)
RE_SUB_LETTER = re.compile(r"^\s*\(?([a-eđ])\s*\)\s*|^\s*([a-e])\s*\.\s*(?=\\\(|\s+\S)\s*")
RE_SUB_DIGIT = re.compile(r"^\s*\(?(\d{1,2}(?:\.\d)?)\s*\)\s*|^\s*(\d\.\d)\s+(?=\S)|"
                          r"^\s*(\d{1,2})\s*\.\s*(?=\\\(|\s+[^\d\s])\s*")
RE_OPTION = re.compile(r"^\s*([A-D])\s*[\.\)]\s*")
RE_OPTION_INLINE = re.compile(r"(?:(?<=\s)|^)([A-D])\s*\.\s+")
RE_NOISE = re.compile(
    r"^(Đề bài|Lời giải( chi tiết)?|[-–—\s]*HẾT[-–—\s\.]*|.*Ad Tag.*|.*ZMEDIA.*|Họ (và )?tên.*|Số báo danh.*|"
    r"Cán bộ coi thi.*|Chữ (ký|kí).*|\(?Đề thi (gồm|có).*|\(?Thí sinh không.*|Giám thị.*|.*loigiaihay\.com.*|"
    r"HƯỚNG DẪN GIẢI.*|THỰC HIỆN:.*|MÔN TOÁN\s*[-–]?\s*)$", re.I)
RE_HEADER_TABLE = re.compile(r"SỞ GIÁO DỤC|KỲ THI|KÌ THI|ĐỀ CHÍNH THỨC|Thời gian làm bài|Môn thi", re.I)
RE_SOL_START = re.compile(r"^\s*(Cách giải|Lời giải|Giải|Hướng dẫn giải|Lời giải chi tiết)\s*[:\.]?\s*$", re.I)
RE_METHOD = re.compile(r"^\s*(Phương pháp( giải)?|Phân tích)\s*[:\.]?", re.I)


def qnum(s: str) -> int | None:
    s = s.strip().upper()
    if s.isdigit():
        return int(s)
    return ROMAN.get(s)


def section_type(title: str) -> str:
    t = title.lower()
    if "đúng sai" in t or "đúng/sai" in t or "đúng - sai" in t:
        return "true_false"
    if "trả lời ngắn" in t:
        return "short_answer"
    if "trắc nghiệm" in t:
        return "mcq"
    if "tự luận" in t:
        return "essay"
    return "unknown"


@dataclass
class Node:
    kind: str                  # question | sub
    label: str                 # printed label, e.g. "Câu 3", "a", "2"
    level: int                 # 0 question, 1 sub, 2 sub-sub
    lines: list[str] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)
    imgs: list[str] = field(default_factory=list)
    options: dict[str, str] = field(default_factory=dict)
    children: list["Node"] = field(default_factory=list)
    style: str | None = None   # marker style of children at level 1: letter|digit
    number: int | None = None
    points: str | None = None
    has_table: bool = False

    def own_text(self) -> str:
        return "\n".join(self.lines).strip()

    def full_text(self) -> str:
        parts = [self.own_text()]
        for c in self.children:
            parts.append(f"{c.label}) {c.full_text()}")
        return "\n".join(p for p in parts if p).strip()


@dataclass
class Section:
    ordinal: int
    title: str
    stype: str
    preamble: list[str] = field(default_factory=list)
    questions: list[Node] = field(default_factory=list)


def _split_inline_options(text: str) -> list[tuple[str, str]] | None:
    """'A. 8. B. -8. C. 32. D. 64.' -> [(A,'8.'),…] when >=2 option letters appear in order."""
    hits = [m for m in RE_OPTION_INLINE.finditer(text)]
    letters = [m.group(1) for m in hits]
    if len(hits) >= 2 and letters == sorted(letters) and letters[0] == "A" and text.lstrip().startswith("A"):
        out = []
        for k, m in enumerate(hits):
            end = hits[k + 1].start() if k + 1 < len(hits) else len(text)
            out.append((m.group(1), text[m.end():end].strip()))
        return out
    return None


def _sub_marker(text: str):
    m = RE_SUB_LETTER.match(text)
    if m:
        return "letter", (m.group(1) or m.group(2)), text[m.end():]
    m = RE_SUB_DIGIT.match(text)
    if m:
        return "digit", (m.group(1) or m.group(2) or m.group(3)), text[m.end():]
    return None


def _next_label(lab: str) -> str | None:
    if re.fullmatch(r"[a-d]", lab):
        return chr(ord(lab) + 1)
    if re.fullmatch(r"\d{1,2}", lab):
        return str(int(lab) + 1)
    m = re.fullmatch(r"(\d)\.(\d)", lab)
    if m:
        return f"{m.group(1)}.{int(m.group(2)) + 1}"
    return None


def split_inline_subs(text: str) -> list[str]:
    """'a.\\(…\\) b.\\(…\\) c.\\(…\\)' or 'a) … b) …' on one line -> one line per sub-question.
    Only consecutive labels are split, and only outside formulas."""
    sm = _sub_marker(text)
    if not sm:
        return [text]
    lab = sm[1]
    out, start, pos = [], 0, 0
    while True:
        nl = _next_label(lab)
        if nl is None:
            break
        pat = re.compile(rf"(?:(?<=[\s\.;,])|(?<=\\\)))\(?{re.escape(nl)}\s*[\)\.]\s*(?=\S)")
        found = None
        for m in pat.finditer(text, pos + 1):
            before = text[:m.start()]
            if before.count("\\(") == before.count("\\)"):  # not inside a formula
                found = m
                break
        if not found:
            break
        out.append(text[start:found.start()].strip())
        start = pos = found.start()
        lab = nl
    out.append(text[start:].strip())
    return [o for o in out if o]


def parse_questions(blocks: list[Block], is_solution: bool = False) -> list[Section]:
    sections: list[Section] = []
    cur_q: Node | None = None
    cur_l1: Node | None = None
    cur_l2: Node | None = None

    def sec() -> Section:
        if not sections:
            sections.append(Section(0, "", "unknown"))
        return sections[-1]

    for b in blocks:
        t = b.text.strip()
        if not t:
            continue
        if not is_solution and re.match(r"^[-–—\s\.]*HẾT[-–—\s\.]*$", t, re.I):
            break
        if b.tag == "table" and RE_HEADER_TABLE.search(t) and cur_q is None:
            continue
        if RE_NOISE.match(t) and t != IMG_MARK:
            continue
        if cur_q is None and not sections and RE_HEADER_TABLE.search(t) and len(t) < 120:
            continue
        # --- section header
        if RE_SECTION.match(t) and not RE_QHEAD.match(t) and len(t) < 400:
            sections.append(Section(len(sections), t, section_type(t)))
            cur_q = cur_l1 = cur_l2 = None
            continue
        # --- question header
        m = RE_QHEAD.match(t)
        if m and qnum(m.group(1)) is not None:
            rest = t[m.end():].strip()
            mp = re.match(r"^\(\s*(\d+(?:[,\.]\d+)?)\s*(?:điểm|đ)\s*\)\s*[\.:\-–]?\s*", rest, re.I)
            if mp:  # "Câu I. (1,5 điểm)": points after the punctuation
                rest = rest[mp.end():].strip()
            pts = RE_POINTS.search(m.group(2) or "") or mp
            cur_q = Node("question", t[:m.end()].strip(" .:-–"), 0, number=qnum(m.group(1)),
                         points=pts.group(1) if pts else None)
            cur_q.block_ids.append(b.i)
            sec().questions.append(cur_q)
            cur_l1 = cur_l2 = None
            if rest:
                t = rest  # fall through: the remainder may itself be a sub marker / stem text
            else:
                continue
        if cur_q is None:
            sec().preamble.append(t)
            continue
        # --- MCQ options
        opts = _split_inline_options(t)
        mo = RE_OPTION.match(t)
        if opts or (mo and not _sub_marker(t)):
            if opts is None:
                opts = [(mo.group(1), t[mo.end():].strip())]
            for k, v in opts:
                cur_q.options[k] = v
            cur_q.block_ids.append(b.i)
            cur_q.imgs.extend(b.imgs)
            continue
        # --- sub-question markers
        if _sub_marker(t):
            for part in split_inline_subs(t):
                sm = _sub_marker(part)
                if not sm:  # defensive: keep text with the deepest open node
                    (cur_l2 or cur_l1 or cur_q).lines.append(part)
                    continue
                style, lab, rest = sm
                if cur_q.style is None:
                    cur_q.style = style
                if style == cur_q.style:
                    cur_l1 = Node("sub", lab, 1)
                    cur_q.children.append(cur_l1)
                    cur_l2 = None
                    target = cur_l1
                elif cur_l1 is not None:
                    cur_l2 = Node("sub", lab, 2)
                    cur_l1.children.append(cur_l2)
                    target = cur_l2
                else:  # other style before any level-1 node: treat as level 1 anyway
                    cur_l1 = Node("sub", lab, 1)
                    cur_q.children.append(cur_l1)
                    target = cur_l1
                if rest.strip():
                    target.lines.append(rest.strip())
                target.block_ids.append(b.i)
                target.imgs.extend(b.imgs)
                target.has_table |= b.tag == "table"
            continue
        target = cur_l2 or cur_l1 or cur_q
        target.lines.append(t)
        target.block_ids.append(b.i)
        target.imgs.extend(b.imgs)
        target.has_table |= b.tag == "table"
    return sections


def flat_questions(sections: list[Section]) -> list[tuple[Section, int, Node]]:
    return [(s, k, q) for s in sections for k, q in enumerate(s.questions)]


# --------------------------------------------------------------------------- solutions
def _norm(s: str) -> str:
    s = re.sub(r"\\[a-zA-Z]+|[{}\\\s]", "", s.lower())
    return s[:220]


def _sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


@dataclass
class SolChunk:
    label: str
    number: int | None
    lines: list[str]
    imgs: list[str]


def solution_chunks(blocks: list[Block]) -> tuple[list[SolChunk], list[str]]:
    """Cut the solution container at question headers. Returns (chunks, preamble lines)."""
    chunks: list[SolChunk] = []
    pre: list[str] = []
    for b in blocks:
        t = b.text.strip()
        if not t or (RE_NOISE.match(t) and t != IMG_MARK):
            continue
        m = RE_QHEAD.match(t)
        if m and qnum(m.group(1)) is not None:
            chunks.append(SolChunk(t[:m.end()].strip(" .:-–"), qnum(m.group(1)), [], []))
            rest = t[m.end():].strip()
            if rest:
                chunks[-1].lines.append(rest)
            continue
        if chunks:
            chunks[-1].lines.append(t)
            chunks[-1].imgs.extend(b.imgs)
        else:
            pre.append(t)
    return chunks, pre


def align(problem_qs: list[Node], chunks: list[SolChunk]) -> list[tuple[int | None, float]]:
    """Monotonic DP alignment problem question i -> chunk j (or None). Score = text sim + 0.25·[same number]."""
    n, m = len(problem_qs), len(chunks)
    pa = [_norm(q.full_text() + " " + " ".join(q.options.values())) for q in problem_qs]
    cb = [_norm(" ".join(c.lines[:6])) for c in chunks]
    S = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            s = _sim(pa[i][:160], cb[j][:220])
            if problem_qs[i].number is not None and problem_qs[i].number == chunks[j].number:
                s += 0.25
            S[i][j] = s
    NEG = -1e9
    dp = [[NEG] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]
    for j in range(m + 1):
        dp[0][j] = 0.0
    for i in range(1, n + 1):
        dp[i][0] = 0.0
        bt[i][0] = "up"
        for j in range(1, m + 1):
            opts = [(dp[i - 1][j], "up"), (dp[i][j - 1], "left")]
            if S[i - 1][j - 1] >= 0.2:
                opts.append((dp[i - 1][j - 1] + S[i - 1][j - 1], "diag"))
            dp[i][j], bt[i][j] = max(opts, key=lambda x: x[0])
    res: list[tuple[int | None, float]] = [(None, 0.0)] * n
    i, j = n, m
    while i > 0 and j > 0:
        mv = bt[i][j]
        if mv == "diag":
            res[i - 1] = (j - 1, round(S[i - 1][j - 1], 3))
            i, j = i - 1, j - 1
        elif mv == "up":
            i -= 1
        else:
            j -= 1
    return res


def worked_part(lines: list[str]) -> list[str]:
    """Drop the restated problem + 'Phương pháp' block; keep what follows the last 'Cách giải'/'Lời giải'."""
    idx = None
    for k, ln in enumerate(lines):
        if RE_SOL_START.match(ln) or re.match(r"^\s*(Cách giải|Lời giải)\s*:", ln, re.I):
            idx = k
    if idx is not None:
        out = lines[idx + 1:]
        first = re.sub(r"^\s*(Cách giải|Lời giải)\s*:\s*", "", lines[idx], flags=re.I).strip()
        return ([first] if first else []) + out
    return lines


def split_by_labels(lines: list[str], labels: list[str]) -> dict[str, list[str]]:
    """Cut the worked part into per-sub chunks by the (ordered) labels of the problem's children."""
    out: dict[str, list[str]] = {}
    if not labels:
        return out
    starts: list[tuple[int, str]] = []
    k0 = 0
    for lab in labels:
        pat = re.compile(rf"^\s*\(?{re.escape(lab)}\s*[\)\.]")
        for k in range(k0, len(lines)):
            if pat.match(lines[k]):
                starts.append((k, lab))
                k0 = k + 1
                break
    for n_, (k, lab) in enumerate(starts):
        end = starts[n_ + 1][0] if n_ + 1 < len(starts) else len(lines)
        out[lab] = lines[k:end]
    return out


def parse_key_table(chunks_pre: list[str], all_lines: list[str]) -> dict[int, str]:
    """MCQ key such as '1.B 2.C …' or a 'Câu | 1 | 2 …' / 'Đáp án | B | C …' table. Returns {number: letter}."""
    key: dict[int, str] = {}
    lines = chunks_pre + all_lines
    for k, ln in enumerate(lines):
        pairs = re.findall(r"(?<!\d)(\d{1,2})\s*[\.\-:–]\s*([A-D])\b", ln)
        if len(pairs) >= 3:
            for n_, l_ in pairs:
                key.setdefault(int(n_), l_)
        if "|" in ln and re.match(r"^\s*(Câu|Câu hỏi)\s*\|", ln, re.I) and k + 1 < len(lines):
            nums = [c.strip() for c in ln.split("|")[1:]]
            nxt = lines[k + 1]
            if re.match(r"^\s*(Đáp án|ĐA)\s*\|", nxt, re.I):
                lets = [c.strip() for c in nxt.split("|")[1:]]
                for a, b in zip(nums, lets):
                    if a.isdigit() and re.fullmatch(r"[A-D]", b):
                        key.setdefault(int(a), b)
    return key
