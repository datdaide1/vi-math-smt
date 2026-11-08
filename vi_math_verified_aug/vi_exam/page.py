"""Exam page -> (metadata, problem blocks, solution blocks).

A loigiaihay exam article has two ``div.box-question`` containers: the first holds the paper ("Đề bài"), the
second the site's worked solution ("Lời giải" / "Lời giải chi tiết"; its title is occasionally wrong, e.g.
"Câu 2", so the role is positional). Inside, content is Word-exported HTML: ``p`` / ``table`` / ``ul`` / ``li``
nodes, sometimes wrapped in ``div.WordSectionN``. Formulas are MathJax source (``\\( … \\)``) kept verbatim.

``blocks()`` flattens a container into an ordered list of *leaf blocks* (document order, never keyed by any
printed number). A ``<br>`` inside a leaf splits it into several lines that keep the same DOM path. Images are
recorded per block; tables become one block (cells joined by " | ", rows by newline) unless they contain
question headers, in which case their cells are descended into.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, NavigableString, Tag

BLOCK_TAGS = {"p", "div", "li", "ul", "ol", "table", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "section",
              "center", "tbody", "thead", "tr", "td", "th", "pre", "figure"}
LEAF_TAGS = {"p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "td", "th"}
SKIP_TAGS = {"script", "style", "noscript", "iframe", "button", "ins", "form", "input"}
IMG_MARK = "⟨IMG⟩"
RE_QHEAD_ANY = re.compile(r"(^|\n)\s*(Câu|Bài|CÂU|BÀI)\s*(\d+|[IVX]+)\b")


@dataclass
class Block:
    i: int
    path: str
    tag: str
    text: str
    imgs: list[str] = field(default_factory=list)


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s).replace("\xa0", " ").replace("​", "")
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    return "\n".join(x.strip() for x in s.split("\n")).strip()


def _inline_text(node: Tag, imgs: list[str]) -> str:
    out = []
    for d in node.children:
        if isinstance(d, NavigableString):
            if d.__class__.__name__ in ("Comment", "Doctype", "Declaration", "ProcessingInstruction"):
                continue
            out.append(str(d))
        elif isinstance(d, Tag):
            if d.name in SKIP_TAGS:
                continue
            if d.name == "br":
                out.append("\n")
            elif d.name == "img":
                src = d.get("data-src") or d.get("src") or ""
                imgs.append(src)
                out.append(f" {IMG_MARK} ")
            elif d.name in ("sup",):
                out.append("^" + _inline_text(d, imgs))
            elif d.name in ("sub",):
                out.append("_" + _inline_text(d, imgs))
            else:
                out.append(_inline_text(d, imgs))
    return "".join(out)


def _has_block_child(t: Tag) -> bool:
    return any(isinstance(c, Tag) and c.name in BLOCK_TAGS for c in t.children)


def _table_text(t: Tag, imgs: list[str]) -> str:
    rows = []
    for tr in t.find_all("tr"):
        cells = [norm_text(_inline_text(td, imgs)).replace("\n", " ") for td in tr.find_all(["td", "th"])]
        rows.append(" | ".join(cells))
    return "\n".join(rows)


def blocks(container: Tag) -> list[Block]:
    out: list[Block] = []

    def emit(path: str, tag: str, text: str, imgs: list[str]):
        text = norm_text(text)
        lines = [x for x in text.split("\n")] if tag != "table" else [text]
        first = True
        for ln in lines:
            if not ln.strip() and not (first and imgs):
                continue
            out.append(Block(len(out), path, tag, ln.strip(), imgs if first else []))
            first = False
        if first and imgs:  # image-only block
            out.append(Block(len(out), path, tag, IMG_MARK, imgs))

    def walk(node: Tag, path: str):
        idx = 0
        run: list = []  # consecutive inline children of a mixed container form one block

        def flush():
            if not run:
                return
            imgs: list[str] = []
            txt = []
            for r in run:
                if isinstance(r, NavigableString):
                    if r.__class__.__name__ == "NavigableString":
                        txt.append(str(r))
                elif r.name == "br":
                    txt.append("\n")
                elif r.name == "img":
                    imgs.append(r.get("data-src") or r.get("src") or "")
                    txt.append(f" {IMG_MARK} ")
                else:
                    txt.append(_inline_text(r, imgs))
            emit(f"{path}/#inline[{idx}]", "text", "".join(txt), imgs)
            run.clear()

        for ch in node.children:
            if isinstance(ch, NavigableString) or (isinstance(ch, Tag) and ch.name not in BLOCK_TAGS
                                                   and ch.name not in LEAF_TAGS and ch.name not in SKIP_TAGS):
                run.append(ch)
                continue
            flush()
            if not isinstance(ch, Tag) or ch.name in SKIP_TAGS:
                continue
            cls = ".".join(ch.get("class", []))
            p = f"{path}/{ch.name}{'.' + cls if cls else ''}[{idx}]"
            idx += 1
            if ch.get("id", "").startswith("question-title-idx"):
                continue
            if "rating-section" in cls or "block_gopy" in cls or ch.get("id", "").startswith("pcinread"):
                continue
            if ch.name == "table":
                imgs: list[str] = []
                tt = _table_text(ch, imgs)
                if RE_QHEAD_ANY.search(tt):  # layout table holding questions -> descend cells
                    for k, td in enumerate(ch.find_all(["td", "th"])):
                        walk(td, f"{p}/td[{k}]")
                else:
                    emit(p, "table", tt, imgs)
                continue
            if ch.name in BLOCK_TAGS and _has_block_child(ch):
                # mixed content: inline text directly inside a container is emitted in place
                walk(ch, p)
                continue
            imgs = []
            emit(p, ch.name, _inline_text(ch, imgs), imgs)
        flush()

    walk(container, "")
    for k, b in enumerate(out):
        b.i = k
    return out


# --------------------------------------------------------------------------- metadata
RE_YEAR_HDR = re.compile(r"NĂM\s+HỌC\s*:?\s*(20\d\d)\s*[-–—]\s*(20\d\d)", re.I)
RE_YEAR_TITLE = re.compile(r"(?:năm\s+)?(20[12]\d)\b", re.I)
UNIVERSITY_SCHOOLS = re.compile(r"(Năng Khiếu|Khoa học Tự nhiên|KHTN|ĐHSP|Sư phạm|Đại học|ĐHQG|Ngoại ngữ|ĐH )", re.I)


def page_title(soup: BeautifulSoup) -> str:
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        return norm_text(og["content"])
    h1 = soup.find("h1")
    return norm_text(h1.get_text(" ")) if h1 else norm_text(soup.title.get_text() if soup.title else "")


def parse_title(title: str) -> dict:
    """'Đề thi vào 10 môn Toán Hà Nội 2026 có đáp án …' -> province/school, year, kind."""
    t = re.sub(r"\s*\|.*$", "", title).strip()           # 'Đề thi … năm 2018 | Đề thi vào 10 môn Toán'
    t = re.sub(r"\s+có\s+(đáp án|lời giải).*$", "", t, flags=re.I).strip()
    t = re.sub(r"\s*-\s*Loigiaihay.*$", "", t, flags=re.I)
    low = t.lower()
    if re.search(r"tham khảo|thi thử|đề số|dự đoán|đề ôn", low):
        kind = "mock"
    elif re.search(r"minh họa|minh hoạ", low):
        kind = "minh_hoa"
    elif re.search(r"học sinh giỏi|hsg", low):
        kind = "hsg9"
    elif re.search(r"trường|chuyên|năng khiếu", low):
        kind = "school"
    else:
        kind = "official"
    years = RE_YEAR_TITLE.findall(t)
    year = int(years[-1]) if years else None
    m = re.search(r"môn\s+Toán\s+(?:lớp\s*9\s+)?(.*?)\s*(?:năm\s+)?(?:học\s+)?(20[12]\d)(?:\s*[-–]\s*20\d\d)?\s*(?:\(.*\))?\s*$", t, re.I)
    place = m.group(1).strip(" -–,") if m else None
    if place:
        place = re.sub(r"^(tỉnh|thành phố|TP\.?)\s+", "", place, flags=re.I).strip()
        place = re.sub(r"\s*\(Vòng.*$", "", place, flags=re.I)
    return {"title_clean": t, "kind": kind, "title_year": year, "place": place}


PROVINCE_CANON = {
    "tp hồ chí minh": "TP Hồ Chí Minh", "thành phố hồ chí minh": "TP Hồ Chí Minh", "hồ chí minh": "TP Hồ Chí Minh",
    "tp. hồ chí minh": "TP Hồ Chí Minh", "tphcm": "TP Hồ Chí Minh", "tp.hcm": "TP Hồ Chí Minh", "hcm": "TP Hồ Chí Minh",
    "thừa thiên huế": "Huế", "huế": "Huế", "bà rịa - vũng tàu": "Bà Rịa - Vũng Tàu", "bà rịa vũng tàu": "Bà Rịa - Vũng Tàu",
}


SCHOOL_PROVINCE = [(r"Hồ Chí Minh|HCM", "TP Hồ Chí Minh"), (r"Khoa học Tự nhiên|KHTN|Hà Nội|ĐHSP Hà Nội", "Hà Nội")]


def canon_province(p: str | None) -> str | None:
    if not p:
        return None
    if re.search(r"trường|THPT|Chuyên|Năng Khiếu", p, re.I):  # school exam: province of the school
        for pat, prov in SCHOOL_PROVINCE:
            if re.search(pat, p, re.I):
                return prov
        return None
    k = p.strip().lower()
    return PROVINCE_CANON.get(k, p.strip())


def parse_page(html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    title = page_title(soup)
    boxes = soup.select("div.box-question")
    prob = blocks(boxes[0]) if boxes else []
    sol = blocks(boxes[1]) if len(boxes) > 1 else []
    box_titles = []
    for b in boxes:
        t = b.select_one('[id^="question-title-idx"]')
        box_titles.append(norm_text(t.get_text(" ")) if t else None)
    head = "\n".join(b.text for b in prob[:25])
    hdr_year = RE_YEAR_HDR.search(head)
    meta = parse_title(title)
    ogimg = soup.find("meta", property="og:image")
    ogimg_date = None
    if ogimg and ogimg.get("content"):
        mm = re.search(r"/(20\d\d)/(\d\d)(\d\d)/", ogimg["content"])
        if mm:
            ogimg_date = f"{mm.group(1)}-{mm.group(2)}-{mm.group(3)}"
    exam_date = None
    md = re.search(r"Ngày thi\s*:?\s*(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(20\d\d)", head, re.I)
    if md:
        exam_date = f"{md.group(3)}-{int(md.group(2)):02d}-{int(md.group(1)):02d}"
    else:
        md = re.search(r"[Nn]gày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(20\d\d)", head)
        if md:
            exam_date = f"{md.group(3)}-{int(md.group(2)):02d}-{int(md.group(1)):02d}"
    official_mark = bool(re.search(r"ĐỀ\s+CHÍNH\s+THỨC|ĐỀ\s+THI\s+CHÍNH\s+THỨC", head, re.I))
    so_gd = re.search(r"SỞ\s+GIÁO\s+DỤC\s+(?:VÀ|&)\s+ĐÀO\s+TẠO\s*\n?\s*([^\n|]{2,40})", head, re.I)
    return {
        "title": title, **meta,
        "header_year": int(hdr_year.group(1)) if hdr_year else None,
        "official_mark": official_mark,
        "header_issuer": norm_text(so_gd.group(0)).replace("\n", " ") if so_gd else None,
        "exam_date": exam_date, "og_image_date": ogimg_date,
        "box_titles": box_titles, "problem_blocks": prob, "solution_blocks": sol,
    }


def issuer_type(kind: str, place: str | None, header_issuer: str | None) -> str:
    if kind == "mock":
        return "private"
    if kind == "school" and place and UNIVERSITY_SCHOOLS.search(place):
        return "university"
    return "state"
