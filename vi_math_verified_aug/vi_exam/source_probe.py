"""Probe candidate sources for the 2024 / HSG Toán 9 / chuyên gaps WITHOUT crawling them (X2, 2026-11-07).

Per host: robots.txt (cached, via ``crawl.Fetcher``) + at most one sample page found by web search. For the sample we
measure whether the problem text is HTML text with LaTeX/MathML or only images/PDF embeds. Output:
``data/vi_exam/source_probe.jsonl``.

  python -m vi_math_verified_aug.vi_exam.source_probe
"""
from __future__ import annotations

import json
import re
import urllib.parse

from bs4 import BeautifulSoup

from .crawl import DATA, USER_AGENT, Fetcher

CANDIDATES = [
    # (source, role, sample url found by web search; None = robots only)
    ("thi.tuyensinh247.com", "2024 Toán vào 10, all provinces + chuyên",
     "https://thi.tuyensinh247.com/dap-an-de-thi-vao-lop-10-mon-toan-da-nang-2024-c29a81475.html"),
    ("hoatieu.vn", "2024/2026 đáp án by province; HSG Toán 9 cấp tỉnh",
     "https://hoatieu.vn/de-thi-hoc-sinh-gioi-toan-9-cap-tinh-quang-nam-244959"),
    ("vietjack.com", "HSG Toán 9 by Sở (2025-2026)",
     "https://vietjack.com/de-thi/de-thi-hoc-sinh-gioi-toan-9-so-gddt-hung-yen-nam-2025-2026.jsp"),
    ("vndoc.com", "HSG Toán 9 collections", "https://vndoc.com/50-de-thi-hoc-sinh-gioi-lop-9-mon-toan-cap-huyen-tinh-223609"),
    ("thcs.toanmath.com", "HSG Toán 9 cấp tỉnh 2023-2026 (index)",
     "https://thcs.toanmath.com/2025/03/de-chon-hoc-sinh-gioi-tinh-toan-9-nam-2024-2025-so-gddt-tuyen-quang.html"),
    ("2048.vn", "Ôn vào 10 Toán (incl. Hà Nội 2023-2024)", "https://2048.vn/on-vao-10/toan?page=12"),
    ("hoc247.net", "chuyên Toán (Sở HCM 2019) / thi thử",
     "https://hoc247.net/tu-lieu/de-thi-tuyen-sinh-lop-10-mon-toan-chuyen-nam-2019-so-gd-dt-tp-hcm-co-dap-an-doc5519.html"),
    ("hcm.edu.vn", "official Sở GD-ĐT TP HCM (robots only)", None),
    ("znews.vn", "newspaper: HN 2024 đề + gợi ý đáp án",
     "https://znews.vn/de-thi-va-goi-y-dap-an-mon-toan-tuyen-sinh-lop-10-o-ha-noi-post1479826.html"),
    ("tienphong.vn", "newspaper: HN 2024 official đề + đáp án", "https://tienphong.vn/post-1645927.tpo"),
    ("dspace.ctu.edu.vn", "Toán học & Tuổi trẻ HSG lời giải (scans)", "https://dspace.ctu.edu.vn/jspui/handle/123456789/57527"),
]


def robots_summary(f: Fetcher, host: str) -> dict:
    url = f"https://{host}/robots.txt"
    txt, meta = f.get(url)
    rp = f.robots(url)
    lines = [ln.strip() for ln in (txt or "").splitlines() if ln.strip() and not ln.startswith("#")]
    dis = [ln for ln in lines if ln.lower().startswith("disallow")]
    delay = None
    if rp is not None:
        try:
            delay = rp.crawl_delay(USER_AGENT)
        except Exception:  # noqa: BLE001
            delay = None
    return {"robots_status": (meta or {}).get("status"), "n_disallow": len(dis), "disallow_sample": dis[:12],
            "crawl_delay": delay}


def page_features(html: str) -> dict:
    s = BeautifulSoup(html, "lxml")
    for t in s(["script", "style", "noscript", "header", "footer", "nav"]):
        t.decompose()
    body = s.find("article") or s.find("main") or s.body or s
    text = body.get_text("\n", strip=True)
    raw = str(body)
    imgs = body.find_all("img")
    return {
        "text_chars": len(text),
        "mathjax_inline": raw.count("\\(") + raw.count("\\["),
        "dollar_math": len(re.findall(r"\$[^$\n]{1,80}\$", text)),
        "mathml": raw.count("<math"),
        "n_img": len(imgs),
        "n_img_formula_like": sum(1 for i in imgs if re.search(r"(latex|codecogs|mathtex|equation|\.svg|math)",
                                                                (i.get("src") or i.get("data-src") or ""), re.I)),
        "pdf_or_iframe": len(body.find_all("iframe")) + len(re.findall(r"\.pdf", raw, re.I)),
        "question_headers": len(re.findall(r"(?m)^\s*(Câu|Bài)\s*(\d+|[IVX]+)\b", text)),
        "has_loi_giai": bool(re.search(r"Lời giải|Hướng dẫn giải|Đáp án", text)),
        "paywall_hint": bool(re.search(r"(tài khoản Pro|VIP|nâng cấp|Tải về.*Pro|trả phí|mua)", text, re.I)),
    }


def main():
    f = Fetcher()
    out = []
    for host, role, sample in CANDIDATES:
        row = {"source": host, "role": role, "sample_url": sample, **robots_summary(f, host)}
        if sample:
            row["sample_allowed"] = f.allowed(sample)
            if row["sample_allowed"]:
                html, meta = f.get(sample)
                row["sample_status"] = (meta or {}).get("status")
                if html:
                    row.update(page_features(html))
        out.append(row)
        print(json.dumps(row, ensure_ascii=False)[:400], flush=True)
    (DATA / "source_probe.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out),
                                             encoding="utf-8")
    print(f.stats)


if __name__ == "__main__":
    main()
