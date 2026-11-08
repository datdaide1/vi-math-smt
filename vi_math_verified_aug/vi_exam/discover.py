"""Discover Vi-ExamMath source pages on loigiaihay.com (robots.txt: ``Allow: /``).

Two independent discovery routes, unioned:
  1. **Listings**: hub ``/de-thi-vao-10-mon-toan-c949.html`` -> per-province / per-school event listings
     (``-e<id>.html``) -> exam articles (``-a<id>.html``). Mock/compilation listings ("đề tham khảo", "tổng hợp
     50 đề") are recorded but their articles are not collected (teacher-made, not state-issued).
  2. **Monthly article sitemaps** (``sitemap_articles.xml`` -> ``monthly-detail-m<M>-y<Y>.xml``): every article
     whose slug looks like a grade-10 entrance Toán exam or an HSG Toán 9 paper. This route also gives the
     sitemap ``lastmod`` (publication-time upper bound used for ``first_seen``) and finds articles that no
     listing links to.

Output: ``data/vi_exam/exam_index.jsonl`` (one row per candidate URL, with discovery evidence).

CLI: ``python -m vi_math_verified_aug.vi_exam.discover [--offline] [--from-year 2017]``
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.parse

from bs4 import BeautifulSoup

from .crawl import DATA, Fetcher

BASE = "https://loigiaihay.com"
HUB = BASE + "/de-thi-vao-10-mon-toan-c949.html"
SITEMAP_INDEX = BASE + "/sitemap_articles.xml"
INDEX_OUT = DATA / "exam_index.jsonl"
LISTINGS_OUT = DATA / "listings.jsonl"

RE_ART = re.compile(r"-a\d+\.html$")
RE_EVT = re.compile(r"-e\d+\.html$")
# slug patterns (ASCII slugs, no diacritics)
RE_ENTRANCE = re.compile(r"(vao-(lop-)?10|tuyen-sinh-(lop-)?10|vao-thpt)")
RE_MATH = re.compile(r"(^|-)toan(-|$)")
RE_HSG9 = re.compile(r"(hoc-sinh-gioi|hsg).*(toan).*(9|lop-9)|(toan).*(9|lop-9).*(hoc-sinh-gioi|hsg)")
RE_MOCK = re.compile(r"(tham-khao|thi-thu|de-so-\d|de-on|on-thi|de-cuong|tong-hop|bo-de|luyen-thi|du-doan|de-xuat)")


def classify_listing(url: str, title: str) -> str:
    slug = url.rsplit("/", 1)[-1]
    if "tham-khao" in slug or "thi thử" in title.lower():
        return "mock"
    if "tong-hop" in slug:
        return "compilation"
    if "truong-chuyen" in slug or "các trường" in title:
        return "school"
    return "province"


def slug_kind(url: str) -> str | None:
    slug = url.rsplit("/", 1)[-1]
    if RE_HSG9.search(slug):
        return "hsg9"
    if RE_ENTRANCE.search(slug) and RE_MATH.search(slug):
        if "chuyen-de" in slug or "trac-nghiem-toan-9" in slug:
            return None
        return "mock" if RE_MOCK.search(slug) else "entrance"
    return None


def absolute(href: str) -> str:
    return urllib.parse.urljoin(BASE + "/", href.split("#")[0])


def discover(fetcher: Fetcher, from_year: int = 2017) -> list[dict]:
    rows: dict[str, dict] = {}

    def add(url: str, **ev):
        r = rows.setdefault(url, {"url": url, "via": [], "listing_titles": [], "sitemap_lastmod": None,
                                  "slug_kind": slug_kind(url)})
        if ev.get("via") and ev["via"] not in r["via"]:
            r["via"].append(ev["via"])
        if ev.get("listing_title") and ev["listing_title"] not in r["listing_titles"]:
            r["listing_titles"].append(ev["listing_title"])
        if ev.get("lastmod"):
            r["sitemap_lastmod"] = min(filter(None, [r["sitemap_lastmod"], ev["lastmod"]]))
        if ev.get("listing_kind"):
            r.setdefault("listing_kinds", [])
            if ev["listing_kind"] not in r["listing_kinds"]:
                r["listing_kinds"].append(ev["listing_kind"])

    # ---- route 1: listings ----
    hub_html, _ = fetcher.get(HUB)
    listings = []
    if hub_html:
        soup = BeautifulSoup(hub_html, "lxml")
        seen = set()
        for a in soup.find_all("a", href=True):
            u = absolute(a["href"])
            if RE_EVT.search(u) and "mon-toan" in u and "vao-10" in u and u not in seen:
                seen.add(u)
                t = a.get_text(" ", strip=True)
                listings.append({"listing_url": u, "listing_title": t, "kind": classify_listing(u, t)})
    for L in listings:
        html, meta = fetcher.get(L["listing_url"])
        L["fetched_at"] = (meta or {}).get("fetched_at")
        L["n_articles"] = 0
        if not html:
            continue
        soup = BeautifulSoup(html, "lxml")
        for a in soup.find_all("a", href=True):
            u = absolute(a["href"])
            if not RE_ART.search(u) or not u.startswith(BASE):
                continue
            title = a.get_text(" ", strip=True)[:200]
            if not re.search(r"(vào|lớp)\s*10", title, re.I):
                continue
            L["n_articles"] += 1
            if L["kind"] in ("mock", "compilation"):
                continue  # recorded in listings.jsonl count only; teacher-made papers are out of scope
            add(u, via="listing", listing_title=title, listing_kind=L["kind"])
    # ---- route 2: monthly article sitemaps ----
    idx, _ = fetcher.get(SITEMAP_INDEX)
    months = re.findall(r"<loc>\s*(\S+monthly-detail-m(\d+)-y(\d{4})\.xml)\s*</loc>", idx or "")
    n_sm = 0
    for sm_url, _m, y in months:
        if int(y) < from_year:
            continue
        xml, _ = fetcher.get(sm_url)
        n_sm += 1
        for block in re.findall(r"<url>(.*?)</url>", xml or "", re.S):
            loc = re.search(r"<loc>\s*(\S+?)\s*</loc>", block)
            if not loc:
                continue
            u = loc.group(1)
            k = slug_kind(u)
            if k in ("entrance", "hsg9"):
                lm = re.search(r"<lastmod>\s*(\S+?)\s*</lastmod>", block)
                add(u, via="sitemap", lastmod=lm.group(1) if lm else None)
            elif k == "mock" and u in rows:
                lm = re.search(r"<lastmod>\s*(\S+?)\s*</lastmod>", block)
                add(u, lastmod=lm.group(1) if lm else None)
    LISTINGS_OUT.write_text("".join(json.dumps(L, ensure_ascii=False) + "\n" for L in listings), encoding="utf-8")
    for r in rows.values():  # a mock-looking slug linked from an official province listing ("đề số 1/2" = two
        if r["slug_kind"] == "mock" and "province" in r.get("listing_kinds", []):  # papers): page title decides
            r["slug_kind"] = "entrance"
    out = sorted(rows.values(), key=lambda r: r["url"])
    INDEX_OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    print(f"listings={len(listings)} sitemaps_read={n_sm} candidates={len(out)} stats={fetcher.stats}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--from-year", type=int, default=2017)
    a = ap.parse_args()
    discover(Fetcher(offline=a.offline), a.from_year)


if __name__ == "__main__":
    main()
