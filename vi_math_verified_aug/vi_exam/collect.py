"""Fetch every discovered exam page, then look up its earliest Wayback capture (first_seen evidence).

One sequential process (one crawler), cache-first, ≤1 request / 2 s per host policy from ``crawl.Fetcher``.

  python -m vi_math_verified_aug.vi_exam.collect pages      # exam articles listed in exam_index.jsonl
  python -m vi_math_verified_aug.vi_exam.collect wayback    # CDX earliest capture -> data/vi_exam/first_seen.jsonl
  python -m vi_math_verified_aug.vi_exam.collect all

Wayback: ``/cdx/search/cdx?url=<page>&output=json&fl=timestamp,statuscode&limit=3`` (CDX returns captures in
ascending time order, so the first 200 row is the earliest archived copy). ``first_seen`` itself is computed in
``build.py`` as the minimum of {Wayback earliest, sitemap lastmod, og:image upload date}: each is an upper bound
on when the text was public.
"""
from __future__ import annotations

import json
import sys
import urllib.parse

from .crawl import DATA, Fetcher, read_cache
from .discover import INDEX_OUT

FIRST_SEEN = DATA / "first_seen.jsonl"


def cdx_url(page: str) -> str:
    return ("https://web.archive.org/cdx/search/cdx?url=" + urllib.parse.quote(page, safe="")
            + "&output=json&fl=timestamp,statuscode&limit=3")


def wayback_earliest(page: str, fetcher: Fetcher) -> dict:
    body, meta = fetcher.get(cdx_url(page))
    out = {"url": page, "cdx_status": (meta or {}).get("status"), "cdx_fetched_at": (meta or {}).get("fetched_at"),
           "wayback_first": None}
    if body:
        try:
            rows = json.loads(body)
        except json.JSONDecodeError:
            rows = []
        for ts, code in rows[1:]:
            if code == "200":
                out["wayback_first"] = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}T{ts[8:10]}:{ts[10:12]}:{ts[12:14]}Z"
                break
    return out


def index_rows():
    with INDEX_OUT.open(encoding="utf-8") as fh:
        for ln in fh:
            r = json.loads(ln)
            if r.get("slug_kind") in ("entrance", "hsg9"):
                yield r


def run_pages(f: Fetcher):
    n = ok = 0
    for r in index_rows():
        n += 1
        html, meta = f.get(r["url"])
        ok += html is not None
    print(f"pages: {ok}/{n} available; stats={f.stats}", flush=True)


def run_wayback(f: Fetcher):
    rows = []
    for r in index_rows():
        rows.append(wayback_earliest(r["url"], f))
    FIRST_SEEN.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
    print(f"wayback: {sum(1 for x in rows if x['wayback_first'])}/{len(rows)} archived; stats={f.stats}", flush=True)


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    f = Fetcher()
    if what in ("pages", "all"):
        run_pages(f)
    if what in ("wayback", "all"):
        run_wayback(f)


if __name__ == "__main__":
    main()
