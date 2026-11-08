"""Polite caching HTTP client for Vi-ExamMath collection (X2, 2026-11).

Policy (04_data_eval.md §3.1, EXEC brief X2):
  * robots.txt of each host is fetched once and obeyed; an unreadable robots.txt fails closed (no fetch);
  * at most one network request per ``MIN_INTERVAL`` (2.0 s) per process, raised to a robots ``Crawl-delay``;
  * a generic User-Agent with no personal data;
  * every response is cached at ``data/vi_exam/raw/<host>/<sha1(url)>.html`` with a ``.json`` sidecar holding
    url, final_url, status, fetched_at (UTC ISO-8601) and the body sha1. A cached 200/404 is never re-fetched.

The cache layout is deliberately simple so that page snapshots are data, independent of any parser version.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import time
import urllib.parse
import urllib.robotparser
from datetime import datetime, timezone

import requests

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "vi_exam"
RAW = DATA / "raw"
FETCH_LOG = DATA / "fetch_log.jsonl"

USER_AGENT = "Mozilla/5.0 (compatible; ViExamMath-research-crawler/1.0; non-commercial academic research)"
MIN_INTERVAL = 2.0


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def cache_paths(url: str) -> tuple[pathlib.Path, pathlib.Path]:
    host = urllib.parse.urlsplit(url).netloc.replace(":", "_")
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()
    return RAW / host / f"{h}.html", RAW / host / f"{h}.json"


def read_cache(url: str) -> tuple[str | None, dict | None]:
    body, meta = cache_paths(url)
    if not meta.exists():
        return None, None
    m = json.loads(meta.read_text(encoding="utf-8"))
    if m.get("status") == 200 and body.exists():
        return body.read_text(encoding="utf-8"), m
    return None, m


class Fetcher:
    """Cache-first fetcher. ``offline=True`` never touches the network (rebuilds are reproducible)."""

    def __init__(self, offline: bool = False, min_interval: float = MIN_INTERVAL, timeout: int = 40):
        self.offline = offline
        self.min_interval = min_interval
        self.timeout = timeout
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "vi,en;q=0.5"})
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self._last = 0.0
        self.stats = {"network": 0, "cache": 0, "robots_blocked": 0, "errors": 0}

    def _sleep_turn(self, delay: float) -> None:
        wait = max(self.min_interval, delay) - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    def _raw_get(self, url: str, delay: float = 0.0) -> requests.Response | None:
        self._sleep_turn(delay)
        try:
            r = self.s.get(url, timeout=self.timeout)
        except requests.RequestException:
            self.stats["errors"] += 1
            return None
        self.stats["network"] += 1
        r.encoding = r.encoding if r.encoding and r.encoding.lower() != "iso-8859-1" else "utf-8"
        return r

    def _store(self, url: str, r: requests.Response) -> dict:
        body, meta = cache_paths(url)
        body.parent.mkdir(parents=True, exist_ok=True)
        text = r.text if r.status_code == 200 else ""
        body.write_text(text, encoding="utf-8")
        m = {"url": url, "final_url": r.url, "status": r.status_code, "fetched_at": utcnow(),
             "sha1_body": hashlib.sha1(text.encode("utf-8")).hexdigest(), "user_agent": USER_AGENT}
        meta.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
        FETCH_LOG.parent.mkdir(parents=True, exist_ok=True)
        with FETCH_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"url": url, "status": r.status_code, "fetched_at": m["fetched_at"]}) + "\n")
        return m

    def robots(self, url: str) -> urllib.robotparser.RobotFileParser | None:
        p = urllib.parse.urlsplit(url)
        base = f"{p.scheme}://{p.netloc}"
        if base in self._robots:
            return self._robots[base]
        rurl = base + "/robots.txt"
        txt, meta = read_cache(rurl)
        if meta is None and not self.offline:
            r = self._raw_get(rurl)
            if r is not None:
                meta = self._store(rurl, r)
                txt = r.text if r.status_code == 200 else ""
        rp = None
        if meta is not None:
            rp = urllib.robotparser.RobotFileParser()
            # 4xx robots => everything allowed (RFC 9309); body otherwise
            rp.parse((txt or "").splitlines() if meta.get("status") == 200 else [])
        self._robots[base] = rp
        return rp

    def allowed(self, url: str) -> bool:
        rp = self.robots(url)
        return rp is not None and rp.can_fetch(USER_AGENT, url)

    def get(self, url: str, refresh: bool = False) -> tuple[str | None, dict | None]:
        if not refresh:
            body, meta = read_cache(url)
            if meta is not None and (meta.get("status") in (200, 404, 410)):
                self.stats["cache"] += 1
                return body, meta
        if self.offline:
            return None, None
        if not self.allowed(url):
            self.stats["robots_blocked"] += 1
            return None, {"url": url, "status": "robots_disallowed"}
        rp = self.robots(url)
        delay = float(rp.crawl_delay(USER_AGENT) or 0) if rp else 0.0
        for attempt in range(3):
            r = self._raw_get(url, delay)
            if r is None:
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code in (429, 503):
                ra = r.headers.get("Retry-After", "")
                time.sleep(float(ra) if ra.isdigit() else 30 * (attempt + 1))
                continue
            m = self._store(url, r)
            return (r.text if r.status_code == 200 else None), m
        return None, {"url": url, "status": "failed"}
