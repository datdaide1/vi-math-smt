"""NIM chat client for W5 (both generation arms).

* One process-wide rate limiter: requests are *started* at most ``rpm`` per minute
  (sliding 60 s window + minimum spacing). Retries go through the same limiter.
* Every call (success or failure) is appended as one JSON line to
  ``logs/api/W5/<run>.jsonl``: ts, provider, model, params, request, raw response,
  latency, status. The API key is never logged.
* The log doubles as a cache: a request whose (model, messages, params, tag) hash
  already has a 200 response in any W5 log is answered from the log, so a crashed
  pilot can be resumed without re-spending calls.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import deque
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "logs" / "api" / "W5"
NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

GEN_MODEL = "deepseek-ai/deepseek-v4.1-flash"          # generator / editor / formalizer A / solution writer
CHECK_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"       # back-formalizer (round trip) / judge


def _load_key() -> str:
    k = os.environ.get("NVIDIA_API_KEY")
    if k:
        return k
    for line in open(ROOT / ".env", encoding="utf-8"):
        if line.startswith("NVIDIA_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise RuntimeError("NVIDIA_API_KEY not found")


class RateLimiter:
    def __init__(self, rpm: float):
        self.rpm = rpm
        self.min_gap = 60.0 / rpm
        self.starts: deque[float] = deque()
        self.lock = threading.Lock()
        self.last = 0.0

    def acquire(self):
        while True:
            with self.lock:
                now = time.time()
                while self.starts and now - self.starts[0] > 60.0:
                    self.starts.popleft()
                if len(self.starts) < self.rpm and now - self.last >= self.min_gap:
                    self.starts.append(now)
                    self.last = now
                    return
                wait = max(self.min_gap - (now - self.last),
                           (60.0 - (now - self.starts[0]) + 0.05) if len(self.starts) >= self.rpm else 0)
            time.sleep(max(wait, 0.05))


def req_hash(model: str, messages: list, params: dict, tag: str) -> str:
    s = json.dumps([model, messages, params, tag], ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


class NimClient:
    def __init__(self, run: str, rpm: float = 23.0, timeout: float = 400.0, max_retries: int = 3):
        self.key = _load_key()
        self.limiter = RateLimiter(rpm)
        self.timeout = timeout
        self.max_retries = max_retries
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = LOG_DIR / f"{run}.jsonl"
        self.log_lock = threading.Lock()
        self.cache: dict[str, dict] = {}
        self.stats = {"api_calls": 0, "cache_hits": 0, "errors": 0}
        self._load_cache()

    def _load_cache(self):
        for p in sorted(LOG_DIR.glob("*.jsonl")):
            for line in open(p, encoding="utf-8"):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("status") == 200 and r.get("hash") and r.get("content") is not None:
                    self.cache[r["hash"]] = r

    def _log(self, rec: dict):
        with self.log_lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def chat(self, model: str, messages: list, *, temperature: float = 0.0, max_tokens: int = 4096,
             top_p: float = 1.0, tag: str = "", extra: dict | None = None, stage: str = "") -> dict:
        """Return {content, reasoning, usage, latency, cached, hash}. Raises on final failure."""
        params = {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}
        if extra:
            params.update(extra)
        h = req_hash(model, messages, params, tag)
        if h in self.cache:
            self.stats["cache_hits"] += 1
            r = self.cache[h]
            return {"content": r["content"], "reasoning": r.get("reasoning"), "usage": r.get("usage") or {},
                    "latency": r.get("latency"), "cached": True, "hash": h}
        body = {"model": model, "messages": messages, **params}
        last_err = None
        for attempt in range(self.max_retries + 1):
            self.limiter.acquire()
            t0 = time.time()
            status, raw, err = None, None, None
            try:
                resp = requests.post(NIM_URL, headers={"Authorization": f"Bearer {self.key}",
                                                       "Accept": "application/json"},
                                     json=body, timeout=self.timeout)
                status = resp.status_code
                raw = resp.text
            except Exception as e:  # timeout / connection
                err = repr(e)
            lat = time.time() - t0
            self.stats["api_calls"] += 1
            content = reasoning = None
            usage = {}
            if status == 200:
                try:
                    j = json.loads(raw)
                    msg = j["choices"][0]["message"]
                    content = msg.get("content")
                    reasoning = msg.get("reasoning_content") or msg.get("reasoning")
                    usage = j.get("usage") or {}
                    finish = j["choices"][0].get("finish_reason")
                except Exception as e:
                    err = f"parse: {e!r}"
                    status = -1
            rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "provider": "nim", "model": model,
                   "params": params, "tag": tag, "stage": stage, "hash": h, "attempt": attempt,
                   "request": messages, "status": status, "error": err, "latency": round(lat, 2),
                   "raw": raw if status != 200 else None, "content": content, "reasoning": reasoning,
                   "usage": usage, "finish_reason": finish if status == 200 else None}
            self._log(rec)
            if status == 200 and content is not None:
                self.cache[h] = rec
                return {"content": content, "reasoning": reasoning, "usage": usage, "latency": lat,
                        "cached": False, "hash": h}
            self.stats["errors"] += 1
            last_err = f"status={status} err={err} raw={(raw or '')[:200]}"
            if status in (400, 401, 403, 404, 410, 422):
                break
            time.sleep(min(60, 5 * (2 ** attempt)))
        raise RuntimeError(f"NIM call failed: {last_err}")


GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GEMINI_JUDGE = "gemini-3.5-flash-lite"


def _load_gemini_keys() -> list[str]:
    for line in open(ROOT / ".env", encoding="utf-8"):
        if line.startswith("GEMINI_API_KEYS="):
            v = line.split("=", 1)[1].strip().strip('"')
            return [k.strip() for k in v.split(",") if k.strip()]
    return []


class GeminiClient:
    """Free-tier Gemini pool, used ONLY as a second-family judge on training-side generated items.

    Each key has its own limiter (conservative ``rpm_per_key``); keys are used round-robin;
    a 429 puts that key on a cool-down. Logged to the same W5 log dir as NIM calls (key index
    only, never the key)."""

    def __init__(self, run: str, rpm_per_key: float = 8.0, timeout: float = 120.0, max_retries: int = 4):
        self.keys = _load_gemini_keys()
        if not self.keys:
            raise RuntimeError("GEMINI_API_KEYS not found")
        self.limiters = [RateLimiter(rpm_per_key) for _ in self.keys]
        self.cool = [0.0] * len(self.keys)
        self.rr = 0
        self.rr_lock = threading.Lock()
        self.timeout = timeout
        self.max_retries = max_retries
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = LOG_DIR / f"{run}.gemini.jsonl"
        self.log_lock = threading.Lock()
        self.cache: dict[str, dict] = {}
        self.stats = {"api_calls": 0, "cache_hits": 0, "errors": 0}
        for p in sorted(LOG_DIR.glob("*.gemini.jsonl")):
            for line in open(p, encoding="utf-8"):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("status") == 200 and r.get("content") is not None:
                    self.cache[r["hash"]] = r

    def _pick(self) -> int:
        while True:
            with self.rr_lock:
                now = time.time()
                for _ in range(len(self.keys)):
                    i = self.rr
                    self.rr = (self.rr + 1) % len(self.keys)
                    if self.cool[i] <= now:
                        return i
            time.sleep(1.0)

    def chat(self, prompt: str, *, model: str = GEMINI_JUDGE, temperature: float = 0.0,
             max_tokens: int = 2048, tag: str = "", stage: str = "") -> dict:
        params = {"temperature": temperature, "maxOutputTokens": max_tokens}
        messages = [{"role": "user", "content": prompt}]
        h = req_hash(model, messages, params, tag)
        if h in self.cache:
            self.stats["cache_hits"] += 1
            r = self.cache[h]
            return {"content": r["content"], "usage": r.get("usage") or {}, "latency": r.get("latency"),
                    "cached": True, "hash": h}
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}}
        last = None
        for attempt in range(self.max_retries + 1):
            i = self._pick()
            self.limiters[i].acquire()
            t0 = time.time()
            status = raw = err = None
            try:
                resp = requests.post(GEMINI_URL.format(model=model),
                                     headers={"x-goog-api-key": self.keys[i], "Content-Type": "application/json"},
                                     json=body, timeout=self.timeout)
                status, raw = resp.status_code, resp.text
            except Exception as e:
                err = repr(e)
            lat = time.time() - t0
            self.stats["api_calls"] += 1
            content, usage = None, {}
            if status == 200:
                try:
                    j = json.loads(raw)
                    content = "".join(p.get("text", "") for p in j["candidates"][0]["content"]["parts"])
                    usage = j.get("usageMetadata") or {}
                except Exception as e:
                    status, err = -1, f"parse: {e!r}"
            rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "provider": "gemini", "model": model,
                   "key_index": i, "params": params, "tag": tag, "stage": stage, "hash": h,
                   "attempt": attempt, "request": messages, "status": status, "error": err,
                   "latency": round(lat, 2), "raw": raw if status != 200 else None,
                   "content": content, "usage": usage}
            with self.log_lock:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if status == 200 and content is not None:
                self.cache[h] = rec
                return {"content": content, "usage": usage, "latency": lat, "cached": False, "hash": h}
            self.stats["errors"] += 1
            last = f"status={status} err={err} raw={(raw or '')[:200]}"
            if status == 429:
                self.cool[i] = time.time() + 65
                continue
            if status in (400, 401, 403, 404):
                break
            time.sleep(min(30, 3 * (2 ** attempt)))
        raise RuntimeError(f"Gemini call failed: {last}")


def parse_json_block(text: str):
    """Extract the first top-level JSON object from a model reply (handles ```json fences)."""
    if text is None:
        return None
    t = text.strip()
    if "```" in t:
        parts = t.split("```")
        for p in parts[1::2]:
            p2 = p[4:] if p.lstrip().lower().startswith("json") else p
            p2 = p2.strip()
            if p2.lower().startswith("json"):
                p2 = p2[4:]
            try:
                return json.loads(p2)
            except Exception:
                pass
    start = t.find("{")
    while start >= 0:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[start:i + 1])
                    except Exception:
                        break
        start = t.find("{", start + 1)
    return None
