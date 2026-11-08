"""Rate-limited, fully logged LLM clients for W3 (training-side items ONLY).

* NIM (OpenAI-compatible): token bucket at <= 10 RPM for the whole W3 process (EXEC_BRIEF budget).
* Gemini (REST, key pool GEMINI_API_KEYS, header x-goog-api-key): different-family judge, <= 10 RPM,
  exponential back-off on 429.
* Every call (success or failure) is appended to ``logs/api/W3/<provider>_<date>.jsonl`` with
  ts, provider, model, params, request, raw response, latency, item_id.
* Guard: any item id containing ``/test/`` (Vi-GSM8K / Vi-MATH test) or ``vi_exam`` is refused —
  test and benchmark items must never reach NIM or Gemini.
Keys are read from ``.env`` and never printed or logged.
"""
from __future__ import annotations

import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "logs" / "api" / "W3"
NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
GEM_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

CALLS = {"nim": 0, "nim_errors": 0, "gemini": 0, "gemini_errors": 0}


class ForbiddenItem(RuntimeError):
    pass


def _env() -> dict:
    env = {}
    p = ROOT / ".env"
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def guard(item_id: str) -> None:
    low = (item_id or "").lower()
    if (not item_id or "/test/" in low or "test" in low.split("/") or "vi_exam" in low or "benchmark" in low
            or not low.split("/")[1:2] == ["train"]):
        raise ForbiddenItem(f"refusing to send item {item_id!r} to an external API (test/benchmark item)")


class Bucket:
    """Minimum spacing between request starts: 60/rpm seconds (strict, process-wide)."""

    def __init__(self, rpm: float):
        self.dt = 60.0 / rpm
        self.next = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next)
            self.next = t + self.dt
        if t > now:
            time.sleep(t - now)


NIM_BUCKET = Bucket(9.5)  # <= 10 RPM budget for W3, small safety margin
GEM_BUCKET = Bucket(10)


def _log(provider: str, rec: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    p = LOG_DIR / f"{provider}_{time.strftime('%Y%m%d')}.jsonl"
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _post(url: str, headers: dict, body: dict, timeout: float) -> tuple[int, str]:
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def nim_chat(item_id: str, model: str, messages: list[dict], *, temperature: float = 0.0, max_tokens: int = 4096,
             seed: int = 42, role: str = "", prompt_file: str = "", tries: int = 4, timeout: float = 300) -> str:
    """One chat completion on NIM. Returns the message content ('' on final failure)."""
    guard(item_id)
    key = _env()["NVIDIA_API_KEY"]
    params = {"temperature": temperature, "max_tokens": max_tokens, "seed": seed, "top_p": 1.0}
    body = {"model": model, "messages": messages, **params}
    for k in range(tries):
        NIM_BUCKET.wait()
        t0 = time.time()
        try:
            status, raw = _post(NIM_URL, {"Authorization": f"Bearer {key}"}, body, timeout)
        except Exception as e:  # network
            status, raw = -1, f"{type(e).__name__}: {e}"
        dt = round(time.time() - t0, 2)
        CALLS["nim"] += 1
        content = ""
        if status == 200:
            try:
                content = json.loads(raw)["choices"][0]["message"]["content"] or ""
            except Exception:
                status = -2
        if status != 200:
            CALLS["nim_errors"] += 1
        _log("nim", {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "provider": "nim", "model": model, "params": params,
                     "item_id": item_id, "role": role, "prompt_file": prompt_file, "attempt": k + 1,
                     "request": messages, "status": status, "raw_response": raw, "latency_s": dt})
        if status == 200 and content:
            return content
        if status in (400, 404, 422):
            break
        time.sleep(min(60, 5 * 2 ** k) + random.random())
    return ""


_GEM_KEYS: list[str] = []
_GEM_I = 0


def gemini(item_id: str, model: str, prompt: str, *, temperature: float = 0.0, max_tokens: int = 2048,
           role: str = "", prompt_file: str = "", tries: int = 6, json_mode: bool = True) -> str:
    """One Gemini generateContent call with key rotation and 429 back-off. Training-side items only."""
    global _GEM_KEYS, _GEM_I
    guard(item_id)
    if not _GEM_KEYS:
        _GEM_KEYS = [k.strip() for k in _env().get("GEMINI_API_KEYS", "").split(",") if k.strip()]
        if not _GEM_KEYS:
            raise RuntimeError("GEMINI_API_KEYS not set")
    params = {"temperature": temperature, "maxOutputTokens": max_tokens}
    if json_mode:
        params["responseMimeType"] = "application/json"
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": params}
    for k in range(tries):
        GEM_BUCKET.wait()
        key_idx = _GEM_I % len(_GEM_KEYS)
        _GEM_I += 1
        t0 = time.time()
        try:
            status, raw = _post(GEM_URL.format(model=model), {"x-goog-api-key": _GEM_KEYS[key_idx]}, body, 120)
        except Exception as e:
            status, raw = -1, f"{type(e).__name__}: {e}"
        dt = round(time.time() - t0, 2)
        CALLS["gemini"] += 1
        text = ""
        if status == 200:
            try:
                text = "".join(p.get("text", "") for p in json.loads(raw)["candidates"][0]["content"]["parts"])
            except Exception:
                status = -2
        if status != 200:
            CALLS["gemini_errors"] += 1
        _log("gemini", {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "provider": "gemini", "model": model,
                        "params": params, "item_id": item_id, "role": role, "prompt_file": prompt_file,
                        "attempt": k + 1, "key_slot": key_idx, "request": prompt, "status": status,
                        "raw_response": raw, "latency_s": dt})
        if status == 200 and text:
            return text
        if status in (400, 404):
            break
        time.sleep(min(90, (10 if status == 429 else 4) * 2 ** k) + random.random())
    return ""
