"""G1 pilot: one call interface over NIM, Claude (Anthropic SDK) and Gemini (REST).

call(model_key, system, user, task=..., item=..., **opts) -> dict(text, usage, latency, cost, cached, ok, err)

* Every call (success or failure) is appended to logs/api/G1/<provider>.jsonl with ts, provider, model,
  params, request, raw response, latency, usage, cost. Keys are never logged or printed.
* The logs double as a cache keyed by sha1(model, system, user, params, tag): a crashed run resumes
  without re-spending calls.
* NIM: one key, one process-wide limiter (<= NIM_RPM request starts per rolling minute, all NIM models).
* Claude: list prices (per M tokens) Haiku 5.5 0.10/0.50, Sonnet 5.5 2/10, Opus 5.5 4/20; direct calls;
  output_config.effort set explicitly; hard budget stop CLAUDE_BUDGET (USD, all G1 logs summed).
* Gemini: key pool (GEMINI_API_KEYS), header x-goog-api-key, round-robin, per-key cool-down on 429.
* Guard: refuses any item id that is not a Vi-GSM8K *train* seed or a G1 generated item for NIM/Gemini.
"""
from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from collections import deque
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "logs" / "api" / "G1"
LOG_DIR.mkdir(parents=True, exist_ok=True)

NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
GEM_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
NIM_RPM = 38            # shared token bucket across the 3 NIM models (one key, 40 RPM cap)
NIM_INFLIGHT = 35       # NIM limits by RPM, not concurrency; latency 40-600 s needs many in flight
CLAUDE_INFLIGHT = 24
GEM_LANE_INFLIGHT = 3   # requests in flight per (key, model) lane (was 1: lanes were latency-bound)
GEM_LANE_RPM = 12       # per (key, model) lane; probe: 48/48 OK, no 429 (logs/api/G1/gemini_probe.jsonl)
CLAUDE_BUDGET = 9.70  # hard stop below the user's $10 cap

PRICES = {  # USD per 1M tokens (input, output)
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0),
}

# model_key -> (provider, model id, default effort for Claude)
MODELS = {
    "deepseek": ("nim", "deepseek-ai/deepseek-v4.1-flash"),
    "nemotron": ("nim", "nvidia/nemotron-3-ultra-550b-a55b"),
    "glm": ("nim", "z-ai/glm-5.3-flash"),
    "haiku": ("claude", "claude-haiku-5-5"),
    "sonnet": ("claude", "claude-sonnet-5-5"),
    "gemini": ("gemini", "gemini-3.5-flash-lite"),
    "gemini31": ("gemini", "gemini-3.1-flash-lite"),
    "opus": ("claude", "claude-opus-5-5"),
    "gemini_judge": ("gemini", "gemini-3.5-flash-lite"),
}
GENERATORS = ["haiku", "sonnet", "gemini", "gemini31", "deepseek", "nemotron", "glm"]


def _env() -> dict:
    env = {}
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


ENV = _env()


class BudgetExceeded(RuntimeError):
    pass


class ForbiddenItem(RuntimeError):
    pass


def guard(item: str, provider: str):
    """item ids: 'seed:<seed_id>' (Vi-GSM8K train dev seed) or 'gen:<...>' (G1 generated item)."""
    if provider == "claude":
        return
    low = (item or "").lower()
    if not (low.startswith("seed:") or low.startswith("gen:")) or "test" in low or "exam" in low:
        raise ForbiddenItem(f"refusing to send {item!r} to {provider}")


class RateLimiter:
    def __init__(self, rpm: float):
        self.rpm, self.gap = rpm, 60.0 / rpm
        self.starts: deque = deque()
        self.lock = threading.Lock()
        self.last = 0.0

    def acquire(self):
        while True:
            with self.lock:
                now = time.time()
                while self.starts and now - self.starts[0] > 60.0:
                    self.starts.popleft()
                if len(self.starts) < self.rpm and now - self.last >= self.gap:
                    self.starts.append(now)
                    self.last = now
                    return
                wait = max(self.gap - (now - self.last),
                           (60.05 - (now - self.starts[0])) if len(self.starts) >= self.rpm else 0)
            time.sleep(max(wait, 0.05))


_log_lock = threading.Lock()
_cache: dict = {}
_cache_loaded = False
_claude_spent = 0.0
_spend_lock = threading.Lock()
NIM_LIMIT = RateLimiter(NIM_RPM)
NIM_SEM = threading.Semaphore(NIM_INFLIGHT)
CLAUDE_SEM = threading.Semaphore(CLAUDE_INFLIGHT)
STATS = {"calls": 0, "cache_hits": 0, "errors": 0, "rate_limited": 0}


_load_lock = threading.Lock()


def _load_cache():
    global _cache_loaded, _claude_spent
    if _cache_loaded:
        return
    with _load_lock:
        if _cache_loaded:
            return
        _load_cache_locked()


def _load_cache_locked():
    global _cache_loaded, _claude_spent
    for p in sorted(LOG_DIR.glob("*.jsonl")):
        for line in open(p, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("provider") == "claude":
                _claude_spent += float(r.get("cost") or 0)
            if r.get("ok") and r.get("hash"):
                _cache[r["hash"]] = r
    _cache_loaded = True


def claude_spent() -> float:
    _load_cache()
    return _claude_spent


def _log(provider: str, rec: dict):
    line = json.dumps(rec, ensure_ascii=False) + "\n"
    with _log_lock:
        for k in range(20):  # Windows: transient PermissionError when another process reads the file
            try:
                with open(LOG_DIR / f"{provider}.jsonl", "a", encoding="utf-8") as f:
                    f.write(line)
                return
            except PermissionError:
                time.sleep(0.25 * (k + 1))
        raise


def _hash(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- providers
def _nim(model, system, user, params):
    """Streaming (SSE) request: the NIM gateway returns 504 after ~300 s on non-streamed calls, which
    slow reasoning models (deepseek-v4.1-flash) hit routinely. Raw = assembled message + final usage chunk."""
    body = {"model": model, "messages": ([{"role": "system", "content": system}] if system else []) +
            [{"role": "user", "content": user}], **params, "stream": True,
            "stream_options": {"include_usage": True}}
    with NIM_SEM:
        NIM_LIMIT.acquire()
        t0 = time.time()
        out = _nim_stream(body)
        if out[0] == 200:
            out[3]["http_latency"] = round(time.time() - t0, 2)
        return out


def _nim_stream(body):
    r = requests.post(NIM_URL, headers={"Authorization": f"Bearer {ENV['NVIDIA_API_KEY']}",
                                        "Accept": "text/event-stream"}, json=body, timeout=(30, 900), stream=True)
    if r.status_code != 200:
        return r.status_code, r.text, None, {}, None
    content, reasoning, u, finish, ttft, t0 = [], [], {}, None, None, time.time()
    for line in r.iter_lines(decode_unicode=False):
        if not line or not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            break
        try:
            j = json.loads(data.decode("utf-8"))
        except Exception:
            continue
        if j.get("usage"):
            u = j["usage"]
        for ch in j.get("choices") or []:
            d = ch.get("delta") or {}
            if d.get("content"):
                content.append(d["content"])
                if ttft is None:
                    ttft = time.time() - t0
            rc = d.get("reasoning_content") or d.get("reasoning")
            if rc:
                reasoning.append(rc)
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
    text = "".join(content)
    usage = {"input_tokens": u.get("prompt_tokens", 0), "output_tokens": u.get("completion_tokens", 0),
             "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
             "reasoning_chars": sum(map(len, reasoning))}
    raw = json.dumps({"content": text, "reasoning": "".join(reasoning), "usage": u, "finish_reason": finish},
                     ensure_ascii=False)
    if not u and not text:
        return -3, raw, None, {}, finish
    return 200, raw, text, usage, finish


_anth = None


def _claude(model, system, user, params):
    global _anth
    import anthropic
    if _anth is None:
        _anth = anthropic.Anthropic(api_key=ENV["ANTHROPIC_API_KEY"], max_retries=0, timeout=600,
                                    default_headers={"anthropic-workspace-id": ENV["ANTHROPIC_WORKSPACE_ID"]})
    kw = {"model": model, "max_tokens": params["max_tokens"],
          "output_config": {"effort": params["effort"]},
          "messages": [{"role": "user", "content": user}]}
    if system:
        kw["system"] = ([{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
                        if params.get("cache_system") else system)
    try:
        with CLAUDE_SEM:
            t_http = time.time()
            with _anth.messages.stream(**kw) as s:
                m = s.get_final_message()
            t_http = time.time() - t_http
    except anthropic.RateLimitError as e:
        return 429, str(e), None, {}, None
    except anthropic.APIStatusError as e:
        return e.status_code, str(e), None, {}, None
    except anthropic.APIConnectionError as e:
        return -2, str(e), None, {}, None
    text = "".join(b.text for b in m.content if b.type == "text")
    u = m.usage
    usage = {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
             "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
             "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0, "http_latency": round(t_http, 2)}
    return 200, m.model_dump_json(), text, usage, m.stop_reason


_gem_keys = [k.strip() for k in ENV.get("GEMINI_API_KEYS", "").split(",") if k.strip()]
GEM_OBS = {"429": 0, "per_lane_429": {}, "503": 0}
_lanes: dict = {}          # model -> queue of free key indices (one lane per (key, model): 1 request in flight)
_lane_lim: dict = {}       # (model, key) -> RateLimiter
_lane_cool: dict = {}      # (model, key) -> time until which the lane is cooling down after a 429
_lanes_lock = threading.Lock()


def _lane_get(model):
    import queue
    with _lanes_lock:
        if model not in _lanes:
            q = queue.Queue()
            for _ in range(GEM_LANE_INFLIGHT):
                for i in range(len(_gem_keys)):
                    q.put(i)
            for i in range(len(_gem_keys)):
                _lane_lim[(model, i)] = RateLimiter(GEM_LANE_RPM)
                _lane_cool[(model, i)] = 0.0
            _lanes[model] = q
    while True:
        i = _lanes[model].get()
        wait = _lane_cool[(model, i)] - time.time()
        if wait <= 0:
            return i
        _lanes[model].put(i)
        time.sleep(min(wait, 2.0))


def _gemini(model, system, user, params):
    i = _lane_get(model)
    try:
        _lane_lim[(model, i)].acquire()
        body = {"contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"temperature": params["temperature"], "maxOutputTokens": params["max_tokens"]}}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        t_http = time.time()
        r = requests.post(GEM_URL.format(model=model), headers={"x-goog-api-key": _gem_keys[i],
                                                                "Content-Type": "application/json"},
                          json=body, timeout=300)
        t_http = time.time() - t_http
        raw = r.text
        if r.status_code == 429:
            GEM_OBS["429"] += 1
            k = f"{model}#{i}"
            GEM_OBS["per_lane_429"][k] = GEM_OBS["per_lane_429"].get(k, 0) + 1
            _lane_cool[(model, i)] = time.time() + 60
        if r.status_code == 503:
            GEM_OBS["503"] += 1
    finally:
        _lanes[model].put(i)
    if r.status_code != 200:
        return r.status_code, raw, None, {"key_index": i}, None
    j = json.loads(raw)
    cand = (j.get("candidates") or [{}])[0]
    text = "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts", []) if not p.get("thought"))
    u = j.get("usageMetadata") or {}
    usage = {"input_tokens": u.get("promptTokenCount", 0),
             "output_tokens": (u.get("candidatesTokenCount", 0) or 0) + (u.get("thoughtsTokenCount", 0) or 0),
             "reasoning_tokens": u.get("thoughtsTokenCount"), "key_index": i, "http_latency": round(t_http, 2)}
    return 200, raw, text, usage, cand.get("finishReason")


# ---------------------------------------------------------------- public
def call(model_key: str, system: str, user: str, *, task: str, item: str, temperature: float = 0.0,
         max_tokens: int = 8192, effort: str | None = None, tag: str = "", retries: int = 5,
         cache_system: bool = False) -> dict:
    _load_cache()
    global _claude_spent
    provider, model = MODELS[model_key]
    guard(item, provider)
    if provider == "claude":
        params = {"max_tokens": max_tokens, "effort": effort or "low"}
        if cache_system:
            params["cache_system"] = True
    else:
        params = {"temperature": temperature, "max_tokens": max_tokens}
    h = _hash(model, system, user, params, tag, task, item)
    if h in _cache:
        STATS["cache_hits"] += 1
        r = _cache[h]
        return {"text": r["text"], "usage": r.get("usage") or {}, "latency": r.get("latency"),
                "cost": r.get("cost", 0.0), "cached": True, "ok": True, "finish": r.get("finish")}
    last = None
    for attempt in range(retries + 1):
        if provider == "claude":
            with _spend_lock:
                if _claude_spent > CLAUDE_BUDGET:
                    raise BudgetExceeded(f"Claude spend ${_claude_spent:.2f} > ${CLAUDE_BUDGET}")
        t0 = time.time()
        try:
            fn = {"nim": _nim, "claude": _claude, "gemini": _gemini}[provider]
            status, raw, text, usage, finish = fn(model, system, user, params)
        except Exception as e:  # network / parse
            status, raw, text, usage, finish = -1, repr(e), None, {}, None
        lat = time.time() - t0
        STATS["calls"] += 1
        cost = 0.0
        if provider == "claude" and status == 200:
            pi, po = PRICES[model]
            cost = (usage["input_tokens"] * pi + usage.get("cache_write", 0) * pi * 1.25 +
                    usage.get("cache_read", 0) * pi * 0.1 + usage["output_tokens"] * po) / 1e6
            with _spend_lock:
                _claude_spent += cost
        ok = status == 200 and text is not None
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "provider": provider, "model": model,
               "model_key": model_key, "task": task, "item": item, "tag": tag, "params": params,
               "hash": h, "attempt": attempt, "request": {"system": system, "user": user},
               "status": status, "ok": ok, "raw": raw, "text": text, "finish": finish,
               "latency": round(lat, 2), "usage": usage, "cost": round(cost, 6)}
        _log(provider, rec)
        if ok:
            _cache[h] = rec
            return {"text": text, "usage": usage, "latency": lat, "cost": cost, "cached": False, "ok": True,
                    "finish": finish}
        STATS["errors"] += 1
        last = f"status={status} {str(raw)[:300]}"
        if status == 429:
            STATS["rate_limited"] += 1
            if provider != "gemini":  # gemini: the lane cools down, another key takes the retry
                time.sleep(min(90, 10 * (attempt + 1)) + random.random() * 3)
            continue
        if status in (400, 401, 403, 404, 422):
            break
        time.sleep(min(60, 4 * 2 ** attempt))
    return {"text": None, "usage": {}, "latency": None, "cost": 0.0, "cached": False, "ok": False, "err": last}


def parse_json(text: str):
    """First top-level JSON object in a reply (handles ``` fences)."""
    if not text:
        return None
    t = text.strip()
    if "```" in t:
        for p in t.split("```")[1::2]:
            p2 = p.strip()
            if p2.lower().startswith("json"):
                p2 = p2[4:].strip()
            try:
                return json.loads(p2)
            except Exception:
                pass
    start = t.find("{")
    while start >= 0:
        depth, in_str, esc = 0, False, False
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
