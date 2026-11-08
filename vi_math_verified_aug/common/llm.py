"""The one async LLM client: Claude (anthropic SDK + Message Batches API), Gemini (REST key pool), NIM (off).

Usage::

    from vi_math_verified_aug.common.llm import LLMClient, LLMRequest
    client = LLMClient(ws="G1", budget_usd=5.0)
    req = LLMRequest(model="claude-haiku-5-5", messages=[{"role": "user", "content": "1+1?"}],
                     max_tokens=256, temperature=0.0, item_id="seed:abc", prompt_id="answer_judge",
                     prompt_version=1)
    resp = client.run(client.call(req))          # or: await client.call(req) inside your own loop
    resp.ok, resp.text, resp.cost_usd, resp.cache_hit

Guarantees
----------
* **Every attempt is logged** (``apilog``) to ``logs/api/<ws>/<provider>.jsonl``, cache hits included
  (``cache_hit: true``, without repeating request/response bodies). Keys are never logged.
* **Cache**: successful responses are stored under ``request_hash`` = sha256 of the canonical request
  (provider, model, system, messages, generation params, ``extra``, prompt id+version+sha, ``sample_idx``,
  ``cache_tag``). ``item_id`` and ``tag`` are metadata and NOT part of the key: identical text is one call.
  Use ``sample_idx`` to draw several independent samples of the same prompt.
* **Budget**: hard cap per ``budget_scope`` (default = ws), reserved before and settled after each call
  (see ``costs``). Unknown Claude price -> refuse to call.
* **Routing guard**: Gemini and NIM receive only requests marked ``split="train"`` or ``"dev"``; test /
  benchmark items (or unmarked ones) are refused (fail closed). Claude accepts every split.
* **NIM** is kept but disabled unless ``enable_nim=True`` or ``VIMATH_ENABLE_NIM=1``.
* **Retries per error class** (``RetryPolicy``): rate_limit -> lane cool-down + adaptive TPM cut, retried on
  another key if one exists; overloaded/server/network/timeout -> exponential backoff with jitter;
  content (empty / blocked) -> retried once; client (400/404/422) -> no retry; auth (401/403) -> the key's
  lane is disabled and another key is tried, else ``AuthError``.

API failures do not raise: they return ``LLMResponse(ok=False, error_class=..., error=...)``.
``BudgetExceeded``, ``ForbiddenRoute``, ``ProviderDisabled``, ``UnknownPrice`` and ``AuthError`` do raise.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Protocol

from .apilog import ApiLogger, utc_now
from .cache import ResponseCache
from .costs import BudgetExceeded, CostTracker, Price, UnknownPrice, cost_usd, load_prices, price_for, \
    worst_case_cost
from .env import read_env, secret_values, split_keys
from .hashing import stable_hash
from .paths import cache_dir
from .ratelimit import Lane, LaneConfig, LanePool

__all__ = ["LLMRequest", "LLMResponse", "LLMClient", "RetryPolicy", "RawResult", "ClaudeProvider",
           "GeminiProvider", "NimProvider", "ClaudeBatches", "ForbiddenRoute", "ProviderDisabled", "AuthError",
           "BudgetExceeded", "UnknownPrice", "parse_json", "DEFAULT_LANES"]

SAFE_SPLITS = {"train", "dev"}          # the only splits that may go to Gemini / NIM
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
NIM_BASE = "https://integrate.api.nvidia.com/v1"
ANTHROPIC_BASE = "https://api.anthropic.com"

DEFAULT_LANES: dict[str, LaneConfig] = {
    # one lane per (key, model). G1 probe: 12 RPM/lane gave 48/48 OK (logs/api/G1/gemini_probe.jsonl);
    # 1 in flight per lane, token rate learned from 429s (G1: long outputs -> per-key token-rate 429s).
    "gemini": LaneConfig(rpm=12, burst=1, max_inflight=1, min_tpm=2_000),
    # one lane per model, one key, workspace-level limits; generous defaults, 429s back off adaptively.
    "claude": LaneConfig(rpm=50, burst=5, max_inflight=16, base_cooldown_s=15),
    # T6 §6.4: 38 RPM per 40-RPM quota (5% headroom under 40), ONE bucket regardless of number of keys.
    "nim": LaneConfig(rpm=38, burst=1, max_inflight=35, base_cooldown_s=20),
}


class ForbiddenRoute(RuntimeError):
    """A test/benchmark (or unmarked) item was about to be sent to Gemini or NIM."""


class ProviderDisabled(RuntimeError):
    pass


class AuthError(RuntimeError):
    pass


# =================================================================================================== request
@dataclass
class LLMRequest:
    model: str
    messages: list[dict[str, str]]
    system: str | None = None
    max_tokens: int = 2048
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None                   # Gemini / NIM only (Claude has no seed); always in the hash
    stop: list[str] | None = None
    json_mode: bool = False                   # Gemini responseMimeType / NIM response_format; ignored by Claude
    extra: dict[str, Any] = field(default_factory=dict)   # merged into the provider body (e.g. Claude
    #   {"output_config": {"effort": "low"}}, Gemini {"generationConfig": {"thinkingConfig": {...}}})
    provider: str | None = None               # inferred from the model id if None
    prompt_id: str | None = None
    prompt_version: int | None = None
    prompt_sha256: str | None = None
    item_id: str | None = None
    split: str | None = None                  # train | dev | test | benchmark  (routing guard)
    sample_idx: int = 0
    cache_tag: str = ""
    tag: str = ""                             # free metadata, logged, not hashed
    use_cache: bool = True

    @classmethod
    def from_prompt(cls, rendered: Any, model: str, **kw) -> "LLMRequest":
        """Build from ``prompts.RenderedPrompt`` (system/user text + id/version/sha)."""
        return cls(model=model, system=rendered.system or None,
                   messages=[{"role": "user", "content": rendered.user}], prompt_id=rendered.prompt_id,
                   prompt_version=rendered.version, prompt_sha256=rendered.sha256, **kw)

    def params(self) -> dict[str, Any]:
        p = {"max_tokens": self.max_tokens, "temperature": self.temperature, "top_p": self.top_p,
             "seed": self.seed, "stop": self.stop, "json_mode": self.json_mode or None}
        p = {k: v for k, v in p.items() if v is not None}
        if self.extra:
            p["extra"] = self.extra
        return p

    def request_hash(self, provider: str) -> str:
        return stable_hash({"provider": provider, "model": self.model, "system": self.system,
                            "messages": self.messages, "params": self.params(), "prompt_id": self.prompt_id,
                            "prompt_version": self.prompt_version, "prompt_sha256": self.prompt_sha256,
                            "sample_idx": self.sample_idx, "cache_tag": self.cache_tag})

    def est_input_tokens(self) -> int:
        chars = len(self.system or "") + sum(len(str(m.get("content", ""))) for m in self.messages)
        return chars // 3 + 16           # deliberately high for Vietnamese (diacritics tokenise poorly)


@dataclass
class LLMResponse:
    ok: bool
    text: str | None
    request_hash: str
    provider: str
    model: str
    finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    cost_usd: float = 0.0
    latency_ms: int | None = None
    cache_hit: bool = False
    model_returned: str | None = None
    attempts: int = 0
    error_class: str | None = None
    error: str | None = None
    raw: Any = None

    @property
    def truncated(self) -> bool:
        return (self.finish_reason or "").lower() in {"max_tokens", "length", "max_output_tokens"}

    def json(self) -> Any:
        return parse_json(self.text)

    def cache_payload(self) -> dict[str, Any]:
        return {"text": self.text, "finish_reason": self.finish_reason, "usage": self.usage,
                "model_returned": self.model_returned}


@dataclass
class RawResult:
    """What a provider adapter returns for one HTTP attempt (already normalised)."""
    status: int                                   # HTTP status; -1 network, -2 timeout
    text: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None
    model_returned: str | None = None
    raw: Any = None
    error: str | None = None
    retry_after_s: float | None = None
    content_error: bool = False                   # 200 but empty/blocked


# =================================================================================================== retries
@dataclass
class RetryPolicy:
    max_attempts: dict[str, int] = field(default_factory=lambda: {
        "rate_limit": 8, "overloaded": 5, "server": 4, "network": 4, "timeout": 3, "content": 2,
        "client": 1, "auth": 1})
    base_delay_s: dict[str, float] = field(default_factory=lambda: {
        "overloaded": 8.0, "server": 4.0, "network": 2.0, "timeout": 5.0, "content": 1.0})
    max_delay_s: float = 90.0
    jitter: float = 0.25
    total_attempts: int = 12

    def delay(self, cls: str, n_in_class: int) -> float:
        base = self.base_delay_s.get(cls, 0.0)
        if base <= 0:
            return 0.0
        d = min(self.max_delay_s, base * 2 ** (n_in_class - 1))
        return d * (1 + self.jitter * (2 * random.random() - 1))


def classify(r: RawResult) -> str | None:
    """Error class of an attempt, or None for success."""
    s = r.status
    if s == 200:
        return "content" if r.content_error or r.text is None else None
    if s == 429:
        return "rate_limit"
    if s in (529, 503):
        return "overloaded"
    if s == -2 or s == 408:
        return "timeout"
    if s == -1:
        return "network"
    if s in (401, 403):
        return "auth"
    if s >= 500:
        return "server"
    return "client"


def _retry_after(headers: Any) -> float | None:
    try:
        v = headers.get("retry-after") if headers is not None else None
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


# =================================================================================================== providers
class Provider(Protocol):
    name: str
    base_url: str

    def n_keys(self) -> int: ...

    def lib_version(self) -> str: ...

    async def send(self, req: LLMRequest, key_index: int) -> RawResult: ...

    async def aclose(self) -> None: ...


HttpPost = Callable[[str, dict, dict, float], Awaitable[tuple[int, dict, str]]]
"""async (url, headers, json_body, timeout) -> (status, headers, body_text)."""


class _Httpx2Post:
    """Default REST transport (httpx2, which the anthropic SDK already depends on). Lazy import."""

    def __init__(self):
        self._client = None

    async def __call__(self, url: str, headers: dict, body: dict, timeout: float) -> tuple[int, dict, str]:
        import httpx2
        if self._client is None:
            self._client = httpx2.AsyncClient(timeout=httpx2.Timeout(timeout, connect=30.0))
        try:
            r = await self._client.post(url, headers=headers, json=body, timeout=timeout)
        except httpx2.TimeoutException as e:
            return -2, {}, f"timeout: {type(e).__name__}"
        except httpx2.TransportError as e:
            return -1, {}, f"network: {type(e).__name__}: {e}"
        return r.status_code, {k.lower(): v for k, v in r.headers.items()}, r.text

    async def aclose(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None


class GeminiProvider:
    name = "gemini"
    base_url = GEMINI_BASE

    def __init__(self, keys: list[str], *, post: HttpPost | None = None, timeout: float = 600.0):
        self._keys = list(keys)
        self._post = post or _Httpx2Post()
        self.timeout = timeout

    def n_keys(self) -> int:
        return len(self._keys)

    def lib_version(self) -> str:
        return "rest-v1beta"

    def body(self, req: LLMRequest) -> dict:
        role = {"assistant": "model", "model": "model"}
        b: dict[str, Any] = {"contents": [{"role": role.get(m["role"], "user"), "parts": [{"text": m["content"]}]}
                                          for m in req.messages]}
        if req.system:
            b["systemInstruction"] = {"parts": [{"text": req.system}]}
        gc = {"maxOutputTokens": req.max_tokens, "temperature": req.temperature, "topP": req.top_p,
              "seed": req.seed, "stopSequences": req.stop,
              "responseMimeType": "application/json" if req.json_mode else None}
        b["generationConfig"] = {k: v for k, v in gc.items() if v is not None}
        return _deep_merge(b, req.extra) if req.extra else b

    async def send(self, req: LLMRequest, key_index: int) -> RawResult:
        url = f"{self.base_url}/models/{req.model}:generateContent"
        headers = {"x-goog-api-key": self._keys[key_index], "Content-Type": "application/json"}
        status, hdrs, text = await self._post(url, headers, self.body(req), self.timeout)
        try:
            j = json.loads(text) if text else None
        except json.JSONDecodeError:
            j = None
        if status != 200:
            return RawResult(status, raw=j if j is not None else text, error=(text or "")[:500],
                             retry_after_s=_retry_after(hdrs) or self._retry_info(j))
        return self.parse(j if isinstance(j, dict) else {})

    @staticmethod
    def _retry_info(j: Any) -> float | None:
        try:
            for d in j["error"].get("details", []):
                if str(d.get("@type", "")).endswith("RetryInfo"):
                    return float(str(d["retryDelay"]).rstrip("s"))
        except Exception:
            return None
        return None

    @staticmethod
    def parse(j: dict) -> RawResult:
        cands = j.get("candidates") or []
        u = j.get("usageMetadata") or {}
        thoughts = u.get("thoughtsTokenCount") or 0
        usage = {"input_tokens": u.get("promptTokenCount", 0) or 0,
                 "output_tokens": (u.get("candidatesTokenCount", 0) or 0) + thoughts,
                 "reasoning_tokens": thoughts or None}
        if not cands:
            block = (j.get("promptFeedback") or {}).get("blockReason")
            return RawResult(200, None, usage, block or "NO_CANDIDATES", j.get("modelVersion"), j,
                             error=f"no candidates ({block})", content_error=True)
        c = cands[0]
        parts = (c.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        finish = c.get("finishReason")
        empty = not text and finish not in ("MAX_TOKENS",)
        return RawResult(200, text, usage, finish, j.get("modelVersion"), j,
                         error=f"empty text (finish={finish})" if empty else None, content_error=empty)

    async def aclose(self):
        if hasattr(self._post, "aclose"):
            await self._post.aclose()


class NimProvider:
    """OpenAI-compatible NVIDIA NIM endpoint. Kept for completeness; disabled by default (T6 §4.1: one 40-RPM
    quota, so one lane however many keys exist). Non-streaming: G1 found the gateway returns 504 after ~300 s
    on non-streamed slow reasoning models; port G1's SSE reader (pilot_gen/clients.py) before heavy use."""
    name = "nim"
    base_url = NIM_BASE

    def __init__(self, key: str, *, post: HttpPost | None = None, timeout: float = 900.0):
        self._key = key
        self._post = post or _Httpx2Post()
        self.timeout = timeout

    def n_keys(self) -> int:
        return 1

    def lib_version(self) -> str:
        return "rest-openai-v1"

    def body(self, req: LLMRequest) -> dict:
        msgs = ([{"role": "system", "content": req.system}] if req.system else []) + list(req.messages)
        b = {"model": req.model, "messages": msgs, "max_tokens": req.max_tokens, "temperature": req.temperature,
             "top_p": req.top_p, "seed": req.seed, "stop": req.stop,
             "response_format": {"type": "json_object"} if req.json_mode else None}
        b = {k: v for k, v in b.items() if v is not None}
        return _deep_merge(b, req.extra) if req.extra else b

    async def send(self, req: LLMRequest, key_index: int) -> RawResult:
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        status, hdrs, text = await self._post(f"{self.base_url}/chat/completions", headers, self.body(req),
                                              self.timeout)
        try:
            j = json.loads(text) if text else None
        except json.JSONDecodeError:
            j = None
        if status != 200 or not isinstance(j, dict):
            return RawResult(status if status != 200 else -1, raw=j if j is not None else text,
                             error=(text or "")[:500], retry_after_s=_retry_after(hdrs))
        ch = (j.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        u = j.get("usage") or {}
        usage = {"input_tokens": u.get("prompt_tokens", 0), "output_tokens": u.get("completion_tokens", 0),
                 "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens")}
        text_out = msg.get("content")
        return RawResult(200, text_out, usage, ch.get("finish_reason"), j.get("model"), j,
                         content_error=not text_out, error=None if text_out else "empty content")

    async def aclose(self):
        if hasattr(self._post, "aclose"):
            await self._post.aclose()


class ClaudeProvider:
    """anthropic SDK (async). Header ``anthropic-workspace-id`` from ``ANTHROPIC_WORKSPACE_ID``. SDK retries are
    off (``max_retries=0``): retrying is this module's job. Uses streaming (``messages.stream``) so long
    ``max_tokens`` never trips the SDK's non-streaming time guard. ``sdk`` is injectable for tests."""
    name = "claude"
    base_url = ANTHROPIC_BASE

    def __init__(self, api_key: str | None = None, workspace_id: str | None = None, *, sdk: Any = None,
                 timeout: float = 900.0):
        self._key = api_key
        self._ws = workspace_id
        self._sdk = sdk
        self.timeout = timeout

    @property
    def sdk(self):
        if self._sdk is None:
            import anthropic
            headers = {"anthropic-workspace-id": self._ws} if self._ws else None
            self._sdk = anthropic.AsyncAnthropic(api_key=self._key, max_retries=0, timeout=self.timeout,
                                                 default_headers=headers)
        return self._sdk

    def n_keys(self) -> int:
        return 1

    def lib_version(self) -> str:
        try:
            import anthropic
            return f"anthropic-{anthropic.__version__}"
        except Exception:
            return "anthropic-unknown"

    @staticmethod
    def params(req: LLMRequest) -> dict:
        p: dict[str, Any] = {"model": req.model, "max_tokens": req.max_tokens, "messages": list(req.messages)}
        if req.system:
            p["system"] = req.system
        if req.temperature is not None:
            p["temperature"] = req.temperature
        if req.top_p is not None:
            p["top_p"] = req.top_p
        if req.stop:
            p["stop_sequences"] = req.stop
        return _deep_merge(p, req.extra) if req.extra else p

    @staticmethod
    def parse_message(m: dict) -> RawResult:
        text = "".join(b.get("text", "") for b in m.get("content") or [] if b.get("type") == "text")
        u = m.get("usage") or {}
        usage = {"input_tokens": u.get("input_tokens", 0) or 0, "output_tokens": u.get("output_tokens", 0) or 0,
                 "cache_read": u.get("cache_read_input_tokens", 0) or 0,
                 "cache_write": u.get("cache_creation_input_tokens", 0) or 0}
        stop = m.get("stop_reason")
        empty = not text and stop != "max_tokens"
        return RawResult(200, text, usage, stop, m.get("model"), m, content_error=empty,
                         error=f"empty text (stop={stop})" if empty else None)

    @staticmethod
    def _error(e: Exception) -> RawResult:
        status = getattr(e, "status_code", None)
        name = type(e).__name__
        if status is None:
            status = -2 if "Timeout" in name else -1
        resp = getattr(e, "response", None)
        ra = _retry_after(getattr(resp, "headers", None))
        body = getattr(e, "body", None)
        return RawResult(int(status), raw=body if body is not None else str(e), error=f"{name}: {e}"[:500],
                         retry_after_s=ra)

    async def send(self, req: LLMRequest, key_index: int) -> RawResult:
        try:
            async with self.sdk.messages.stream(**self.params(req)) as s:
                m = await s.get_final_message()
        except Exception as e:                                  # SDK errors carry status_code / response
            return self._error(e)
        d = m.model_dump() if hasattr(m, "model_dump") else dict(m)
        return self.parse_message(d)

    # ---------------------------------------------------------------- Message Batches API
    async def batch_create(self, items: list[dict]) -> dict:
        b = await self.sdk.messages.batches.create(requests=items)
        return b.model_dump() if hasattr(b, "model_dump") else dict(b)

    async def batch_retrieve(self, batch_id: str) -> dict:
        b = await self.sdk.messages.batches.retrieve(batch_id)
        return b.model_dump() if hasattr(b, "model_dump") else dict(b)

    async def batch_results(self, batch_id: str) -> list[dict]:
        out = []
        res = await self.sdk.messages.batches.results(batch_id)
        async for r in res:
            out.append(r.model_dump() if hasattr(r, "model_dump") else dict(r))
        return out

    async def aclose(self):
        if self._sdk is not None and hasattr(self._sdk, "close"):
            try:
                await self._sdk.close()
            except Exception:
                pass
            if self._key is not None:          # only drop SDK clients we created
                self._sdk = None


# =================================================================================================== client
def infer_provider(model: str) -> str:
    m = model.lower()
    if m.startswith("claude"):
        return "claude"
    if m.startswith(("gemini", "gemma", "models/gemini")):
        return "gemini"
    if "/" in m:
        return "nim"
    raise ValueError(f"cannot infer provider for model {model!r}; set LLMRequest.provider")


class LLMClient:
    def __init__(self, ws: str, *, run_id: str | None = None, env: dict[str, str] | None = None,
                 budget_usd: float | None = None, budget_scope: str | None = None,
                 cache: bool | ResponseCache = True, state_dir: str | os.PathLike | None = None,
                 log_dir: str | os.PathLike | None = None, providers: dict[str, Any] | None = None,
                 enable_nim: bool | None = None, lanes: dict[str, LaneConfig] | None = None,
                 retry: RetryPolicy | None = None, prices: dict[str, Any] | None = None,
                 max_concurrency: int = 64, log_cache_hits: bool = True,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep):
        self.ws = ws
        self.env = read_env() if env is None else env
        self.state_dir = Path(state_dir) if state_dir else cache_dir()
        self.logger = ApiLogger(ws, run_id=run_id, log_dir=log_dir, secrets=secret_values(self.env))
        if cache is True:
            self.cache: ResponseCache | None = ResponseCache(self.state_dir / "llm_cache.sqlite")
        else:
            self.cache = cache or None
        self.costs = CostTracker(budget_usd, db_path=self.state_dir / "spend.sqlite", scope=budget_scope or ws)
        self.prices = load_prices(prices)
        self.retry = retry or RetryPolicy()
        self.lane_cfg = {**DEFAULT_LANES, **(lanes or {})}
        nim_on = enable_nim if enable_nim is not None else self.env.get("VIMATH_ENABLE_NIM") == "1"
        self._providers: dict[str, Any] = dict(providers or {})
        self._nim_enabled = nim_on or "nim" in self._providers
        self._pools: dict[tuple[str, str], LanePool] = {}
        self._dead: set[tuple[str, str, int]] = set()
        self._sem_n = max_concurrency
        self._sem: asyncio.Semaphore | None = None
        self._inflight: dict[str, asyncio.Future] = {}
        self.log_cache_hits = log_cache_hits
        self.clock, self.sleep = clock, sleep
        self.stats = {"calls": 0, "attempts": 0, "cache_hits": 0, "errors": 0, "rate_limited": 0}
        self.batches = ClaudeBatches(self)

    # ---------------------------------------------------------------- plumbing
    def provider(self, name: str):
        if name in self._providers:
            return self._providers[name]
        if name == "claude":
            key = self.env.get("ANTHROPIC_API_KEY")
            if not key:
                raise ProviderDisabled("ANTHROPIC_API_KEY not set")
            p = ClaudeProvider(key, self.env.get("ANTHROPIC_WORKSPACE_ID") or None)
        elif name == "gemini":
            keys = split_keys(self.env.get("GEMINI_API_KEYS") or self.env.get("GEMINI_API_KEY"))
            if not keys:
                raise ProviderDisabled("GEMINI_API_KEYS not set")
            p = GeminiProvider(keys)
        elif name == "nim":
            if not self._nim_enabled:
                raise ProviderDisabled("NIM is off by default; pass enable_nim=True or set VIMATH_ENABLE_NIM=1")
            key = self.env.get("NVIDIA_API_KEY")
            if not key:
                raise ProviderDisabled("NVIDIA_API_KEY not set")
            p = NimProvider(key)
        else:
            raise ProviderDisabled(f"unknown provider {name!r}")
        self._providers[name] = p
        return p

    def pool(self, provider: str, model: str) -> LanePool:
        k = (provider, model)
        if k not in self._pools:
            prov = self.provider(provider)
            cfg = self.lane_cfg[provider]
            lanes = [Lane(f"{provider}:k{i}:{model}", i, cfg, clock=self.clock) for i in range(prov.n_keys())]
            self._pools[k] = LanePool(lanes, clock=self.clock, sleep=self.sleep)
        return self._pools[k]

    def _semaphore(self) -> asyncio.Semaphore:
        if self._sem is None:
            self._sem = asyncio.Semaphore(self._sem_n)
        return self._sem

    def _guard(self, provider: str, req: LLMRequest) -> None:
        if provider == "nim" and not self._nim_enabled:
            raise ProviderDisabled("NIM is off by default; pass enable_nim=True or set VIMATH_ENABLE_NIM=1")
        if provider in ("gemini", "nim") and (req.split or "").lower() not in SAFE_SPLITS:
            raise ForbiddenRoute(f"item {req.item_id!r} split={req.split!r} may not be sent to {provider} "
                                 f"(only {sorted(SAFE_SPLITS)}; test/benchmark items go to Claude only)")

    def _price(self, provider: str, model: str) -> Price:
        return price_for(model, provider, self.prices)

    def _base_record(self, provider: str, req: LLMRequest, h: str) -> dict:
        prov = self._providers.get(provider)
        return {"provider": provider, "base_url": getattr(prov, "base_url", None), "model_requested": req.model,
                "params": req.params(), "prompt_id": req.prompt_id, "prompt_version": req.prompt_version,
                "prompt_sha256": req.prompt_sha256, "item_id": req.item_id, "split": req.split,
                "sample_idx": req.sample_idx, "tag": req.tag, "request_hash": h,
                "client_lib_version": prov.lib_version() if prov is not None else None}

    # ---------------------------------------------------------------- public API
    async def call(self, req: LLMRequest) -> LLMResponse:
        provider = req.provider or infer_provider(req.model)
        self._guard(provider, req)
        h = req.request_hash(provider)
        self.stats["calls"] += 1
        if self.cache is not None and req.use_cache:
            hit = self.cache.get(h)
            if hit is not None:
                self.stats["cache_hits"] += 1
                resp = LLMResponse(True, hit.get("text"), h, provider, req.model, hit.get("finish_reason"),
                                   hit.get("usage") or {}, 0.0, 0, True, hit.get("model_returned"), 0)
                if self.log_cache_hits:
                    self.logger.log({**self._base_record(provider, req, h), "ok": True, "cache_hit": True,
                                     "cost_usd": 0.0, "model_returned": resp.model_returned,
                                     "finish_reason": resp.finish_reason, "usage": resp.usage})
                return resp
        price = self._price(provider, req.model)
        prov = self.provider(provider)
        pool = self.pool(provider, req.model)
        if not (self.cache is not None and req.use_cache):
            async with self._semaphore():
                return await self._call_live(req, provider, prov, pool, price, h)
        # single-flight: an identical request already in flight is awaited, then served from the cache
        leader = self._inflight.get(h)
        if leader is not None:
            await asyncio.shield(leader)
            self.stats["calls"] -= 1                     # the re-entry below counts it
            return await self.call(req)
        fut = asyncio.get_running_loop().create_future()
        self._inflight[h] = fut
        try:
            async with self._semaphore():
                return await self._call_live(req, provider, prov, pool, price, h)
        finally:
            self._inflight.pop(h, None)
            fut.set_result(None)

    async def _call_live(self, req, provider, prov, pool: LanePool, price: Price, h: str) -> LLMResponse:
        per_class: dict[str, int] = {}
        exclude: set[int] = set()
        attempt = 0
        last: RawResult | None = None
        last_cls: str | None = None
        while True:
            attempt += 1
            est_in = req.est_input_tokens()
            rid = None
            if price.input or price.output:
                rid = self.costs.reserve(worst_case_cost(est_in, req.max_tokens, price))
            dead = {i for (p, m, i) in self._dead if p == provider and m == req.model}
            try:
                lane, ticket = await pool.acquire(est_in, req.max_tokens, exclude=exclude | dead)
            except BaseException:                  # LaneStall / cancellation: do not leak the budget reservation
                self.costs.release(rid)
                raise
            t0 = time.perf_counter()
            try:
                r = await prov.send(req, lane.key_index)
            except Exception as e:                           # adapter bug or unexpected transport error
                r = RawResult(-1, error=f"{type(e).__name__}: {e}"[:500])
            latency_ms = int((time.perf_counter() - t0) * 1000)
            self.stats["attempts"] += 1
            cls = classify(r)
            cost = 0.0
            if r.status == 200 and r.usage:                  # billed (also when the text is empty/truncated)
                cost = cost_usd(r.usage, price)
                self.costs.charge(f"{h}:{uuid.uuid4().hex}", cost, provider=provider, model=req.model, rid=rid)
            else:
                self.costs.release(rid)
            if cls == "rate_limit":
                self.stats["rate_limited"] += 1
                lane.rate_limited(r.retry_after_s)
                lane.finish(ticket, input_tokens=0, output_tokens=0, ok=False)
                if len(pool.lanes) > 1:
                    exclude = {lane.key_index}
            elif cls is None:
                lane.finish(ticket, input_tokens=r.usage.get("input_tokens"),
                            output_tokens=r.usage.get("output_tokens"), ok=True)
                exclude = set()
            else:
                lane.finish(ticket, input_tokens=r.usage.get("input_tokens", 0),
                            output_tokens=r.usage.get("output_tokens", 0), ok=False)
            self.logger.log({**self._base_record(provider, req, h), "lane": lane.name, "attempt": attempt,
                             "status": r.status, "ok": cls is None, "error_class": cls, "error": r.error,
                             "cache_hit": False, "request": {"system": req.system, "messages": req.messages},
                             "response_text": r.text, "raw_response": r.raw, "finish_reason": r.finish_reason,
                             "model_returned": r.model_returned, "usage": r.usage, "latency_ms": latency_ms,
                             "cost_usd": round(cost, 8), "retry_after_s": r.retry_after_s})
            if cls is None:
                resp = LLMResponse(True, r.text, h, provider, req.model, r.finish_reason, r.usage, cost, latency_ms,
                                   False, r.model_returned, attempt, raw=r.raw)
                if self.cache is not None and req.use_cache:
                    self.cache.put(h, provider, req.model, resp.cache_payload())
                return resp
            self.stats["errors"] += 1
            last, last_cls = r, cls
            per_class[cls] = per_class.get(cls, 0) + 1
            if cls == "auth":
                self._dead.add((provider, req.model, lane.key_index))
                lane.cooldown_until = float("inf")
                alive = [ln for ln in pool.lanes if (provider, req.model, ln.key_index) not in self._dead]
                if not alive:
                    raise AuthError(f"{provider}: every key was refused ({r.status}): {r.error}")
                continue
            if per_class[cls] >= self.retry.max_attempts.get(cls, 1) or attempt >= self.retry.total_attempts:
                break
            d = self.retry.delay(cls, per_class[cls])
            if d > 0:
                await self.sleep(d)
        return LLMResponse(False, None, h, provider, req.model, last.finish_reason if last else None,
                           last.usage if last else {}, 0.0, None, False, last.model_returned if last else None,
                           attempt, last_cls, last.error if last else None, last.raw if last else None)

    async def map(self, reqs: Iterable[LLMRequest], *, return_exceptions: bool = False) -> list:
        """Run requests concurrently (bounded by ``max_concurrency`` and the lanes); results in input order."""
        return await asyncio.gather(*(self.call(r) for r in reqs), return_exceptions=return_exceptions)

    def run(self, coro: Awaitable[Any]) -> Any:
        """Sync helper: run ``coro`` on a fresh event loop and close HTTP clients afterwards."""
        async def _wrap():
            try:
                return await coro
            finally:
                await self.aclose()
        return asyncio.run(_wrap())

    async def aclose(self) -> None:
        for p in self._providers.values():
            if hasattr(p, "aclose"):
                await p.aclose()
        self._sem = None

    def close(self) -> None:
        if self.cache is not None:
            self.cache.close()
        self.costs.close()

    def summary(self) -> dict:
        return {**self.stats, **self.costs.summary(),
                "lanes": {f"{p}:{m}": pool.snapshot() for (p, m), pool in self._pools.items()}}


# =================================================================================================== batches
class ClaudeBatches:
    """Claude Message Batches API (50% price, results within 24 h).

    ``submit`` -> ``status`` -> ``collect`` (or ``run`` for all three). Cached requests never reach the API;
    identical requests are sent once. A manifest ``<state_dir>/batches/<batch_id>.json`` keeps the requests so
    a crashed process can ``collect`` later. Charges use the id ``batch:<batch_id>:<custom_id>``, so collecting
    the same batch twice never double-counts.
    """

    def __init__(self, client: LLMClient):
        self.c = client
        self._reservations: dict[str, int] = {}

    def _dir(self) -> Path:
        d = self.c.state_dir / "batches"
        d.mkdir(parents=True, exist_ok=True)
        return d

    async def submit(self, reqs: list[LLMRequest]) -> dict:
        c = self.c
        todo: dict[str, LLMRequest] = {}
        cached: list[str] = []
        for r in reqs:
            prov = r.provider or infer_provider(r.model)
            if prov != "claude":
                raise ValueError(f"batches are Claude-only; got {r.model!r}")
            h = r.request_hash("claude")
            if c.cache is not None and r.use_cache and h in c.cache:
                cached.append(h)
            else:
                todo.setdefault(h, r)
        manifest = {"batch_id": None, "created_utc": utc_now(), "n_requests": len(reqs), "n_cached": len(cached),
                    "items": {h: dataclasses.asdict(r) for h, r in todo.items()}}
        if not todo:
            return manifest
        est = sum(worst_case_cost(r.est_input_tokens(), r.max_tokens, c._price("claude", r.model), batch=True)
                  for r in todo.values())
        rid = c.costs.reserve(est)
        prov = c.provider("claude")
        try:
            b = await prov.batch_create([{"custom_id": h, "params": ClaudeProvider.params(r)}
                                         for h, r in todo.items()])
        except Exception:
            c.costs.release(rid)
            raise
        manifest["batch_id"] = b["id"]
        manifest["reserved_usd"] = est
        self._reservations[b["id"]] = rid
        (self._dir() / f"{b['id']}.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        c.logger.log({"provider": "claude", "event": "batch_submit", "batch_id": b["id"],
                      "n_items": len(todo), "n_cached": len(cached), "reserved_usd": round(est, 6),
                      "request_hashes": list(todo), "client_lib_version": prov.lib_version()})
        return manifest

    async def status(self, batch_id: str) -> dict:
        return await self.c.provider("claude").batch_retrieve(batch_id)

    async def collect(self, batch_id: str) -> dict[str, LLMResponse]:
        c = self.c
        man = json.loads((self._dir() / f"{batch_id}.json").read_text(encoding="utf-8"))
        items = {h: LLMRequest(**d) for h, d in man["items"].items()}
        prov = c.provider("claude")
        out: dict[str, LLMResponse] = {}
        for entry in await prov.batch_results(batch_id):
            h = entry.get("custom_id")
            req = items.get(h)
            if req is None:
                continue
            res = entry.get("result") or {}
            base = {**c._base_record("claude", req, h), "batch_id": batch_id, "attempt": 1, "cache_hit": False,
                    "request": {"system": req.system, "messages": req.messages}}
            if res.get("type") == "succeeded":
                r = ClaudeProvider.parse_message(res.get("message") or {})
                cost = cost_usd(r.usage, c._price("claude", req.model), batch=True)
                c.costs.charge(f"batch:{batch_id}:{h}", cost, provider="claude", model=req.model)
                cls = classify(r)
                ok = cls is None
                resp = LLMResponse(ok, r.text, h, "claude", req.model, r.finish_reason, r.usage, cost, None, False,
                                   r.model_returned, 1, cls, r.error, r.raw)
                if ok and c.cache is not None and req.use_cache:
                    c.cache.put(h, "claude", req.model, resp.cache_payload())
                c.logger.log({**base, "status": 200, "ok": ok, "error_class": cls, "error": r.error,
                              "response_text": r.text, "raw_response": r.raw, "finish_reason": r.finish_reason,
                              "model_returned": r.model_returned, "usage": r.usage, "cost_usd": round(cost, 8)})
            else:
                kind = res.get("type") or "unknown"
                err = json.dumps(res.get("error"), ensure_ascii=False, default=str)[:500] if res.get("error") else kind
                resp = LLMResponse(False, None, h, "claude", req.model, attempts=1, error_class=f"batch_{kind}",
                                   error=err, raw=res)
                c.logger.log({**base, "status": None, "ok": False, "error_class": f"batch_{kind}", "error": err,
                              "raw_response": res, "cost_usd": 0.0})
            out[h] = resp
        c.costs.release(self._reservations.pop(batch_id, None))
        return out

    async def run(self, reqs: list[LLMRequest], *, poll_s: float = 30.0, timeout_s: float | None = None
                  ) -> list[LLMResponse]:
        """Submit, wait until the batch has ended, collect; returns responses in input order (cache hits
        included)."""
        man = await self.submit(reqs)
        got: dict[str, LLMResponse] = {}
        if man["batch_id"]:
            t0 = self.c.clock()
            while True:
                st = await self.status(man["batch_id"])
                if st.get("processing_status") == "ended":
                    break
                if timeout_s is not None and self.c.clock() - t0 > timeout_s:
                    raise TimeoutError(f"batch {man['batch_id']} not ended after {timeout_s}s; collect() later")
                await self.c.sleep(poll_s)
            got = await self.collect(man["batch_id"])
        out = []
        for r in reqs:
            h = r.request_hash("claude")
            if h in got:
                out.append(got[h])
            else:                                            # was cached at submit time
                out.append(await self.c.call(r))
        return out


# =================================================================================================== helpers
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


def parse_json(text: str | None) -> Any:
    """First JSON object/array in a model reply (fenced block first, then a balanced-brace scan)."""
    if not text:
        return None
    for m in _FENCE.finditer(text):
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                return dec.raw_decode(text, i)[0]
            except json.JSONDecodeError:
                continue
    return None
