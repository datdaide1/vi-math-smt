"""common/llm.py, common/costs.py, common/cache.py — all transports mocked; no network, no keys."""
from __future__ import annotations

import asyncio
import json
import sys
import types

import pytest

from vi_math_verified_aug.common import costs as costs_mod
from vi_math_verified_aug.common.costs import BudgetExceeded, CostTracker, Price, cost_usd
from vi_math_verified_aug.common.llm import (AuthError, ClaudeProvider, ForbiddenRoute, GeminiProvider, LLMClient,
                                             LLMRequest, ProviderDisabled, RawResult, RetryPolicy, parse_json)
from vi_math_verified_aug.common.ratelimit import LaneConfig

FAST = LaneConfig(rpm=60_000, burst=1_000, max_inflight=1_000)
SECRET = "AIzaSECRETKEY0123456789"


class FakeClock:
    def __init__(self):
        self.t = 1000.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    async def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s
        await asyncio.sleep(0)


class FakeProvider:
    """Scripted provider: ``script`` is a list of RawResult (or callables(req, key) -> RawResult) consumed in
    order; when exhausted, every call succeeds."""

    def __init__(self, name="claude", n_keys=1, script=None, usage=None):
        self.name = name
        self.base_url = f"https://fake/{name}"
        self._n = n_keys
        self.script = list(script or [])
        self.calls: list[tuple[str, int]] = []
        self.usage = usage or {"input_tokens": 10, "output_tokens": 20}

    def n_keys(self):
        return self._n

    def lib_version(self):
        return "fake-1"

    async def send(self, req, key_index):
        self.calls.append((req.messages[-1]["content"], key_index))
        await asyncio.sleep(0)
        if self.script:
            r = self.script.pop(0)
            return r(req, key_index) if callable(r) else r
        return RawResult(200, f"answer to {req.messages[-1]['content']}", dict(self.usage), "end_turn", req.model,
                         {"echo": req.messages[-1]["content"]})

    async def aclose(self):
        pass


def make_client(tmp_path, providers, *, budget=None, prices=None, clock=None, retry=None, ws="TEST", **kw):
    clock = clock or FakeClock()
    return LLMClient(ws, env={"GEMINI_API_KEYS": SECRET}, state_dir=tmp_path / "state", log_dir=tmp_path / "logs",
                     providers=providers, budget_usd=budget,
                     prices=prices or {"claude-test": (1000.0, 1000.0)},  # $0.001 / token: easy arithmetic
                     lanes={"claude": FAST, "gemini": FAST, "nim": FAST}, clock=clock.now, sleep=clock.sleep,
                     retry=retry, **kw)


def req(content="q", model="claude-test", **kw):
    return LLMRequest(model=model, messages=[{"role": "user", "content": content}], max_tokens=100, **kw)


def log_lines(tmp_path, ws="TEST", provider="claude"):
    p = tmp_path / "logs" / f"{provider}.jsonl"
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []


# ------------------------------------------------------------------------------------------------- basics
def test_success_logged_cached_then_cache_hit(tmp_path):
    fp = FakeProvider()
    c = make_client(tmp_path, {"claude": fp})
    r1 = asyncio.run(c.call(req("2+2?", prompt_id="answer_judge", prompt_version=1, item_id="i1")))
    assert r1.ok and r1.text == "answer to 2+2?" and not r1.cache_hit
    assert r1.cost_usd == pytest.approx(30 * 0.001)
    r2 = asyncio.run(c.call(req("2+2?", prompt_id="answer_judge", prompt_version=1, item_id="other")))
    assert r2.cache_hit and r2.text == r1.text and r2.cost_usd == 0.0
    assert len(fp.calls) == 1                                      # second call never reached the provider
    recs = log_lines(tmp_path)
    assert [r["cache_hit"] for r in recs] == [False, True]
    live = recs[0]
    for k in ("ts_utc", "ws", "provider", "model_requested", "params", "prompt_id", "prompt_version", "request",
              "raw_response", "latency_ms", "usage", "cost_usd", "request_hash", "attempt", "client_lib_version"):
        assert k in live, k
    assert live["request"]["messages"][0]["content"] == "2+2?"
    assert c.costs.spent() == pytest.approx(0.03)


def test_hash_depends_on_content_and_sample_idx_not_item_id():
    a, b = req("x", item_id="1"), req("x", item_id="2")
    assert a.request_hash("claude") == b.request_hash("claude")
    assert a.request_hash("claude") != req("x", sample_idx=1).request_hash("claude")
    assert a.request_hash("claude") != req("x", temperature=0.5).request_hash("claude")
    assert a.request_hash("claude") != req("x", prompt_version=2).request_hash("claude")


def test_single_flight_identical_concurrent_requests_pay_once(tmp_path):
    fp = FakeProvider()
    c = make_client(tmp_path, {"claude": fp})

    async def go():
        return await asyncio.gather(*(c.call(req("same")) for _ in range(5)))
    rs = asyncio.run(go())
    assert len(fp.calls) == 1 and all(r.ok for r in rs)
    assert sum(r.cost_usd for r in rs) == pytest.approx(c.costs.spent()) == pytest.approx(0.03)


# ------------------------------------------------------------------------------------------------- retries
def test_retry_overloaded_then_success(tmp_path):
    clock = FakeClock()
    fp = FakeProvider(script=[RawResult(529, error="overloaded"), RawResult(503, error="unavailable")])
    c = make_client(tmp_path, {"claude": fp}, clock=clock)
    r = asyncio.run(c.call(req()))
    assert r.ok and r.attempts == 3
    assert len([s for s in clock.sleeps if s >= 1]) == 2           # exponential backoff slept twice
    assert [x["error_class"] for x in log_lines(tmp_path)] == ["overloaded", "overloaded", None]


def test_client_error_not_retried(tmp_path):
    fp = FakeProvider(script=[RawResult(400, error="bad request")])
    c = make_client(tmp_path, {"claude": fp})
    r = asyncio.run(c.call(req()))
    assert not r.ok and r.error_class == "client" and r.attempts == 1 and len(fp.calls) == 1
    assert c.costs.spent() == 0 and c.costs.reserved() == 0     # reservation released
    assert c.cache.get(r.request_hash) is None                    # failures are not cached


def test_network_errors_give_up_after_policy(tmp_path):
    fp = FakeProvider(script=[RawResult(-1, error="conn reset")] * 10)
    c = make_client(tmp_path, {"claude": fp})
    r = asyncio.run(c.call(req()))
    assert not r.ok and r.error_class == "network" and r.attempts == RetryPolicy().max_attempts["network"]


def test_rate_limit_moves_to_other_key_and_cools_lane(tmp_path):
    fp = FakeProvider(name="gemini", n_keys=2,
                      script=[RawResult(429, error="quota", retry_after_s=30.0)])
    c = make_client(tmp_path, {"gemini": fp})
    r = asyncio.run(c.call(req(model="gemini-x", split="train")))
    assert r.ok and r.attempts == 2
    assert fp.calls[0][1] != fp.calls[1][1]                        # retried on the other key
    pool = c.pool("gemini", "gemini-x")
    hit = [ln for ln in pool.lanes if ln.n_429 == 1][0]
    assert hit.cooldown_until - c.clock() == pytest.approx(30.0, abs=0.01)


def test_auth_error_disables_key_then_raises_when_none_left(tmp_path):
    fp = FakeProvider(name="gemini", n_keys=2, script=[RawResult(403, error="denied")])
    c = make_client(tmp_path, {"gemini": fp})
    assert asyncio.run(c.call(req("a", model="gemini-x", split="dev"))).ok      # second key works
    fp2 = FakeProvider(name="claude", script=[RawResult(401, error="bad key")])
    c2 = make_client(tmp_path / "b", {"claude": fp2})
    with pytest.raises(AuthError):
        asyncio.run(c2.call(req()))


# ------------------------------------------------------------------------------------------------- routing guard
@pytest.mark.parametrize("split", ["test", "benchmark", None, "TEST"])
def test_test_items_never_go_to_gemini(tmp_path, split):
    fp = FakeProvider(name="gemini")
    c = make_client(tmp_path, {"gemini": fp})
    with pytest.raises(ForbiddenRoute):
        asyncio.run(c.call(req(model="gemini-x", split=split)))
    assert fp.calls == []


def test_test_items_may_go_to_claude(tmp_path):
    c = make_client(tmp_path, {"claude": FakeProvider()})
    assert asyncio.run(c.call(req(split="test"))).ok


def test_nim_off_by_default(tmp_path):
    c = LLMClient("TEST", env={"NVIDIA_API_KEY": "nvapi-xxxxxxxxxxxx"}, state_dir=tmp_path, log_dir=tmp_path / "l")
    with pytest.raises(ProviderDisabled):
        asyncio.run(c.call(req(model="deepseek-ai/deepseek-v4.1-flash", split="train")))


# ------------------------------------------------------------------------------------------------- budget / costs
def test_hard_budget_cap_under_concurrency_and_exact_accounting(tmp_path):
    fp = FakeProvider()
    # worst case per call = (est_in 16 + max_tokens 100) * 0.001 = 0.116 ; actual = 30 * 0.001 = 0.03
    c = make_client(tmp_path, {"claude": fp}, budget=0.5)

    async def go():
        return await asyncio.gather(*(c.call(req(f"q{i}")) for i in range(20)), return_exceptions=True)
    rs = asyncio.run(go())
    ok = [r for r in rs if not isinstance(r, BaseException)]
    refused = [r for r in rs if isinstance(r, BudgetExceeded)]
    assert len(ok) + len(refused) == 20 and refused and ok
    assert len(fp.calls) == len(ok)                              # refused calls never hit the API
    assert c.costs.spent() == pytest.approx(sum(r.cost_usd for r in ok))
    assert c.costs.spent() <= 0.5 and c.costs.reserved() == 0


def test_no_overcount_with_cache_hits_and_restarts(tmp_path):
    """G1 bug: concurrent cache loads inflated spend. Re-running the same requests (same process, then a fresh
    client on the same state dir = a restarted process) must not add a cent."""
    fp = FakeProvider()
    c = make_client(tmp_path, {"claude": fp})

    async def go(cl):
        return await asyncio.gather(*(cl.call(req(f"q{i}")) for i in range(10)))
    first = asyncio.run(go(c))
    spent = c.costs.spent()
    assert spent == pytest.approx(sum(r.cost_usd for r in first)) == pytest.approx(0.3)
    again = asyncio.run(go(c))
    assert all(r.cache_hit for r in again) and c.costs.spent() == pytest.approx(spent)
    c2 = make_client(tmp_path, {"claude": FakeProvider()})
    third = asyncio.run(go(c2))
    assert all(r.cache_hit for r in third)
    assert c2.costs.spent() == pytest.approx(spent)               # persisted, not re-summed from logs


def test_unknown_claude_price_refuses(tmp_path):
    c = make_client(tmp_path, {"claude": FakeProvider()})
    with pytest.raises(costs_mod.UnknownPrice):
        asyncio.run(c.call(req(model="claude-mystery-9")))


def test_cost_tracker_idempotent_and_shared_across_instances(tmp_path):
    db = tmp_path / "spend.sqlite"
    a = CostTracker(1.0, db_path=db, scope="X")
    b = CostTracker(1.0, db_path=db, scope="X")
    assert a.charge("c1", 0.25) and not a.charge("c1", 0.25)
    assert b.charge("c2", 0.25)
    assert a.spent() == pytest.approx(0.5) and b.spent() == pytest.approx(0.5)
    rid = a.reserve(0.4)
    with pytest.raises(BudgetExceeded):
        a.reserve(0.2)
    a.release(rid)
    assert a.reserve(0.2)
    assert CostTracker(None, db_path=db, scope="other").spent() == 0


def test_cost_formula():
    p = Price(4.0, 20.0)
    u = {"input_tokens": 1000, "output_tokens": 500, "cache_read": 2000, "cache_write": 100}
    full = (1000 * 4 + 500 * 20 + 2000 * 4 * 0.1 + 100 * 4 * 1.25) / 1e6
    assert cost_usd(u, p) == pytest.approx(full)
    assert cost_usd(u, p, batch=True) == pytest.approx(full / 2)


# ------------------------------------------------------------------------------------------------- Gemini REST
def test_gemini_rest_body_headers_parse_and_key_never_logged(tmp_path):
    seen = []

    async def post(url, headers, body, timeout):
        seen.append((url, headers, body))
        if len(seen) == 1:  # 429 with RetryInfo in the body; echo the key in the error text to test scrubbing
            return 429, {}, json.dumps({"error": {"code": 429, "message": f"quota for key {SECRET}", "details": [
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "37s"}]}})
        return 200, {}, json.dumps({
            "candidates": [{"content": {"parts": [{"text": "think", "thought": True}, {"text": "42"}]},
                            "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 3, "thoughtsTokenCount": 5},
            "modelVersion": "gemini-x-001"})

    gp = GeminiProvider([SECRET], post=post)
    c = make_client(tmp_path, {"gemini": gp})
    r = asyncio.run(c.call(LLMRequest(model="gemini-x", system="sys", messages=[{"role": "user", "content": "hi"}],
                                      max_tokens=64, temperature=0.0, seed=7, json_mode=True, split="train")))
    assert r.ok and r.text == "42" and r.model_returned == "gemini-x-001"
    assert r.usage["input_tokens"] == 11 and r.usage["output_tokens"] == 8 and r.usage["reasoning_tokens"] == 5
    url, headers, body = seen[-1]
    assert url.endswith("/models/gemini-x:generateContent") and SECRET not in url
    assert headers["x-goog-api-key"] == SECRET
    assert body["systemInstruction"]["parts"][0]["text"] == "sys"
    assert body["generationConfig"] == {"maxOutputTokens": 64, "temperature": 0.0, "seed": 7,
                                        "responseMimeType": "application/json"}
    lane = c.pool("gemini", "gemini-x").lanes[0]
    assert lane.n_429 == 1 and lane.tpm_limit is not None           # adaptive token-rate engaged
    text = (tmp_path / "logs" / "gemini.jsonl").read_text(encoding="utf-8")
    assert SECRET not in text and "REDACTED" in text


def test_gemini_empty_candidates_is_content_error_retried_once(tmp_path):
    async def post(url, headers, body, timeout):
        return 200, {}, json.dumps({"promptFeedback": {"blockReason": "SAFETY"}, "usageMetadata": {}})
    c = make_client(tmp_path, {"gemini": GeminiProvider([SECRET], post=post)})
    r = asyncio.run(c.call(req(model="gemini-x", split="train")))
    assert not r.ok and r.error_class == "content" and r.attempts == 2


# ------------------------------------------------------------------------------------------------- Claude SDK
class _Msg:
    def __init__(self, d):
        self.d = d

    def model_dump(self):
        return self.d


def _claude_msg(text="ok", model="claude-test", i=12, o=7):
    return {"id": "msg_1", "model": model, "stop_reason": "end_turn", "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": i, "output_tokens": o, "cache_read_input_tokens": 0,
                      "cache_creation_input_tokens": 0}}


class FakeSDK:
    """Mimics the parts of anthropic.AsyncAnthropic that ClaudeProvider uses."""

    def __init__(self, fail_first=None):
        self.kwargs = []
        self.fail_first = list(fail_first or [])
        self.batch_requests = None
        self.results_calls = 0
        sdk = self

        class _Stream:
            def __init__(self, kw):
                self.kw = kw

            async def __aenter__(self):
                if sdk.fail_first:
                    raise sdk.fail_first.pop(0)
                return self

            async def __aexit__(self, *a):
                return False

            async def get_final_message(self):
                return _Msg(_claude_msg("re: " + self.kw["messages"][-1]["content"], self.kw["model"]))

        class _Batches:
            async def create(self, requests):
                sdk.batch_requests = requests
                return _Msg({"id": "msgbatch_1", "processing_status": "in_progress"})

            async def retrieve(self, bid):
                return _Msg({"id": bid, "processing_status": "ended"})

            async def results(self, bid):
                sdk.results_calls += 1

                async def gen():
                    for i, r in enumerate(sdk.batch_requests):
                        if i == len(sdk.batch_requests) - 1:
                            yield _Msg({"custom_id": r["custom_id"],
                                        "result": {"type": "errored", "error": {"type": "invalid_request"}}})
                        else:
                            m = _claude_msg("b:" + r["params"]["messages"][-1]["content"], r["params"]["model"],
                                            100, 100)
                            yield _Msg({"custom_id": r["custom_id"], "result": {"type": "succeeded", "message": m}})
                return gen()

        class _Messages:
            batches = _Batches()

            def stream(self, **kw):
                sdk.kwargs.append(kw)
                return _Stream(kw)

        self.messages = _Messages()


class _StatusError(Exception):
    def __init__(self, status, headers=None):
        super().__init__(f"status {status}")
        self.status_code = status
        self.response = types.SimpleNamespace(headers=headers or {})


def test_claude_provider_params_and_error_mapping(tmp_path):
    sdk = FakeSDK(fail_first=[_StatusError(529), _StatusError(429, {"retry-after": "2"})])
    c = make_client(tmp_path, {"claude": ClaudeProvider(sdk=sdk)})
    r = asyncio.run(c.call(LLMRequest(model="claude-test", system="S", messages=[{"role": "user", "content": "x"}],
                                      max_tokens=50, temperature=0.0, seed=1, stop=["##"],
                                      extra={"output_config": {"effort": "low"}})))
    assert r.ok and r.text == "re: x" and r.attempts == 3
    kw = sdk.kwargs[-1]
    assert kw == {"model": "claude-test", "max_tokens": 50, "messages": [{"role": "user", "content": "x"}],
                  "system": "S", "temperature": 0.0, "stop_sequences": ["##"], "output_config": {"effort": "low"}}
    assert [x["error_class"] for x in log_lines(tmp_path)] == ["overloaded", "rate_limit", None]


def test_claude_sdk_built_with_workspace_header(monkeypatch):
    captured = {}
    fake = types.ModuleType("anthropic")
    fake.__version__ = "0-fake"

    class AsyncAnthropic:
        def __init__(self, **kw):
            captured.update(kw)
    fake.AsyncAnthropic = AsyncAnthropic
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    cp = ClaudeProvider("sk-ant-test-key-123", "wrkspc_42")
    cp.sdk  # noqa: B018 - lazily constructs
    assert captured["default_headers"] == {"anthropic-workspace-id": "wrkspc_42"}
    assert captured["max_retries"] == 0 and captured["api_key"] == "sk-ant-test-key-123"


def test_claude_message_batches_run_collect_idempotent(tmp_path):
    sdk = FakeSDK()
    c = make_client(tmp_path, {"claude": ClaudeProvider(sdk=sdk)}, budget=5.0)
    asyncio.run(c.call(req("cached-one")))                       # pre-cached: must not be sent in the batch
    reqs = [req("cached-one"), req("a"), req("b"), req("a"), req("c")]
    out = asyncio.run(c.batches.run(reqs, poll_s=1))
    assert len(sdk.batch_requests) == 3                           # cached and duplicate requests removed
    assert all(len(r["custom_id"]) <= 64 for r in sdk.batch_requests)
    assert [o.ok for o in out] == [True, True, True, True, False]  # last batch entry errored ("c")
    assert out[1].text == "b:a" and out[3].text == "b:a" and out[0].cache_hit
    assert out[1].cost_usd == pytest.approx(200 * 0.001 * 0.5)     # 50% batch price
    spent = c.costs.spent()
    bid = "msgbatch_1"
    asyncio.run(c.batches.collect(bid))                           # collecting again never double-counts
    assert c.costs.spent() == pytest.approx(spent) and c.costs.reserved() == 0
    assert asyncio.run(c.call(req("b"))).cache_hit                # batch results feed the cache
    recs = log_lines(tmp_path)
    assert any(r.get("event") == "batch_submit" for r in recs)
    assert sum(1 for r in recs if r.get("batch_id") == bid and r.get("error_class") == "batch_errored") == 2


# ------------------------------------------------------------------------------------------------- helpers
@pytest.mark.parametrize("text,exp", [
    ('```json\n{"a": 1}\n```', {"a": 1}),
    ('Verdict: {"verdict": "valid", "x": [1, {"y": "}"}]} trailing', {"verdict": "valid", "x": [1, {"y": "}"}]}),
    ("no json here", None),
    ("[1, 2]", [1, 2]),
])
def test_parse_json(text, exp):
    assert parse_json(text) == exp
