"""G2 shared plumbing: the common LLM client (logs/api/G2, hard $ cap), prompt rendering, judge/solve calls,
verdict -> label mapping, and a resumable JSONL store under data/pilot/g2/.

Every call goes through ``vi_math_verified_aug.common.llm.LLMClient`` (cache + budget + routing guard). Every item
sent to Gemini is a Vi-GSM8K *train* dev seed or an item generated from one (split="dev").
"""
from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

from ..common.llm import LLMClient, LLMRequest, parse_json
from ..common.prompts import PromptRegistry
from ..common.ratelimit import LaneConfig
from . import seeds as S
from .util import OP_INSTR, answer_values, last_boxed, prompt as g1_prompt, same_answer

ROOT = S.ROOT
G1 = S.OUT                                   # data/pilot/generator (G1, read-only here)
OUT = ROOT / "data" / "pilot" / "g2"
OUT.mkdir(parents=True, exist_ok=True)
BUDGET = 7.9                                 # hard cap below the user's $8 Claude budget (scope G2)
REG = PromptRegistry()

MODELS = {
    "opus": "claude-opus-5-5", "sonnet": "claude-sonnet-5-5", "haiku": "claude-haiku-5-5",
    "g38": "gemini-3.8-flash", "g37": "gemini-3.7-flash", "g36": "gemini-3.6-flash", "g3fp": "gemini-3-flash-preview", "g31l": "gemini-3.1-flash-lite", "g35l": "gemini-3.5-flash-lite",
}
LANES = {"gemini": LaneConfig(rpm=10, burst=1, max_inflight=1, min_tpm=2_000, base_cooldown_s=30),
         "claude": LaneConfig(rpm=50, burst=5, max_inflight=16, base_cooldown_s=15)}

_client: LLMClient | None = None


def client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient("G2", budget_usd=BUDGET, lanes=LANES, max_concurrency=400)
    return _client


_sems: dict = {}
PENDING = {"claude": 16, "gemini": 6}   # requests admitted per model; LLMClient reserves budget BEFORE waiting
#                                       for a lane, so unbounded admission would reserve the whole cap


async def call(rq: LLMRequest):
    """client().call with per-model admission control (keeps budget reservations ~= requests in flight)."""
    prov = "claude" if rq.model.startswith("claude") else "gemini"
    s = _sems.get(rq.model)
    if s is None:
        s = _sems[rq.model] = asyncio.Semaphore(PENDING[prov])
    async with s:
        return await client().call(rq)


# ------------------------------------------------------------------ requests
def req(model_key: str, system: str | None, user: str, *, item: str, effort: str | None = None,
        max_tokens: int = 4000, temperature: float | None = None, cache_system: bool = False,
        prompt_id: str | None = None, version: int | None = None, sha: str | None = None, tag: str = "",
        sample_idx: int = 0) -> LLMRequest:
    model = MODELS[model_key]
    extra: dict = {}
    sysarg = system or None
    if model.startswith("claude"):
        extra["output_config"] = {"effort": effort or "low"}
        if cache_system and system:
            extra["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
            sysarg = None
        temperature = None
    return LLMRequest(model=model, system=sysarg, messages=[{"role": "user", "content": user}],
                      max_tokens=max_tokens, temperature=temperature, extra=extra, item_id=item, split="dev",
                      prompt_id=prompt_id, prompt_version=version, prompt_sha256=sha, tag=tag,
                      sample_idx=sample_idx)


def render(pid: str, version: int, **kw):
    p = REG.get(pid, version)
    return p.render(**kw)


def op_instr(s: dict) -> str:
    return OP_INSTR[s["op"]] + (s.get("partner_text") or "") if s["op"] == "M6" else OP_INSTR[s["op"]]


JUDGE_CFG = {   # judge key -> (model key, prompt version, effort, max_tokens, temperature)
    "opus_v2": ("opus", 2, "medium", 3000, None),
    "g38_v2": ("g38", 2, None, 8192, 0.0),
    "g37_v2": ("g37", 2, None, 8192, 0.0),
    "g36_v2": ("g36", 2, None, 8192, 0.0),
    "g3fp_v2": ("g3fp", 2, None, 8192, 0.0),
    "opusR_v2": ("opus", 2, "medium", 3000, None),     # Opus test-retest (independent sample, sample_idx=1)
    "g31l_v2": ("g31l", 2, None, 4096, 0.0),
    "g31l_v3": ("g31l", 3, None, 6144, 0.0),
    "haiku_v2": ("haiku", 2, "low", 4000, None),
    "haiku_v3": ("haiku", 3, "low", 5000, None),
    "haikum_v3": ("haiku", 3, "medium", 8000, None),
    "sonnet_v3": ("sonnet", 3, "low", 4000, None),
}


def judge_request(jkey: str, seed_text: str, op: str, op_instruction: str, problem: str, item: str) -> LLMRequest:
    mk, v, eff, mt, temp = JUDGE_CFG[jkey]
    rp = render("validity_judge", v, seed=seed_text, op=op, op_instruction=op_instruction, problem=problem)
    return req(mk, rp.system, rp.user, item=item, effort=eff, max_tokens=mt, temperature=temp,
               cache_system=True, prompt_id=rp.prompt_id, version=rp.version, sha=rp.sha256, tag=jkey,
               sample_idx=1 if jkey.startswith("opusR") else 0)


def solve_request(model_key: str, problem: str, item: str, effort: str = "low", sample_idx: int = 0,
                  max_tokens: int = 2500, temperature: float | None = 0.6) -> LLMRequest:
    rp = render("solution_writer", 2, problem=problem)
    return req(model_key, None, rp.user, item=item, effort=effort, max_tokens=max_tokens, temperature=temperature,
               prompt_id=rp.prompt_id, version=rp.version, sha=rp.sha256, tag=f"solve_{model_key}",
               sample_idx=sample_idx)


# ------------------------------------------------------------------ labels
BOOLS = ["atomic", "no_conflict", "complete", "single_question", "realistic"]


def _nat(v) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def lab_v2(v: dict) -> dict:
    """judge_v2 verdict -> labels (identical to G1 analyze.lab)."""
    cs = set(v.get("codes") or [])
    valid = all(v.get(b) is True for b in BOOLS) and _nat(v.get("natural")) >= 4
    return {"valid": valid, "teacher": v.get("teacher_would_write") is True, "new": v.get("new_vs_seed") is True,
            "op": v.get("op_followed") is True, "natural": _nat(v.get("natural")), "leakj": "NL-LEAK" in cs,
            "absurd": "NL-ABSURD" in cs, "tail": "NL-TACKED" in cs, "neardupj": "NL-NEARDUP" in cs,
            "fluency": "NL-FLUENCY" in cs, "unansw": "NL-UNANSW" in cs}


def lab_v3(v: dict) -> dict:
    """validity_judge.v3 (decomposed questions) -> the same label set, derived deterministically."""
    t = lambda k: v.get(k) is True  # noqa: E731
    absurd = t("q_noninteger_count") or t("q_implausible_quantity") or t("q_incoherent_setting")
    unansw = t("q_contradiction") or t("q_missing_or_ambiguous")
    nat = _nat(v.get("natural"))
    valid = not (unansw or t("q_multi_question") or absurd) and nat >= 4
    return {"valid": valid, "teacher": t("teacher_would_write"),
            "new": not (t("q_same_structure") or t("q_tail")), "op": t("q_op_followed"), "natural": nat,
            "leakj": t("q_leak"), "absurd": absurd, "tail": t("q_tail"), "neardupj": t("q_same_structure"),
            "fluency": t("q_unnatural_vi") or nat < 4, "unansw": unansw,
            "teacher_derived": valid and t("teacher_would_write") and not (t("q_leak") or t("q_tail")),
            "judge_answer": v.get("answer")}


def labels(jkey: str, verdict: dict | None) -> dict | None:
    if not verdict:
        return None
    v = JUDGE_CFG[jkey][1]
    return lab_v3(verdict) if v == 3 else lab_v2(verdict)


# ------------------------------------------------------------------ store
class Store:
    """One JSONL per (stage) under OUT; rows keyed by 'key'; last row wins; resumable."""

    def __init__(self, stage: str):
        self.path = OUT / f"{stage}.jsonl"
        self.lock = threading.Lock()
        self.rows: dict[str, dict] = {}
        if self.path.exists():
            for l in open(self.path, encoding="utf-8"):
                r = json.loads(l)
                self.rows[r["key"]] = r

    def ok(self, key: str) -> bool:
        r = self.rows.get(key)
        return bool(r and r.get("api_ok"))

    def put(self, key: str, row: dict) -> dict:
        row = {"key": key, **row}
        with self.lock:
            self.rows[key] = row
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row


def resp_meta(r) -> dict:
    return {"api_ok": bool(r.ok), "cost": r.cost_usd, "usage": r.usage, "latency_ms": r.latency_ms,
            "cache_hit": r.cache_hit, "finish": r.finish_reason, "err": r.error}


async def run_judges(store: Store, jobs: list[tuple[str, str, LLMRequest]]):
    """jobs = [(key, jkey, request)] -> store rows {verdict, labels}. Skips keys already OK."""
    async def one(key, jkey, rq):
        if store.ok(key):
            return
        try:
            r = await call(rq)
        except Exception as e:                       # BudgetExceeded etc.: record and continue
            store.put(key, {"jkey": jkey, "api_ok": False, "err": f"{type(e).__name__}: {e}"})
            return
        v = parse_json(r.text) if r.ok else None
        v = v if isinstance(v, dict) else None
        store.put(key, {"jkey": jkey, "verdict": v, "parsed": v is not None, **resp_meta(r)})
    await asyncio.gather(*(one(*j) for j in jobs))


def g1_rows(stage: str, gen: str) -> dict:
    p = G1 / stage / f"{gen}.jsonl"
    d = {}
    if p.exists():
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            d[r["key"]] = r
    return d


__all__ = ["client", "req", "render", "op_instr", "judge_request", "solve_request", "labels", "lab_v2", "lab_v3",
           "Store", "run_judges", "g1_rows", "OUT", "G1", "MODELS", "JUDGE_CFG", "last_boxed", "answer_values",
           "same_answer", "g1_prompt", "Path"]
