"""D1 shared plumbing: paths, the LLM client (ws D1, hard $12 Claude cap), request builders, a resumable JSONL store.

Every Claude/Gemini call goes through ``common.llm.LLMClient`` -> logged under ``logs/api/D1/``, cached, budgeted.
Only generated TRAINING items (split="train") go to Gemini; eval / benchmark items are never sent anywhere.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Iterable, Iterator

from ..common.llm import LLMClient, LLMRequest
from ..common.prompts import PromptRegistry
from ..common.ratelimit import LaneConfig

ROOT = Path(__file__).resolve().parents[2]
POOL = ROOT / "data" / "seeds" / "pool_v1.jsonl"
SLOTS = ROOT / "data" / "train" / "slots_v1.jsonl"
SLOTS_META = ROOT / "data" / "train" / "slots_v1.meta.json"
OUT = ROOT / "data" / "train" / "N"
WORK = OUT / "_work"
CARDS_VI = ROOT / "vi_math_verified_aug" / "pilot_gen" / "cards_v1.json"
CARDS_EN = Path(__file__).with_name("cards_en_v1.json")

WS = "D1"
CAP_USD = 21.0   # lead 2026-11-08 03:30: hard stop at $21 total for D1 ($12 + EN reserve top-ups)
GENERATOR = "claude-haiku-5-5"
SOLVER = "claude-haiku-5-5"
AUDITOR = "claude-opus-5-5"
LITE = "gemini-3.5-flash-lite"   # primary screen model
# 12 lanes = 6 keys x 2 flash-lite models (lead, 23:00). The judge model is recorded per item. NB: D1's probe at 22:47
# found a 500 requests/day/project free-tier cap on 3.1-flash-lite (exhausted on 3/6 keys until ~15:00 UTC+7).
LITE_MODELS = ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite")
LANGS = ("vi", "en")

# ---------------------------------------------------------------- operator schedule (T1 §2.0)
PI = {"M1": .20, "M2": .20, "M3": .25, "M4": .15, "M5": .10, "M6": .10}
RNG_SEED = 20261107          # T1 §2.0 schedule seed
J = 3                        # slots per seed

OP_INSTR = {
    "vi": {"M1": "Đổi các số liệu, giữ nguyên cấu trúc lời giải.",
           "M2": "Đảo bài toán: cho biết kết quả của bài gốc, hỏi một dữ kiện ban đầu.",
           "M3": "Thêm đúng một bước tính hợp lý vào giữa câu chuyện (không thêm vào sau đáp số).",
           "M4": "Thay đổi một quan hệ (thêm↔bớt, gấp k lần, phần trăm…).",
           "M5": "Hỏi một đại lượng khác cần ít nhất hai bước tính.",
           "M6": "Ghép với bài sau thành một bài nhiều bước: "},
    "en": {"M1": "Change the numbers, keep the structure of the solution.",
           "M2": "Invert the problem: give the result of the seed problem and ask for one of the original facts.",
           "M3": "Add exactly one reasonable computation step in the middle of the story (not after the answer).",
           "M4": "Change one relation (add↔take away, k times as many, percent…).",
           "M5": "Ask for a different quantity that needs at least two computation steps.",
           "M6": "Join with the following problem into one multi-step problem: "},
}
TARGET = {"vi": ("khó hơn một bậc", "giữ nguyên độ khó"), "en": ("one level harder", "keep the same difficulty")}
GRADE = {"vi": {"gsm8k": "tiểu học và THCS (lớp 3–7)", "math": "THCS và THPT (lớp 7–10)"},
         "en": {"gsm8k": "elementary and middle school (grades 3–7)", "math": "middle and high school (grades 7–10)"}}
PROMPT_ID = {"vi": {"gen": "native_generator", "solve": "solution_writer"},
             "en": {"gen": "native_generator_en", "solve": "solution_writer_en"}}
PROMPT_VER = {"gen": 3, "solve": 2, "lite": 1, "audit": 2}
LITE_LANG = {"vi": ("Vietnamese", "Vietnam"), "en": ("English", "the United States")}

# generator configs: "A" = G2's validated setting (effort low, adaptive thinking);
#                    "B" = effort low, thinking disabled (Haiku 5.5 accepts disabled at effort <= high)
GEN_CFG = {"A": {"effort": "low", "thinking": None, "max_tokens": 6000},
           "B": {"effort": "low", "thinking": {"type": "disabled"}, "max_tokens": 3000}}

REG = PromptRegistry()
_client: LLMClient | None = None
LANES = {"gemini": LaneConfig(rpm=10, burst=1, max_inflight=1, min_tpm=2_000, base_cooldown_s=30),
         "claude": LaneConfig(rpm=240, burst=10, max_inflight=32, base_cooldown_s=15)}  # 04:25: real-time hv3 throughput


def client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient(WS, budget_usd=CAP_USD, lanes=LANES, max_concurrency=200)
    return _client


def render(pid: str, version: int, **kw):
    return REG.get(pid, version).render(**kw)


def cards(lang: str) -> dict[str, dict]:
    p = CARDS_VI if lang == "vi" else CARDS_EN
    return {c["id"]: c for c in json.loads(p.read_text(encoding="utf-8"))}


def card_text(c: dict, lang: str) -> str:
    if lang == "vi":
        items = "; ".join(f"{a} ({b})" for a, b in c["vat_pham"])
        return (f"bối cảnh: {c['boi_canh']}; tên người: {', '.join(c['ten'])}; vật phẩm và giá: {items}; "
                f"đơn vị: {', '.join(c['don_vi'])}")
    items = "; ".join(f"{a} ({b})" for a, b in c["items"])
    return f"setting: {c['setting']}; names: {', '.join(c['names'])}; items and prices: {items}; units: {', '.join(c['units'])}"


def claude_req(model: str, system: str | None, user: str, *, item: str, effort: str = "low",
               thinking: dict | None = None, max_tokens: int = 4000, cache_system: bool = False,
               prompt_id: str | None = None, version: int | None = None, sha: str | None = None, tag: str = "",
               sample_idx: int = 0) -> LLMRequest:
    extra: dict[str, Any] = {"output_config": {"effort": effort}}
    if thinking:
        extra["thinking"] = thinking
    sysarg = system or None
    if cache_system and system:
        extra["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        sysarg = None
    return LLMRequest(model=model, system=sysarg, messages=[{"role": "user", "content": user}], max_tokens=max_tokens,
                      extra=extra, item_id=item, split="train", prompt_id=prompt_id, prompt_version=version,
                      prompt_sha256=sha, tag=tag, sample_idx=sample_idx)


def seed_text(seed: dict, lang: str) -> str:
    return seed["problem"] if lang == "vi" else seed["en"]["problem"]


def gen_request(slot: dict, seed: dict, partner: dict | None, lang: str, cfg: str = "B") -> LLMRequest:
    op = slot["op"]
    instr = OP_INSTR[lang][op] + (seed_text(partner, lang) if op == "M6" and partner else "")
    hard, same = TARGET[lang]
    kind = "gsm8k" if seed.get("dataset") == "gsm8k" else "math"
    rp = render(PROMPT_ID[lang]["gen"], PROMPT_VER["gen"], grade_hint=GRADE[lang][kind], seed_vi=seed_text(seed, lang),
                op=op, op_instruction=instr, card=card_text(cards(lang)[slot["card"]], lang),
                target=hard if op in ("M2", "M3", "M6") else same)
    c = GEN_CFG[cfg]
    return claude_req(GENERATOR, rp.system, rp.user, item=f"N:{lang}:{slot['key']}", effort=c["effort"],
                      thinking=c["thinking"], max_tokens=c["max_tokens"], cache_system=True,
                      prompt_id=rp.prompt_id, version=rp.version, sha=rp.sha256, tag=f"D1_gen_{lang}_{cfg}")


def solve_request(problem: str, lang: str, key: str, sample_idx: int = 0) -> LLMRequest:
    rp = render(PROMPT_ID[lang]["solve"], PROMPT_VER["solve"], problem=problem)
    return claude_req(SOLVER, None, rp.user, item=f"N:{lang}:{key}", effort="low", max_tokens=3000,
                      prompt_id=rp.prompt_id, version=rp.version, sha=rp.sha256, tag=f"D1_solve_{lang}",
                      sample_idx=sample_idx)


def audit_request(seed_txt: str, op: str, op_instruction: str, problem: str, key: str, lang: str = "vi") -> LLMRequest:
    """Opus 5.5 + validity_judge.v2 (the judge of record, G1/G2) for VI items; its EN twin validity_judge_en.v1
    (D5, same fields/codes/labels) for EN items. Effort medium, rubric cached."""
    pid, ver = ("validity_judge", PROMPT_VER["audit"]) if lang == "vi" else ("validity_judge_en", 1)
    rp = render(pid, ver, seed=seed_txt, op=op, op_instruction=op_instruction, problem=problem)
    return claude_req(AUDITOR, rp.system, rp.user, item=f"N:{key}", effort="medium", max_tokens=3000,
                      cache_system=True, prompt_id=rp.prompt_id, version=rp.version, sha=rp.sha256, tag="D1_audit")


# ---------------------------------------------------------------- IO
def read_jsonl(p: str | os.PathLike) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def iter_jsonl(p: str | os.PathLike) -> Iterator[dict]:
    with open(p, encoding="utf-8") as f:
        for l in f:
            if l.strip():
                yield json.loads(l)


def write_jsonl(p: str | os.PathLike, rows: Iterable[dict]) -> int:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, p)
    return n


def sha256_file(p: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Store:
    """Append-only JSONL keyed by ``key``; last row wins; resumable."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.rows: dict[str, dict] = {}
        if self.path.exists():
            for r in iter_jsonl(self.path):
                self.rows[r["key"]] = r

    def __contains__(self, key: str) -> bool:
        return key in self.rows

    def get(self, key: str) -> dict | None:
        return self.rows.get(key)

    def put(self, key: str, row: dict) -> dict:
        row = {"key": key, **row}
        with self.lock:
            self.rows[key] = row
            with open(self.path, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row


def resp_meta(r) -> dict:
    return {"api_ok": bool(r.ok), "cost": r.cost_usd, "usage": r.usage, "finish": r.finish_reason,
            "err": r.error, "cache_hit": r.cache_hit, "model_returned": r.model_returned}
