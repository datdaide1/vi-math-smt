"""The ONE common validity screen for the "+" arms (N+ and Y+): FROZEN 2026-11-07 ~23:05 (UTC+7). Stable API for G4.

    from vi_math_verified_aug.scale_gen.screen import SCREEN_SPEC, screen, labels

Do not change anything here (prompt, models, packing, thresholds) without telling the lead: both arms must call
identical code. Settings:
* prompt ``validity_lite.v1`` (``prompts/validity_lite.v1.md``; the same text for VI and EN, only <<language>> /
  <<country>> change: Vietnamese/Vietnam, English/the United States);
* models: gemini-3.5-flash-lite and gemini-3.1-flash-lite, alternating per pack over 6 keys (12 lanes, lead
  23:00); the model that judged each item is stored in its row (``model``);
* 10 items per call (input order), temperature 0, JSON mode, max_tokens 4096; a partial reply is retried once in
  halves; 180 s watchdog per call, a timed-out / rate-limited model is benched 600 s and the pack goes to the other;
* decision: ``lite_ok`` = no contradiction, ambiguity, multi-question, non-integer count, implausible quantity,
  incoherent setting or leak flag, and natural >= 4;
* training items only (requests are marked split="train"; the LLMClient routing guard refuses anything else).

``screen(store, lang, items)``: items = [(key, problem_text)], results are appended to ``store`` (a
``scale_gen.common.Store`` or any object with ``get(key)`` / ``put(key, row)``); already-judged keys are skipped.
"""
from __future__ import annotations

import asyncio
from typing import Callable

from .common import LITE_LANG, LITE_MODELS, PROMPT_VER, Store
from .lite import BENCH_S, FLAWS, PACK, PACK_TIMEOUT_S, judge_all, labels

SCREEN_SPEC = {"prompt": f"validity_lite.v{PROMPT_VER['lite']}", "models": list(LITE_MODELS), "pack": PACK,
               "temperature": 0.0, "json_mode": True, "max_tokens": 4096, "timeout_s": PACK_TIMEOUT_S,
               "bench_s": BENCH_S, "flaws": list(FLAWS), "natural_min": 4, "language_country": LITE_LANG,
               "frozen": "2026-11-07 23:05 UTC+7 (D1)",
               "status": "flash-lite part NOT APPLIED since 2026-11-08 02:05 (Gemini quota; lead): common screen = "
                         "Haiku-v3 only (see haiku_v3 / common_pass)"}


async def screen(store, lang: str, items: list[tuple[str, str]], *, concurrency: int = 14,
                 stop: Callable[[], bool] | None = None, llm=None) -> None:
    """``llm``: your own ``common.llm.LLMClient`` (e.g. ``LLMClient("G4", ...)``) so calls are logged under your ws;
    default = the D1 client (logs/api/D1)."""
    from . import lite as _lite
    _lite._client_override = llm
    try:
        await judge_all(store, lang, items, concurrency=concurrency, stop=stop)
    finally:
        _lite._client_override = None


def screen_sync(store_path: str, lang: str, items: list[tuple[str, str]]) -> Store:
    """Convenience wrapper: run the screen into ``store_path`` (JSONL) with the D1 client (logs/api/D1)."""
    from .common import client
    st = Store(store_path)
    client().run(screen(st, lang, items))
    return st


# ===================================================================================================================
# Part 2 of the COMMON "+" screen (lead, 2026-11-07 ~23:10, from G4's calibration on 358 Opus-labelled Y items:
# lite alone 10% [7-15] absurd among accepts; Haiku-v3 ∧ lite 6% [3-11]).
#   common_pass = lite_ok (part 1, above) ∧ haiku_v3_pass
#   haiku_v3_pass = ¬absurd ∧ ¬unansw ∧ teacher_would_write   (labels = lab_v3, identical to G2 / pilot_symbolic)
# Haiku 5.5, effort low, max_tokens 4000, rubric in the cached system block, Message Batches API.
# Prompt: validity_judge.v3 (VI items) / validity_judge_en.v3 (EN items; E1 R4 mirror, same keys and anchors).
# op_instruction = OP_INSTR[lang][op] (+ the partner seed's text for M6), as in G2's judge calls.
# ===================================================================================================================
from .common import AUDITOR as _AUD  # noqa: E402,F401  (keeps the import surface explicit)
from .common import OP_INSTR, claude_req, render  # noqa: E402

HV3 = {"model": "claude-haiku-5-5", "effort": "low", "max_tokens": 4000,
       "prompt": {"vi": ("validity_judge", 3), "en": ("validity_judge_en", 3)}}
SCREEN_SPEC["haiku_v3"] = {**HV3, "prompt": {k: f"{a}.v{b}" for k, (a, b) in HV3["prompt"].items()},
                           "pass": "not absurd and not unansw and teacher_would_write (lab_v3)"}


def op_instruction(lang: str, op: str, partner_text: str | None = None) -> str:
    return OP_INSTR[lang][op] + (partner_text or "" if op == "M6" else "")


def haiku_v3_request(seed_text: str, op: str, problem: str, *, lang: str, item_key: str,
                     partner_text: str | None = None):
    pid, ver = HV3["prompt"][lang]
    rp = render(pid, ver, seed=seed_text, op=op, op_instruction=op_instruction(lang, op, partner_text), problem=problem)
    return claude_req(HV3["model"], rp.system, rp.user, item=item_key, effort=HV3["effort"],
                      max_tokens=HV3["max_tokens"], cache_system=True, prompt_id=rp.prompt_id, version=rp.version,
                      sha=rp.sha256, tag=f"common_screen_hv3_{lang}")


def lab_v3(v: dict) -> dict:
    """validity_judge.v3 verdict -> labels (verbatim rule of G2 g2_common.lab_v3 / pilot_symbolic.judge.lab_v3)."""
    t = lambda k: v.get(k) is True  # noqa: E731
    try:
        nat = float(v.get("natural"))
    except (TypeError, ValueError):
        nat = 0.0
    absurd = t("q_noninteger_count") or t("q_implausible_quantity") or t("q_incoherent_setting")
    unansw = t("q_contradiction") or t("q_missing_or_ambiguous")
    valid = not (unansw or t("q_multi_question") or absurd) and nat >= 4
    return {"valid": valid, "teacher": t("teacher_would_write"), "new": not (t("q_same_structure") or t("q_tail")),
            "op": t("q_op_followed"), "natural": nat, "leakj": t("q_leak"), "absurd": absurd, "tail": t("q_tail"),
            "neardupj": t("q_same_structure"), "fluency": t("q_unnatural_vi") or nat < 4, "unansw": unansw,
            "judge_answer": v.get("answer")}


def haiku_v3_parse(text: str | None) -> dict:
    """-> {verdict, labels, hv3_pass, parsed}"""
    from ..common.llm import parse_json
    v = parse_json(text) if text else None
    if not isinstance(v, dict):
        return {"verdict": None, "labels": None, "hv3_pass": False, "parsed": False}
    lab = lab_v3(v)
    return {"verdict": v, "labels": lab, "parsed": True,
            "hv3_pass": (not lab["absurd"]) and (not lab["unansw"]) and lab["teacher"]}


def common_pass(lite_row: dict | None, hv3_row: dict | None) -> bool:
    """COMMON "+" screen as of 2026-11-08 02:05 (lead): Haiku-v3 only. ``lite_row`` is ignored (flash-lite not
    applied: Gemini free quota exhausted on both models x 6 keys until ~15:00 +07). G4 calibration: absurd among
    accepts 8% [5-12] Haiku-v3 alone vs 6% [3-11] Haiku-v3 ∧ lite."""
    return bool(hv3_row and hv3_row.get("hv3_pass"))


__all__ = ["SCREEN_SPEC", "screen", "screen_sync", "labels", "HV3", "haiku_v3_request", "haiku_v3_parse", "lab_v3",
           "common_pass", "op_instruction"]
