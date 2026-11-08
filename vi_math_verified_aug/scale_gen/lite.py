"""gemini flash-lite packed validity / absurdity screen (prompts/validity_lite.v1.md), training items only.

* Up to ``PACK`` items per call: the free tier is bound per key and model by requests/day (D1 probe: 500 RPD on
  3.1-flash-lite) and tokens/min, so packing saves quota (see the prompt's front matter).
* 12 lanes = 6 keys x ``LITE_MODELS`` (alternating per pack; the judge model is stored per item).
* Watchdog: every call is wrapped in ``asyncio.wait_for(PACK_TIMEOUT_S)``; a model whose call times out or is
  rate-limited is benched for ``BENCH_S`` and the pack moves to the other model. (A lane-cooldown wait of hours
  -- a daily-quota 429 -- can otherwise stall the whole job.)
* A pack whose reply misses items is retried once split in halves; items still missing are recorded with
  ``api_ok: false`` and retried by the next ``judge_all``.
"""
from __future__ import annotations

import asyncio
import itertools
import time

from ..common.llm import LLMRequest, parse_json
from .common import LITE_LANG, LITE_MODELS, PROMPT_VER, Store, client, render

PACK = 10
PACK_TIMEOUT_S = 180
BENCH_S = 600
FLAWS = ("contradiction", "ambiguous", "multi_question", "noninteger_count", "implausible", "incoherent", "leak")
_bench: dict[str, float] = {}
_client_override = None      # another workstream (G4) may pass its own LLMClient so calls log under its ws


def _cl():
    return _client_override if _client_override is not None else client()
_rr = itertools.count()


def labels(v: dict) -> dict:
    t = lambda k: v.get(k) is True  # noqa: E731
    try:
        nat = float(v.get("natural"))
    except (TypeError, ValueError):
        nat = 0.0
    absurd = t("noninteger_count") or t("implausible") or t("incoherent")
    unansw = t("contradiction") or t("ambiguous")
    return {"absurd": absurd, "unansw": unansw, "multi": t("multi_question"), "leakj": t("leak"), "natural": nat,
            "lite_ok": not (absurd or unansw or t("multi_question") or t("leak")) and nat >= 4}


def pack_request(lang: str, items: list[tuple[str, str]], model: str = LITE_MODELS[0]
                 ) -> tuple[LLMRequest, dict[str, str]]:
    ids = {f"p{i + 1}": key for i, (key, _) in enumerate(items)}
    body = "\n\n".join(f"[{pid}]\n{prob.strip()}" for pid, (_, prob) in zip(ids, items))
    language, country = LITE_LANG[lang]
    rp = render("validity_lite", PROMPT_VER["lite"], language=language, country=country, items=body)
    req = LLMRequest(model=model, system=rp.system, messages=[{"role": "user", "content": rp.user}],
                     max_tokens=4096, temperature=0.0, json_mode=True, split="train",
                     item_id=f"N:{lang}:pack:{items[0][0]}", prompt_id=rp.prompt_id, prompt_version=rp.version,
                     prompt_sha256=rp.sha256, tag=f"D1_lite_{lang}")
    return req, ids


def _models_in_order() -> list[str]:
    i = next(_rr)
    ms = list(LITE_MODELS[i % len(LITE_MODELS):]) + list(LITE_MODELS[:i % len(LITE_MODELS)])
    now = time.time()
    live = [m for m in ms if _bench.get(m, 0) <= now]
    return live or sorted(ms, key=lambda m: _bench.get(m, 0))


async def _call(lang: str, items: list[tuple[str, str]]):
    err = None
    for model in _models_in_order():
        req, ids = pack_request(lang, items, model)
        try:
            r = await asyncio.wait_for(_cl().call(req), timeout=PACK_TIMEOUT_S)
        except asyncio.TimeoutError:
            _bench[model] = time.time() + BENCH_S
            err = f"timeout {PACK_TIMEOUT_S}s on {model}"
            continue
        except Exception as e:                   # ForbiddenRoute would be a routing bug: never swallow it
            if type(e).__name__ == "ForbiddenRoute":
                raise
            _bench[model] = time.time() + BENCH_S
            err = f"{type(e).__name__}: {e}"[:300]
            continue
        if r.ok:
            return r, ids, model, None
        if r.error_class == "rate_limit":
            _bench[model] = time.time() + BENCH_S
        err = f"{r.error_class}: {(r.error or '')[:200]}"
    return None, None, None, err


async def judge_pack(store: Store, lang: str, items: list[tuple[str, str]], depth: int = 0) -> None:
    r, ids, model, err = await _call(lang, items)
    got: dict[str, dict] = {}
    if r is not None:
        arr = parse_json(r.text)
        if isinstance(arr, dict):
            arr = arr.get("items") or arr.get("results") or [arr]
        if isinstance(arr, list):
            for v in arr:
                if isinstance(v, dict) and str(v.get("id")) in ids:
                    got[str(v["id"])] = v
        err = None if got else "unparsable"
        missing = [(ids[p], prob) for p, (_, prob) in zip(ids, items) if p not in got]
    else:
        missing = list(items)
    for pid, v in got.items():
        store.put(ids[pid], {"verdict": {k: v.get(k) for k in (*FLAWS, "natural")}, **labels(v), "api_ok": True,
                             "pack": len(items), "model": model})
    if missing:
        if r is not None and depth == 0:          # partial / unparsable reply: retry once in halves
            h = max(1, len(missing) // 2)
            parts = [missing[:h], missing[h:]] if len(missing) > 1 else [missing]
            await asyncio.gather(*(judge_pack(store, lang, p, 1) for p in parts if p))
        else:
            for key, _ in missing:
                store.put(key, {"api_ok": False, "err": str(err)[:300], "pack": len(items)})


async def judge_all(store: Store, lang: str, items: list[tuple[str, str]], concurrency: int = 14,
                    stop: "callable | None" = None) -> None:
    """Judge ``items`` in the given order. ``stop()`` -> True ends the job early (critical-path mode)."""
    todo = [(k, p) for k, p in items if not (store.get(k) or {}).get("api_ok")]
    packs = [todo[i:i + PACK] for i in range(0, len(todo), PACK)]
    sem = asyncio.Semaphore(concurrency)

    async def one(p):
        async with sem:
            if stop is not None and stop():
                return
            await judge_pack(store, lang, p)
    await asyncio.gather(*(one(p) for p in packs))
