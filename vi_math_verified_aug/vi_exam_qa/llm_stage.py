"""Q1 Claude stages over the Message Batches API, through ``common/llm.py`` (run with ``E:\\venv_tools``).

  set PYTHONPATH=. & E:\\venv_tools\\Scripts\\python.exe -m vi_math_verified_aug.vi_exam_qa.llm_stage run solve2 \
      [--limit N] [--stage-budget USD] [--max-tokens N] [--effort high]
  ... run adjudicate | audit | audit_judge
  ... spend                                    # Q1 spend so far (ledger .cache/vimath/spend.sqlite, scope Q1)

One process drives a stage end to end: it submits chunks whose *worst-case* cost (input estimate + max_tokens at
the batch output price) fits under the hard cap, polls, collects, writes one row per item to
``data/vi_exam_qa/<stage>.jsonl`` and submits the next chunk. Reservations live in this process, so never run two
stages at once. The client's cap is min($10 Q1 cap, spent + --stage-budget).

Routing: every request is ``split="benchmark"`` → common/llm refuses Gemini/NIM for it; the model is Claude only.
All attempts are logged by common/llm to ``logs/api/Q1/claude.jsonl``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

from ..common.llm import LLMClient, LLMRequest
from ..common.prompts import load_prompt
from ..common.costs import price_for, load_prices
from . import data as D

POLL_S = 30.0
STAGES = ("solve2", "adjudicate", "audit", "audit_judge", "audit2", "audit2_judge")
PROMPT = {"solve2": "q1_solve", "adjudicate": "q1_adjudicate", "audit": "q1_audit_solve",
          "audit_judge": "q1_audit_judge", "audit2": "q1_audit_solve", "audit2_judge": "q1_audit_judge"}
DEFAULT_MAX_TOKENS = {"solve2": 16000, "adjudicate": 16000, "audit": 16000, "audit_judge": 12000,
                      "audit2": 16000, "audit2_judge": 12000}
SAMPLE_IDX = {"audit2": 1}     # audit #2 is a fresh draw even for items audit #1 already re-solved (no cache hit)
REF_CHARS = 3000               # reference-solution clip for the adjudicator (--ref-chars)


# ----------------------------------------------------------------------------------------------- inputs
def _clip(s: str | None, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else "…" + s[-n:]


def _head(s: str | None, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + "\n…[truncated]"


def item_info(it: dict) -> str:
    flags = (it.get("qa") or {}).get("flags") or []
    return (f"answer_type tag: {it.get('answer_type')}; topic: {it.get('topic')} {it.get('topics')}; "
            f"certifiable tag: {it.get('symbolically_certifiable')}; has_figure: {it.get('has_figure')}; "
            f"images: {it.get('n_images')}; table: {it.get('has_table')}; flags: {flags or 'none'}; "
            f"exam: {(it.get('provenance') or {}).get('province')} {(it.get('provenance') or {}).get('exam_year')}")


def keys_block(it: dict, ktab: dict, ck: dict | None, s2: dict | None) -> str:
    sk = it.get("sympy_key")
    lines = []
    if sk:
        desc = {k: sk.get(k) for k in ("key_type", "definition", "equation", "system", "target", "substituted",
                                         "variables", "variable", "values_str") if sk.get(k) not in (None, {}, [])}
        lines.append("[sympy] SymPy program key (non-Claude; computed from the LaTeX it parsed): "
                     + json.dumps(desc, ensure_ascii=False))
    else:
        lines.append("[sympy] absent (the SymPy translator could not key this item)")
    prov = it.get("provisional_answer") or {}
    if prov.get("value_final") not in (None, ""):
        lines.append(f"[page] answer extracted by regex from the reference solution (non-Claude): "
                     f"{prov.get('value_final')}   (method {prov.get('method')}, extractor confidence "
                     f"{prov.get('confidence')})")
    else:
        lines.append("[page] absent (no answer could be extracted from the reference solution)")
    c1 = (ck or {}).get("answer") or (it.get("claude_blind_key") or {}).get("answer")
    if c1:
        lines.append(f"[claude1] Claude blind solve #1 (Opus, low effort) final answer: {c1}\n"
                     f"  end of its reasoning: {_clip((ck or {}).get('response_tail'), 400)}")
    else:
        lines.append("[claude1] absent")
    if s2 and s2.get("ok") and s2.get("text"):
        from_box = s2.get("final_answer") or "(no \\boxed answer)"
        lines.append(f"[claude2] Claude blind solve #2 (Opus, high effort, different prompt) final answer: {from_box}\n"
                     f"  end of its reasoning: {_clip(s2.get('text'), 600)}")
    else:
        lines.append("[claude2] absent")
    return "\n".join(lines)


def reference(priv: dict | None) -> str:
    priv = priv or {}
    txt = _head(priv.get("solution_text"), REF_CHARS) or "(none aligned)"
    if priv.get("conclusion"):
        txt += f"\n[conclusion line] {priv['conclusion']}"
    return txt


def last_box(text: str | None) -> str | None:
    """Brace-balanced last \\boxed{...} (no grader import: this runs in the tools venv)."""
    if not text:
        return None
    i = text.rfind("\\boxed")
    while i >= 0:
        j = text.find("{", i)
        if j >= 0:
            depth = 0
            for k in range(j, len(text)):
                if text[k] == "{":
                    depth += 1
                elif text[k] == "}":
                    depth -= 1
                    if depth == 0:
                        return text[j + 1:k].strip()
        i = text.rfind("\\boxed", 0, i)
    return None


# ----------------------------------------------------------------------------------------------- requests
CACHE_SYSTEM = True   # Opus 5.5 caches prefixes >= 512 tokens; the shared system prompt is read at 0.05-0.1x


def _cached(extra: dict, system: str) -> dict:
    """Put the (identical-for-all-items) system prompt in a cached block. Overrides the plain ``system`` string in
    the provider params (``_deep_merge``); the request hash covers ``extra``, so cached/uncached are distinct."""
    if not CACHE_SYSTEM:
        return extra
    return {**extra, "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]}


def build_requests(stage: str, ids: list[str], max_tokens: int, effort: str) -> list[tuple[str, LLMRequest]]:
    prompt = load_prompt(PROMPT[stage])
    items = {it["id"]: it for it in D.core_items() if it["id"] in set(ids)}
    extra = {"output_config": {"effort": effort}}
    out = []
    if stage in ("solve2", "audit", "audit2"):
        for i in ids:
            r = prompt.render(problem=items[i]["problem"])
            out.append((i, LLMRequest.from_prompt(r, D.MODEL, max_tokens=max_tokens, extra=_cached(extra, r.system), item_id=i,
                                                   split="benchmark", tag=f"Q1:{stage}",
                                                   sample_idx=SAMPLE_IDX.get(stage, 0))))
    elif stage == "adjudicate":
        ktab = {r["id"]: r for r in D.iter_jsonl(D.KEYS_TABLE) if r["id"] in set(ids)}
        ck = D.claude_keys_by_id(set(ids))
        priv = D.private_by_id(set(ids))
        s2 = D.stage_rows("solve2")
        for i in ids:
            it = items[i]
            r = prompt.render(problem=it["problem"], item_info=item_info(it),
                              keys=keys_block(it, ktab.get(i, {}), ck.get(i), s2.get(i)),
                              reference_solution=reference(priv.get(i)))
            out.append((i, LLMRequest.from_prompt(r, D.MODEL, max_tokens=max_tokens, extra=_cached(extra, r.system), item_id=i,
                                                   split="benchmark", tag=f"Q1:{stage}")))
    elif stage in ("audit_judge", "audit2_judge"):
        pre = "audit" if stage == "audit_judge" else "audit2"
        mism = {r["id"]: r for r in D.iter_jsonl(D.OUT / f"{pre}_mismatches.jsonl")}
        priv = D.private_by_id(set(ids))
        aud = D.stage_rows(pre)
        for i in ids:
            m = mism[i]
            a = aud.get(i) or {}
            ans = f"final answer: {a.get('final_answer')}\nend of reasoning: {_clip(a.get('text'), 1500)}"
            r = prompt.render(problem=items[i]["problem"], gold=json.dumps(m["gold"], ensure_ascii=False),
                              audit_answer=ans, reference_solution=reference(priv.get(i)))
            out.append((i, LLMRequest.from_prompt(r, D.MODEL, max_tokens=max_tokens, extra=_cached(extra, r.system), item_id=i,
                                                   split="benchmark", tag=f"Q1:{stage}")))
    return out


def stage_ids(stage: str) -> list[str]:
    sel = D.load_json(D.SELECTION)
    if stage == "solve2":
        return list(sel["adjudicate"])
    if stage == "adjudicate":                      # triage order (T1 → T2 → T3) once solve #2 is graded
        return list(sel.get("adjudicate_order") or sel["adjudicate"])
    if stage == "audit":
        return list(sel["audit_sample"])
    if stage == "audit_judge":
        return [r["id"] for r in D.iter_jsonl(D.OUT / "audit_mismatches.jsonl")]
    if stage == "audit2":
        return list(D.load_json(D.OUT / "audit2_sample.frozen.json")["ids"])
    if stage == "audit2_judge":
        return [r["id"] for r in D.iter_jsonl(D.OUT / "audit2_mismatches.jsonl")]
    raise ValueError(stage)


LEDGER = D.OUT / "batches.jsonl"


def ledger_submit(stage: str, batch_id: str, pairs: list[tuple[str, str]], effort: str) -> None:
    D.append_jsonl(LEDGER, [{"event": "submit", "stage": stage, "batch_id": batch_id, "effort": effort,
                             "pairs": pairs, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}])


def ledger_collected(batch_id: str) -> None:
    D.append_jsonl(LEDGER, [{"event": "collected", "batch_id": batch_id,
                             "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}])


def pending_batches(stage: str | None = None) -> dict[str, dict]:
    sub, done = {}, set()
    for r in D.iter_jsonl(LEDGER):
        if r["event"] == "submit" and (stage is None or r["stage"] == stage):
            sub[r["batch_id"]] = r
        elif r["event"] == "collected":
            done.add(r["batch_id"])
    return {b: r for b, r in sub.items() if b not in done}


def pending_items(stage: str) -> set[str]:
    return {i for r in pending_batches(stage).values() for i, _ in r["pairs"]}


async def recover(stage: str | None, wait: bool) -> None:
    """Collect batches submitted by a process that died (ledger + common/llm batch manifest)."""
    client = LLMClient(D.WS, budget_usd=D.CAP_USD)
    from dataclasses import fields
    names = {f.name for f in fields(LLMRequest)}
    for bid, sub in pending_batches(stage).items():
        while True:
            st = await client.batches.status(bid)
            if st.get("processing_status") == "ended" or not wait:
                break
            await asyncio.sleep(POLL_S)
        if st.get("processing_status") != "ended":
            _log(f"[recover] {bid}: {st.get('processing_status')} {st.get('request_counts')}")
            continue
        man = json.loads((client.state_dir / "batches" / f"{bid}.json").read_text(encoding="utf-8"))
        reqs = {h: LLMRequest(**{k: v for k, v in d.items() if k in names}) for h, d in man["items"].items()}
        got = await client.batches.collect(bid)
        rows = []
        for i, h in sub["pairs"]:
            resp = got.get(h)
            rows.append(row_from(sub["stage"], i, reqs[h], resp, bid, sub["effort"]) if resp is not None else
                        {"item_id": i, "stage": sub["stage"], "ok": False, "error": "missing in results",
                         "batch_id": bid})
        D.append_jsonl(D.STAGE_OUT[sub["stage"]], rows)
        ledger_collected(bid)
        _log(f"[recover] {bid}: ok {sum(1 for r in rows if r.get('ok'))}/{len(rows)}, "
             f"cost ${sum(r.get('cost_usd', 0) for r in rows):.4f}, Q1 spent ${client.costs.spent():.4f}")
    await client.aclose()
    client.close()


def done_ids(stage: str) -> set[str]:
    return {i for i, r in D.stage_rows(stage).items() if r.get("ok")}


def row_from(stage: str, item_id: str, req: LLMRequest, resp, batch_id: str | None, effort: str) -> dict:
    text = resp.text
    ok = bool(resp.ok) and not resp.truncated
    row = {"item_id": item_id, "stage": stage, "ok": ok, "text": text, "finish_reason": resp.finish_reason,
           "usage": resp.usage, "cost_usd": round(resp.cost_usd or 0.0, 6), "cache_hit": resp.cache_hit,
           "request_hash": resp.request_hash, "batch_id": batch_id, "model": req.model,
           "model_returned": resp.model_returned, "prompt_id": req.prompt_id, "prompt_version": req.prompt_version,
           "prompt_sha256": req.prompt_sha256, "effort": effort, "max_tokens": req.max_tokens,
           "error_class": resp.error_class, "error": resp.error,
           "collected_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if stage in ("solve2", "audit", "audit2"):
        row["final_answer"] = last_box(text)
    return row


# ----------------------------------------------------------------------------------------------- driver
def _log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


async def drive(stage: str, limit: int | None, stage_budget: float | None, max_tokens: int, effort: str,
                max_chunk: int, ids_override: list[str] | None = None) -> None:
    probe = LLMClient(D.WS, budget_usd=D.CAP_USD, cache=False)
    spent0 = probe.costs.spent()
    probe.close()
    cap = D.CAP_USD if stage_budget is None else min(D.CAP_USD, spent0 + stage_budget)
    client = LLMClient(D.WS, budget_usd=cap)
    _log(f"[{stage}] Q1 spent so far ${spent0:.4f}; client cap ${cap:.4f}")
    ids = ids_override or stage_ids(stage)
    have = done_ids(stage) | pending_items(stage)
    todo_ids = [i for i in ids if i not in have]
    if limit is not None:
        todo_ids = todo_ids[:limit]
    pairs = build_requests(stage, todo_ids, max_tokens, effort)
    price = price_for(D.MODEL, "claude", load_prices())
    # cached requests (identical text sent before) are answered from the cache without a batch
    todo: list[tuple[str, LLMRequest]] = []
    for i, req in pairs:
        h = req.request_hash("claude")
        if client.cache is not None and h in client.cache:
            resp = await client.call(req)
            D.append_jsonl(D.STAGE_OUT[stage], [row_from(stage, i, req, resp, None, effort)])
        else:
            todo.append((i, req))
    _log(f"[{stage}] {len(pairs)} requested, {len(pairs) - len(todo)} from cache, {len(todo)} to batch")
    outstanding: dict[str, dict[str, list[tuple[str, LLMRequest]]]] = {}
    while todo or outstanding:
        rem = client.costs.remaining()
        worst = [(r.est_input_tokens() * price.input + r.max_tokens * price.output) / 1e6 * price.batch_mult
                 for _, r in todo[:max_chunk]]
        n, acc = 0, 0.0
        for w in worst:
            if acc + w > (rem or 0) - 1e-6:
                break
            acc += w
            n += 1
        if todo and n > 0 and (n >= min(len(todo), 10) or not outstanding):
            chunk, todo = todo[:n], todo[n:]
            man = await client.batches.submit([r for _, r in chunk])
            by_hash: dict[str, list[tuple[str, LLMRequest]]] = {}
            for i, r in chunk:
                by_hash.setdefault(r.request_hash("claude"), []).append((i, r))
            if man["batch_id"]:
                outstanding[man["batch_id"]] = by_hash
                ledger_submit(stage, man["batch_id"], [(i, r.request_hash("claude")) for i, r in chunk], effort)
                _log(f"[{stage}] submitted {man['batch_id']} n={len(chunk)} reserved ${man.get('reserved_usd', 0):.3f}"
                     f" (left to submit {len(todo)})")
            continue
        if not outstanding:
            _log(f"[{stage}] budget exhausted: remaining ${rem:.4f} cannot cover the next request; "
                 f"{len(todo)} items not submitted")
            break
        await asyncio.sleep(POLL_S)
        for bid in list(outstanding):
            st = await client.batches.status(bid)
            if st.get("processing_status") != "ended":
                continue
            got = await client.batches.collect(bid)
            rows = []
            pend = outstanding.pop(bid)
            for h, lst in pend.items():
                resp = got.get(h)
                for i, req in lst:
                    if resp is None:
                        rows.append({"item_id": i, "stage": stage, "ok": False, "error": "missing in results",
                                     "batch_id": bid})
                    else:
                        rows.append(row_from(stage, i, req, resp, bid, effort))
            D.append_jsonl(D.STAGE_OUT[stage], rows)
            ledger_collected(bid)
            cost = sum(r.get("cost_usd", 0) for r in rows)
            nok = sum(1 for r in rows if r.get("ok"))
            _log(f"[{stage}] collected {bid}: ok {nok}/{len(rows)}, cost ${cost:.4f}, "
                 f"Q1 spent ${client.costs.spent():.4f}")
    await client.aclose()
    _log(f"[{stage}] done. Q1 spent ${client.costs.spent():.4f}")
    client.close()


def spend() -> None:
    c = LLMClient(D.WS, budget_usd=D.CAP_USD, cache=False)
    print(json.dumps(c.costs.summary()))
    c.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("stage", choices=STAGES)
    r.add_argument("--limit", type=int)
    r.add_argument("--stage-budget", type=float)
    r.add_argument("--max-tokens", type=int)
    r.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    r.add_argument("--max-chunk", type=int, default=400)
    r.add_argument("--ids-file")
    r.add_argument("--ref-chars", type=int, default=3000)
    sub.add_parser("spend")
    rc = sub.add_parser("recover")
    rc.add_argument("--stage", choices=STAGES)
    rc.add_argument("--no-wait", action="store_true")
    a = ap.parse_args()
    if a.cmd == "spend":
        spend()
        return
    if a.cmd == "recover":
        asyncio.run(recover(a.stage, not a.no_wait))
        return
    global REF_CHARS
    REF_CHARS = a.ref_chars
    ids = None
    if a.ids_file:
        ids = [ln.strip() for ln in open(a.ids_file, encoding="utf-8") if ln.strip()]
    asyncio.run(drive(a.stage, a.limit, a.stage_budget, a.max_tokens or DEFAULT_MAX_TOKENS[a.stage], a.effort,
                      a.max_chunk, ids))


if __name__ == "__main__":
    sys.exit(main())
