"""MTA-vi (T4 §3) for damaged seed rows, Claude-based (no NIM): mask -> fired glossary + CR -> translate (Haiku)
-> unmask -> G0/G1 deterministic gates -> G2 blind solve-check -> G3 MQM span judge (Gemini, other family)
-> <= 2 Haiku post-edit rounds with the external feedback -> escalation to Sonnet (translate + 1 post-edit).

Acceptance (S3 brief): G0 (placeholder bijection, LaTeX/structure parity) and G1 (numbers, final answer, glossary,
CR) hold, AND a blind solve of the Vietnamese problem reproduces the gold (G2 = PASS), AND the span judge reports no
major/critical error. An unavailable judge does not block (recorded).

G2 solver ladder (the solver never sees the English or the gold): gemini-3.1-flash-lite (temperature 0, then a
second sample at 0.6) -> Haiku -> Sonnet. The English control solve (same ladder) separates FAIL (English solved,
Vietnamese not: the wording changed the math) from UNINFORMATIVE (nobody solves it); only PASS accepts.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field

from ..data_fix.common import canon_num
from .checks import fired, g0, g1
from .common import Store, call, render, req, resp_meta

MASK_ORDER = [
    ("A", re.compile(r"\[asy\].*?\[/asy\]", re.S)),
    ("E", re.compile(r"\\begin\{(\w+\*?)\}.*?\\end\{\1\}", re.S)),
    ("D", re.compile(r"(?<!\\)\$\$.*?(?<!\\)\$\$|\\\[.*?\\\]", re.S)),   # not the "$\$$" of MATH amounts
    ("M", re.compile(r"\\\(.*?\\\)|(?<!\\)\$(?:\\\$|[^$])+?(?<!\\)\$", re.S)),
    ("C", re.compile(r"<<[^<>]*>>")),
    ("U", re.compile(r"https?://\S+")),
]
TAG_RE = re.compile(r"<([A-Z])(\d+)/>")


def _boxed_spans(text: str) -> list[tuple[int, int]]:
    out = []
    for m in re.finditer(r"\\boxed\s*\{", text):
        i, d = m.end(), 1
        while i < len(text) and d:
            if text[i] == "\\":
                i += 2
                continue
            d += (text[i] == "{") - (text[i] == "}")
            i += 1
        if d == 0:
            out.append((m.start(), i))
    return out


def mask(text: str, dataset: str, spans: dict[str, str]) -> str:
    """Math / code / annotations -> ``<Xn/>`` (numbering shared across problem + solution)."""
    order = MASK_ORDER if dataset == "math" else [m for m in MASK_ORDER if m[0] in ("C", "U")]
    for tag, rx in order:
        def rep(m, tag=tag):
            k = f"{tag}{len(spans) + 1}"
            spans[k] = m.group(0)
            return f"<{k}/>"
        text = rx.sub(rep, text)
    if dataset == "math":
        # a \boxed{} left in prose AFTER the math masks (boxes inside $..$ / \[..\] / align are already inside a
        # span; masking them first nested a <Bn/> inside another span — the run-1 G0 failures)
        for s, e in reversed(_boxed_spans(text)):
            k = f"B{len(spans) + 1}"
            spans[k] = text[s:e]
            text = text[:s] + f"<{k}/>" + text[e:]
    return text


def unmask(text: str, spans: dict[str, str]) -> str:
    for _ in range(3):                        # a span may itself contain a tag (\boxed{$x$})
        new = TAG_RE.sub(lambda m: spans.get(m.group(1) + m.group(2), m.group(0)), text)
        if new == text:
            break
        text = new
    return text


def parse_blocks(out: str | None) -> tuple[str | None, str | None]:
    out = re.sub(r"<think>.*?</think>", "", out or "", flags=re.S)
    p = re.search(r"<problem>\s*(.*?)\s*</problem>", out, re.S)
    s = re.search(r"<solution>\s*(.*?)\s*</solution>", out, re.S)
    return (p.group(1) if p else None), (s.group(1) if s else None)


def _fmt(gl: list[dict], rules: list[dict]) -> tuple[str, str]:
    gs = "\n".join(f"   - {e['line']}" for e in gl[:15]) or "   (không có)"
    rs = "\n".join(f"   - {c['id']}: {c['line']}" for c in rules) or "   (không có)"
    return gs, rs


def typed_gold(row: dict) -> dict:
    from ..eval.math_verify import infer_gold
    raw = canon_num(row["gold_raw_en"]) if row["dataset"] == "gsm8k" else row["gold_raw_en"]
    return infer_gold(raw, problem=row["problem_en"], locale="mixed")


# ------------------------------------------------------------------------------------------------ G2 solve
SOLVERS = ("flash_lite35", "flash_lite", "flash_lite35@0.6", "haiku", "sonnet")


async def solve_once(problem: str, lang: str, solver: str, item: str, gold: dict) -> dict:
    from ..eval.math_verify import grade
    model_key, _, temp = solver.partition("@")
    rp = render("seed_solve" if lang == "vi" else "seed_solve_en", problem=problem)
    sample = 1 if temp else 0
    rq = req(model_key, rp, item=item, max_tokens=4096 if model_key.startswith("flash_lite") else 3000,
             temperature=float(temp) if temp else 0.0, tag=f"S3-solve-{lang}-{solver}", sample_idx=sample,
             effort="low")
    try:
        r = await call(rq)
    except Exception as e:                       # BudgetExceeded / ForbiddenRoute / ProviderDisabled
        return {"solver": solver, "model": rq.model, "ok": None, "err": f"{type(e).__name__}: {e}"[:300]}
    if not r.ok:
        return {"solver": solver, "ok": None, "err": (r.error or r.error_class or "")[:300], **resp_meta(r)}  # meta has model
    g = grade(r.text, gold)
    return {"solver": solver, "model": rq.model, "ok": bool(g["correct"]), "span": g["span"], "method": g["method"],
            "cost": r.cost_usd, "cache_hit": r.cache_hit}


async def solve_ladder(problem: str, lang: str, item: str, gold: dict, ladder=SOLVERS) -> dict:
    """Blind solve up the ladder until one solver reproduces the gold. Returns {pass, by, attempts}."""
    attempts = []
    for s in ladder:
        a = await solve_once(problem, lang, s, item, gold)
        attempts.append(a)
        if a.get("ok"):
            return {"pass": True, "by": s, "attempts": attempts}
    return {"pass": False, "by": None, "attempts": attempts}


async def g2(problem_vi: str, problem_en: str, item: str, gold: dict) -> tuple[str, list[str], dict]:
    vi = await solve_ladder(problem_vi, "vi", item, gold)
    if vi["pass"]:
        return "PASS", [], {"vi": vi}
    en = await solve_ladder(problem_en, "en", item, gold)
    spans = [a.get("span") for a in vi["attempts"] if a.get("ok") is False]
    if not en["pass"]:
        return "UNINFORMATIVE", [f"G2 solve-check: no solver reproduced the gold {gold['answer_canonical']!r} on "
                                 f"the Vietnamese problem (got {spans}); the English control is not solved "
                                 f"either"], {"vi": vi, "en": en}
    return "FAIL", [f"G2 solve-check: solvers got {spans} on the Vietnamese problem but the gold is "
                    f"{gold['answer_canonical']!r} and the English problem is solved correctly — the Vietnamese "
                    f"wording likely changes the math (check quantity relations, quantifier scope, direction, "
                    f"units, missing conditions)"], {"vi": vi, "en": en}


# ------------------------------------------------------------------------------------------------ G3 judge
async def g3(row: dict, vi_p: str, vi_s: str, item: str, gold: dict,
             judges: tuple[str, ...] = ("flash_lite35", "flash_lite", "haiku")) -> tuple[bool, list[str], dict]:
    """MQM span judge; the first judge in ``judges`` that answers parsably decides (later ones are fallbacks)."""
    rp = render("seed_mqm", gold=gold["answer_canonical"], problem_en=row["problem_en"], problem_vi=vi_p,
                solution_en=row["solution_en"], solution_vi=vi_s)
    for mk in judges:  # noqa: B007
        rq = req(mk, rp, item=item, max_tokens=4096, temperature=0.0, json_mode=mk.startswith("flash_lite"),
                 tag=f"S3-mqm-{mk}", effort="low")
        try:
            r = await call(rq)
        except Exception as e:
            continue
        if not r.ok:
            continue
        from ..common.llm import parse_json
        v = parse_json(r.text)
        if isinstance(v, dict) and isinstance(v.get("errors"), list):
            bad = [e for e in v["errors"] if str(e.get("severity", "")).lower() in ("major", "critical")]
            fb = [f"G3 judge ({e.get('severity')}, {e.get('category')}): '{e.get('span_vi')}' — "
                  f"{e.get('explanation')}" for e in bad]
            return not bad, fb, {"judge": mk, "errors": v["errors"]}
    return True, [], {"judge": None, "errors": None, "note": "judge unavailable/unparsable (not blocking)"}


# ------------------------------------------------------------------------------------------------ driver
@dataclass
class Job:
    row: dict
    gold: dict
    spans: dict = field(default_factory=dict)
    masked_p: str = ""
    masked_s: str = ""
    history: list = field(default_factory=list)


async def _translate(job: Job, model_key: str, prev: tuple[str, str] | None, fails: list[str], rnd: int,
                     gs: str, rs: str) -> tuple[str | None, str | None, dict]:
    item = f"seed:{job.row['seed_id']}"
    if prev is None:
        rp = render("seed_translate", glossary=gs, rules=rs, problem=job.masked_p, solution=job.masked_s)
    else:
        rp = render("seed_postedit", glossary=gs, rules=rs, feedback="\n".join(f"- {f}" for f in fails),
                    problem=job.masked_p, solution=job.masked_s, vi_problem=prev[0], vi_solution=prev[1])
    rq = req(model_key, rp, item=item, max_tokens=8000, tag=f"S3-mta-{model_key}-r{rnd}",
             effort="low" if model_key == "haiku" else "medium")
    try:
        r = await call(rq)
    except Exception as e:
        return None, None, {"err": f"{type(e).__name__}: {e}"[:300]}
    if not r.ok:
        return None, None, {"err": r.error, **resp_meta(r)}
    p, s = parse_blocks(r.text)
    return p, s, resp_meta(r)


def _tag_fails(vm_p: str, vm_s: str, spans: dict) -> list[str]:
    keys = [a + b for a, b in TAG_RE.findall(vm_p + "\n" + vm_s)]
    if sorted(keys) == sorted(spans):
        return []
    missing = sorted(set(spans) - set(keys))
    extra = sorted(set(keys) - set(spans))
    dup = sorted({k for k in keys if keys.count(k) > 1})
    return [f"G0 placeholder mismatch: missing {missing} extra {extra} duplicated {dup} — every tag exactly once"]


async def _gate(job: Job, vm_p: str, vm_s: str) -> tuple[list[str], dict, str, str]:
    r = job.row
    vi_p, vi_s = unmask(vm_p, job.spans), unmask(vm_s, job.spans)
    fails = _tag_fails(vm_p, vm_s, job.spans) + g0(vi_p, vi_s, r["problem_en"], r["solution_en"], r["dataset"]) + \
        g1(vi_p, vi_s, r["problem_en"], r["solution_en"], r["gold_raw_en"], r["dataset"])
    checks: dict = {"G0G1": list(fails)}
    if fails:                         # cheap gates first: no solver/judge calls on a structurally broken draft
        return fails, checks, vi_p, vi_s
    item = f"seed:{r['seed_id']}"
    st, fb, det = await g2(vi_p, r["problem_en"], item, job.gold)
    checks["G2"] = st
    checks["G2_detail"] = det
    if st != "PASS":
        return fb, checks, vi_p, vi_s
    ok, fb3, det3 = await g3(r, vi_p, vi_s, item, job.gold)
    checks["G3"] = {"pass": ok, **det3}
    return ([] if ok else fb3), checks, vi_p, vi_s


async def translate_row(row: dict, store: Store) -> dict:
    key = row["seed_id"]
    old = store.get(key)
    if old and old.get("status") in ("ACCEPT", "FAIL"):
        return old
    job = Job(row, typed_gold(row))
    job.masked_p = mask(row["problem_en"], row["dataset"], job.spans)
    job.masked_s = mask(row["solution_en"], row["dataset"], job.spans)
    gl, rules = fired(row["problem_en"] + "\n" + row["solution_en"])
    gs, rs = _fmt(gl, rules)
    plan = [("haiku", 0), ("haiku", 1), ("haiku", 2), ("sonnet", 0), ("sonnet", 1)]
    prev, fails = None, []
    vi_p = vi_s = None
    status = "FAIL"
    for model_key, rnd in plan:
        if model_key == "sonnet" and rnd == 0:
            prev_in, fails_in = (prev, fails) if prev else (None, [])     # escalate: Sonnet post-edits the last
        else:                                                              # draft with its feedback
            prev_in, fails_in = (prev, fails) if rnd > 0 else (None, [])
        p, s, meta = await _translate(job, model_key, prev_in, fails_in, rnd, gs, rs)
        if p is None or s is None:
            fails = ["output must contain exactly one <problem>...</problem> and one <solution>...</solution> block"]
            job.history.append({"model": model_key, "round": rnd, "fails": fails, "meta": meta})
            if "BudgetExceeded" in str(meta.get("err")):
                status = "BUDGET"
                break
            continue
        fails, checks, vi_p, vi_s = await _gate(job, p, s)
        job.history.append({"model": model_key, "round": rnd, "fails": fails, "checks": checks,
                            "vi_problem": vi_p, "vi_solution": vi_s, "cost": meta.get("cost")})
        prev = (p, s)
        if not fails:
            status = "ACCEPT"
            break
    last = job.history[-1] if job.history else {}
    lc = last.get("checks") or {}
    if status == "FAIL" and lc.get("G2") == "PASS" and not lc.get("G0G1") and lc.get("G3", {}).get("pass") is False:
        # T4 §3.5 G3b: only the span judge objects (G0/G1/G2 pass) -> tie-break with a second judge family
        ok, fb3, det3 = await g3(row, vi_p, vi_s, f"seed:{key}", job.gold, judges=("sonnet",))
        job.history.append({"model": "J2-sonnet", "round": "tiebreak", "fails": [] if ok else fb3,
                            "checks": {"G3b": {"pass": ok, **det3}}})
        if ok:
            status = "ACCEPT"
            job.history[-1]["accepted_draft"] = f"{last['model']}-r{last['round']}"
    if status == "FAIL" and all("meta" in h and h["meta"].get("err") for h in job.history):
        status = "ERROR"                                   # infrastructure only: retried on the next run
    rec = {"seed_id": key, "source": row["source"], "index": row["index"], "status": status,
           "accepted_by": (job.history[-1]["model"] + f"-r{job.history[-1]['round']}") if status == "ACCEPT"
           else None, "problem_vi": vi_p if status == "ACCEPT" else None,
           "solution_vi": vi_s if status == "ACCEPT" else None, "history": job.history,
           "glossary_fired": [e["id"] for e in gl], "rules_fired": [c["id"] for c in rules],
           "n_spans": len(job.spans)}
    return store.put(key, rec)


async def run_rows(rows: list[dict], store: Store, concurrency: int = 8) -> None:
    sem = asyncio.Semaphore(concurrency)
    done = 0

    async def one(r):
        nonlocal done
        async with sem:
            try:
                rec = await translate_row(r, store)
            except Exception as e:
                rec = store.put(r["seed_id"], {"seed_id": r["seed_id"], "status": "ERROR",
                                               "err": f"{type(e).__name__}: {e}"[:500]})
            done += 1
            print(f"[mta] {done}/{len(rows)} {r['source']}#{r['index']}: {rec.get('status')} "
                  f"{rec.get('accepted_by') or ''}", flush=True)
    await asyncio.gather(*(one(r) for r in rows))


def run(limit: int | None = None, only: list[str] | None = None) -> dict:
    from .common import WORK, client, read_jsonl
    det = {d["seed_id"]: d for d in read_jsonl(WORK / "detect.jsonl")}
    aligned = read_jsonl(WORK / "aligned.jsonl")
    extra = set()
    from .assemble import current_rows             # solve-check FAILs (Gemini or Haiku batch): EN solved, VI not
    extra = {r["seed_id"] for r in current_rows() if r["solvecheck"] == "FAIL"}
    rows = [r for r in aligned if det[r["seed_id"]]["retranslate"] or r["seed_id"] in extra]
    if only:
        rows = [r for r in rows if r["seed_id"] in set(only) or f"{r['source']}#{r['index']}" in set(only)]
    if limit:
        rows = rows[:limit]
    store = Store("mta")
    print(f"[mta] {len(rows)} rows ({len(extra)} from the solve-check)", flush=True)
    c = client()
    asyncio.run(_main(rows, store, c))
    from collections import Counter
    st = Counter(store.get(r["seed_id"])["status"] for r in rows if store.get(r["seed_id"]))
    out = {"n": len(rows), "status": dict(st), "spend": c.costs.summary()}
    print(json.dumps(out, ensure_ascii=False))
    return out


async def _main(rows, store, c):
    try:
        await run_rows(rows, store)
    finally:
        await c.aclose()
