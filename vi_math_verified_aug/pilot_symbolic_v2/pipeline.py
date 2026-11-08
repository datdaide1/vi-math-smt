"""G4 arm-Y pipeline v2, paired with G3 on the same 60 seeds / 180 (seed, operator) slots and cached IRs.

    python -m vi_math_verified_aug.pilot_symbolic_v2.pipeline mutate   # fixed operators (M6 dropped, story checks)
    python -m vi_math_verified_aug.pilot_symbolic_v2.pipeline render   # E (edit only) / R1 (edit + fluency) / R2 (IR)
    python -m vi_math_verified_aug.pilot_symbolic_v2.pipeline cycle    # blind gemini-3.1-flash-lite back-formalization
    python -m vi_math_verified_aug.pilot_symbolic_v2.pipeline gates    # leak / answer-in-text / chrF / cos / dedup

Variants (same slot, same mutated program P2):
  E   G3's renderer: M1 deterministic substitution, M2-M5 Haiku editor.v3 (minimal edit). Rows identical to G3 are
      copied from G3 run_b (render, cycle, judges); only slots whose program changed under the fixes are re-run.
  R1  E's tagged output -> Haiku fluency.v1 (localization card, Vietnamese conventions, natural phrasing; money tags
      rewritten to the localized đồng values; never the answer) -> tag check AGAIN -> cycle check AGAIN.
  R2  Haiku ir_render.v1 writes the problem from the localized program's labelled givens and relations plus the
      card; never sees the seed text or the answer -> tag check -> cycle check.
R1/R2 target program = ``localize(P2)`` (USD -> đồng, re-executed; see localize.py).
Outputs: data/pilot/symbolic_v2/{mutate,render_E,render_R1,render_R2,cycle_E,cycle_R1,cycle_R2,gates}.jsonl
"""
from __future__ import annotations

import asyncio
import json
import random
import re
import sys
from collections import Counter
from fractions import Fraction

from ..pilot_symbolic import seeds as SD
from ..pilot_symbolic.editor import check_output, editor_view, final_text
from ..pilot_symbolic.gates import Deduper, item_gates
from ..pilot_symbolic.pipeline import MAX_BACK_ATTEMPTS, MAX_EDITOR_ATTEMPTS, _edits, _feedback, _fx, \
    accepted_programs, load_ir
from ..symbolic import Program, execute, is_semantic_tail, mutate, render_substitutions, tagged_text
from ..symbolic import operators as OPS
from ..symbolic.numerals import contains_value
from ..symbolic.render import _TAG
from . import roles as RL
from .common import Store, call, g3, parse_json, render, req, resp_meta
from .localize import card_for, ir_view, keep_units, localize, money_map

SEED = SD.SEED
VARIANTS = ("E", "R1", "R2", "R1b")
COS_VI = 0.85


def P_of(d: dict) -> Program:
    return Program.from_dict(d)


# ============================================================================ 1. mutate (fixed operators)
async def _story_check(P, op, desc, key, n):
    rp = render("story", seed=P.text, op=op, change=desc)
    rq = req("haiku", rp, item=f"gen:G4:{key}", effort="low", max_tokens=3000, tag=f"story_{op}",
             cache_system=True)
    r = await call(rq)
    v = parse_json(r.text) if r.ok else None
    v = v if isinstance(v, dict) else {}
    ok = v.get("type_ok") is True and v.get("story_meaning") is True
    return {"n": n, "desc": desc, "verdict": v, "accept": ok, **resp_meta(r)}


def _row_from(P, P2, edits, sl, source):
    return {"ir2": P2.to_dict(), "edits": [e.to_dict() for e in edits],
            "editor_edits": [e.to_editor_dict() for e in edits],
            "answer": _fx(execute(P2)[P2.query]), "seed_answer": _fx(execute(P)[P.query]), "source": source}


async def mutate_slot(st: Store, sl: dict, P: Program | None, g3row: dict | None):
    if st.get(sl["key"]):
        return
    base = {k: sl[k] for k in ("seed_id", "index", "slot", "op")}
    op = sl["op"]
    if P is None:
        st.put(sl["key"], {**base, "ok": False, "reason": "no_ir", "same_as_g3": False})
        return
    if op == "M6":
        st.put(sl["key"], {**base, "ok": False, "reason": "dropped in G4 (M6 compose)", "same_as_g3": False})
        return
    if op in ("M1", "M2"):          # unchanged operators: G3's program verbatim
        if g3row and g3row.get("ok"):
            st.put(sl["key"], {**base, "ok": True, "same_as_g3": True, "source": "g3",
                               **{k: g3row[k] for k in ("ir2", "edits", "editor_edits", "answer", "seed_answer")}})
        else:
            st.put(sl["key"], {**base, "ok": False, "reason": (g3row or {}).get("reason"), "same_as_g3": True})
        return
    # M3/M4/M5: candidates = G3's program, then fresh draws with role-filtered templates
    cands, seen, log = [], set(), []
    if g3row and g3row.get("ok"):
        cands.append(("g3", P_of(g3row["ir2"]), _edits(g3row)))
    OPS._templates_for = RL.role_templates
    try:
        for r in range(1, RL.MAX_DRAWS + 1):
            res = mutate(op, P, random.Random(f"{sl['key']}:{SEED}:g4r{r}"))
            if res.ok:
                cands.append((f"r{r}", res.program, res.edits))
            elif r == 1:
                log.append({"draw": "r1", "not_applicable": res.reason})
    finally:
        OPS._templates_for = RL.ORIG_TEMPLATES
    checks, chosen = [], None
    for src, P2, edits in cands:
        sig = json.dumps([e.to_editor_dict() for e in edits], ensure_ascii=False, sort_keys=True)
        if sig in seen:
            continue
        seen.add(sig)
        ok, why = RL.det_check(op, P, P2, edits)
        try:
            stail = bool(is_semantic_tail(P, P2))
        except Exception:                                        # noqa: BLE001
            stail = None
        if not ok:
            checks.append({"source": src, "det_ok": False, "det_reason": why, "semantic_tail": stail})
            continue
        if sum(1 for c in checks if c.get("det_ok")) >= RL.MAX_CHECKS:
            break
        desc = RL.describe(op, P, P2, edits)
        hc = await _story_check(P, op, desc, f"{sl['key']}:{src}", len(checks))
        checks.append({"source": src, "det_ok": True, "semantic_tail": stail, **hc})
        if hc["accept"]:
            chosen = (src, P2, edits)
            break
    row = {**base, "n_candidates": len(seen), "checks": checks, "draw_log": log}
    if chosen is None:
        why = "no candidate passed the story-role checks" if cands else "operator not applicable"
        st.put(sl["key"], {**row, "ok": False, "reason": why, "same_as_g3": False})
        return
    src, P2, edits = chosen
    st.put(sl["key"], {**row, "ok": True, "same_as_g3": src == "g3", **_row_from(P, P2, edits, sl, src)})


async def stage_mutate():
    progs = accepted_programs()
    g3m = g3("mutate")
    st = Store("mutate")
    await asyncio.gather(*(mutate_slot(st, sl, progs.get(sl["seed_id"]), g3m.get(sl["key"]))
                           for sl in SD.load_slots()))
    rows = list(st.rows.values())
    print("mutate:", dict(sorted(Counter((r["op"], r["ok"], r.get("same_as_g3")) for r in rows).items())))


# ============================================================================ 2. render E / R1 / R2
async def _llm_tagged(key_prompt: str, view: dict, check, item: str, tag: str) -> dict:
    attempts, fb, out, issues = [], "", None, ["not run"]
    for a in range(MAX_EDITOR_ATTEMPTS):
        rp = render(key_prompt, **view)
        rq = req("haiku", rp, item=item, effort="low", max_tokens=6000, tag=tag, extra_user=fb,
                 cache_system=key_prompt != "editor")
        r = await call(rq)
        j = parse_json(r.text) if r.ok else None
        out = j.get("text_tagged") if isinstance(j, dict) else None
        warn = (j.get("warning") or "") if isinstance(j, dict) else ""
        issues = ["no text_tagged in output"] if not out else check(out)
        attempts.append({"attempt": a + 1, "issues": issues, "text_tagged": out, "warning": warn, **resp_meta(r)})
        if not issues:
            break
        fb = ("BẢN TRƯỚC (bị từ chối):\n" + (out or r.text or "")[:4000] + "\n\nLỖI THẺ:\n" +
              "\n".join(f"- {i}" for i in issues[:10]) + "\n\nHãy sửa các lỗi trên và chỉ trả về đối tượng JSON.")
    ok = not issues
    return {"tag_ok": ok, "tag_attempts": len(attempts), "attempts": attempts,
            "text_tagged": out if ok else None, "text": final_text(out) if ok else None,
            "warning": attempts[-1]["warning"] if attempts else ""}


async def render_E(st: Store, m: dict, P: Program, g3r: dict | None):
    if st.get(m["key"]):
        return
    base = {k: m[k] for k in ("seed_id", "index", "slot", "op")}
    P2 = P_of(m["ir2"])
    edits = _edits(m)
    if m["op"] == "M1":
        R = render_substitutions(P2, edits)
        st.put(m["key"], {**base, "variant": "E", "same_as_g3": m["same_as_g3"], "renderer": "deterministic",
                          "tag_ok": True, "text_tagged": tagged_text(R), "text": R.text, "warning": "",
                          "target": m["answer"]})
        return
    if m["same_as_g3"] and g3r is not None:
        st.put(m["key"], {**base, "variant": "E", "same_as_g3": True, "renderer": "g3_copy",
                          "tag_ok": g3r["tag_ok"], "text_tagged": g3r.get("text_tagged"), "text": g3r.get("text"),
                          "warning": g3r.get("warning", ""), "target": m["answer"]})
        return
    view = editor_view(P, P2, edits, m["op"])
    res = await _llm_tagged("editor", view, lambda o: check_output(o, P, P2, m["op"]), f"gen:G4:E:{m['key']}",
                            f"editor_{m['op']}")
    st.put(m["key"], {**base, "variant": "E", "same_as_g3": False, "renderer": "haiku_editor_v3", **res,
                      "view": view, "target": m["answer"]})


async def render_R1(st: Store, m: dict, P: Program, e_row: dict, variant: str = "R1"):
    """R1: fluency.v1, 1 USD -> 1.000 đồng. R1b: fluency.v2 (unit rule fixed), 1 USD -> 10.000 đồng."""
    if st.get(m["key"]):
        return
    base = {k: m[k] for k in ("seed_id", "index", "slot", "op")}
    P2 = P_of(m["ir2"])
    PL, info = localize(P2, usd=10000 if variant == "R1b" else 1000)
    tgt = _fx(execute(PL)[PL.query])
    if not e_row.get("tag_ok"):
        st.put(m["key"], {**base, "variant": variant, "tag_ok": False, "reason": "no E render", "target": tgt,
                          "localized": info["rescaled"]})
        return
    cid, card = card_for(m["seed_id"])
    view = {"text_tagged": e_row["text_tagged"], "card": card, "money": money_map(PL, info),
            "keep_units": keep_units(PL)}
    pk = "fluency2" if variant == "R1b" else "fluency"
    res = await _llm_tagged(pk, view, lambda o: check_output(o, P, PL, m["op"]), f"gen:G4:{variant}:{m['key']}",
                            f"{pk}_{m['op']}")
    st.put(m["key"], {**base, "variant": variant, **res, "view": view, "card": cid, "target": tgt,
                      "localized": info["rescaled"], "localize_why": info["why"], "ir_target": PL.to_dict()})


async def render_R2(st: Store, m: dict, P: Program):
    if st.get(m["key"]):
        return
    base = {k: m[k] for k in ("seed_id", "index", "slot", "op")}
    P2 = P_of(m["ir2"])
    PL, info = localize(P2)
    tgt = _fx(execute(PL)[PL.query])
    cid, card = card_for(m["seed_id"])
    view = {**ir_view(PL), "card": card}
    res = await _llm_tagged("irr", view, lambda o: check_output(o, P, PL, m["op"]), f"gen:G4:R2:{m['key']}",
                            f"irrender_{m['op']}")
    st.put(m["key"], {**base, "variant": "R2", **res, "view": view, "card": cid, "target": tgt,
                      "localized": info["rescaled"], "localize_why": info["why"], "ir_target": PL.to_dict()})


async def stage_render():
    progs = accepted_programs()
    mut = [r for r in Store("mutate").rows.values() if r.get("ok")]
    g3r = g3("render")
    sE, s1, s2 = Store("render_E"), Store("render_R1"), Store("render_R2")
    await asyncio.gather(*(render_E(sE, m, progs[m["seed_id"]], g3r.get(m["key"])) for m in mut))
    await asyncio.gather(*(render_R1(s1, m, progs[m["seed_id"]], sE.get(m["key"])) for m in mut),
                         *(render_R2(s2, m, progs[m["seed_id"]]) for m in mut))
    for v, s in (("E", sE), ("R1", s1), ("R2", s2)):
        print(v, "tag_ok", dict(sorted(Counter((r["op"], r["tag_ok"]) for r in s.rows.values()).items())))


async def stage_render_r1b():
    progs = accepted_programs()
    mut = [r for r in Store("mutate").rows.values() if r.get("ok")]
    sE, sb = Store("render_E"), Store("render_R1b")
    await asyncio.gather(*(render_R1(sb, m, progs[m["seed_id"]], sE.get(m["key"]), "R1b") for m in mut))
    print("R1b tag_ok", dict(sorted(Counter((r["op"], r["tag_ok"]) for r in sb.rows.values()).items())))


# ============================================================================ 3. cycle check
async def cycle_one(st: Store, key: str, rrow: dict, prog: Program):
    r0 = st.get(key)
    if r0 and r0.get("done") and not r0.get("api_fail"):
        return
    text, target = rrow["text"], Fraction(rrow["target"])
    attempts, fb, P3, errs, ans = [], "", None, [], None
    for a in range(MAX_BACK_ATTEMPTS):
        rp = render("back", problem=text)
        rq = req("g31l", rp, item=f"gen:G4:{rrow['variant']}:{key}", max_tokens=6144, temperature=0.0,
                 json_mode=True, tag="back_formalize", extra_user=fb)
        r = await call(rq)
        P3, errs = (None, [f"api: {r.error_class}: {r.error}"]) if not r.ok else \
            load_ir(f"back:{key}", text, r.text, None)
        ans = None
        if P3 is not None:
            try:
                ans = execute(P3)[P3.query]
            except Exception as e:                       # noqa: BLE001
                errs = errs + [f"exec: {e}"]
                P3 = None
        attempts.append({"attempt": a + 1, "errors": errs, "answer": _fx(ans), **resp_meta(r)})
        if P3 is not None and ans is not None:
            break
        if not r.ok:
            break
        fb = _feedback(r.text or "", [e for e in errs if not e.startswith("[gold]")] or ["could not execute"])
    api_fail = bool(attempts) and not attempts[-1]["api_ok"]
    st.put(key, {"seed_id": rrow["seed_id"], "op": rrow["op"], "variant": rrow["variant"], "done": True,
                 "api_fail": api_fail, "back_answer": _fx(ans), "target": _fx(target),
                 "cycle_ok": ans is not None and ans == target, "executed": ans is not None,
                 "n_attempts": len(attempts), "attempts": attempts,
                 "ir3": P3.to_dict() if P3 is not None else None, "source": "g4"})


async def stage_cycle(variants=VARIANTS):
    m = Store("mutate").rows
    g3c = g3("cycle")
    jobs = []
    for v in variants:
        st = Store(f"cycle_{v}")
        for r in Store(f"render_{v}").rows.values():
            if not r.get("tag_ok"):
                continue
            if v == "E" and m[r["key"]]["same_as_g3"] and r["key"] in g3c and not st.get(r["key"]):
                c = g3c[r["key"]]
                st.put(r["key"], {**{k: c[k] for k in ("seed_id", "op", "back_answer", "target", "cycle_ok",
                                                      "executed", "n_attempts")},
                                  "variant": "E", "done": True, "api_fail": False, "source": "g3"})
                continue
            jobs.append(cycle_one(st, r["key"], r, None))
    await asyncio.gather(*jobs)
    for v in variants:
        rows = Store(f"cycle_{v}").rows.values()
        print(v, "cycle:", dict(sorted(Counter((r["op"], r["cycle_ok"]) for r in rows).items())),
              "api_fail", sum(bool(r.get("api_fail")) for r in rows))


# ============================================================================ 4. gates
_DAY = re.compile(r"thứ\s+(hai|ba|tư|năm|sáu|bảy)\b", re.I)


def answer_untagged(tagged: str, ans: Fraction) -> bool:
    """G3 §6 fix (a): the answer value written OUTSIDE the given tags (day names excluded)."""
    blank = _TAG.sub(lambda mm: " " * len(mm.group(0)), tagged or "")
    blank = _DAY.sub(lambda mm: " " * len(mm.group(0)), blank)
    return contains_value(blank, ans)


def stage_gates():
    seeds = SD.load_seeds()
    st = Store("gates")
    st.rows = {}
    cos = Store("cosine").rows
    for v in VARIANTS:
        rnd, cyc = Store(f"render_{v}").rows, Store(f"cycle_{v}").rows
        dd = Deduper()
        for sl in SD.load_slots():
            k = sl["key"]
            r = rnd.get(k)
            if not r or not r.get("tag_ok"):
                continue
            ans = Fraction(r["target"])
            g = item_gates(r["text"], seeds[sl["seed_id"]]["text"], ans, sl["op"])
            g["answer_untagged"] = answer_untagged(r["text_tagged"], ans)
            c = cyc.get(k, {})
            cs = cos.get(f"{v}:{k}", {})
            decontam_ok = cs.get("cos_eval_max", 1.0) < COS_VI
            xdup_ok = (cs.get("cos_other_seed_item_max") or 0.0) < COS_VI
            pre = c.get("cycle_ok", False) and g["leak_ok"] and not g["answer_untagged"] and g["chrf_ok"] \
                and decontam_ok and xdup_ok
            dup_ok, jac, who = dd.check_and_add(k, sl["seed_id"], r["text"]) if pre else (None, None, None)
            st.rows[f"{v}:{k}"] = {"key": f"{v}:{k}", "slot": k, "variant": v, "seed_id": sl["seed_id"],
                                   "op": sl["op"], **g, "cycle_ok": c.get("cycle_ok", False),
                                   "cos_eval_max": cs.get("cos_eval_max"), "decontam_ok": decontam_ok,
                                   "cos_seed": cs.get("cos_seed"), "cos_xseed_max": cs.get("cos_other_seed_item_max"),
                                   "xseed_dup_ok": xdup_ok, "dedup_ok": dup_ok, "max_jaccard": jac, "dup_of": who,
                                   "warning": bool(r.get("warning")), "survivor": bool(pre and dup_ok)}
    st.rewrite()
    for v in VARIANTS:
        rows = [r for r in st.rows.values() if r["variant"] == v]
        print(v, "survivors", sum(r["survivor"] for r in rows), "/", len(rows))


def main(argv):
    stage = argv[1] if len(argv) > 1 else ""
    if stage == "mutate":
        asyncio.run(stage_mutate())
    elif stage == "render":
        asyncio.run(stage_render())
    elif stage == "render_r1b":
        asyncio.run(stage_render_r1b())
    elif stage == "cycle":
        asyncio.run(stage_cycle(tuple(argv[2].split(",")) if len(argv) > 2 else VARIANTS))
    elif stage == "gates":
        stage_gates()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
