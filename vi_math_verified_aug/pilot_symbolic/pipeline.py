"""G3 arm-Y pipeline, stages 1-5 (resumable; every LLM call cached and logged under logs/api/G3).

    python -m vi_math_verified_aug.pilot_symbolic.pipeline formalize   # 1. Haiku 5.5 -> IR, <= 3 attempts
    python -m vi_math_verified_aug.pilot_symbolic.pipeline mutate      # 2. X4 operators per slot (CPU only)
    python -m vi_math_verified_aug.pilot_symbolic.pipeline render      # 3. M1 deterministic; M2-M6 Haiku editor
    python -m vi_math_verified_aug.pilot_symbolic.pipeline cycle       # 4. blind gemini-3.1-flash-lite back-formalization
    python -m vi_math_verified_aug.pilot_symbolic.pipeline gates       # 5. leak / answer-in-text / chrF / dedup /
                                                                       #    bge-m3 decontam + cross-seed dup (cos < 0.85)
Outputs: data/pilot/symbolic/{formalize,mutate,render,cycle,gates}.jsonl
"""
from __future__ import annotations

import asyncio
import json
import random
import sys
import time
from collections import Counter
from fractions import Fraction

from ..symbolic import (IRError, ExecError, Program, execute, from_expr_json, is_answer_tail, is_identity,
                        is_semantic_tail, mutate, render_substitutions, validate_program, gate)
from ..symbolic import operators as OPS
from ..symbolic.render import TextEdit
from . import seeds as SD
from .common import OUT, Store, call, parse_json, render, req, resp_meta
from .editor import check_output, editor_view, final_text
from .gates import Deduper, item_gates

SEED = SD.SEED
MAX_FORMALIZE_ATTEMPTS = 3
MAX_EDITOR_ATTEMPTS = 2
MAX_BACK_ATTEMPTS = 2
COS_VI = 0.85        # X5 calibration (exec/X5_decontam.md §5): bge-m3 Vi-Vi


def _fx(v) -> str | None:
    return None if v is None else str(Fraction(v))


# ============================================================================ 1. formalize
def _feedback(prev: str, errs: list[str]) -> str:
    return ("PREVIOUS ATTEMPT (rejected by the validator):\n" + (prev or "")[:6000] + "\n\nVALIDATOR ERRORS:\n" +
            "\n".join(f"- {e}" for e in errs[:12]) + "\n\nFix every error and output the corrected JSON object only.")


def load_ir(sid: str, text: str, out_text: str | None, gold=None) -> tuple[Program | None, list[str]]:
    j = parse_json(out_text) if out_text else None
    if not isinstance(j, dict):
        return None, ["json: no JSON object in the output"]
    try:
        P = from_expr_json(sid, text, j)
    except (IRError, ExecError, ValueError, KeyError, TypeError) as e:
        return None, [f"ir: {e}"]
    try:
        errs = [str(v) for v in validate_program(P, gold)]
    except Exception as e:                                   # validator crash = reject, recorded
        return None, [f"validator: {type(e).__name__}: {e}"]
    return P, errs


async def formalize_seed(st: Store, s: dict):
    r0 = st.get(s["seed_id"])
    if r0 and r0.get("done"):
        return
    attempts, fb = [], ""
    P = None
    for a in range(MAX_FORMALIZE_ATTEMPTS):
        rp = render("formalize", text=s["text"], en_text=s["en_text"], gold=s["gold"])
        rq = req("haiku", rp, item=f"seed:{s['seed_id']}", effort="low", max_tokens=8000, tag="formalize",
                 extra_user=fb)
        r = await call(rq)
        P, errs = (None, [f"api: {r.error_class}: {r.error}"]) if not r.ok else \
            load_ir(s["seed_id"], s["text"], r.text, s["gold"])
        attempts.append({"attempt": a + 1, "errors": errs, **resp_meta(r)})
        if r.ok and P is not None and not errs:
            break
        fb = _feedback(r.text or "", errs)
        P = None
    st.put(s["seed_id"], {"seed_id": s["seed_id"], "index": s["index"], "done": True, "accepted": P is not None,
                          "n_attempts": len(attempts), "attempts": attempts,
                          "ir": P.to_dict() if P is not None else None})


async def stage_formalize():
    st = Store("formalize")
    seeds = list(SD.load_seeds().values())
    await asyncio.gather(*(formalize_seed(st, s) for s in seeds))
    rows = [st.get(s["seed_id"]) for s in seeds]
    acc = sum(r["accepted"] for r in rows)
    p1 = sum(r["accepted"] and r["n_attempts"] == 1 for r in rows)
    print(f"formalize: accepted {acc}/{len(rows)} (pass@1 {p1})")


def accepted_programs() -> dict[str, Program]:
    st = Store("formalize")
    return {k: Program.from_dict(r["ir"]) for k, r in st.rows.items() if r.get("accepted")}


# ============================================================================ 2. mutate
_GATE_LOG: Counter = Counter()
_orig_gate = OPS.gate


def _counting_gate(P, P2, op=""):
    why = _orig_gate(P, P2, op)
    _GATE_LOG[(op, "pass" if why is None else why.split(":")[0])] += 1
    return why


def stage_mutate():
    OPS.gate = _counting_gate                      # instrument X4's internal gate (read-only wrapper)
    progs = accepted_programs()
    partners = [progs[k] for k in sorted(progs)]
    st = Store("mutate")
    for sl in SD.load_slots():
        if st.get(sl["key"]):
            continue
        P = progs.get(sl["seed_id"])
        row = {**sl}
        if P is None:
            st.put(sl["key"], {**row, "ok": False, "reason": "no_ir"})
            continue
        _GATE_LOG.clear()
        rng = random.Random(f"{sl['key']}:{SEED}")
        t0 = time.perf_counter()
        kw = {"partners": partners} if sl["op"] == "M6" else {}
        res = mutate(sl["op"], P, rng, **kw)
        dt = time.perf_counter() - t0
        gl = {f"{k[1]}": v for k, v in _GATE_LOG.items()}
        row.update({"ok": res.ok, "reason": res.reason, "secs": round(dt, 3), "gate_calls": gl})
        if res.ok:
            P2 = res.program
            # independent re-check of the X4 guarantees on the returned program
            chk = {"gate": gate(P, P2, sl["op"]), "tail": is_answer_tail(P, P2), "identity": is_identity(P, P2)}
            try:
                chk["semantic_tail"] = is_semantic_tail(P, P2)
            except Exception as e:                                    # pragma: no cover
                chk["semantic_tail"] = f"error: {e}"
            env0, env2 = execute(P), execute(P2)
            row.update({"ir2": P2.to_dict(), "edits": [e.to_dict() for e in res.edits],
                        "editor_edits": [e.to_editor_dict() for e in res.edits],
                        "answer": _fx(env2[P2.query]), "seed_answer": _fx(env0[P.query]),
                        "partner": P2.meta.get("partner"), "recheck": chk,
                        "m1_solver": P2.meta.get("m1_solver")})
        st.put(sl["key"], row)
    OPS.gate = _orig_gate
    rows = list(st.rows.values())
    by = Counter((r["op"], r["ok"]) for r in rows)
    print("mutate:", dict(sorted(by.items())))


# ============================================================================ 3. render
def _edits(row) -> list[TextEdit]:
    out = []
    for d in row["edits"]:
        d = dict(d)
        if "span" in d and d["span"] is not None:
            d["span"] = tuple(d["span"])
        out.append(TextEdit(**d))
    return out


async def render_slot(st: Store, row: dict, P: Program, seeds: dict, progs: dict):
    r0 = st.get(row["key"])
    if r0 and r0.get("done"):
        return
    P2 = Program.from_dict(row["ir2"])
    edits = _edits(row)
    op = row["op"]
    base = {k: row[k] for k in ("seed_id", "index", "slot", "op")}
    if op == "M1":
        R = render_substitutions(P2, edits)
        errs = [str(v) for v in validate_program(R, P2.answer if P2.answer is not None else execute(P2)[P2.query])]
        st.put(row["key"], {**base, "done": True, "renderer": "deterministic", "tag_ok": True, "tag_attempts": 0,
                            "text": R.text, "revalidate_errors": errs, "warning": ""})
        return
    partner = progs.get(row.get("partner")) if op == "M6" else None
    view = editor_view(P, P2, edits, op)
    attempts, fb, out, issues = [], "", None, ["not run"]
    for a in range(MAX_EDITOR_ATTEMPTS):
        rp = render("editor", **view)
        rq = req("haiku", rp, item=f"gen:G3:{row['key']}", effort="low", max_tokens=6000, tag=f"editor_{op}",
                 extra_user=fb)
        r = await call(rq)
        j = parse_json(r.text) if r.ok else None
        out = j.get("text_tagged") if isinstance(j, dict) else None
        warn = (j.get("warning") or "") if isinstance(j, dict) else ""
        issues = ["no text_tagged in output"] if not out else check_output(out, P, P2, op, partner)
        attempts.append({"attempt": a + 1, "issues": issues, "text_tagged": out, "warning": warn, **resp_meta(r)})
        if not issues:
            break
        fb = ("BẢN SỬA TRƯỚC (bị từ chối):\n" + (out or r.text or "")[:4000] + "\n\nLỖI THẺ:\n" +
              "\n".join(f"- {i}" for i in issues[:10]) + "\n\nHãy sửa các lỗi trên và chỉ trả về đối tượng JSON.")
    ok = not issues
    st.put(row["key"], {**base, "done": True, "renderer": "haiku_editor_v3", "tag_ok": ok,
                        "tag_attempts": len(attempts), "attempts": attempts, "view": view,
                        "text_tagged": out if ok else None, "text": final_text(out) if ok else None,
                        "warning": attempts[-1]["warning"] if attempts else ""})


async def stage_render():
    progs = accepted_programs()
    seeds = SD.load_seeds()
    st = Store("render")
    rows = [r for r in Store("mutate").rows.values() if r.get("ok")]
    await asyncio.gather(*(render_slot(st, r, progs[r["seed_id"]], seeds, progs) for r in rows))
    rr = [st.get(r["key"]) for r in rows]
    print("render: tag_ok", Counter((r["op"], r["tag_ok"]) for r in rr))


# ============================================================================ 4. cycle check
async def cycle_slot(st: Store, rrow: dict, mrow: dict):
    r0 = st.get(rrow["key"])
    if r0 and r0.get("done") and not r0.get("api_fail", bool(r0.get("attempts")) and not r0["attempts"][-1].get("api_ok", True)):
        return
    text = rrow["text"]
    target = Fraction(mrow["answer"])
    attempts, fb, P3, errs, ans = [], "", None, [], None
    for a in range(MAX_BACK_ATTEMPTS):
        rp = render("back", problem=text)
        rq = req("g31l", rp, item=f"gen:G3:{rrow['key']}", max_tokens=6144, temperature=0.0, json_mode=True,
                 tag="back_formalize", extra_user=fb)
        r = await call(rq)
        P3, errs = (None, [f"api: {r.error_class}: {r.error}"]) if not r.ok else \
            load_ir(f"back:{rrow['key']}", text, r.text, None)
        ans = None
        if P3 is not None:
            try:
                ans = execute(P3)[P3.query]
            except Exception as e:                       # exec failure: retry with the error (never the answer)
                errs = errs + [f"exec: {e}"]
                P3 = None
        attempts.append({"attempt": a + 1, "errors": errs, "answer": _fx(ans), **resp_meta(r)})
        if P3 is not None and ans is not None:
            break
        if not r.ok:
            break
        fb = _feedback(r.text or "", [e for e in errs if not e.startswith("[gold]")] or ["could not execute"])
    P2 = Program.from_dict(mrow["ir2"])
    struct_same = None
    if P3 is not None:
        try:
            struct_same = P3.canonical() == P2.canonical()
        except Exception:
            struct_same = None
    api_fail = bool(attempts) and not attempts[-1]["api_ok"]   # provider failure: not a verdict; re-run later
    st.put(rrow["key"], {"seed_id": rrow["seed_id"], "op": rrow["op"], "done": True, "api_fail": api_fail,
                         "back_answer": _fx(ans), "target": _fx(target),
                         "cycle_ok": ans is not None and ans == target, "executed": ans is not None,
                         "validator_flags": errs if P3 is not None else [], "struct_same": struct_same,
                         "n_attempts": len(attempts), "attempts": attempts,
                         "ir3": P3.to_dict() if P3 is not None else None})


async def stage_cycle():
    st = Store("cycle")
    m = Store("mutate").rows
    rows = [r for r in Store("render").rows.values() if r.get("tag_ok")]
    await asyncio.gather(*(cycle_slot(st, r, m[r["key"]]) for r in rows))
    print("cycle:", Counter((st.get(r["key"])["op"], st.get(r["key"])["cycle_ok"]) for r in rows),
          "api_fail:", sum(bool(st.get(r["key"]).get("api_fail")) for r in rows))


# ============================================================================ 5. common gates
def stage_gates():
    seeds = SD.load_seeds()
    m, rnd, cyc = Store("mutate").rows, Store("render").rows, Store("cycle").rows
    cos = Store("cosine").rows             # bge-m3 signals (pilot_symbolic.cosine); X5 tau = 0.85 Vi-Vi
    st = Store("gates")
    st.rows = {}
    dd = Deduper()
    for sl in SD.load_slots():               # slot order = schedule order (deterministic dedup)
        k = sl["key"]
        r = rnd.get(k)
        if not r or not r.get("tag_ok"):
            continue
        mr = m[k]
        g = item_gates(r["text"], seeds[sl["seed_id"]]["text"], Fraction(mr["answer"]), sl["op"])
        c = cyc.get(k, {})
        cs = cos.get(k, {})
        decontam_ok = cs.get("cos_eval_max", 1.0) < COS_VI          # missing embedding = fail closed
        xdup_ok = (cs.get("cos_other_seed_item_max") or 0.0) < COS_VI
        pre = c.get("cycle_ok", False) and g["leak_ok"] and not g["answer_in_text"] and g["chrf_ok"]             and decontam_ok and xdup_ok
        dup_ok, jac, who = dd.check_and_add(k, sl["seed_id"], r["text"]) if pre else (None, None, None)
        st.rows[k] = {"key": k, "seed_id": sl["seed_id"], "op": sl["op"], **g, "cycle_ok": c.get("cycle_ok", False),
                      "cos_eval_max": cs.get("cos_eval_max"), "decontam_ok": decontam_ok,
                      "cos_seed": cs.get("cos_seed"), "cos_xseed_max": cs.get("cos_other_seed_item_max"),
                      "xseed_dup_ok": xdup_ok,
                      "dedup_ok": dup_ok, "max_jaccard": jac, "dup_of": who,
                      "survivor": bool(pre and dup_ok)}
    st.rewrite()
    print("gates: survivors", Counter((r["op"], r["survivor"]) for r in st.rows.values()))


def main(argv):
    stage = argv[1] if len(argv) > 1 else ""
    if stage == "formalize":
        asyncio.run(stage_formalize())
    elif stage == "mutate":
        stage_mutate()
    elif stage == "render":
        asyncio.run(stage_render())
    elif stage == "cycle":
        asyncio.run(stage_cycle())
    elif stage == "gates":
        stage_gates()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
