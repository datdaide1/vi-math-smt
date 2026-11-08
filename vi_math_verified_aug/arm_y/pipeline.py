"""G5 arm-Y pipeline (v3 operators, VI renderers R1b/R2b, EN mirror from the same IR). Resumable, round-based:
every stage sends attempt k for all pending items at once (direct calls, or Message Batches when cfg.batch and the
round has >= 100 Claude calls).

    python -m vi_math_verified_aug.arm_y.pipeline <run_dir> init v3|pool <n_seeds> <variants> [batch]
    python -m vi_math_verified_aug.arm_y.pipeline <run_dir> formalize|mutate|render|cycle|gates|vgate

variants (comma list): R1b = E (G3 editor.v3) + fluency.v2 with 1 USD = 10.000 đồng; R2b = ir_render.v1 from the
localized IR (same scale); EN = ir_render_en.v1 from the same mutated IR (USD kept). Cycle check: blind Haiku
back-formalizer (back_formalizer.v1 / back_formalizer_en.v1); the Gemini flash-lite free tier was exhausted on
2026-11-07 (500 RPD/project, G4 §8). Validity gate: Haiku 5.5 validity_judge.v3 (G2 rubric), rule from calib.py.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

from ..pilot_gen import seeds as G1S
from ..pilot_gen.util import OP_INSTR, chrf, leak_regex
from ..pilot_symbolic.editor import check_output, editor_view, final_text
from ..pilot_symbolic.gates import Deduper
from ..pilot_symbolic.judge import lab_v3
from ..pilot_symbolic.pipeline import _edits, _feedback, _fx, load_ir
from ..pilot_symbolic_v2 import roles as RL
from ..pilot_symbolic_v2.localize import card_for, ir_view, keep_units, localize, money_map
from ..pilot_symbolic_v2.pipeline import answer_untagged
from ..symbolic import Program, execute, render_substitutions, tag_issues, tagged_text
from . import ops as OP
from .common import ROOT, Store, call, call_many, call_many_chunks, parse_json, render, req, resp_meta, spent
from .irview_en import ir_view_en

SEED = 20261108
J = 3
CHRF_MAX = 0.90


def cfg(run: Path) -> dict:
    return json.load(open(run / "config.json"))


def P_of(d) -> Program:
    return Program.from_dict(d)


# ============================================================================ init (seeds + slots)
def _pilot_indices() -> set[int]:
    out = set()
    for p in ("data/pilot/generator/seeds.jsonl", "data/pilot/g2/seeds40.jsonl", "data/pilot/symbolic/seeds.jsonl",
              "data/pilot/symbolic_v3/seeds.jsonl"):
        f = ROOT / p
        if f.exists():
            out |= {json.loads(l)["index"] for l in open(f, encoding="utf-8") if l.strip()}
    return out


def init(run: Path, source: str, n: int, variants: list[str], batch: bool):
    run.mkdir(parents=True, exist_ok=True)
    if source == "v3":                         # fresh dev seeds (T1 §2.0 dev pool), disjoint from G1/G2/G3
        vi = [json.loads(l) for l in open(G1S.VI, encoding="utf-8")]
        en = [json.loads(l) for l in open(G1S.EN, encoding="utf-8")]
        excl = G1S.R1_CORRUPT | G1S.R1_HALF_ENGLISH | _pilot_indices()
        pool = []
        for v, e in zip(vi, en):
            sid = G1S.seed_id(v["index"])
            if int(sid, 16) % 20 or v["index"] in excl:
                continue
            pool.append({"seed_id": sid, "index": v["index"], "text": v["question"], "en_text": e["question"],
                         "gold": str(e["ground_truth"]).replace(",", "")})
        pool.sort(key=lambda d: d["index"])
        seeds = sorted(random.Random(20261109).sample(pool, n), key=lambda d: d["index"])
        srng = random.Random(SEED)
        ops, w = list(OP.PI_V3), list(OP.PI_V3.values())
        slots = [{"key": f"{s['seed_id']}:{j}", "seed_id": s["seed_id"], "slot": j, "op": srng.choices(ops, w)[0]}
                 for s in seeds for j in range(1, J + 1)]
    else:                                      # S3 pool_v1 (GSM8K story seeds; dev + pilot + dropped seeds excluded)
        excl = _pilot_indices()
        src = ROOT / "data/seeds/pool_v1.jsonl"
        if src.exists():
            rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
            dropped = set()
        else:                                  # pool_v1 not frozen yet: S3's aligned rows minus its decontam drops
            src = ROOT / "data/seeds/_work/aligned.jsonl"
            rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
            dropped = {json.loads(l)["seed_id"] for l in open(ROOT / "data/seeds/_work/decontam.jsonl", encoding="utf-8")
                       if l.strip() and json.loads(l).get("drop")}
        pool = []
        for r in rows:
            if r.get("dataset") != "gsm8k" or r.get("dev") or r.get("index") in excl or r.get("drop")                     or r["seed_id"] in dropped:
                continue
            gold = r.get("gold") or r.get("gold_en") or r.get("gold_raw_en") or r.get("answer")
            pool.append({"seed_id": r["seed_id"], "index": r.get("index"), "text": r["problem_vi"],
                         "en_text": r["problem_en"], "gold": str(gold).replace(",", ""), "seed_source": src.name})
        # slots_v1 priority order (D1 scale_gen/slots.py, lead 23:15): all j=1 slots first, then j=2, then j=3; within
        # j by sha1("slots_v1|<key>"). Membership-independent, so the order equals D1's on the shared pool_v1.
        by_id = {d["seed_id"]: d for d in pool}
        order = sorted(((j, hashlib.sha1(f"slots_v1|{sid}:{j}".encode()).hexdigest(), sid) for sid in by_id
                        for j in range(1, J + 1)))[:n]
        slots = []
        for rank, (j, _, sid) in enumerate(order):
            rng = random.Random(f"G5|{SEED}|{sid}|{j}")            # per-slot op draw, pi_v3 (Y's own schedule)
            slots.append({"key": f"{sid}:{j}", "seed_id": sid, "slot": j, "rank": rank,
                          "op": rng.choices(list(OP.PI_V3), list(OP.PI_V3.values()))[0]})
        keep = {sl["seed_id"] for sl in slots}
        seeds = sorted((by_id[x] for x in keep), key=lambda d: d["seed_id"])
    for name, rows in (("seeds", seeds), ("slots", slots)):
        with open(run / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump({"source": source, "n_seeds": len(seeds), "variants": variants, "batch": batch, "pi": OP.PI_V3,
               "J": J}, open(run / "config.json", "w"), indent=1)
    print(f"init: {len(seeds)} seeds, {len(slots)} slots, ops {dict(Counter(s['op'] for s in slots))}")


def extend(run: Path, j: int, max_new: int | None = None):
    """Breadth-first wave j (lead 23:20): append slot j, in slots_v1 rank order, for every formalized seed that has no
    Y+ item yet. Existing slots and all stage rows are kept, so every stage simply resumes on the new slots."""
    slots = load(run, "slots")
    have = {s["key"] for s in slots}
    cyc = Store(run, "cycle_R2b").rows
    yplus = {r["slot"].split(":")[0] for r in Store(run, "vgate").rows.values()
             if r.get("pass") and (cyc.get(r["slot"]) or {}).get("cycle_ok")}       # screen pass AND Qwen cycle ok
    ok = set()
    for l in open(run / "formalize.jsonl", encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            if r.get("accepted"):
                ok.add(r["key"])
    pp = ROOT / "data" / "seeds" / "pool_v1.jsonl"
    if pp.exists():                  # waves only on pool_v1 seeds whose text is unchanged since formalization
        seeds = {s["seed_id"]: s for s in load(run, "seeds")}
        same = set()
        for l in open(pp, encoding="utf-8"):
            if l.strip():
                x = json.loads(l)
                if x.get("dataset") == "gsm8k" and x["seed_id"] in seeds and x["problem"] == seeds[x["seed_id"]]["text"]:
                    same.add(x["seed_id"])
        ok &= same
    # expected-yield order (lead 04:00): seeds whose earlier item failed stochastically (tag, gates, trace mismatch,
    # screen reject, cycle fail) first; structural failures (operator not applicable) last; slots_v1 hash within tier.
    # Seeds with a screen pass still awaiting the cycle check are potential Y+ and are skipped.
    vg = Store(run, "vgate").rows
    pending = {r["slot"].split(":")[0] for r in vg.values() if r.get("pass")
               and not (cyc.get(r["slot"]) or {}).get("done")}
    mut_ok = set()                                              # streamed (RAM rule)
    for l in open(run / "mutate.jsonl", encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            if r.get("ok"):
                mut_ok.add(r["seed_id"])
    cand = sorted((0 if sid in mut_ok else 1, hashlib.sha1(f"slots_v1|{sid}:{j}".encode()).hexdigest(), sid)
                  for sid in ok if sid not in yplus and sid not in pending and f"{sid}:{j}" not in have)
    cand = [(h, sid) for _, h, sid in cand]
    if max_new is not None:
        cand = cand[:max_new]
    rank = max((s.get("rank", 0) for s in slots), default=-1) + 1
    new = []
    for _, sid in cand:
        rng = random.Random(f"G5|{SEED}|{sid}|{j}")
        new.append({"key": f"{sid}:{j}", "seed_id": sid, "slot": j, "rank": rank,
                    "op": rng.choices(list(OP.PI_V3), list(OP.PI_V3.values()))[0]})
        rank += 1
    with open(run / "slots.jsonl", "a", encoding="utf-8") as f:
        for r in new:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"extend j={j}: +{len(new)} slots (seeds with Y+ so far: {len(yplus)}; formalized seeds {len(ok)})")


def yplus_report(run: Path) -> dict:
    vg = Store(run, "vgate").rows
    gates = Store(run, "gates").rows
    cyc = Store(run, "cycle_R2b").rows
    slots = load(run, "slots")
    yp = [r for r in vg.values() if r.get("variant") == "R2b" and r.get("pass")
          and (cyc.get(r["slot"]) or {}).get("cycle_ok") and (gates.get(f"R2b:{r['slot']}") or {}).get("leak_ok")]
    out = {"slots": len(slots), "slots_by_j": dict(Counter(s["slot"] for s in slots)),
           "VI_Yplus_items(screen&cycle&gates)": len(yp),
           "VI_Yplus_distinct_seeds": len({r["slot"].split(":")[0] for r in yp}),
           "screen_pass_awaiting_cycle": sum(1 for r in vg.values() if r.get("pass") and not (cyc.get(r["slot"]) or {}).get("done")),
           "yplus_items": sum(bool(r.get("pass")) for r in vg.values()),
           "yplus_distinct_seeds": len({r["slot"].split(":")[0] for r in vg.values() if r.get("pass")}),
           "haiku_leg_pass_items": sum(bool(r.get("haiku_pass")) for r in vg.values()),
           "haiku_leg_pass_distinct_seeds": len({r["slot"].split(":")[0] for r in vg.values() if r.get("haiku_pass")}),
           "lite_pending": sum(bool(r.get("lite_pending")) for r in vg.values()),
           "spent_g5": round(spent(), 3)}
    print(json.dumps(out))
    return out


LITE_FIELDS = ("key", "seed_id", "slot", "op", "tag_ok", "text", "text_tagged", "target", "warning", "done")


def render_lite(run: Path, v: str) -> dict[str, dict]:
    """render_<v> rows without the IR payload (laptop RAM rule: <= 300 MB per process)."""
    out: dict[str, dict] = {}
    p = run / f"render_{v}.jsonl"
    if p.exists():
        for l in open(p, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                out[r["key"]] = {k: r.get(k) for k in LITE_FIELDS}
    return out


def solve_lite(run: Path, v: str) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    p = run / f"solve_{v}.jsonl"
    if p.exists():
        for l in open(p, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                out[r["key"]] = r.get("trace")
    return out


def load(run, name):
    return [json.loads(l) for l in open(run / f"{name}.jsonl", encoding="utf-8") if l.strip()]


# ============================================================================ formalize
async def stage_formalize(run: Path):
    c = cfg(run)
    st = Store(run, "formalize")
    seeds = {s["seed_id"]: s for s in load(run, "seeds")}
    fb: dict[str, str] = {}
    for a in range(3):
        todo = [s for s in seeds.values() if not (st.get(s["seed_id"]) or {}).get("done")]
        if not todo:
            break
        reqs = []
        for s in todo:
            rp = render("formalize", text=s["text"], en_text=s["en_text"], gold=s["gold"])
            reqs.append(req("haiku", rp, item=f"seed:{s['seed_id']}", effort="low", max_tokens=8000, tag="formalize",
                            extra_user=fb.get(s["seed_id"], "")))
        resps = await call_many(reqs, c["batch"], f"formalize#{a + 1}")
        for s, r in zip(todo, resps):
            P, errs = (None, [f"api: {r.error_class}"]) if not r.ok else load_ir(s["seed_id"], s["text"], r.text, s["gold"])
            ok = r.ok and P is not None and not errs
            prev = (st.get(s["seed_id"]) or {}).get("attempts", [])
            att = prev + [{"attempt": a + 1, "errors": errs[:8], **resp_meta(r)}]
            st.put(s["seed_id"], {"seed_id": s["seed_id"], "done": ok or a == 2, "accepted": ok, "n_attempts": a + 1,
                                  "attempts": att, "ir": P.to_dict() if ok else None})
            fb[s["seed_id"]] = _feedback(r.text or "", errs)
    rows = list(st.rows.values())
    print(f"formalize: accepted {sum(r['accepted'] for r in rows)}/{len(rows)}; spent ${spent():.3f}")


def programs(run) -> dict[str, Program]:
    return {k: P_of(r["ir"]) for k, r in Store(run, "formalize").rows.items() if r.get("accepted")}


# ============================================================================ mutate (+ story check)
async def stage_mutate(run: Path, chunk: int = 400):
    """Lean (RAM rule): done keys are streamed, IRs are loaded only for pending seeds, and slots are processed in
    chunks of ``chunk`` (candidates -> one story-check round -> rows appended) so memory stays bounded."""
    c = cfg(run)
    done = set()
    for l in open(run / "mutate.jsonl", encoding="utf-8") if (run / "mutate.jsonl").exists() else []:
        if l.strip():
            done.add(json.loads(l)["key"])
    todo_slots = [sl for sl in load(run, "slots") if sl["key"] not in done]
    need = {sl["seed_id"] for sl in todo_slots}
    irs = {}
    for l in open(run / "formalize.jsonl", encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            if r["key"] in need and r.get("accepted"):
                irs[r["key"]] = r["ir"]
    st = Store(run, "mutate", keep_rows=False)
    print(f"mutate: {len(todo_slots)} pending slots", flush=True)
    for ci in range(0, len(todo_slots), chunk):
        await _mutate_chunk(run, c, st, todo_slots[ci:ci + chunk], irs)
        print(f"mutate: {min(ci + chunk, len(todo_slots))}/{len(todo_slots)} spent ${spent():.3f}", flush=True)
    rows = [json.loads(l) for l in open(run / "mutate.jsonl", encoding="utf-8") if l.strip()]
    print("mutate:", dict(sorted(Counter((r["op"], r["ok"]) for r in rows).items())), f"spent ${spent():.3f}")


async def _mutate_chunk(run, c, st, slots, irs):
    pend = []                                   # (slot, P, cands, log)
    for sl in slots:
        if sl["seed_id"] not in irs:
            st.put(sl["key"], {**sl, "ok": False, "reason": "no_ir"})
            continue
        P = P_of(irs[sl["seed_id"]])
        cands, log = OP.candidates(sl["op"], P, sl["key"])
        if not cands:
            st.put(sl["key"], {**sl, "ok": False, "reason": "no candidate passed the v3 constraints", "log": log})
            continue
        pend.append((sl, P, cands, log))
    # one Haiku story-check round for every M3/M4/M5 candidate
    jobs = [(i, j) for i, (sl, P, cands, _) in enumerate(pend) if sl["op"] in ("M3", "M4", "M5")
            for j in range(len(cands))]
    reqs = []
    for i, j in jobs:
        sl, P, cands, _ = pend[i]
        src, P2, edits = cands[j]
        rp = render("story", seed=P.text, op=sl["op"], change=RL.describe(sl["op"], P, P2, edits))
        reqs.append(req("haiku", rp, item=f"gen:G5:{sl['key']}:{src}", effort="low", max_tokens=3000,
                        tag=f"story_{sl['op']}", cache_system=True))
    resps = await call_many(reqs, c["batch"], "story")
    verdict = {}
    for (i, j), r in zip(jobs, resps):
        v = parse_json(r.text) if r.ok else None
        v = v if isinstance(v, dict) else {}
        verdict[(i, j)] = {"verdict": v, "accept": v.get("type_ok") is True and v.get("story_meaning") is True}
    for i, (sl, P, cands, log) in enumerate(pend):
        chosen, checks = None, []
        for j, (src, P2, edits) in enumerate(cands):
            if sl["op"] in ("M3", "M4", "M5"):
                checks.append({"source": src, **verdict[(i, j)]})
                if not verdict[(i, j)]["accept"]:
                    continue
            chosen = (src, P2, edits)
            break
        if chosen is None:
            st.put(sl["key"], {**sl, "ok": False, "reason": "story check rejected all candidates", "checks": checks,
                               "log": log})
            continue
        src, P2, edits = chosen
        st.put(sl["key"], {**sl, "ok": True, "source": src, "checks": checks, "ir2": P2.to_dict(),
                           "edits": [e.to_dict() for e in edits], "answer": _fx(execute(P2)[P2.query]),
                           "seed_answer": _fx(execute(P)[P.query])})


# ============================================================================ render
def _tagcheck(variant, out, P, PT, op):
    if variant.startswith("EN"):
        return tag_issues(out, PT, seed=None)
    return check_output(out, P, PT, op)


async def _render_rounds(run, c, variant, items, build, prompt_key, tagname, sample_idx: int = 0):
    """items: list of (key, row-base, P, PT, view). Two attempts; the retry gets only the tag issues."""
    st = Store(run, f"render_{variant}")
    fb: dict[str, str] = {}
    for a in range(2):
        todo = [it for it in items if not (st.get(it[0]) or {}).get("done")]
        if not todo:
            break
        reqs = [req("haiku", render(prompt_key, **it[4]), item=f"gen:G5:{variant}:{it[0]}", effort="low",
                    max_tokens=6000, tag=f"{tagname}_{it[1]['op']}", extra_user=fb.get(it[0], ""), sample_idx=sample_idx,
                    cache_system=prompt_key != "editor") for it in todo]
        async for off, resps in call_many_chunks(reqs, c["batch"], f"{variant}#{a + 1}"):
          for (key, base, P, PT, view), r in zip(todo[off:off + len(resps)], resps):
              P = P_of(P) if isinstance(P, dict) else P          # programs may be passed as dicts (RAM rule)
              PT = P_of(PT) if isinstance(PT, dict) else PT
              j = parse_json(r.text) if r.ok else None
              out = j.get("text_tagged") if isinstance(j, dict) else None
              warn = (j.get("warning") or "") if isinstance(j, dict) else ""
              issues = ["no text_tagged in output"] if not out else _tagcheck(variant, out, P, PT, base["op"])
              ok = not issues
              st.put(key, {**base, "variant": variant, "done": ok or a == 1, "tag_ok": ok, "tag_attempts": a + 1,
                           "issues": issues[:6], "text_tagged": out if ok else None,
                           "text": final_text(out) if ok else None, "warning": warn,
                           "target": _fx(execute(PT)[PT.query]), "ir_target": PT.to_dict(), **resp_meta(r)})
              fb[key] = ("BẢN TRƯỚC (bị từ chối):\n" if not variant.startswith("EN") else "PREVIOUS VERSION (rejected):\n") + \
                  (out or r.text or "")[:4000] + ("\n\nLỖI THẺ:\n" if variant != "EN" else "\n\nTAG ERRORS:\n") + \
                  "\n".join(f"- {i}" for i in issues[:10])
    rows = list(st.rows.values())
    print(variant, "tag_ok", sum(r["tag_ok"] for r in rows), "/", len(rows), f"spent ${spent():.3f}")


async def stage_render(run: Path):
    c = dict(cfg(run))
    if os.environ.get("ARMY_RENDER_VARIANTS"):        # e.g. R2b only: EN is rendered only for selected slots
        c["variants"] = os.environ["ARMY_RENDER_VARIANTS"].split(",")
    have = set()
    for v in c["variants"]:
        p = run / f"render_{v}.jsonl"
        if p.exists():
            for l in open(p, encoding="utf-8"):
                if l.strip():
                    x = json.loads(l)
                    if x.get("done"):
                        have.add((v, x["key"]))
    mut = []                                    # streamed: only mutated slots still missing a render (RAM rule)
    for l in open(run / "mutate.jsonl", encoding="utf-8"):
        if l.strip():
            x = json.loads(l)
            if x.get("ok") and any((v, x["key"]) not in have for v in c["variants"]):
                mut.append(x)
    need = {m["seed_id"] for m in mut}
    progs = {}
    for l in open(run / "formalize.jsonl", encoding="utf-8"):
        if l.strip():
            x = json.loads(l)
            if x["key"] in need and x.get("accepted"):
                progs[x["key"]] = x["ir"]                  # dicts; Programs are built on demand
    print(f"render: {len(mut)} slots pending", flush=True)
    base = lambda m: {k: m[k] for k in ("seed_id", "slot", "op")}  # noqa: E731
    if "R1b" in c["variants"]:                 # E first (M1 deterministic, M2-M5 editor.v3)
        sE = Store(run, "render_E")
        items = []
        for m in mut:
            if sE.get(m["key"]):
                continue
            P, P2 = P_of(progs[m["seed_id"]]), P_of(m["ir2"])
            if m["op"] == "M1":
                R = render_substitutions(P2, _edits(m))
                sE.put(m["key"], {**base(m), "variant": "E", "done": True, "tag_ok": True, "text_tagged": tagged_text(R),
                                  "text": R.text, "target": m["answer"]})
                continue
            items.append((m["key"], base(m), P, P2, editor_view(P, P2, _edits(m), m["op"])))
        await _render_rounds(run, c, "E", items, None, "editor", "editor")
        sE = Store(run, "render_E")
        items = []
        for m in mut:
            e = sE.get(m["key"])
            if not e or not e.get("tag_ok"):
                continue
            P, P2 = P_of(progs[m["seed_id"]]), P_of(m["ir2"])
            PL, info = localize(P2, usd=10000)
            items.append((m["key"], base(m), P, PL, {"text_tagged": e["text_tagged"], "card": card_for(m["seed_id"])[1],
                                                     "money": money_map(PL, info), "keep_units": keep_units(PL)}))
        await _render_rounds(run, c, "R1b", items, None, "fluency", "fluency2")
    if "R2b" in c["variants"]:
        items = []
        for m in mut:
            PL, _ = localize(P_of(m["ir2"]), usd=10000)
            items.append((m["key"], base(m), progs[m["seed_id"]], PL.to_dict(),
                          {**ir_view(PL), "card": card_for(m["seed_id"])[1]}))
        await _render_rounds(run, c, "R2b", items, None, "irr", "irrender")
    if "EN" in c["variants"]:
        items = []
        for m in mut:
            items.append((m["key"], base(m), None, m["ir2"], ir_view_en(P_of(m["ir2"]))))
        await _render_rounds(run, c, "EN", items, None, "irr_en", "irrender_en")


async def stage_en_selected(run: Path, sel_path: Path):
    """EN mirror on D5's selection (part 1, Y-): the wave-1 slot of every seeds_minus + reserve_minus seed, rendered
    in English from the SAME mutated IR (ir_render_en.v1), then D1's shared solver (stage_solve EN)."""
    c = cfg(run)
    sel = json.load(open(sel_path, encoding="utf-8"))
    want = set()
    sids = (list(sel["seeds_minus"]) + list(sel.get("reserve_minus") or [])) if "seeds_minus" in sel         else list(sel["needs_EN"]["Y"])                  # part 2 (S+): only seeds with no EN Y item yet
    for sid in sids:
        k = sel["slots"]["Y"].get(sid)
        if k:
            want.add(k)
    # lean loading (laptop RAM rule <= 300 MB): stream the stores, keep only the selected slots / seeds
    mut = {}
    for l in open(run / "mutate.jsonl", encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            if r["key"] in want:
                mut[r["key"]] = r
    need = {m["seed_id"] for m in mut.values()}
    progs = {}
    for l in open(run / "formalize.jsonl", encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            if r["key"] in need and r.get("accepted"):
                progs[r["key"]] = r["ir"]
    items = []
    for k in sorted(want):
        m = mut.get(k)
        if not m or not m.get("ok") or m["seed_id"] not in progs:
            continue
        items.append((k, {x: m[x] for x in ("seed_id", "slot", "op")}, None, m["ir2"], ir_view_en(P_of(m["ir2"]))))
    del mut, progs
    print(f"EN selected slots {len(want)}, with a mutated IR {len(items)}", flush=True)
    if "EN" not in c["variants"]:
        c["variants"] = c["variants"] + ["EN"]
        json.dump(c, open(run / "config.json", "w"), indent=1)
    await _render_rounds(run, c, "EN", items, None, "irr_en", "irrender_en")
    await stage_solve(run, ["EN"])


async def stage_en_retry(run: Path, seeds_file: Path):
    """EN retry (lead 05:35): ONE fresh EN render (attempt_no 2: same IR, same slot, new sample) for the S+ seeds whose
    first EN Y item failed (or is missing) while N-en passed, plus the double fails. Variant EN2; '+'-only: first
    attempt flags are unchanged (R8), and EN2 rows are never Y- eligible."""
    c = cfg(run)
    sids = [l.strip() for l in open(seeds_file, encoding="utf-8") if l.strip()]
    want = {f"{sid}:1" for sid in sids}
    mut = {}
    for l in open(run / "mutate.jsonl", encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            if r["key"] in want and r.get("ok"):
                mut[r["key"]] = r
    items = [(k, {x: m[x] for x in ("seed_id", "slot", "op")}, None, m["ir2"], ir_view_en(P_of(m["ir2"])))
             for k, m in sorted(mut.items())]
    print(f"EN2 retry: {len(want)} seeds, {len(items)} with a mutated IR", flush=True)
    if "EN2" not in c["variants"]:
        c["variants"] = c["variants"] + ["EN2"]
        json.dump(c, open(run / "config.json", "w"), indent=1)
    await _render_rounds(run, c, "EN2", items, None, "irr_en", "irrender_en2", sample_idx=1)
    await stage_solve(run, ["EN2"])


# ============================================================================ cycle (blind Haiku back-formalization)
def _verdict(r, x, a, v):
    P3, errs = (None, [f"api: {x.error_class}"]) if not x.ok else load_ir(f"back:{r['key']}", r["text"], x.text, None)
    ans = None
    if P3 is not None:
        try:
            ans = execute(P3)[P3.query]
        except Exception as e:                     # noqa: BLE001
            errs = errs + [f"exec: {e}"]
    return ans, errs


async def _cycle_gemini(run: Path, v: str, rows: list):
    """Free gemini flash-lite lanes (6 keys x {3.5, 3.1}); each item streams its row to the store, so a watchdog can
    kill and resume. API failures are not verdicts: the row stays not-done and falls to the Haiku pass."""
    st = Store(run, f"cycle_{v}")
    pk = "back_en" if v.startswith("EN") else "back"

    async def one(r):
        model = "g35l" if int(hashlib.sha1(r["key"].encode()).hexdigest(), 16) % 2 else "g31l"
        fb = ""
        for a in range(2):
            x = await call(req(model, render(pk, problem=r["text"]), item=f"gen:G5:{v}:{r['key']}", max_tokens=6144,
                               temperature=0.0, json_mode=True, tag=f"back_{v}_gem", extra_user=fb))
            if not x.ok:
                st.put(r["key"], {"variant": v, "done": False, "api_fail": True, "backend": model, **resp_meta(x)})
                return
            ans, errs = _verdict(r, x, a, v)
            if ans is not None or a == 1:
                st.put(r["key"], {"seed_id": r["seed_id"], "op": r["op"], "variant": v, "done": True,
                                  "executed": ans is not None, "back_answer": _fx(ans), "target": r["target"],
                                  "cycle_ok": ans is not None and ans == Fraction(r["target"]), "n_attempts": a + 1,
                                  "backend": model, **resp_meta(x)})
                return
            fb = _feedback(x.text or "", [e for e in errs if not e.startswith("[gold]")] or ["could not execute"])
    todo = [r for r in rows if not (st.get(r["key"]) or {}).get("done")]
    await asyncio.gather(*(one(r) for r in todo))


async def stage_cycle(run: Path, variants=None, backend: str = "haiku"):
    c = cfg(run)
    for v in variants or c["variants"]:
        rows = [r for r in Store(run, f"render_{v}").rows.values() if r.get("tag_ok")]
        if backend == "gemini":
            await _cycle_gemini(run, v, rows)
        st = Store(run, f"cycle_{v}")
        fb: dict[str, str] = {}
        for a in range(2):
            todo = [r for r in rows if not (st.get(r["key"]) or {}).get("done")]
            if not todo:
                break
            pk = "back_en" if v.startswith("EN") else "back"
            reqs = [req("haiku", render(pk, problem=r["text"]), item=f"gen:G5:{v}:{r['key']}", effort="low",
                        max_tokens=8000, tag=f"back_{v}", extra_user=fb.get(r["key"], "")) for r in todo]
            print(f"cycle_{v}: Haiku pass on {len(todo)} items (attempt {a + 1})", flush=True)
            resps = await call_many(reqs, c["batch"], f"cycle_{v}#{a + 1}")
            for r, x in zip(todo, resps):
                P3, errs = (None, [f"api: {x.error_class}"]) if not x.ok else load_ir(f"back:{r['key']}", r["text"], x.text, None)
                ans = None
                if P3 is not None:
                    try:
                        ans = execute(P3)[P3.query]
                    except Exception as e:     # noqa: BLE001
                        errs = errs + [f"exec: {e}"]
                        P3 = None
                executed = ans is not None
                st.put(r["key"], {"seed_id": r["seed_id"], "op": r["op"], "variant": v, "done": executed or a == 1,
                                  "executed": executed, "back_answer": _fx(ans), "target": r["target"],
                                  "cycle_ok": executed and ans == Fraction(r["target"]), "n_attempts": a + 1,
                                  **resp_meta(x)})
                fb[r["key"]] = _feedback(x.text or "", [e for e in errs if not e.startswith("[gold]")] or ["could not execute"])
        rr = list(st.rows.values())
        print(v, "cycle_ok", sum(r["cycle_ok"] for r in rr), "/", len(rr), f"spent ${spent():.3f}")


# ============================================================================ gates
def stage_gates(run: Path):
    c = cfg(run)
    seeds = {s["seed_id"]: s for s in load(run, "seeds")}
    slots = load(run, "slots")
    st = Store(run, "gates")
    st.rows = {}
    for v in c["variants"]:
        rnd, cyc = render_lite(run, v), Store(run, f"cycle_{v}").rows
        dd = Deduper()
        for sl in slots:
            k = sl["key"]
            r = rnd.get(k)
            if not r or not r.get("tag_ok"):
                continue
            ans = Fraction(r["target"])
            seed_text = seeds[sl["seed_id"]]["en_text" if v.startswith("EN") else "text"]
            lk = leak_regex(r["text"])
            cf = chrf(r["text"], seed_text)
            g = {"leak_ok": not lk, "answer_untagged": answer_untagged(r["text_tagged"], ans), "chrf": round(cf, 4),
                 "chrf_ok": sl["op"] == "M1" or cf <= CHRF_MAX, "cycle_ok": bool((cyc.get(k) or {}).get("cycle_ok"))}
            pre = g["cycle_ok"] and g["leak_ok"] and not g["answer_untagged"] and g["chrf_ok"]
            dup_ok = dd.check_and_add(k, sl["seed_id"], r["text"])[0] if pre else None
            st.rows[f"{v}:{k}"] = {"key": f"{v}:{k}", "slot": k, "variant": v, "op": sl["op"], **g,
                                   "dedup_ok": dup_ok, "survivor": bool(pre and dup_ok)}
    st.rewrite()
    for v in c["variants"]:
        rows = [r for r in st.rows.values() if r["variant"] == v]
        print(v, "tag-passing", len(rows), "survivors", sum(r["survivor"] for r in rows))


# ============================================================================ validity gate (Haiku v3)
def vgate_rule(lab: dict) -> bool:
    """Accept rule chosen in calib.py (see exec/G5_Y_scale.md §2)."""
    return not lab["absurd"] and not lab["unansw"] and lab["teacher"]


async def _judge_round(run, c, todo, model, tag):
    seeds = {s["seed_id"]: s for s in load(run, "seeds")}
    rend = {v: Store(run, f"render_{v}").rows for v in c["variants"]}
    reqs = []
    for g in todo:
        r = rend[g["variant"]][g["slot"]]
        s = seeds[g["slot"].split(":")[0]]
        rp = render("judge3", seed=s["en_text"] if g["variant"].startswith("EN") else s["text"], op=g["op"],
                    op_instruction=OP_INSTR[g["op"]], problem=r["text"])
        reqs.append(req(model, rp, item=f"gen:G5:{g['key']}", effort="low", max_tokens=4000, cache_system=True,
                        tag=tag))
    resps = await call_many(reqs, c["batch"], tag)
    out = []
    for r in resps:
        v = parse_json(r.text) if r.ok else None
        v = v if isinstance(v, dict) else None
        out.append((lab_v3(v) if v else None, r))
    return out


async def stage_vgate(run: Path):
    """The COMMON screen of the + arms, called through D1's frozen ``scale_gen.screen`` (identical code for N+ and
    Y+): flash-lite ``screen`` (validity_lite.v1) AND ``haiku_v3_request`` / ``haiku_v3_parse`` -> ``common_pass``.
    Cheapest-first: only gate survivors whose solver trace matches the program answer are screened; the free
    flash-lite leg runs first and the paid Haiku leg only on lite_ok items (pass needs both, so the result is the
    same as screening everything)."""
    from ..scale_gen import screen as SCR
    from ..scale_gen.solver import agree
    from . import common as CM
    from .export import ans_str
    c = cfg(run)
    seeds = {s["seed_id"]: s for s in load(run, "seeds")}
    gates = Store(run, "gates").rows
    rend = {v: render_lite(run, v) for v in c["variants"]}
    sol = {v: solve_lite(run, v) for v in c["variants"]}
    st = Store(run, "vgate")
    elig = []
    pre = os.environ.get("ARMY_SCREEN_POSTCYCLE") != "1"  # lead 03:40: the (paid, Modal) cycle check runs LAST,
    # only on items that pass gates + trace-match + this screen; export still requires the cycle for Y+
    for g in gates.values():                              # still requires the gate survivor flag for Y+
        if g["variant"].startswith("EN") and os.environ.get("ARMY_SCREEN_EN") != "1":
            continue                                      # EN part 1 is Y- only (D5 selection); screen on request
        if pre:
            if not (g["leak_ok"] and not g["answer_untagged"] and g["chrf_ok"]):
                continue
        elif not g["survivor"]:
            continue
        tr = sol[g["variant"]].get(g["slot"])
        lang = "en" if g["variant"].startswith("EN") else "vi"
        if tr and agree(tr, ans_str(rend[g["variant"]][g["slot"]]["target"]), lang):
            elig.append(g)
    lite = Store(run, "screen_lite")
    hv3_only = c.get("screen") == "hv3_only"     # lead 02:10: Gemini quota gone -> common screen = Haiku-v3 only
    for lang, vs in (("vi", [v for v in c["variants"] if not v.startswith("EN")]),
                     ("en", [v for v in c["variants"] if v.startswith("EN")])):
        items = [(g["key"], rend[g["variant"]][g["slot"]]["text"]) for g in elig if g["variant"] in vs]
        for _ in range(0 if hv3_only else 2):                                   # second pass retries packs that failed
            if items:
                try:
                    await asyncio.wait_for(SCR.screen(lite, lang, items, llm=CM.client()), timeout=1800)
                except Exception as e:                       # noqa: BLE001  free quota gone: lite stays pending
                    print(f"lite screen unavailable ({type(e).__name__}: {str(e)[:160]}); lite_ok stays pending",
                          flush=True)
                    break
    hv = Store(run, "screen_haiku")

    def lite_known(k):
        return bool((lite.get(k) or {}).get("api_ok"))
    # paid Haiku leg: on lite_ok items, and on items whose lite verdict is still pending (quota), never on lite rejects
    todo = [g for g in elig if ((lite.get(g["key"]) or {}).get("lite_ok") or not lite_known(g["key"]))
            and not (hv.get(g["key"]) or {}).get("parsed")]
    reqs = []
    for g in todo:
        lang = "en" if g["variant"].startswith("EN") else "vi"
        s = seeds[g["slot"].split(":")[0]]
        reqs.append(SCR.haiku_v3_request(s["en_text"] if lang == "en" else s["text"], g["op"],
                                         rend[g["variant"]][g["slot"]]["text"], lang=lang, item_key=f"Y:{g['key']}"))
    async for off, resps in call_many_chunks(reqs, c["batch"], "screen_haiku"):
        for g, r in zip(todo[off:off + len(resps)], resps):
            hv.put(g["key"], {**SCR.haiku_v3_parse(r.text if r.ok else None), **resp_meta(r)})
    for g in elig:
        lr, hr = lite.get(g["key"]), hv.get(g["key"])
        if hv3_only:
            st.put(g["key"], {"variant": g["variant"], "slot": g["slot"], "op": g["op"], "trace_match": True,
                              "lite_ok": "not applied", "haiku_pass": (hr or {}).get("hv3_pass"),
                              "screen": "hv3_only", "pass": bool((hr or {}).get("hv3_pass"))})
            continue
        st.put(g["key"], {"variant": g["variant"], "slot": g["slot"], "op": g["op"], "trace_match": True,
                          "lite_ok": (lr or {}).get("lite_ok") if (lr or {}).get("api_ok") else None,
                          "lite_model": (lr or {}).get("model"), "haiku_pass": (hr or {}).get("hv3_pass"),
                          "lite_pending": not lite_known(g["key"]), "pass": SCR.common_pass(lr, hr)})
    rows = list(st.rows.values())
    for v in c["variants"]:
        rr = [r for r in rows if r.get("variant") == v]
        print(v, "eligible (survivor & trace match)", len(rr), "lite ok", sum(bool(r.get("lite_ok")) for r in rr),
              "screen pass (Y+)", sum(bool(r.get("pass")) for r in rr), f"spent ${spent():.3f}", flush=True)
    yplus_report(run)


# ============================================================================ solve (the ONE shared solution writer)
async def stage_solve(run: Path, variants=None):
    """D1's shared writer (scale_gen.solver: Haiku 5.5, effort low, solution_writer[_en].v2, max_tokens 3000), unchanged,
    on every tag-passing item, so arms N and Y build training rows identically."""
    from ..scale_gen.solver import solve_request
    c = cfg(run)
    for v in variants or c["variants"]:
        lang = "en" if v.startswith("EN") else "vi"
        st = Store(run, f"solve_{v}")
        rows = [r for r in Store(run, f"render_{v}").rows.values()
                if r.get("tag_ok") and not (st.get(r["key"]) or {}).get("api_ok")]
        reqs = [solve_request(r["text"], lang, f"Y:{lang}:{r['key']}") for r in rows]
        async for off, resps in call_many_chunks(reqs, c["batch"], f"solve_{v}"):
            for r, x in zip(rows[off:off + len(resps)], resps):
                st.put(r["key"], {"variant": v, "trace": x.text if x.ok else None, **resp_meta(x)})
        rr = list(st.rows.values())
        print(v, "solved", sum(bool(r.get("trace")) for r in rr), "/", len(rr), f"spent ${spent():.3f}")


def main(argv):
    run = Path(argv[1])
    if not run.is_absolute():
        run = ROOT / run
    stage = argv[2]
    if stage == "init":
        init(run, argv[3], int(argv[4]), argv[5].split(","), len(argv) > 6 and argv[6] == "batch")   # pool: n = #slots
    elif stage == "en_retry":
        asyncio.run(stage_en_retry(run, Path(argv[3])))
    elif stage == "en_selected":
        asyncio.run(stage_en_selected(run, Path(argv[3])))
    elif stage == "extend":
        extend(run, int(argv[3]), int(argv[4]) if len(argv) > 4 else None)
    elif stage == "status":
        yplus_report(run)
    elif stage == "formalize":
        asyncio.run(stage_formalize(run))
    elif stage == "mutate":
        asyncio.run(stage_mutate(run))
    elif stage == "render":
        asyncio.run(stage_render(run))
    elif stage == "cycle":
        vs = argv[3].split(",") if len(argv) > 3 and argv[3] != "-" else None
        asyncio.run(stage_cycle(run, vs, argv[4] if len(argv) > 4 else "haiku"))
    elif stage == "gates":
        stage_gates(run)
    elif stage == "solve":
        asyncio.run(stage_solve(run, argv[3].split(",") if len(argv) > 3 else None))
    elif stage == "vgate":
        asyncio.run(stage_vgate(run))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
