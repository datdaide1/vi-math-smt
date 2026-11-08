"""Export arm-Y training pools in the N layout (scale_gen/finalize.py): data/train/Y/<lang>/{pool.jsonl,
train_10k_plus.jsonl, train_10k_minus.jsonl, SHA256SUMS, manifest.json}.

    python -m vi_math_verified_aug.arm_y.export <run_dir> <variant> <lang> items   # -> <run>/decontam_items_<lang>.json
    modal run -m vi_math_verified_aug.arm_y.decontam_modal --items ... --out <run>/decontam_<lang>.json
    python -m vi_math_verified_aug.arm_y.export <run_dir> <variant> <lang> write [out_dir]

Training row = {question, solution, answer} from the ONE shared solution writer (scale_gen.solver.training_row:
solution = clean_trace(trace), answer = the trace's own last \\boxed{}), identical to arm N.
* **Y- (unverified)** = parse + tag check (every given tagged with its value, no stray numeral) + leak regex +
  answer value not written outside the tags + a usable solver trace (has \\boxed{}) + decontamination
  (13-gram overlap = 0 and bge-m3 cos < 0.85 vs the eval targets of the same language). Mirrors N- (no extra check).
* **Y+ (verified)** = Y- AND blind cycle check (back-formalized answer == program answer, exact) AND chrF <= 0.90
  vs the seed (M2-M5) AND 5-gram dedup AND trace answer == program answer (X1 grader, scale_gen.checks.agree)
  AND the COMMON screen shared with N+ (arm_y.screen: D1 flash-lite lite_ok AND Haiku-v3 not absurd / not
  unanswerable / teacher-acceptable).
* Subsamples: priority sha1("G5sel|<slot key>") (shared across languages); K = min(10,000, |Y+|, |Y-|);
  minus = first K pool rows in priority order, plus = first K Y+ rows (size-matched, E1 R2).
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

from .common import ROOT, Store
from .pipeline import cfg, load

OUT = ROOT / "data" / "train" / "Y"
TAU = 0.85
K = 10_000


def ans_str(s: str) -> str:
    f = Fraction(s)
    if f.denominator == 1:
        return str(f.numerator)
    d = f.denominator
    for p in (2, 5):
        while d % p == 0:
            d //= p
    return f"{float(f):.10f}".rstrip("0").rstrip(".") if d == 1 else f"{f.numerator}/{f.denominator}"


def _grams(text: str, n: int = 13) -> set:
    w = re.findall(r"\w+", (text or "").lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def _prio(slot: str) -> str:
    return hashlib.sha1(f"G5sel|{slot}".encode()).hexdigest()


def items_file(run: Path, variant: str, lang: str):
    rows = [r for r in Store(run, f"render_{variant}").rows.values() if r.get("tag_ok")]
    p = run / f"decontam_items_{lang}.json"
    json.dump([{"id": r["key"], "text": r["text"], "lang": lang} for r in rows], open(p, "w", encoding="utf-8"),
              ensure_ascii=False)
    print("items", len(rows), "->", p)


def build_rows(run: Path, variant: str, lang: str, dec: dict | None, grams: set | None) -> tuple[list, Counter]:
    from ..scale_gen.solver import SOLVER_SPEC, training_row
    c = cfg(run)
    rend = {}                                                  # streamed, IR payload dropped (RAM rule)
    for l in open(run / f"render_{variant}.jsonl", encoding="utf-8"):
        if l.strip():
            x = json.loads(l)
            x["n_ops"] = len((x.pop("ir_target", None) or {}).get("ops") or [])
            x.pop("attempts", None)
            rend[x["key"]] = x
    gates = Store(run, "gates").rows
    vg = Store(run, "vgate").rows
    from .pipeline import solve_lite
    sol = {k: {"trace": t} for k, t in solve_lite(run, variant).items()}
    cyc = Store(run, f"cycle_{variant}").rows                 # primary (Qwen3-32B-AWQ) verdicts
    cyc2 = Store(run, f"cycle_glm_{variant}").rows            # secondary label (NIM glm-5.3-flash / Gemini)
    scr = Store(run, "screen_haiku").rows                     # common screen (Haiku-v3 leg; lite not applied)
    rej: Counter = Counter()
    pool = []
    seeds = {x["seed_id"]: x for x in load(run, "seeds")}
    pv1 = {}
    pp = ROOT / "data" / "seeds" / "pool_v1.jsonl"
    if pp.exists():                                    # final rows come from pool_v1 seeds only (lead 02:05)
        for l in open(pp, encoding="utf-8"):
            if l.strip():
                x = json.loads(l)
                pv1[x["seed_id"]] = x
    for k, r in sorted(rend.items()):
        if pv1 and r["seed_id"] not in pv1:
            rej["not in pool_v1"] += 1
            continue
        if not r.get("tag_ok"):
            rej["tag"] += 1
            continue
        g = gates.get(f"{variant}:{k}", {})
        if not g.get("leak_ok", False) or g.get("answer_untagged", True):
            rej["leak/answer-in-text"] += 1
            continue
        if dec is not None:
            d = dec.get(k)
            if d is None or d["cos_max"] >= TAU or (_grams(r["text"]) & grams):
                rej["decontam"] += 1
                continue
        else:
            d = {}
        tr = (sol.get(k) or {}).get("trace")
        target = ans_str(r["target"])
        row = training_row(r["text"], tr, lang=lang, verified_answer=target) if tr else None
        if row is None:
            rej["no boxed trace"] += 1
            continue
        v = vg.get(f"{variant}:{k}", {})
        hv = scr.get(f"{variant}:{k}") or {}
        cy = cyc.get(k) or {}
        tm = bool(row["trace_matches_verified"])
        screen = hv.get("hv3_pass") if hv.get("parsed") else None          # None = not screened yet
        cycle = cy.get("cycle_ok") if cy.get("done") and cy.get("cycle_model") else None   # None = no Qwen verdict
        checks = {"cycle_ok": cycle, "cycle_model": cy.get("cycle_model"), "chrf_ok": bool(g.get("chrf_ok")),
                  "trace_matches_program": tm, "screen_lite": "not applied", "screen_haiku": screen}
        # plus: True = every check passed; False = some check failed; None = pending (never a placeholder False)
        if not tm or not g.get("chrf_ok", False) or screen is False or cycle is False:
            plus = False
        elif screen is None or cycle is None:
            plus = None
        else:
            plus = True
        row.pop("trace_matches_verified", None)
        pool.append({"id": f"Y/{lang}/{k}" + ("#a2" if variant == "EN2" else ""), "lang": lang, "arm": "Y", "plus": plus, **row,
                     "program_answer": target, "slot": k, "slot_j": r.get("slot"), "seed_id": r["seed_id"],
                     "op": r["op"], "ir_ops": r.get("n_ops"),
                     "source": "vi_gsm8k_train", "verified": plus, "wave": r.get("slot"),
                     "attempt_no": 2 if variant == "EN2" else 1,
                     # Y- = wave-1 first-attempt items only (lead 03:20); EN2 retries are '+'-only (lead 05:35)
                     "minus_eligible": r.get("slot") == 1 and variant != "EN2",
                     "cycle_model": cy.get("cycle_model"),
                     "cycle_glm_ok": (cyc2.get(k) or {}).get("cycle_ok") if (cyc2.get(k) or {}).get("done") else None,
                     "seed_text_changed_in_pool_v1": bool(pv1) and (pv1[r["seed_id"]]["problem"] !=
                                                                    seeds[r["seed_id"]]["text"]),
                     "gates": {**checks, "chrf": g.get("chrf"), "cos_max": d.get("cos_max"), "cos_tid": d.get("tid")},
                     "provenance": {"arm": "Y", "workstream": "G5", "renderer": variant,
                                    "renderer_model": "claude-haiku-5-5",
                                    "renderer_prompt": "ir_render_en.v1" if lang == "en" else
                                    ("ir_render.v1" if variant == "R2b" else "editor.v3+fluency.v2"),
                                    "formalizer": "claude-haiku-5-5 formalizer.v2", "operators": "X4 + G5 v3 constraints",
                                    "pi": c["pi"], "cycle": "back_formalizer[_en].v1 (gemini flash-lite / Haiku fallback)",
                                    "screen": "arm_y.screen (common to N+/Y+): validity_lite.v1 flash-lite lite_ok AND validity_judge.v3 Haiku 5.5", "solver": SOLVER_SPEC,
                                    "seed_source": c["source"], "api_log": "logs/api/G5/",
                                    "created": time.strftime("%Y-%m-%d")}})
    return pool, rej


def write(run: Path, variant: str, lang: str, out: Path | None = None, decontam: bool = True,
          extra: tuple = ()):
    dec = grams = None
    if decontam:
        dec = {d["id"]: d for d in json.load(open(run / f"decontam_{lang}.json", encoding="utf-8"))}
        tg = json.load(open(ROOT / "data/seeds/_work/embed/inputs.json", encoding="utf-8"))["targets"]
        grams = set()
        for t in tg:
            if t["lang"] == lang:
                grams |= _grams(t["text"])
    pool, rej = build_rows(run, variant, lang, dec, grams)
    for xv in extra:                                       # e.g. EN2 retries ('+'-only), own decontam file
        xdec = {d["id"]: d for d in json.load(open(run / f"decontam_{xv.lower()}.json", encoding="utf-8"))}             if decontam else None
        xp, xr = build_rows(run, xv, lang, xdec, grams)
        pool += xp
        rej.update({f"{xv}:{k}": v for k, v in xr.items()})
    by = sorted(pool, key=lambda r: _prio(r["slot"]))
    kk = min(K, len(pool), sum(r["plus"] is True for r in pool))
    minus = [r for r in by if r["minus_eligible"]][:kk]
    plus = [r for r in by if r["plus"] is True][:kk]
    d = (out or OUT) / lang
    d.mkdir(parents=True, exist_ok=True)
    files = {"pool.jsonl": pool, "train_10k_plus.jsonl": plus, "train_10k_minus.jsonl": minus}
    sums = {}
    for fn, rows in files.items():
        with open(d / fn, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        sums[fn] = hashlib.sha256((d / fn).read_bytes()).hexdigest()
    (d / "SHA256SUMS").write_text("".join(f"{h}  {fn}\n" for fn, h in sums.items()), encoding="utf-8", newline="\n")
    man = {"arm": "Y", "lang": lang, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "workstream": "G5",
           "run_dir": str(run.relative_to(ROOT)).replace("\\", "/"), "renderer": variant, "config": cfg(run),
           "definitions": __doc__, "n_pool": len(pool), "n_plus_eligible": sum(r["plus"] is True for r in pool),
           "n_plus_pending": sum(r["plus"] is None for r in pool), "n_plus_false": sum(r["plus"] is False for r in pool),
           "subsample_k": kk, "train_10k_shortfall": K - kk, "rejections_before_pool": dict(rej),
           "plus_failures": dict(Counter(k for r in pool if r["plus"] is False for k, v in r["gates"].items()
                                         if v is False)),
           "files": sums, "op_mix": {"pool": dict(Counter(r["op"] for r in pool)),
                                     "plus": dict(Counter(r["op"] for r in plus)),
                                     "minus": dict(Counter(r["op"] for r in minus))}}
    json.dump(man, open(d / "manifest.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps({k: man[k] for k in ("n_pool", "n_plus_eligible", "subsample_k", "rejections_before_pool")}))
    return man


if __name__ == "__main__":
    run = Path(sys.argv[1])
    run = run if run.is_absolute() else ROOT / run
    if sys.argv[4] == "items":
        items_file(run, sys.argv[2], sys.argv[3])
    else:
        write(run, sys.argv[2], sys.argv[3], Path(sys.argv[5]) if len(sys.argv) > 5 else None)
