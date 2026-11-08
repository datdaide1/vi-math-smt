"""Build the N-arm outputs for one language: pool.jsonl (EVERY generated item with its labels; D5 selects the final
training sets from it), provisional train_10k_{plus,minus}.jsonl, SHA256SUMS, manifest.json.

Definitions (T1 §2.4, 01 §3.5, lead decisions 2026-11-07; deviations in exec/D1_N_scale.md):
* training row = {question, solution, answer}: ``solution`` = clean_trace(trace of the ONE shared solution writer,
  ``scale_gen.solver``), ``answer`` = that trace's own last \\boxed{} (same construction as arm Y).
* ``minus_eligible`` (N-, unverified) = generator parse/format checks (``checks.parse_gen``) ∧ the trace has a
  \\boxed{} ∧ decontamination gate (X5 G4: 13-gram exact/no-diacritics/templated; bge-m3 cos >= .85 same-language /
  >= .80 cross-language vs data/eval v1.1 + Vi-ExamMath; MinHash near-dup within the arm). No agreement filter.
* ``verified`` (N+) = minus_eligible ∧ trace answer == generator answer (X1 grader) ∧ the COMMON "+" screen
  (``scale_gen.screen``: Haiku-v3 ``hv3_pass``). Items never reached by the (early-stopped) Haiku-v3 screen have
  ``verified = None`` (unknown), not False.
* DEVIATION (lead, 2026-11-08 02:05): the flash-lite part of the common screen is NOT APPLIED: the Gemini free
  quota (500 requests/day/project/model) was exhausted on both flash-lite models across all 6 keys until ~15:00.
  Lite verdicts that exist (first items in rank order) are kept in ``flags.lite`` as information only.
* provisional subsamples: slots_v1 rank order (breadth over seeds first); minus = first K minus_eligible items,
  plus = first K verified items, K = min(k, |plus|, |minus|). D5 owns the final selection (data/train/final/).
"""
from __future__ import annotations

import json
import time
from collections import Counter

from . import checks as C
from .common import (AUDITOR, GEN_CFG, GENERATOR, LITE_MODELS, OUT, PROMPT_ID, PROMPT_VER, REG, SLOTS, SOLVER, WORK,
                     Store, sha256_file, write_jsonl)

K = 10_000


def prompt_info() -> dict:
    out = {}
    ids = [(PROMPT_ID[l][r], PROMPT_VER[r]) for l in ("vi", "en") for r in ("gen", "solve")]
    ids += [("validity_lite", PROMPT_VER["lite"]), ("validity_judge", 3), ("validity_judge_en", 3),
            ("validity_judge", PROMPT_VER["audit"])]
    for pid, v in ids:
        p = REG.get(pid, v)
        out[f"{p.id}.v{p.version}"] = {"sha256": p.sha256, "path": "prompts/" + p.path.replace("\\", "/").split("/")[-1]}
    return out


def build(lang: str, cfg: str, slots: list[dict], *, require_embed: bool = True, k: int = K,
          seeds: dict | None = None, retry_slots: set | None = None) -> dict:
    """``retry_slots``: EN-retry items (lead 05:35): attempt_no=2, "+"-only (never in minus, never in R8 rates)."""
    retry_slots = retry_slots or set()
    gen, solve, lite, hv3, dec = (Store(WORK / lang / f"{s}.jsonl") for s in ("gen", "solve", "lite", "hv3", "decontam"))
    counts, rej = Counter(), Counter()
    pool = []
    gp = REG.get(PROMPT_ID[lang]["gen"], PROMPT_VER["gen"])
    sp = REG.get(PROMPT_ID[lang]["solve"], PROMPT_VER["solve"])
    for sl in sorted(slots, key=lambda s: s["rank"]):
        key = f"{lang}|{cfg}|{sl['key']}"
        g = gen.get(key)
        if g is None:
            continue
        counts["generated"] += 1
        s = solve.get(key) or {}
        lv = lite.get(key) or {}
        hv = hv3.get(key) or {}
        d = dec.get(key) or {}
        fmt_ok = bool(g.get("fmt_ok"))
        trace_ok = fmt_ok and bool(s.get("api_ok") and s.get("boxed"))
        dec_checked = bool(d) and (d.get("embedded") or not require_embed)
        dec_ok = trace_ok and dec_checked and bool(d.get("pass"))
        agree = bool(s.get("agree")) if s.get("api_ok") else None
        lite_ok = bool(lv.get("lite_ok")) if lv.get("api_ok") else None
        hv3_pass = bool(hv.get("hv3_pass")) if hv.get("api_ok") else None
        minus_ok = dec_ok
        if not minus_ok:
            verified = False
        elif agree is False or hv3_pass is False:
            verified = False
        elif agree and hv3_pass:
            verified = True
        else:
            verified = None                      # a "+" gate not reached (early-stopped screen)
        # ---- counts per gate (manipulation check, E1 R8)
        counts["format_ok"] += fmt_ok
        if not fmt_ok:
            for f, v in (g.get("fmt_flags") or {}).items():
                rej[f"format:{f}"] += bool(v)
        counts["trace_ok"] += trace_ok
        if trace_ok and not dec_checked:
            counts["decontam_unchecked"] += 1
        if trace_ok and dec_checked and not d.get("pass"):
            for f in ("g4_ngram", "g4_cos", "g5_dup"):
                rej[f"decontam:{f}"] += bool(d.get(f))
        counts["minus_eligible"] += minus_ok
        if minus_ok:
            counts["agree"] += bool(agree)
            counts["lite_judged"] += lite_ok is not None
            counts["lite_ok"] += bool(lite_ok)
            counts["hv3_judged"] += hv3_pass is not None
            counts["hv3_pass"] += bool(hv3_pass)
            counts["verified"] += verified is True
            counts["verified_unknown"] += verified is None
        hl = hv.get("labels") or {}
        rec = {
            "id": f"N/{lang}/{sl['key']}", "lang": lang, "arm": "N", "slot_id": sl["key"], "rank": sl["rank"],
            "seed_id": sl["seed_id"], "source": sl.get("source"), "dataset": sl.get("dataset"), "op": sl["op"],
            "card": sl["card"], "partner": sl.get("partner"),
            "seed_screens": ({"solvecheck": (seeds[sl["seed_id"]].get("screens") or {}).get("solvecheck"),
                              "seed_provenance": seeds[sl["seed_id"]].get("provenance"),
                              "pending2": (seeds[sl["seed_id"]].get("screens") or {}).get("solvecheck") == "PENDING2"}
                             if seeds and sl["seed_id"] in seeds else None),
            "question": g.get("problem"),
            "solution": C.clean_trace(s["trace"]) if trace_ok else None,
            "answer": s.get("boxed") if trace_ok else None,
            "generator_answer": g.get("answer"),
            "minus_eligible": minus_ok, "verified": verified,
            "attempt_no": 2 if sl["key"] in retry_slots else 1, "plus_only": sl["key"] in retry_slots,
            "flags": {
                "format_ok": fmt_ok, "format": g.get("fmt_flags"), "gen_finish": g.get("finish"),
                "trace_ok": trace_ok, "solve_agree": agree,
                "lite_applied": False, "lite_ok": lite_ok, "lite": ({x: lv.get(x) for x in ("absurd", "unansw", "multi", "leakj", "natural")}
                                             | {"model": lv.get("model")}) if lv.get("api_ok") else None,
                "hv3_pass": hv3_pass, "hv3": ({x: hl.get(x) for x in ("absurd", "unansw", "teacher", "valid", "new",
                                                                        "tail", "leakj", "natural")}
                                              if hv.get("api_ok") else None),
                "decontam_pass": bool(d.get("pass")) if dec_checked else None,
                "decontam": {x: d.get(x) for x in ("g4_ngram", "g4_cos", "g5_dup", "cos_same", "cos_x")} if d else None,
            },
            "provenance": {"generator": GENERATOR, "gen_prompt": f"{gp.id}.v{gp.version}", "gen_prompt_sha256": gp.sha256,
                           "gen_cfg": cfg, "gen_params": dict(GEN_CFG[cfg]), "gen_request_hash": g.get("request_hash"),
                           "solver": SOLVER, "solver_prompt": f"{sp.id}.v{sp.version}", "solver_prompt_sha256": sp.sha256,
                           "screen": "scale_gen.screen: validity_judge[_en].v3 Haiku-v3 only (flash-lite not applied: "
                                     "Gemini quota, lead 2026-11-08 02:05)",
                           "api_log": "logs/api/D1/", "workstream": "D1"},
        }
        pool.append(rec)
    minus_rows = [r for r in pool if r["minus_eligible"] and not r["plus_only"]]
    plus_rows = [r for r in pool if r["minus_eligible"] and r["verified"] is True]
    kk = min(k, len(plus_rows), len(minus_rows))
    if not plus_rows:          # e.g. EN part 1 (S- seeds only, no "+" screen): minus file stands alone
        kk_minus = min(k, len(minus_rows))
    else:
        kk_minus = kk
    train = lambda r: {x: r[x] for x in ("id", "question", "solution", "answer", "slot_id", "seed_id", "op", "lang",  # noqa
                                         "verified", "provenance")}
    d = OUT / lang
    d.mkdir(parents=True, exist_ok=True)
    files = {"pool.jsonl": pool, "train_10k_plus.jsonl": [train(r) for r in plus_rows[:kk]],
             "train_10k_minus.jsonl": [train(r) for r in minus_rows[:kk_minus]]}
    for fn, rows in files.items():
        write_jsonl(d / fn, rows)
    sums = {fn: sha256_file(d / fn) for fn in files}
    (d / "SHA256SUMS").write_text("".join(f"{h}  {fn}\n" for fn, h in sums.items()), encoding="utf-8", newline="\n")
    man = {
        "arm": "N", "lang": lang, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "workstream": "D1",
        "status": "pool final for the slots listed; train_10k_* PROVISIONAL (D5 selects data/train/final/)",
        "deviations": ["common '+' screen = Haiku-v3 only; flash-lite NOT APPLIED (Gemini free quota exhausted on both "
                       "flash-lite models x 6 keys until ~15:00 +07; lead decision 2026-11-08 02:05). Existing lite "
                       "verdicts kept as information in flags.lite."],
        "definitions": __doc__, "gen_cfg": cfg, "gen_params": GEN_CFG[cfg],
        "models": {"generator": GENERATOR, "solver": SOLVER, "screen_lite": list(LITE_MODELS),
                   "screen_haiku_v3": "claude-haiku-5-5", "auditor": AUDITOR},
        "prompts": prompt_info(), "slots_file": "data/train/slots_v1.jsonl",
        "slots_sha256": sha256_file(SLOTS) if SLOTS.exists() else None, "n_slots_requested": len(slots),
        "counts": dict(counts), "rejections": dict(rej), "subsample_k": kk, "files": sums,
        "n_rows": {fn: len(r) for fn, r in files.items()},
        "pool_v1_sha256": sha256_file(SLOTS.parents[1] / "seeds" / "pool_v1.jsonl")
        if (SLOTS.parents[1] / "seeds" / "pool_v1.jsonl").exists() else None,
        "seed_solvecheck_in_pool": dict(Counter((r.get("seed_screens") or {}).get("solvecheck") for r in pool)),
        "n_seeds": {"pool": len({r["seed_id"] for r in pool}), "minus_eligible": len({r["seed_id"] for r in minus_rows}),
                    "verified": len({r["seed_id"] for r in plus_rows})},
        "op_mix": {"pool": dict(Counter(r["op"] for r in pool)), "verified": dict(Counter(r["op"] for r in plus_rows))},
    }
    (d / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    return man
