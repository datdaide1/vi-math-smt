"""G3 report tables (markdown) from data/pilot/symbolic/ and logs/api/G3.

    python -m vi_math_verified_aug.pilot_symbolic.report  -> data/pilot/symbolic/<run>/report_tables.md, metrics.json
"""
from __future__ import annotations

import json
import os
import statistics as stt
from collections import Counter, defaultdict
from datetime import datetime

from ..pilot_gen.util import fmt_ci, kappa
from . import seeds as SD
from .common import OUT, ROOT, run_dir

OPS = ["M1", "M2", "M3", "M4", "M5", "M6"]


def L(path) -> dict:
    d = {}
    if path.exists():
        for l in open(path, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                d[r["key"]] = r
    return d


def ts(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def main():
    R = run_dir()
    out: list[str] = []
    P = out.append
    slots = SD.load_slots()
    form = L(OUT / "formalize.jsonl")
    mut, ren, cyc, gat = L(R / "mutate.jsonl"), L(R / "render.jsonl"), L(R / "cycle.jsonl"), L(R / "gates.jsonl")
    jo, so, j2 = L(R / "judge_opus.jsonl"), L(R / "solve_opus.jsonl"), L(R / "judge_second.jsonl")
    js = L(R / "judge_opus_seeds.jsonl")
    metrics: dict = {}

    # ---------------------------------------------------------------- 1. formalization
    acc = [r for r in form.values() if r["accepted"]]
    p1 = sum(r["n_attempts"] == 1 for r in acc)
    reasons = Counter()
    for r in form.values():
        for a in r["attempts"]:
            for e in a["errors"]:
                reasons[e.split("]")[0].strip("[").split(":")[0]] += 1
    P("### Stage 1: formalization (Haiku 5.5, formalizer.v2, ≤ 3 attempts with validator feedback)\n")
    P(f"- accepted: {fmt_ci(len(acc), len(form))}; pass@1 {fmt_ci(p1, len(form))}")
    P(f"- violations seen across all {sum(r['n_attempts'] for r in form.values())} attempts (code: count): "
      + ", ".join(f"`{k}` {v}" for k, v in reasons.most_common()))
    P(f"- rejected seeds: " + "; ".join(f"idx {r['index']}: {r['attempts'][-1]['errors'][0][:110]}"
                                        for r in form.values() if not r["accepted"]) + "\n")
    metrics["formalize"] = {"accepted": len(acc), "n": len(form), "pass1": p1, "violations": dict(reasons)}

    # ---------------------------------------------------------------- 2. stage-by-stage yield
    stages = ["slots", "IR", "mutated", "tag ok", "cycle ok", "no leak/answer", "chrF ≤ .90", "cos < .85",
              "dedup", "survivor"]
    tab = {op: Counter() for op in OPS + ["all"]}
    for sl in slots:
        k, op = sl["key"], sl["op"]
        flags = {"slots": True, "IR": sl["seed_id"] in {r["seed_id"] for r in acc}}
        flags["mutated"] = flags["IR"] and mut.get(k, {}).get("ok", False)
        flags["tag ok"] = flags["mutated"] and ren.get(k, {}).get("tag_ok", False)
        g = gat.get(k, {})
        flags["cycle ok"] = flags["tag ok"] and g.get("cycle_ok", False)
        flags["no leak/answer"] = flags["cycle ok"] and g.get("leak_ok", False) and not g.get("answer_in_text", True)
        flags["chrF ≤ .90"] = flags["no leak/answer"] and g.get("chrf_ok", False)
        flags["cos < .85"] = flags["chrF ≤ .90"] and g.get("decontam_ok", False) and g.get("xseed_dup_ok", False)
        flags["dedup"] = flags["cos < .85"] and bool(g.get("dedup_ok"))
        flags["survivor"] = bool(g.get("survivor"))
        assert flags["survivor"] == flags["dedup"], k
        for s in stages:
            tab[op][s] += flags[s]
            tab["all"][s] += flags[s]
    P("### Stage-by-stage yield per operator (cumulative; count and % of scheduled slots)\n")
    P("| op | " + " | ".join(stages) + " |")
    P("|---|" + "---|" * len(stages))
    for op in OPS + ["all"]:
        n = tab[op]["slots"]
        P(f"| {op} | " + " | ".join(f"{tab[op][s]} ({100 * tab[op][s] / n:.0f}%)" if s != "slots" else str(n)
                                     for s in stages) + " |")
    metrics["yield"] = {op: dict(tab[op]) for op in tab}

    # independent (non-cumulative) gate pass rates on tag-passing renders
    P("\n**Each gate on its own, over all tag-passing renders** (not cumulative):\n")
    P("| op | n | cycle ok | regex leak | answer value in text | chrF ≤ .90 | cos_eval < .85 | cos_xseed < .85 | "
      "median chrF | median cos(seed) |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    for op in OPS + ["all"]:
        rows = [g for g in gat.values() if op == "all" or g["op"] == op]
        n = len(rows)
        P(f"| {op} | {n} | {fmt_ci(sum(g['cycle_ok'] for g in rows), n)} | {sum(bool(g['leak_regex']) for g in rows)} | "
          f"{sum(g['answer_in_text'] for g in rows)} | {fmt_ci(sum(g['chrf_ok'] for g in rows), n)} | "
          f"{sum(g['decontam_ok'] for g in rows)}/{n} | {sum(g['xseed_dup_ok'] for g in rows)}/{n} | "
          f"{stt.median(g['chrf'] for g in rows):.2f} | {stt.median(g['cos_seed'] for g in rows):.3f} |")

    # mutate: reasons and gate-call histogram
    P("\n**Mutation (X4 operators, with the G3 fixes):** NotApplicable reasons and X4 `gate` verdicts over all "
      "candidate programs the operators tried (instrumented wrapper):\n")
    P("| op | applied | NotApplicable (reason: n) | gate verdicts over tried candidates | semantic tail (applied) |")
    P("|---|---|---|---|---|")
    for op in OPS:
        rows = [m for m in mut.values() if m["op"] == op]
        okr = [m for m in rows if m["ok"]]
        na = Counter(m["reason"] for m in rows if not m["ok"])
        gc = Counter()
        for m in rows:
            gc.update(m.get("gate_calls") or {})
        st = sum(m["recheck"]["semantic_tail"] is True for m in okr)
        P(f"| {op} | {len(okr)}/{len(rows)} | " + "; ".join(f"{k}: {v}" for k, v in na.items()) + " | "
          + ", ".join(f"{k} {v}" for k, v in gc.most_common()) + f" | {st}/{len(okr)} |")
    rech = Counter((m["recheck"]["gate"] is None, m["recheck"]["tail"], m["recheck"]["identity"])
                   for m in mut.values() if m["ok"])
    P(f"\nIndependent re-check of every returned program (gate passes, structural tail, identity): {dict(rech)}")

    # render
    P("\n**Rendering:** M1 deterministic (`render_substitutions`, re-validated); M2-M6 Haiku editor.v3, ≤ 2 attempts "
      "(retry gets only the tag-check issues):\n")
    P("| op | renders | tag pass@1 | tag pass ≤2 | editor warning raised | M1 re-validation errors |")
    P("|---|---|---|---|---|---|")
    for op in OPS:
        rows = [r for r in ren.values() if r["op"] == op]
        if not rows:
            continue
        p1 = sum(r["tag_ok"] and r["tag_attempts"] <= 1 for r in rows)
        P(f"| {op} | {len(rows)} | {fmt_ci(p1, len(rows))} | {fmt_ci(sum(r['tag_ok'] for r in rows), len(rows))} | "
          f"{sum(bool(r.get('warning')) for r in rows)} | {sum(bool(r.get('revalidate_errors')) for r in rows)} |")

    # cycle
    P("\n**Cycle check** (gemini-3.1-flash-lite, back_formalizer.v1, blind; ≤ 2 attempts, retry only on "
      "parse/IR/exec errors):\n")
    P("| op | n | executed | answer == program answer | same canonical DAG (among executed) |")
    P("|---|---|---|---|---|")
    for op in OPS + ["all"]:
        rows = [c for c in cyc.values() if op == "all" or c["op"] == op]
        ex = [c for c in rows if c["executed"]]
        P(f"| {op} | {len(rows)} | {len(ex)} | {fmt_ci(sum(c['cycle_ok'] for c in rows), len(rows))} | "
          f"{sum(bool(c.get('struct_same')) for c in ex)}/{len(ex)} |")

    # ---------------------------------------------------------------- 3. quality gate (Opus)
    def lab(k):
        return (jo.get(k) or {}).get("labels")

    def qrow(name, keys, newkeys=None):
        keys = [k for k in keys if lab(k)]
        n = len(keys)
        nk = [k for k in (newkeys if newkeys is not None else keys) if lab(k)]
        c = Counter()
        for k in keys:
            lb = lab(k)
            for f in ("valid", "teacher", "absurd", "tail", "neardupj", "leakj", "fluency", "unansw", "op"):
                c[f] += bool(lb[f])
            c["agree"] += bool(so.get(k, {}).get("agree"))
        newc = sum(bool(lab(k)["new"]) for k in nk)
        nat = stt.mean(lab(k)["natural"] for k in keys) if keys else float("nan")
        return (f"| {name} | {n} | {fmt_ci(c['teacher'], n)} | {fmt_ci(newc, len(nk))} | {fmt_ci(c['leakj'], n)} | "
                f"{fmt_ci(c['absurd'], n)} | {fmt_ci(c['tail'], n)} | {fmt_ci(c['neardupj'], n)} | "
                f"{fmt_ci(c['valid'], n)} | {fmt_ci(c['unansw'], n)} | {fmt_ci(c['fluency'], n)} | {nat:.2f} | "
                f"{fmt_ci(c['agree'], n)} |"), {"n": n, **c, "new": newc, "n_new": len(nk), "natural": nat}

    hdr = ("| set | n | teacher-would-write (gate ≥ 70%) | new, M2-M6 (≥ 60%) | leak, judge (0%) | absurd (≤ 3%) | "
           "tail | near-dup (judge) | valid | unanswerable | fluency issue | natural (mean) | Opus blind solve = "
           "program answer |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    surv = [k for k, g in gat.items() if g["survivor"]]
    P("\n### Quality gate on the survivors (Opus 5.5, validity_judge.v2 = G1 judge of record)\n")
    P(hdr)
    rr, m_all = qrow("**survivors, all ops**", surv, [k for k in surv if gat[k]["op"] != "M1"])
    P(rr)
    metrics["gate_survivors"] = m_all
    for op in OPS:
        ks = [k for k in surv if gat[k]["op"] == op]
        rr, mm = qrow(op, ks, ks if op != "M1" else [])
        P(rr)
        metrics.setdefault("gate_by_op", {})[op] = mm
    P("\n### Same judge on other sets (what the gates remove; seed baseline)\n")
    P(hdr)
    allk = list(gat)
    for name, ks in [("all tag-passing renders", allk),
                     ("cycle check passed", [k for k in allk if gat[k]["cycle_ok"]]),
                     ("cycle check FAILED", [k for k in allk if not gat[k]["cycle_ok"]]),
                     ("chrF > .90 (near-dup gate fired)", [k for k in allk if not gat[k]["chrf_ok"]]),
                     ("answer value in text", [k for k in allk if gat[k]["answer_in_text"]]),
                     ("M2-M6 tag-passing (cf. G1 D2)", [k for k in allk if gat[k]["op"] != "M1"])]:
        rr, mm = qrow(name, ks, [k for k in ks if gat[k]["op"] != "M1"])
        P(rr)
        metrics.setdefault("gate_other", {})[name] = mm
    # seeds baseline (labels only; no solve)
    sk = [k for k, r in js.items() if r.get("labels")]
    c = Counter()
    for k in sk:
        for f in ("valid", "teacher", "absurd", "fluency", "unansw", "leakj"):
            c[f] += bool(js[k]["labels"][f])
    nat = stt.mean(js[k]["labels"]["natural"] for k in sk) if sk else float("nan")
    P(f"| the 60 seed texts themselves (op M0) | {len(sk)} | {fmt_ci(c['teacher'], len(sk))} | – | "
      f"{fmt_ci(c['leakj'], len(sk))} | {fmt_ci(c['absurd'], len(sk))} | – | – | {fmt_ci(c['valid'], len(sk))} | "
      f"{fmt_ci(c['unansw'], len(sk))} | {fmt_ci(c['fluency'], len(sk))} | {nat:.2f} | – |")
    metrics["seed_baseline"] = {"n": len(sk), **c, "natural": nat}

    # editor warning as a predictor
    wk = [k for k in allk if gat[k]["op"] != "M1"]
    w1 = [k for k in wk if ren[k].get("warning")]
    w0 = [k for k in wk if not ren[k].get("warning")]
    P("\n**Editor `warning` as a free filter (M2-M6 tag-passing):**\n")
    P(hdr)
    for name, ks in [("editor raised a warning", w1), ("no warning", w0)]:
        rr, mm = qrow(name, ks, ks)
        P(rr)
        metrics.setdefault("warning", {})[name] = mm

    # ---------------------------------------------------------------- 4. kappa
    P("\n### Second judge: gemini-3.1-flash-lite (validity_judge.v3) vs Opus, same items\n")
    both = [k for k in allk if lab(k) and (j2.get(k) or {}).get("labels")]
    P(f"n = {len(both)} items judged by both (of {len(allk)}).\n")
    P("| label | Cohen's κ | raw agreement | Opus positive | Gemini positive |")
    P("|---|---|---|---|---|")
    for f, g2f in [("teacher", "teacher"), ("new", "new"), ("valid", "valid"), ("absurd", "absurd"),
                   ("tail", "tail"), ("neardupj", "neardupj"), ("leakj", "leakj"), ("unansw", "unansw")]:
        pairs = [(bool(lab(k)[f]), bool(j2[k]["labels"][g2f])) for k in both]
        if not pairs:
            continue
        kp = kappa(pairs)
        P(f"| {f} | {kp:.2f} | {100 * sum(a == b for a, b in pairs) / len(pairs):.0f}% | "
          f"{sum(a for a, _ in pairs)} | {sum(b for _, b in pairs)} |")
        metrics.setdefault("kappa", {})[f] = kp
    if both:
        ga = [k for k in both if j2[k]["labels"].get("judge_answer") is not None]
        from ..pilot_gen.util import answer_values
        from fractions import Fraction
        agree = sum(Fraction(mut[k]["answer"]) in answer_values(j2[k]["labels"]["judge_answer"]) for k in ga)
        P(f"\nGemini judge's own solve (v3 `answer` field) = program answer: {fmt_ci(agree, len(ga))}.")
        sv = [k for k in both if gat[k]["survivor"]]
        P(f"On the {len(sv)} survivors, Gemini teacher-would-write = "
          f"{fmt_ci(sum(bool(j2[k]['labels']['teacher']) for k in sv), len(sv))} (Opus "
          f"{fmt_ci(sum(bool(lab(k)['teacher']) for k in sv), len(sv))}).")

    # ---------------------------------------------------------------- 5. M3 semantic tail
    P("\n### M3 and X4's open question (`is_semantic_tail`: new answer = f(old answer, new constant))\n")
    P("| M3 subset | n | teacher | new | tail (judge) | near-dup (judge) | cycle ok | survivor |")
    P("|---|---|---|---|---|---|---|---|")
    m3 = [k for k in allk if gat[k]["op"] == "M3"]
    for name, ks in [("semantic tail", [k for k in m3 if mut[k]["recheck"]["semantic_tail"] is True]),
                     ("not semantic tail", [k for k in m3 if mut[k]["recheck"]["semantic_tail"] is not True])]:
        ks = [k for k in ks if lab(k)]
        n = len(ks)
        P(f"| {name} | {n} | {sum(lab(k)['teacher'] for k in ks)} | {sum(lab(k)['new'] for k in ks)} | "
          f"{sum(lab(k)['tail'] for k in ks)} | {sum(lab(k)['neardupj'] for k in ks)} | "
          f"{sum(gat[k]['cycle_ok'] for k in ks)} | {sum(gat[k]['survivor'] for k in ks)} |")

    # ---------------------------------------------------------------- 6. cost / time
    logs = [json.loads(l) for l in open(ROOT / "logs/api/G3/claude.jsonl", encoding="utf-8")]
    gl = [json.loads(l) for l in open(ROOT / "logs/api/G3/gemini.jsonl", encoding="utf-8")]
    by = defaultdict(lambda: {"calls": 0, "ok": 0, "usd": 0.0, "lat": []})
    for r in logs + gl:
        t = r.get("tag") or "?"
        t = "editor" if t.startswith("editor") else t
        b = by[(r["model_requested"], t)]
        b["calls"] += 1
        b["ok"] += bool(r.get("ok")) and not r.get("cache_hit")
        b["usd"] += r.get("cost_usd") or 0.0
        if r.get("ok") and not r.get("cache_hit") and r.get("latency_ms"):
            b["lat"].append(r["latency_ms"] / 1000)
    P("\n### API ledger (logs/api/G3; all runs incl. run A; list prices from usage)\n")
    P("| model | task | attempts logged | OK live calls | $ | $ / OK call | latency p50 s |")
    P("|---|---|---|---|---|---|---|")
    tot = 0.0
    for (mdl, t), b in sorted(by.items()):
        tot += b["usd"]
        P(f"| {mdl} | {t} | {b['calls']} | {b['ok']} | {b['usd']:.3f} | "
          f"{(b['usd'] / b['ok'] if b['ok'] else 0):.5f} | {stt.median(b['lat']) if b['lat'] else float('nan'):.1f} |")
    P(f"\n**Claude total: ${tot:.2f}** (cap $6).")
    metrics["claude_usd"] = tot
    # throughput of the gemini cycle run on run B
    cy = [r for r in gl if r.get("tag") == "back_formalize" and r.get("ok") and not r.get("cache_hit")]
    if cy:
        t0, t1 = min(ts(r["ts_utc"]) for r in cy), max(ts(r["ts_utc"]) for r in cy)
        metrics["gemini_cycle_ok_calls"] = len(cy)
        metrics["gemini_cycle_span_min"] = (t1 - t0) / 60
    json.dump(metrics, open(R / "metrics.json", "w"), indent=1, default=str)
    (R / "report_tables.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
