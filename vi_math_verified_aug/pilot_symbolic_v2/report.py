"""G4 report tables -> data/pilot/symbolic_v2/report_tables.md + metrics.json.

    python -m vi_math_verified_aug.pilot_symbolic_v2.report

Variants: G3 (as published: unfixed operators incl. M6, strict answer-in-text rule), E (edit only, fixed
operators), R1 (edit + fluency), R2 (free render from the IR), N (Haiku native_generator.v3 on the same slots;
reference only, no certificate). Quality = Opus 5.5 validity_judge.v2 labels (lab_v2).
"""
from __future__ import annotations

import json
import math
import statistics as stt
from collections import Counter, defaultdict

from ..pilot_gen.util import fmt_ci, kappa, wilson
from ..pilot_symbolic import seeds as SD
from .common import OUT, ROOT, Store, g3

OPS5 = ["M1", "M2", "M3", "M4", "M5"]
YV = ["E", "R1", "R2", "R1b"]
FIELDS = ["teacher", "new", "valid", "absurd", "tail", "neardupj", "unansw", "fluency"]


def binom_two_sided(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    p = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * p)


def pct(k, n):
    return "–" if n == 0 else f"{100 * k / n:.0f}%"


class Data:
    def __init__(self):
        self.slots = SD.load_slots()
        self.seeds = SD.load_seeds()
        self.mut = Store("mutate").rows
        self.render = {v: Store(f"render_{v}").rows for v in YV}
        self.cycle = {v: Store(f"cycle_{v}").rows for v in YV}
        self.gates = Store("gates").rows
        self.jo, self.so, self.js = Store("judge_opus").rows, Store("solve_opus").rows, Store("judge_sonnet").rows
        self.nat = Store("native_N").rows
        self.g3 = {s: g3(s) for s in ("mutate", "render", "cycle", "gates", "judge_opus", "solve_opus")}

    # one record per (variant, slot) with a judged text
    def items(self, v: str) -> list[dict]:
        out = []
        if v == "G3":
            for sl in self.slots:
                k = sl["key"]
                r = self.g3["render"].get(k)
                if not r or not r.get("tag_ok"):
                    continue
                j = (self.g3["judge_opus"].get(k) or {}).get("labels")
                g = self.g3["gates"].get(k, {})
                out.append({"slot": k, "op": sl["op"], "text": r["text"], "lab": j, "survivor": g.get("survivor"),
                            "cycle_ok": g.get("cycle_ok"), "agree": (self.g3["solve_opus"].get(k) or {}).get("agree"),
                            "note": ((self.g3["judge_opus"].get(k) or {}).get("verdict") or {}).get("note"),
                            "warning": bool(r.get("warning")), "son": None})
            return out
        if v == "N":
            for sl in self.slots:
                k = sl["key"]
                r = self.nat.get(k)
                if not r or not r.get("parsed"):
                    continue
                key = f"N:{k}"
                out.append({"slot": k, "op": sl["op"], "text": r["text"], "lab": (self.jo.get(key) or {}).get("labels"),
                            "survivor": True, "cycle_ok": None, "agree": (self.so.get(key) or {}).get("agree"),
                            "note": ((self.jo.get(key) or {}).get("verdict") or {}).get("note"), "warning": False,
                            "son": (self.js.get(key) or {}).get("labels")})
            return out
        for sl in self.slots:
            k = sl["key"]
            r = self.render[v].get(k)
            if not r or not r.get("tag_ok"):
                continue
            key = f"{v}:{k}"
            g = self.gates.get(key, {})
            out.append({"slot": k, "op": sl["op"], "text": r["text"], "lab": (self.jo.get(key) or {}).get("labels"),
                        "survivor": g.get("survivor"), "cycle_ok": g.get("cycle_ok"),
                        "agree": (self.so.get(key) or {}).get("agree"),
                        "note": ((self.jo.get(key) or {}).get("verdict") or {}).get("note"),
                        "warning": bool(r.get("warning")), "son": (self.js.get(key) or {}).get("labels"),
                        "gates": g})
        return out


def quality(its: list[dict]) -> dict:
    J = [i for i in its if i["lab"]]
    n = len(J)
    newset = [i for i in J if i["op"] != "M1"]
    c = lambda f, S=J: sum(bool(i["lab"][f]) for i in S)  # noqa: E731
    ag = [i for i in its if i["agree"] is not None]
    return {"n": n, "teacher": (c("teacher"), n), "new": (c("new", newset), len(newset)),
            "leak": (sum(bool(i["lab"]["leakj"]) for i in J), n), "absurd": (c("absurd"), n), "tail": (c("tail"), n),
            "neardup": (c("neardupj", newset), len(newset)), "valid": (c("valid"), n), "unansw": (c("unansw"), n),
            "fluency": (c("fluency"), n), "natural": round(stt.mean([i["lab"]["natural"] for i in J]), 2) if J else None,
            "agree": (sum(bool(i["agree"]) for i in ag), len(ag)),
            "accept": (sum(i["lab"]["teacher"] and i["lab"]["valid"] and not i["lab"]["absurd"] and not i["lab"]["leakj"]
                           and not i["lab"]["tail"] for i in J), n)}


def gate_verdict(q: dict) -> str:
    t, nw, lk, ab = [q[f][0] / q[f][1] if q[f][1] else float("nan") for f in ("teacher", "new", "leak", "absurd")]
    marks = [t >= .70, nw >= .60, lk == 0, ab <= .03]
    return f"{sum(marks)}/4 ({'PASS' if all(marks) else 'FAIL'})"


def qrow(name, q) -> str:
    return (f"| {name} | {q['n']} | **{fmt_ci(*q['teacher'])}** | {fmt_ci(*q['new'])} | {fmt_ci(*q['leak'])} | "
            f"**{fmt_ci(*q['absurd'])}** | {pct(*q['tail'])} | {pct(*q['neardup'])} | {pct(*q['valid'])} | "
            f"{pct(*q['unansw'])} | {pct(*q['fluency'])} | {q['natural']} | {fmt_ci(*q['agree'])} | "
            f"{pct(*q['accept'])} | {gate_verdict(q)} |")


QHEAD = ("| set | n | teacher (≥70%) | new M2–M5 (≥60%) | leak (0%) | absurd (≤3%) | tail | near-dup M2–M5 | valid | "
         "unansw | fluency issue | natural | Opus solve = target | accept* | gate |\n"
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")


def costs() -> dict:
    by = defaultdict(lambda: [0, 0.0])
    p = ROOT / "logs/api/G4/claude.jsonl"
    if p.exists():
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            if r.get("cache_hit") or not r.get("ok"):
                continue
            t = r.get("tag") or r.get("event") or "?"
            by[t][0] += 1
            by[t][1] += r.get("cost_usd") or 0.0
    return {k: (n, round(c, 4)) for k, (n, c) in sorted(by.items())}


def main():
    D = Data()
    out: list[str] = []
    P = out.append
    M: dict = {}
    nonm6 = [s for s in D.slots if s["op"] != "M6"]

    # ------------------------------------------------------------------ operator fixes
    P("### A. Operator fixes (fixed mutate stage)\n")
    P("| op | slots | G3 mutated | G4 mutated | same program as G3 | new program | rejected by story checks | "
      "candidates checked by Haiku | det-filter rejects | semantic-tail candidates (rejected) |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    for op in OPS5 + ["M6"]:
        S = [s for s in D.slots if s["op"] == op]
        g3m = sum(bool(D.g3["mutate"].get(s["key"], {}).get("ok")) for s in S)
        rows = [D.mut[s["key"]] for s in S]
        ok = sum(r["ok"] for r in rows)
        same = sum(r["ok"] and r.get("same_as_g3") for r in rows)
        rej = sum((not r["ok"]) and "story" in (r.get("reason") or "") for r in rows)
        hk = sum(1 for r in rows for c in r.get("checks", []) if c.get("det_ok"))
        det = sum(1 for r in rows for c in r.get("checks", []) if c.get("det_ok") is False)
        st = [c for r in rows for c in r.get("checks", []) if c.get("semantic_tail")]
        st_rej = sum(1 for c in st if not c.get("accept"))
        P(f"| {op} | {len(S)} | {g3m} | {ok} | {same} | {ok - same} | {rej} | {hk} | {det} | {len(st)} ({st_rej}) |")
    hv = Counter()
    for r in D.mut.values():
        for c in r.get("checks", []):
            if c.get("det_ok"):
                v = c.get("verdict") or {}
                hv[(r["op"], v.get("type_ok"), v.get("story_meaning"))] += 1
    P("\nHaiku story-check verdicts (op, type_ok, story_meaning): " +
      ", ".join(f"{k}: {v}" for k, v in sorted(hv.items(), key=str)) + "\n")

    # ------------------------------------------------------------------ funnel
    P("### B. Stage yield per variant (count and % of the 160 non-M6 slots; G3 row = its 160 non-M6 slots)\n")
    P("| variant | op | slots | mutated | tag ok | cycle ok | no leak/answer | chrF ≤ .90 | cos < .85 | dedup = survivor |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    M["funnel"] = {}
    for v in ["G3"] + YV:
        for op in OPS5 + ["all"]:
            S = [s for s in nonm6 if op == "all" or s["op"] == op]
            cnt = Counter()
            for s in S:
                k = s["key"]
                if v == "G3":
                    mu = D.g3["mutate"].get(k, {}).get("ok", False)
                    r = D.g3["render"].get(k, {})
                    g = D.g3["gates"].get(k, {})
                    noleak = g.get("leak_ok") and not g.get("answer_in_text", True)
                else:
                    mu = D.mut[k]["ok"]
                    r = D.render[v].get(k, {})
                    g = D.gates.get(f"{v}:{k}", {})
                    noleak = g.get("leak_ok") and not g.get("answer_untagged", True)
                f = {"mutated": mu, "tag": mu and r.get("tag_ok", False)}
                f["cycle"] = f["tag"] and bool(g.get("cycle_ok"))
                f["noleak"] = f["cycle"] and bool(noleak)
                f["chrf"] = f["noleak"] and bool(g.get("chrf_ok"))
                f["cos"] = f["chrf"] and bool(g.get("decontam_ok")) and bool(g.get("xseed_dup_ok"))
                f["surv"] = bool(g.get("survivor"))
                for kk, vv in f.items():
                    cnt[kk] += bool(vv)
            n = len(S)
            if op == "all" or v != "G3":
                P(f"| {v} | {op} | {n} | " + " | ".join(f"{cnt[x]} ({pct(cnt[x], n)})" for x in
                                                        ("mutated", "tag", "cycle", "noleak", "chrf", "cos", "surv")) + " |")
            M["funnel"][f"{v}:{op}"] = {**cnt, "slots": n}
    P("")

    # ------------------------------------------------------------------ quality
    P("### C. Quality gate (Opus 5.5, validity_judge.v2) — survivors\n")
    P(QHEAD)
    M["quality"] = {}
    allit = {v: D.items(v) for v in ["G3"] + YV + ["N"]}
    for v in ["G3"] + YV:
        its = allit[v]
        sv = [i for i in its if i["survivor"]]
        q = quality(sv)
        P(qrow(f"**{v} survivors**", q))
        M["quality"][f"{v}:surv"] = q
        if v == "G3":
            q2 = quality([i for i in sv if i["op"] != "M6"])
            P(qrow("G3 survivors, M6 excluded", q2))
            M["quality"]["G3:surv_noM6"] = q2
    qn = quality(allit["N"])
    P(qrow("**N (native v3, same slots), all parsed**", qn))
    M["quality"]["N:all"] = qn
    nsv = [i for i in allit["N"] if i["agree"]]
    P(qrow("N, Opus solve agrees with generator answer", quality(nsv)))
    P("\n\\*accept = teacher ∧ valid ∧ ¬absurd ∧ ¬leak ∧ ¬tail.\n")
    P("#### C2. Same table on ALL tag-passing renders (before cycle check and gates)\n")
    P(QHEAD)
    for v in ["G3"] + YV:
        its = [i for i in allit[v] if v != "G3" or i["op"] != "M6"]
        q = quality(its)
        P(qrow(f"{v} tag-passing{' (no M6)' if v == 'G3' else ''}", q))
        M["quality"][f"{v}:tag"] = q
    P("")
    P("#### C3. Survivors by operator (teacher / absurd / new / n)\n")
    P("| variant | " + " | ".join(OPS5) + " |")
    P("|---|" + "---|" * 5)
    for v in YV + ["N"]:
        cells = []
        for op in OPS5:
            S = [i for i in allit[v] if i["op"] == op and i["survivor"] and i["lab"]]
            if not S:
                cells.append("–")
                continue
            cells.append(f"{pct(sum(i['lab']['teacher'] for i in S), len(S))} / {pct(sum(i['lab']['absurd'] for i in S), len(S))}"
                         f" / {pct(sum(i['lab']['new'] for i in S), len(S)) if op != 'M1' else '–'} / {len(S)}")
        P(f"| {v} | " + " | ".join(cells) + " |")
    P("")

    # ------------------------------------------------------------------ paired
    P("### D. Paired comparisons on the same slots (exact McNemar, two-sided)\n")
    P("| pair | set | n pairs | label | A-only | B-only | A rate | B rate | p |")
    P("|---|---|---|---|---|---|---|---|---|")
    M["paired"] = []
    by = {v: {i["slot"]: i for i in allit[v] if i["lab"]} for v in YV + ["N"]}
    for a, b in (("E", "R1"), ("E", "R2"), ("R1", "R2"), ("R1", "R1b"), ("E", "R1b"), ("R1b", "R2"), ("R1", "N"), ("R2", "N"), ("R1b", "N")):
        for setname in ("tag-passing", "both survive"):
            ks = [k for k in by[a] if k in by[b] and (setname == "tag-passing" or (by[a][k]["survivor"] and by[b][k]["survivor"]))]
            for lab in ("teacher", "absurd", "valid"):
                ao = sum(by[a][k]["lab"][lab] and not by[b][k]["lab"][lab] for k in ks)
                bo = sum(by[b][k]["lab"][lab] and not by[a][k]["lab"][lab] for k in ks)
                ra = sum(by[a][k]["lab"][lab] for k in ks)
                rb = sum(by[b][k]["lab"][lab] for k in ks)
                p = binom_two_sided(min(ao, bo), ao + bo)
                P(f"| {a} vs {b} | {setname} | {len(ks)} | {lab} | {ao} | {bo} | {pct(ra, len(ks))} | {pct(rb, len(ks))} | {p:.3g} |")
                M["paired"].append({"a": a, "b": b, "set": setname, "n": len(ks), "label": lab, "a_only": ao, "b_only": bo, "p": p})
    P("")

    # ------------------------------------------------------------------ faithfulness
    P("### E. Answer faithfulness (separate from text quality)\n")
    P("| variant | cycle pass / tag-passing | Opus agrees: survivors | Opus agrees: cycle passed | Opus agrees: cycle FAILED | "
      "cycle-failed: absurd / unansw |")
    P("|---|---|---|---|---|---|")
    M["faith"] = {}
    for v in ["G3"] + YV:
        its = [i for i in allit[v] if v != "G3" or i["op"] != "M6"]
        cy = sum(bool(i["cycle_ok"]) for i in its)
        f = lambda S: (sum(bool(i["agree"]) for i in S if i["agree"] is not None), sum(i["agree"] is not None for i in S))  # noqa: E731
        sv, cp, cf = [i for i in its if i["survivor"]], [i for i in its if i["cycle_ok"]], [i for i in its if not i["cycle_ok"]]
        cfl = [i for i in cf if i["lab"]]
        P(f"| {v} | {fmt_ci(cy, len(its))} | {fmt_ci(*f(sv))} | {fmt_ci(*f(cp))} | {fmt_ci(*f(cf))} | "
          f"{pct(sum(i['lab']['absurd'] for i in cfl), len(cfl))} / {pct(sum(i['lab']['unansw'] for i in cfl), len(cfl))} |")
        M["faith"][v] = {"cycle": (cy, len(its)), "agree_surv": f(sv), "agree_cp": f(cp), "agree_cf": f(cf)}
    P(f"| N (no certificate) | – | – | – | – | Opus agrees with generator answer: {fmt_ci(*qn['agree'])} |")
    P("")
    P("Cycle pass by operator (tag-passing):\n")
    P("| variant | " + " | ".join(OPS5) + " |")
    P("|---|" + "---|" * 5)
    for v in YV:
        P(f"| {v} | " + " | ".join(
            (lambda S: fmt_ci(sum(bool(i['cycle_ok']) for i in S), len(S)))([i for i in allit[v] if i["op"] == op])
            for op in OPS5) + " |")
    P("")

    # ------------------------------------------------------------------ tag check / localisation
    P("### F. Renderer mechanics\n")
    for v in YV:
        rr = [r for r in D.render[v].values()]
        okn = sum(r["tag_ok"] for r in rr)
        p1 = sum(r["tag_ok"] and r.get("tag_attempts", 0) <= 1 for r in rr)
        warn = sum(bool(r.get("warning")) for r in rr if r["tag_ok"])
        loc = sum(bool(r.get("localized")) for r in rr)
        iss = Counter()
        for r in rr:
            for a in r.get("attempts", []):
                for x in a.get("issues", []):
                    iss[x.split("'")[0].split(":")[0][:40]] += 1
        P(f"- **{v}**: tag pass {okn}/{len(rr)}, pass@1 {p1}; warnings {warn}; USD→đồng rescaled {loc}; "
          f"tag issues over all attempts: " + ", ".join(f"{k} {c}" for k, c in iss.most_common(5)))
        M[f"tag_{v}"] = {"ok": okn, "n": len(rr), "pass1": p1, "warn": warn, "loc": loc}
    for v in YV:
        S = [i for i in allit[v] if i["lab"]]
        w = [i for i in S if i["warning"]]
        nw = [i for i in S if not i["warning"]]
        P(f"- {v} warning flag: teacher {pct(sum(i['lab']['teacher'] for i in w), len(w))} with warning (n={len(w)}) vs "
          f"{pct(sum(i['lab']['teacher'] for i in nw), len(nw))} without (n={len(nw)})")
    P("")

    # ------------------------------------------------------------------ Sonnet kappa
    P("### G. Second judge: Sonnet 5.5 + validity_judge.v3 vs Opus (κ; positive rate Opus / Sonnet)\n")
    P("| set | n | " + " | ".join(FIELDS) + " | Sonnet teacher on survivors |")
    P("|---|---|" + "---|" * len(FIELDS) + "---|")
    M["kappa"] = {}
    for v in ["E", "R1", "R2", "N", "pooled"]:
        S = [i for vv in (["E", "R1", "R2", "N"] if v == "pooled" else [v]) for i in allit[vv] if i["lab"] and i["son"]]
        cells = []
        for f in FIELDS:
            pairs = [(bool(i["lab"][f]), bool(i["son"][f])) for i in S]
            k = kappa(pairs)
            cells.append(f"{k:.2f} ({pct(sum(a for a, _ in pairs), len(pairs))}/{pct(sum(b for _, b in pairs), len(pairs))})")
            M["kappa"][f"{v}:{f}"] = k
        sv = [i for i in S if i["survivor"]]
        P(f"| {v} | {len(S)} | " + " | ".join(cells) + f" | {fmt_ci(sum(i['son']['teacher'] for i in sv), len(sv))} |")
    P("")
    for v in ["E", "R1", "R2", "N"]:
        sv = [i for i in allit[v] if i["survivor"] and i["son"]]
        q = {"teacher": sum(i["son"]["teacher"] for i in sv), "absurd": sum(i["son"]["absurd"] for i in sv),
             "new": sum(i["son"]["new"] for i in sv if i["op"] != "M1"), "n_new": sum(i["op"] != "M1" for i in sv)}
        P(f"- Sonnet-v3 on {v} survivors: teacher {fmt_ci(q['teacher'], len(sv))}, new {fmt_ci(q['new'], q['n_new'])}, "
          f"absurd {fmt_ci(q['absurd'], len(sv))}")
        M["quality"][f"{v}:sonnet_surv"] = q | {"n": len(sv)}
    P("")

    # ------------------------------------------------------------------ examples
    P("### H. Best and worst survivors per variant (Opus verdicts)\n")
    for v in YV + ["N"]:
        sv = [i for i in allit[v] if i["survivor"] and i["lab"]]
        good = sorted([i for i in sv if i["lab"]["teacher"] and i["lab"]["valid"]],
                      key=lambda i: (-i["lab"]["natural"], i["op"] == "M1", -len(i["text"])))
        bad = sorted(sv, key=lambda i: (-(i["lab"]["absurd"] + i["lab"]["unansw"] + i["lab"]["tail"]),
                                        i["lab"]["natural"], i["lab"]["teacher"]))
        P(f"**{v}**\n")
        for tag, L in (("best", good[:3]), ("worst", bad[:3])):
            for i in L:
                codes = ",".join(c for c in ("absurd", "unansw", "tail", "neardupj", "fluency") if i["lab"][c])
                P(f"- {tag} `{i['slot']}` {i['op']} (natural {i['lab']['natural']:.0f}, teacher {i['lab']['teacher']}"
                  f"{', ' + codes if codes else ''}): {i['text']}  \n  Opus: {(i['note'] or '')[:300]}")
        P("")

    # ------------------------------------------------------------------ costs
    P("### I. Claude spend (logs/api/G4, live calls only; batch calls at the 50% price)\n")
    cs = costs()
    tot = sum(c for _, c in cs.values())
    for k, (n, c) in cs.items():
        P(f"- {k}: {n} calls, ${c:.4f}" + (f" (${c / n:.5f}/call)" if n else ""))
    P(f"- **total ${tot:.3f}**\n")
    M["costs"] = cs
    M["spend_total"] = tot
    (OUT / "report_tables.md").write_text("\n".join(out), encoding="utf-8")
    json.dump(M, open(OUT / "metrics.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
    print("\n".join(out))


if __name__ == "__main__":
    main()
