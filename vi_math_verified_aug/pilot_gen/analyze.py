"""G1 pilot analysis (pipeline layout): per-generator metrics with Wilson 95% CIs, cost/latency, judge agreement.

python -m vi_math_verified_aug.pilot_gen.analyze [--since 2026-11-07T16:53:00]
-> data/pilot/generator/metrics.json, items_scored.jsonl

Judge policy (coordinator/user): Claude items -> Gemini judge primary; Gemini items -> Opus primary; NIM items ->
both (an item passes only if both pass; a flag is raised if either raises it). Opus runs on the Opus subsample
(opus_subsample.json) only, so "primary" metrics for Gemini/NIM generators and every answer-correct/yield figure are
on that subsample; Gemini-judge metrics are also given on all 100 items for every generator.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict

from . import clients as C
from . import seeds as S
from .util import answer_in_text, answer_values, chrf, fmt_ci, kappa, leak_regex, wilson

OUT = S.OUT
BOOLS = ["atomic", "no_conflict", "complete", "single_question", "realistic"]
PRIMARY = {"haiku": "gemini_judge", "sonnet": "gemini_judge", "gemini": "opus", "gemini31": "opus",
           "deepseek": "both", "nemotron": "both", "glm": "both"}


def rows(stage, gen):
    p = OUT / stage / f"{gen}.jsonl"
    d = {}
    if p.exists():
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            d[r["key"]] = r
    return d


def valid(v):
    try:
        return all(v.get(b) is True for b in BOOLS) and float(v.get("natural", 0)) >= 4
    except Exception:
        return False


def lab(v: dict) -> dict:
    cs = set(v.get("codes") or [])
    return {"valid": valid(v), "teacher": v.get("teacher_would_write") is True, "new": v.get("new_vs_seed") is True,
            "op": v.get("op_followed") is True, "natural": v.get("natural"),
            "leakj": "NL-LEAK" in cs, "absurd": "NL-ABSURD" in cs, "tail": "NL-TACKED" in cs,
            "neardupj": "NL-NEARDUP" in cs, "fluency": "NL-FLUENCY" in cs, "unansw": "NL-UNANSW" in cs}


def combine(a: dict, b: dict) -> dict:
    out = {}
    for k in a:
        if k == "natural":
            out[k] = min(a[k] or 0, b[k] or 0)
        elif k in ("valid", "teacher", "new", "op"):
            out[k] = a[k] and b[k]
        else:
            out[k] = a[k] or b[k]
    return out


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    f = int(k)
    return xs[f] + (xs[min(f + 1, len(xs) - 1)] - xs[f]) * (k - f)


def ci(k, n):
    p, lo, hi = wilson(k, n)
    return {"k": k, "n": n, "p": p, "lo": lo, "hi": hi, "txt": fmt_ci(k, n)}


def score_items(opus_s):
    seeds = {s["seed_id"]: s for s in S.load()}
    items = []
    for g in C.GENERATORS:
        N, JG, JO, OS = rows("native", g), rows("judge_gemini_judge", g), rows("judge_opus", g), rows("opus_solve", g)
        for sid, s in seeds.items():
            a = N.get(sid)
            it = {"gen": g, "seed_id": sid, "op": s["op"], "in_opus_s": sid in opus_s, "generated": bool(a),
                  "parsed": bool(a and a.get("parsed")), "problem": (a or {}).get("problem"),
                  "answer": (a or {}).get("answer"), "seed": s["text"]}
            if not it["parsed"]:
                items.append(it)
                continue
            av = answer_values(a["answer"])
            it["leak_regex"] = leak_regex(a["problem"])
            it["answer_in_text"] = answer_in_text(a["problem"], av) if av else False
            it["chrf"] = round(chrf(a["problem"], s["text"]), 3)
            jg, jo = JG.get(sid), JO.get(sid)
            it["gj"] = lab(jg["verdict"]) if jg and jg.get("parsed") else None
            it["oj"] = lab(jo["verdict"]) if jo and jo.get("parsed") else None
            it["gj_note"] = (jg or {}).get("verdict", {}).get("note") if jg else None
            it["oj_note"] = (jo or {}).get("verdict", {}).get("note") if jo else None
            pol = PRIMARY[g]
            if pol == "gemini_judge":
                it["prim"] = it["gj"]
            elif pol == "opus":
                it["prim"] = it["oj"]
            else:
                it["prim"] = combine(it["gj"], it["oj"]) if (it["gj"] and it["oj"]) else None
            o = OS.get(sid)
            it["opus_boxed"] = (o or {}).get("boxed")
            it["answer_correct"] = bool(o["agrees"]) if (o and o.get("api_ok")) else None
            if it["prim"]:
                p = it["prim"]
                it["leak"] = bool(it["leak_regex"]) or p["leakj"]
                it["neardup"] = p["neardupj"] or (it["chrf"] > 0.90 and s["op"] != "M1")
            if it["prim"] and it["answer_correct"] is not None:
                it["accepted"] = bool(not it["leak_regex"] and it["prim"]["valid"] and it["answer_correct"])
                it["accepted_strict"] = bool(it["accepted"] and it["prim"]["teacher"] and not it["leak"]
                                             and (it["prim"]["new"] or s["op"] == "M1"))
            items.append(it)
    with open(OUT / "items_scored.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return items


def label_metrics(its, key):
    J = [i for i in its if i.get(key)]
    nn = [i for i in J if i["op"] != "M1"]
    m = {"n_judged": len(J)}
    for f in ("valid", "teacher", "op", "absurd", "tail", "fluency", "unansw"):
        m[f] = ci(sum(bool(i[key][f]) for i in J), len(J))
    m["leak"] = ci(sum(bool(i["leak_regex"]) or i[key]["leakj"] for i in J), len(J))
    m["neardup"] = ci(sum(i[key]["neardupj"] or (i["chrf"] > 0.9 and i["op"] != "M1") for i in J), len(J))
    m["new_M2_M6"] = ci(sum(i[key]["new"] for i in nn), len(nn))
    m["natural_mean"] = st.mean([float(i[key]["natural"] or 0) for i in J]) if J else None
    return m


def main(since: str):
    opus_s = set(json.load(open(OUT / "opus_subsample.json")))
    items = score_items(opus_s)
    logs = []
    for p in sorted(C.LOG_DIR.glob("*.jsonl")):
        for l in open(p, encoding="utf-8"):
            try:
                r = json.loads(l)
            except Exception:
                continue
            if r.get("ts", "") >= since and r.get("task") not in ("smoke", "probe"):
                logs.append(r)
    M = {}
    for g in C.GENERATORS:
        its = [i for i in items if i["gen"] == g]
        sub = [i for i in its if i["in_opus_s"]]
        m = {"generated": sum(i["generated"] for i in its), "parsed": ci(sum(i["parsed"] for i in its),
                                                                         sum(i["generated"] for i in its))}
        # headline on the Opus subsample, primary judge
        sj = [i for i in sub if i["parsed"] and i.get("prim") and i.get("answer_correct") is not None]
        gen_s = [i for i in sub if i["generated"]]
        m["S_n"] = len(gen_s)
        m["S_yield"] = ci(sum(i.get("accepted", False) for i in sj), len(gen_s))
        m["S_yield_strict"] = ci(sum(i.get("accepted_strict", False) for i in sj), len(gen_s))
        m["S_answer_correct"] = ci(sum(i["answer_correct"] for i in sub if i.get("answer_correct") is not None),
                                   sum(1 for i in sub if i.get("answer_correct") is not None))
        m["S_primary"] = label_metrics([i for i in sub if i["parsed"]], "prim")
        m["S_opus"] = label_metrics([i for i in sub if i["parsed"]], "oj")
        m["all_gemini_judge"] = label_metrics([i for i in its if i["parsed"]], "gj")
        if PRIMARY[g] == "gemini_judge":
            m["all_primary"] = m["all_gemini_judge"]
        P = [i for i in its if i["parsed"]]
        m["leak_regex"] = ci(sum(bool(i["leak_regex"]) for i in P), len(P))
        m["answer_in_text"] = ci(sum(i["answer_in_text"] for i in P), len(P))
        m["chrf_median"] = st.median([i["chrf"] for i in P]) if P else None
        F = rows("formalize", g)
        m["formalize_pass1"] = ci(sum(r["pass1"] for r in F.values()), len(F))
        m["formalize_pass3"] = ci(sum(r["pass"] for r in F.values()), len(F))
        m["formalize_flagged"] = ci(sum(bool(r["flags"]) for r in F.values() if r["pass"]),
                                    sum(r["pass"] for r in F.values()))
        fe = defaultdict(int)
        for r in F.values():
            if not r["pass"]:
                e = (r["errors"] or ["?"])[0]
                fe[e.split(":")[0] if not e.startswith("answer") else "wrong answer"] += 1
        m["formalize_fail_reasons"] = dict(fe)
        R = rows("render", g)
        m["render_pass"] = ci(sum(r["pass"] for r in R.values()), len(R))
        re_ = defaultdict(int)
        for r in R.values():
            for e in r["errors"]:
                re_[e.split(" ")[0] + " " + (e.split(" ")[1] if len(e.split(" ")) > 1 else "")] += 1
        m["render_fail_reasons"] = dict(re_)
        D = rows("solve_seeds", g)
        m["solve_seeds"] = ci(sum(r.get("correct", False) for r in D.values()), len(D))
        W = rows("writer_common", g)
        m["solve_common"] = ci(sum(r.get("correct", False) for r in W.values()), len(W))
        model = C.MODELS[g][1]
        lt = defaultdict(lambda: {"lat": [], "hlat": [], "in": [], "out": [], "cost": 0.0, "calls": 0,
                                  "err": defaultdict(int)})
        for r in logs:
            if r.get("model") != model or r.get("model_key") != g:
                continue
            d = lt[r["task"]]
            if not r.get("ok"):
                d["err"][str(r.get("status"))] += 1
                continue
            d["calls"] += 1
            d["lat"].append(r["latency"])
            if (r.get("usage") or {}).get("http_latency") is not None:
                d["hlat"].append(r["usage"]["http_latency"])
            d["in"].append((r.get("usage") or {}).get("input_tokens", 0) or 0)
            d["out"].append((r.get("usage") or {}).get("output_tokens", 0) or 0)
            d["cost"] += r.get("cost") or 0
        m["tasks"] = {t: {"calls": d["calls"], "errors": dict(d["err"]), "lat_med": pct(d["lat"], .5),
                          "lat_p90": pct(d["lat"], .9), "http_n": len(d["hlat"]), "http_med": pct(d["hlat"], .5),
                          "http_p90": pct(d["hlat"], .9), "in_mean": st.mean(d["in"]) if d["in"] else None,
                          "out_mean": st.mean(d["out"]) if d["out"] else None, "cost": d["cost"],
                          "cost_per_call": d["cost"] / d["calls"] if d["calls"] else 0} for t, d in lt.items()}
        M[g] = m
    # judge agreement (items with both verdicts)
    A = defaultdict(list)
    for i in items:
        if i.get("gj") and i.get("oj"):
            for f in ("valid", "teacher", "new", "absurd", "tail", "neardupj", "leakj", "op"):
                A[f].append((i["oj"][f], i["gj"][f]))
                A[f"{f}@{i['gen']}"].append((i["oj"][f], i["gj"][f]))
            A["natural>=4"].append(((i["oj"]["natural"] or 0) >= 4, (i["gj"]["natural"] or 0) >= 4))
    agree = {f: {"n": len(p), "kappa": kappa(p), "agree": sum(x == y for x, y in p) / len(p),
                 "opus_pos": sum(bool(x) for x, _ in p) / len(p), "gem_pos": sum(bool(y) for _, y in p) / len(p)}
             for f, p in A.items() if p}
    jc = defaultdict(lambda: {"calls": 0, "cost": 0.0, "lat": [], "out": [], "in": [], "cache_read": 0, "err": 0})
    for r in logs:
        if r.get("task") in ("judge", "opus_solve"):
            k = f"{r['model_key']}:{r['task']}"
            if not r.get("ok"):
                jc[k]["err"] += 1
                continue
            jc[k]["calls"] += 1
            jc[k]["cost"] += r.get("cost") or 0
            u = r.get("usage") or {}
            jc[k]["lat"].append(u.get("http_latency") or r["latency"])
            jc[k]["out"].append(u.get("output_tokens", 0) or 0)
            jc[k]["in"].append(u.get("input_tokens", 0) or 0)
            jc[k]["cache_read"] += u.get("cache_read", 0) or 0
    jcs = {k: {"calls": v["calls"], "errors": v["err"], "cost": v["cost"],
               "cost_per_call": v["cost"] / v["calls"] if v["calls"] else None, "lat_med": pct(v["lat"], .5),
               "lat_p90": pct(v["lat"], .9), "out_mean": st.mean(v["out"]) if v["out"] else None,
               "in_mean": st.mean(v["in"]) if v["in"] else None, "cache_read_tokens": v["cache_read"]}
           for k, v in jc.items()}
    # observed throughput per provider (successful calls per minute over the active window)
    thr = {}
    for prov in ("nim", "claude", "gemini"):
        ts = sorted(r["ts"] for r in logs if r.get("provider") == prov and r.get("ok"))
        if len(ts) > 1:
            from datetime import datetime
            a, b = datetime.fromisoformat(ts[0]), datetime.fromisoformat(ts[-1])
            mins = max((b - a).total_seconds() / 60, 1e-9)
            thr[prov] = {"calls": len(ts), "minutes": mins, "per_min": len(ts) / mins}
    errs = defaultdict(lambda: defaultdict(int))
    for r in logs:
        if not r.get("ok"):
            errs[f"{r['model_key']}"][str(r.get("status"))] += 1
    res = {"generators": M, "judge_agreement": agree, "judge_calls": jcs, "throughput": thr,
           "errors_by_status": errs, "claude_spent_total": C.claude_spent(), "opus_subsample_n": len(opus_s)}
    json.dump(res, open(OUT / "metrics.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-11-07T16:53:00")
    r = main(ap.parse_args().since)
    for g, m in r["generators"].items():
        print(g, "S_yield", m["S_yield"]["txt"], "| teach", m["S_primary"]["teacher"]["txt"], "| ans",
              m["S_answer_correct"]["txt"], "| gj-teach100", m["all_gemini_judge"]["teacher"]["txt"],
              "| F1", m["formalize_pass1"]["txt"], "F3", m["formalize_pass3"]["txt"], "| R", m["render_pass"]["txt"],
              "| D", m["solve_seeds"]["txt"], "| W", m["solve_common"]["txt"])
    for k, v in r["judge_agreement"].items():
        if "@" not in k:
            print("kappa", k, v)
    print("throughput", r["throughput"])
    print("claude $", r["claude_spent_total"])
