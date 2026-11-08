"""G2 analysis: self-preference (task 1), judge calibration + cascade (task 2), prompt iteration (task 3).

python -m vi_math_verified_aug.pilot_gen.g2_report  -> data/pilot/g2/g2_metrics.json + g2_tables.md
"""
from __future__ import annotations

import json
import math
import random
import statistics as st
from collections import defaultdict

from . import g2_common as G
from . import g2_judges as J
from . import g2_native as N
from .util import chrf, fmt_ci, kappa, leak_regex, wilson, answer_in_text

CLAUDE_GENS = {"haiku", "sonnet"}
FIELDS = ["valid", "teacher", "new", "op", "absurd", "neardupj", "tail", "leakj", "unansw"]
CAND = ["haiku_v2", "haiku_v3", "haikum_v3", "sonnet_v3", "g31l_v2", "g31l_v3", "g38_v2", "g3fp_v2", "opusR_v2"]
OPUS_JUDGE_USD = 0.0147      # G1 measured (cached rubric, effort medium); refreshed from task 3 rows below
OPUS_SOLVE_USD = 0.0083


def load_store(stage):
    return G.Store(stage).rows


def accept(l: dict) -> bool:
    """Judge-side acceptance used throughout: valid ∧ teacher ∧ no leak ∧ no tail."""
    return bool(l and l["valid"] and l["teacher"] and not l["leakj"] and not l["tail"])


def mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def kendall_tau(a: list, b: list) -> float:
    n, s = len(a), 0
    for i in range(n):
        for j in range(i + 1, n):
            x, y = a[i] - a[j], b[i] - b[j]
            s += (x > 0) - (x < 0) if y > 0 else ((x < 0) - (x > 0) if y < 0 else 0)
    return s / (n * (n - 1) / 2)


def pct(k, n):
    return fmt_ci(k, n)


# ------------------------------------------------------------------ data assembly
def calib_table():
    items = J.calib_items("all", False)
    judged = {jk: load_store(f"judge_{jk}") for jk in CAND}
    sol = load_store("solve_haiku")
    for it in items:
        it["anchor_seed"] = it["seed"]["index"] in J.ANCHOR_IDX
        it["L"] = {"opus": G.lab_v2(it["opus"])}
        for jk, rows in judged.items():
            r = rows.get(it["key"])
            if r and r.get("parsed"):
                it["L"][jk] = G.labels(jk, r["verdict"])
        ss = [sol.get(f"{it['key']}#s{k}") for k in range(2)]
        it["haiku_solve2"] = all(s and s.get("agrees_gen") for s in ss) if all(ss) else None
        it["leak_regex"] = bool(leak_regex(it["problem"]))
    return items


def rates(its, key, f):
    xs = [i for i in its if key in i["L"]]
    if f == "new":
        xs = [i for i in xs if i["seed"]["op"] != "M1"]
    return sum(bool(i["L"][key][f]) for i in xs), len(xs)


# ------------------------------------------------------------------ task 1
def task1(items, X="g3fp_v2"):
    nat = [i for i in items if i["set"] == "native" and X in i["L"]]
    out = {"n": len(nat), "per_gen": {}, "kappa": {}, "rank": {}, "selfpref": {}}
    for g in J.GENS:
        its = [i for i in nat if i["gen"] == g]
        row = {}
        for jk in ("opus", X):
            d = {}
            for f in ("valid", "teacher", "new", "absurd", "neardupj", "leakj"):
                k, n = rates(its, jk, f)
                d[f] = (k, n)
            nn = [i for i in its if i["seed"]["op"] != "M1"]
            d["neardup_M2M6"] = (sum(i["L"][jk]["neardupj"] for i in nn), len(nn))
            d["natural"] = st.mean(i["L"][jk]["natural"] for i in its)
            d["yield"] = (sum(i["L"][jk]["valid"] and bool(i["opus_agrees"]) and not i["leak_regex"] for i in its),
                          len(its))
            d["accept"] = (sum(accept(i["L"][jk]) for i in its), len(its))
            row[jk] = d
        out["per_gen"][g] = row
    for f in FIELDS + ["natural>=4", "accept"]:
        if f == "natural>=4":
            pairs = [(i["L"]["opus"]["natural"] >= 4, i["L"][X]["natural"] >= 4) for i in nat]
        elif f == "accept":
            pairs = [(accept(i["L"]["opus"]), accept(i["L"][X])) for i in nat]
        else:
            pairs = [(i["L"]["opus"][f], i["L"][X][f]) for i in nat]
        out["kappa"][f] = {"n": len(pairs), "kappa": kappa(pairs), "agree": sum(a == b for a, b in pairs) / len(pairs),
                           "opus_pos": sum(a for a, _ in pairs) / len(pairs),
                           "x_pos": sum(b for _, b in pairs) / len(pairs)}
    for f in ("valid", "teacher", "yield", "accept"):
        a = [out["per_gen"][g]["opus"][f][0] / out["per_gen"][g]["opus"][f][1] for g in J.GENS]
        b = [out["per_gen"][g][X][f][0] / out["per_gen"][g][X][f][1] for g in J.GENS]
        out["rank"][f] = {"opus": a, "x": b, "tau": kendall_tau(a, b)}
    # self-preference: difference-in-differences of pass rates, seed-cluster bootstrap
    seeds = sorted({i["seed_id"] for i in nat})
    by_seed = defaultdict(list)
    for i in nat:
        by_seed[i["seed_id"]].append(i)

    def did(sample, f):
        cl = [i for s in sample for i in by_seed[s] if i["gen"] in CLAUDE_GENS]
        ge = [i for s in sample for i in by_seed[s] if i["gen"] not in CLAUDE_GENS]
        fn = (lambda l: accept(l)) if f == "accept" else (lambda l: bool(l[f]))
        dc = st.mean(fn(i["L"]["opus"]) - fn(i["L"][X]) for i in cl)
        dg = st.mean(fn(i["L"]["opus"]) - fn(i["L"][X]) for i in ge)
        return dc, dg, dc - dg

    rng = random.Random(7)
    for f in ("valid", "teacher", "accept", "absurd", "new"):
        dc, dg, d = did(seeds, f)
        bs = sorted(did([rng.choice(seeds) for _ in seeds], f)[2] for _ in range(2000))
        cl = [i for i in nat if i["gen"] in CLAUDE_GENS]
        ge = [i for i in nat if i["gen"] not in CLAUDE_GENS]
        fn = (lambda l: accept(l)) if f == "accept" else (lambda l: bool(l[f]))
        disc = lambda its: (sum(fn(i["L"]["opus"]) and not fn(i["L"][X]) for i in its),  # noqa: E731
                            sum(fn(i["L"][X]) and not fn(i["L"]["opus"]) for i in its))
        out["selfpref"][f] = {"opus_minus_g38_claude_items": dc, "opus_minus_g38_gemini_items": dg, "did": d,
                              "ci95": (bs[50], bs[1949]), "discordant_claude(opus+,g38-|opus-,g38+)": disc(cl),
                              "discordant_gemini": disc(ge)}
    return out


# ------------------------------------------------------------------ task 2
def judge_agreement(its, jk):
    xs = [i for i in its if jk in i["L"]]
    res = {"n": len(xs)}
    if not xs:
        return res
    for f in FIELDS + ["natural>=4", "accept"]:
        if f == "natural>=4":
            pairs = [(i["L"]["opus"]["natural"] >= 4, i["L"][jk]["natural"] >= 4) for i in xs]
        elif f == "accept":
            pairs = [(accept(i["L"]["opus"]), accept(i["L"][jk])) for i in xs]
        else:
            pairs = [(bool(i["L"]["opus"][f]), bool(i["L"][jk][f])) for i in xs]
        res[f] = {"kappa": kappa(pairs), "agree": sum(a == b for a, b in pairs) / len(pairs),
                  "opus_pos": sum(a for a, _ in pairs) / len(pairs), "j_pos": sum(b for _, b in pairs) / len(pairs)}
    if "teacher_derived" in xs[0]["L"][jk]:
        pairs = [(bool(i["L"]["opus"]["teacher"]), bool(i["L"][jk]["teacher_derived"])) for i in xs]
        res["teacher_derived"] = {"kappa": kappa(pairs), "agree": sum(a == b for a, b in pairs) / len(pairs)}
    tp = sum(accept(i["L"]["opus"]) and accept(i["L"][jk]) for i in xs)
    jp = sum(accept(i["L"][jk]) for i in xs)
    op = sum(accept(i["L"]["opus"]) for i in xs)
    res["precision"] = (tp, jp)
    res["recall"] = (tp, op)
    return res


def cost_per_call(stage):
    rows = [r for r in load_store(stage).values() if r.get("api_ok") and not r.get("cache_hit")]
    return st.mean(r["cost"] for r in rows) if rows else None, len(rows)


def task2(items):
    hold = [i for i in items if not i["anchor_seed"]]
    out = {"n_holdout": len(hold), "n_native": sum(i["set"] == "native" for i in hold), "judges": {}, "cost": {},
           "by_family": {}}
    for jk in CAND:
        out["judges"][jk] = {"all": judge_agreement(hold, jk),
                             "native": judge_agreement([i for i in hold if i["set"] == "native"], jk)}
        out["cost"][jk] = cost_per_call(f"judge_{jk}")
        fam = {}
        for name, sel in (("haiku_items", lambda i: i["gen"] == "haiku"),
                          ("sonnet_items", lambda i: i["gen"] == "sonnet"),
                          ("gemini_items", lambda i: i["gen"] in ("gemini", "gemini31"))):
            xs = [i for i in hold if sel(i) and jk in i["L"]]
            if xs:
                pairs = [(accept(i["L"]["opus"]), accept(i["L"][jk])) for i in xs]
                fam[name] = {"n": len(xs), "kappa_accept": kappa(pairs),
                             "opus_accept": sum(a for a, _ in pairs) / len(xs),
                             "j_accept": sum(b for _, b in pairs) / len(xs)}
        out["by_family"][jk] = fam
    out["cost"]["solve_haiku"] = cost_per_call("solve_haiku")
    # answer screen: Haiku 2/2 agrees with the generator vs Opus blind-solve agrees
    ns = [i for i in hold if i["set"] == "native" and i["haiku_solve2"] is not None and i["opus_agrees"] is not None]
    pairs = [(bool(i["opus_agrees"]), bool(i["haiku_solve2"])) for i in ns]
    tp = sum(a and b for a, b in pairs)
    out["answer_screen"] = {"n": len(ns), "kappa": kappa(pairs) if pairs else None,
                            "precision": (tp, sum(b for _, b in pairs)), "recall": (tp, sum(a for a, _ in pairs))}
    return out


def cascade_eval(its, j1, j2=None, audit=0.05):
    """Simulate a cascade on items carrying Opus labels. j2=None: single cheap judge, Opus only on audit.
    j1+j2: both accept -> accept; both reject -> reject; disagree -> Opus decides. Returns precision vs Opus,
    recall, Opus share."""
    xs = [i for i in its if j1 in i["L"] and (j2 is None or j2 in i["L"])]
    if not xs:
        return None
    acc, tp, opus_calls = 0, 0, 0
    for i in xs:
        y = accept(i["L"]["opus"])
        a1 = accept(i["L"][j1])
        if j2 is None:
            dec = a1
        else:
            a2 = accept(i["L"][j2])
            if a1 == a2:
                dec = a1
            else:
                opus_calls += 1
                dec = y
        acc += dec
        tp += dec and y
    pos = sum(accept(i["L"]["opus"]) for i in xs)
    return {"n": len(xs), "accepted": acc, "precision": (tp, acc), "recall": (tp, pos),
            "opus_share": opus_calls / len(xs) + audit, "accept_rate": acc / len(xs)}


# ------------------------------------------------------------------ task 3
def task3():
    seeds = {s["seed_id"]: s for s in N.load_seeds()}
    gen = load_store("n_gen")
    jks = [k for k in ("opus_v2", "g36_v2", "g3fp_v2", "haiku_v3", "g31l_v3", "sonnet_v3") if (G.OUT / f"n_judge_{k}.jsonl").exists()]
    J_ = {k: load_store(f"n_judge_{k}") for k in jks}
    OS = load_store("n_opus_solve")
    HS = load_store("n_solve_haiku")
    arms = sorted({r["arm"] for r in gen.values()})
    items = []
    for key, r in gen.items():
        s = seeds[r["seed_id"]]
        it = {"key": key, "arm": r["arm"], "seed_id": r["seed_id"], "op": r["op"], "parsed": r.get("parsed"),
              "problem": r.get("problem"), "answer": r.get("answer"), "gen_cost": r.get("cost"),
              "gen_out": (r.get("usage") or {}).get("output_tokens"), "self_check": r.get("self_check"),
              "seed": s, "L": {}}
        if it["parsed"]:
            av = G.answer_values(it["answer"])
            it["leak_regex"] = bool(leak_regex(it["problem"]))
            it["answer_in_text"] = answer_in_text(it["problem"], av) if av else False
            it["chrf"] = chrf(it["problem"], s["text"])
            for k in jks:
                v = J_[k].get(key)
                if v and v.get("parsed"):
                    it["L"]["opus" if k == "opus_v2" else k] = G.labels(k, v["verdict"])
                    it.setdefault("jcost", {})[k] = v.get("cost")
            o = OS.get(f"{key}#s0")
            it["opus_agrees"] = bool(o["agrees_gen"]) if (o and o.get("api_ok")) else None
            it["opus_solve_cost"] = (o or {}).get("cost")
            hs = [HS.get(f"{key}#s{k}") for k in range(2)]
            it["haiku_solve2"] = all(h and h.get("agrees_gen") for h in hs) if all(hs) else None
        items.append(it)
    out = {"arms": {}, "paired": {}}
    for a in arms:
        its = [i for i in items if i["arm"] == a]
        P = [i for i in its if i["parsed"]]
        d = {"n": len(its), "parsed": (len(P), len(its)),
             "gen_cost_mean": st.mean(i["gen_cost"] or 0 for i in its),
             "gen_out_tokens_mean": st.mean(i["gen_out"] or 0 for i in its)}
        for jk in ("opus", "g36_v2", "g3fp_v2", "haiku_v3", "g31l_v3", "sonnet_v3"):
            X = [i for i in P if jk in i["L"]]
            if not X:
                continue
            nn = [i for i in X if i["op"] != "M1"]
            e = {"n": len(X)}
            for f in ("valid", "teacher", "absurd", "tail", "unansw", "fluency"):
                e[f] = (sum(bool(i["L"][jk][f]) for i in X), len(X))
            e["new_M2M6"] = (sum(bool(i["L"][jk]["new"]) for i in nn), len(nn))
            e["neardup_all"] = (sum(bool(i["L"][jk]["neardupj"]) or (i["chrf"] > .9 and i["op"] != "M1") for i in X),
                                len(X))
            e["neardup_M2M6"] = (sum(bool(i["L"][jk]["neardupj"]) or i["chrf"] > .9 for i in nn), len(nn))
            e["leak"] = (sum(i["leak_regex"] or bool(i["L"][jk]["leakj"]) for i in X), len(X))
            e["natural"] = st.mean(i["L"][jk]["natural"] for i in X)
            Y = [i for i in X if i["opus_agrees"] is not None]
            e["answer_correct"] = (sum(i["opus_agrees"] for i in Y), len(Y))
            e["yield"] = (sum(i["L"][jk]["valid"] and bool(i["opus_agrees"]) and not i["leak_regex"] for i in Y),
                          len(its))
            e["strict_yield"] = (sum(i["L"][jk]["valid"] and bool(i["opus_agrees"]) and not i["leak_regex"]
                                     and i["L"][jk]["teacher"] and not i["L"][jk]["leakj"]
                                     and (i["L"][jk]["new"] or i["op"] == "M1") for i in Y), len(its))
            e["gate"] = {"teacher>=70": e["teacher"][0] / max(1, e["teacher"][1]) >= .70,
                         "new>=60": e["new_M2M6"][0] / max(1, e["new_M2M6"][1]) >= .60,
                         "leak=0": e["leak"][0] == 0,
                         "absurd<=3": e["absurd"][0] / max(1, e["absurd"][1]) <= .03}
            d[jk] = e
        d["answer_in_text"] = (sum(i["answer_in_text"] for i in P), len(P))
        d["chrf_median"] = st.median(i["chrf"] for i in P) if P else None
        sc = [i["self_check"] for i in P if isinstance(i.get("self_check"), dict)]
        d["self_check_all_true"] = (sum(all(v is True for k, v in c.items() if k != "chuoi_phep_tinh_khac_bai_goc")
                                        for c in sc), len(sc))
        out["arms"][a] = d
    # paired McNemar vs v1 on Opus teacher / absurd / accept
    base = {i["seed_id"]: i for i in items if i["arm"] == "v1"}
    for a in arms:
        if a == "v1":
            continue
        res = {}
        for f in ("teacher", "absurd", "valid", "accept"):
            b = c = 0
            for i in items:
                if i["arm"] != a:
                    continue
                j = base.get(i["seed_id"])
                if not (j and "opus" in i["L"] and "opus" in j["L"]):
                    continue
                x = accept(i["L"]["opus"]) if f == "accept" else bool(i["L"]["opus"][f])
                y = accept(j["L"]["opus"]) if f == "accept" else bool(j["L"]["opus"][f])
                b += x and not y
                c += y and not x
            res[f] = {"arm_only": b, "v1_only": c, "p": mcnemar_p(b, c)}
        out["paired"][a] = res
    # judge cost per item on task 3 rows
    out["opus_judge_cost"] = st.mean([i["jcost"]["opus_v2"] for i in items if i.get("jcost", {}).get("opus_v2")]
                                     or [OPUS_JUDGE_USD])
    out["opus_solve_cost"] = st.mean([i["opus_solve_cost"] for i in items if i.get("opus_solve_cost")]
                                     or [OPUS_SOLVE_USD])
    return out, items


# ------------------------------------------------------------------ cascades with the answer screen and costs
COST = {"gen": 0.00115, "haiku_v3": 0.00051, "sonnet_v3": 0.0061, "g31l_v3": 0.0, "screen": 0.00047,
        "opus": 0.0146 + 0.0070}          # per raw item, direct prices (Batch halves every Claude term)


def final_truth(i):
    return accept(i["L"]["opus"]) and bool(i.get("opus_agrees")) and not i.get("leak_regex")


def policy(i, name):
    """-> (decision, opus_called, claude_cost_excl_gen). Every policy also needs the regex leak check and the
    Haiku 2-sample answer screen (2/2 agree with the generator)."""
    scr = bool(i.get("haiku_solve2")) and not i.get("leak_regex")
    L = i["L"]
    h, s, g = (accept(L[k]) if k in L else None for k in ("haiku_v3", "sonnet_v3", "g31l_v3"))
    y = final_truth(i)
    if name == "opus_all":
        return y, True, COST["opus"]
    if name == "haiku":
        return h and scr, False, COST["haiku_v3"] + COST["screen"]
    if name == "sonnet":
        return s and scr, False, COST["sonnet_v3"] + COST["screen"]
    if name == "haiku+g31l→opus":            # disagreement -> Opus
        if not scr:
            return False, False, COST["haiku_v3"] + COST["screen"]
        if h == g:
            return h, False, COST["haiku_v3"] + COST["screen"]
        return y, True, COST["haiku_v3"] + COST["screen"] + COST["opus"]
    if name == "haiku+sonnet→opus":
        if not scr:
            return False, False, COST["haiku_v3"] + COST["screen"]
        if h == s:
            return h, False, COST["haiku_v3"] + COST["sonnet_v3"] + COST["screen"]
        return y, True, COST["haiku_v3"] + COST["sonnet_v3"] + COST["screen"] + COST["opus"]
    if name == "haiku→sonnet→opus":           # Haiku rejects -> reject; Haiku accepts -> Sonnet; disagree -> Opus
        if not (scr and h):
            return False, False, COST["haiku_v3"] + COST["screen"]
        if s:
            return True, False, COST["haiku_v3"] + COST["sonnet_v3"] + COST["screen"]
        return y, True, COST["haiku_v3"] + COST["sonnet_v3"] + COST["screen"] + COST["opus"]
    if name == "haiku→opus":                  # Opus confirms every Haiku accept
        if not (scr and h):
            return False, False, COST["haiku_v3"] + COST["screen"]
        return y, True, COST["haiku_v3"] + COST["screen"] + COST["opus"]
    raise KeyError(name)


POLICIES = ["opus_all", "haiku", "sonnet", "haiku+g31l→opus", "haiku+sonnet→opus", "haiku→sonnet→opus", "haiku→opus"]


def cascades(sets: dict, audit: float = 0.05) -> dict:
    out = {}
    for sname, its in sets.items():
        its = [i for i in its if i.get("opus_agrees") is not None and i.get("haiku_solve2") is not None
               and all(k in i["L"] for k in ("opus", "haiku_v3", "sonnet_v3", "g31l_v3"))]
        for pn in POLICIES:
            dec = [policy(i, pn) for i in its]
            acc = sum(d for d, _, _ in dec)
            tp = sum(d and final_truth(i) for (d, _, _), i in zip(dec, its))
            pos = sum(final_truth(i) for i in its)
            per_raw = COST["gen"] + st.mean(c for _, _, c in dec) + audit * COST["opus"] * (pn != "opus_all")
            share = st.mean(o for _, o, _ in dec) + audit * (pn != "opus_all")
            rate = acc / len(its)
            out[f"{sname}|{pn}"] = {"n": len(its), "precision": (tp, acc), "recall": (tp, pos), "accept_rate": rate,
                                    "opus_share": share, "usd_per_raw": per_raw,
                                    "usd_per_10k_accepted_direct": 1e4 * per_raw / rate if rate else None,
                                    "usd_per_10k_accepted_batch": 1e4 * (COST["gen"] / 2 + (per_raw - COST["gen"]) / 2)
                                    / rate if rate else None}
    return out


def main():
    items = calib_table()
    t1 = {"g3fp_v2": task1(items, "g3fp_v2"), "g38_v2_partial": task1(items, "g38_v2")}
    rt = [i for i in items if "opusR_v2" in i["L"]]
    t1["opus_retest"] = {"n": len(rt), **{f: {"kappa": kappa([(bool(i["L"]["opus"][f]), bool(i["L"]["opusR_v2"][f]))
                                                              for i in rt]),
                                              "g1": sum(bool(i["L"]["opus"][f]) for i in rt),
                                              "retest": sum(bool(i["L"]["opusR_v2"][f]) for i in rt)}
                                          for f in ("valid", "teacher", "new", "absurd", "neardupj")},
                         "accept": {"kappa": kappa([(accept(i["L"]["opus"]), accept(i["L"]["opusR_v2"])) for i in rt]),
                                    "g1": sum(accept(i["L"]["opus"]) for i in rt),
                                    "retest": sum(accept(i["L"]["opusR_v2"]) for i in rt)}}
    t2 = task2(items)
    t3, n_items = task3()
    hold = [i for i in items if not i["anchor_seed"]]
    casc = {}
    for spec in [("haiku_v3", None), ("haikum_v3", None), ("sonnet_v3", None), ("g31l_v3", None), ("g38_v2", None),
                 ("haiku_v3", "g31l_v3"), ("haiku_v3", "g38_v2"), ("sonnet_v3", "g31l_v3"),
                 ("haikum_v3", "g31l_v3")]:
        casc[f"{spec[0]}+{spec[1]}"] = {"calib": cascade_eval(hold, *spec),
                                        "task3": cascade_eval([i for i in n_items if i.get("parsed")], *spec)}
    nat_cal = [i for i in hold if i["set"] == "native"]
    t3p = [i for i in n_items if i.get("parsed")]
    casc2 = cascades({"calib_native": nat_cal, "task3_all": t3p, "task3_v3": [i for i in t3p if i["arm"] == "v3"]})
    res = {"task1": t1, "task2": t2, "task3": t3, "cascade": casc, "cascade_final": casc2}
    json.dump(res, open(G.OUT / "g2_metrics.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False,
              default=str)
    with open(G.OUT / "items_task3.jsonl", "w", encoding="utf-8") as f:
        for i in n_items:
            f.write(json.dumps({k: v for k, v in i.items() if k != "seed"}, ensure_ascii=False, default=str) + "\n")
    print(json.dumps(res, ensure_ascii=False, default=str)[:3000])


if __name__ == "__main__":
    main()
