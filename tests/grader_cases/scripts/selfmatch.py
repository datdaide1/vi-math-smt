"""Self-match: every gold must match itself.

MATH: gold string = last \\boxed{} of the reference solution (the dataset's own gold definition), English from
data/generate/MATH/test/<subj>.json attached by position, Vietnamese from data/test/math/<subj>_vi.jsonl.
GSM8K: ground_truth.  Each gold is typed once with infer_gold(); the prediction is the gold string inside a box
(and, separately, the full reference solution, which goes through extraction).
usage: python selfmatch.py {sympy|math-verify} CROSS_OUT.jsonl EXC_OUT.jsonl
"""
import collections
import glob
import json
import os
import sys
import time

sys.path.insert(0, ".")
from vi_math_verified_aug.eval import math_verify as mv  # noqa: E402

mv.set_backend(sys.argv[1])
cross_out = open(sys.argv[2], "w", encoding="utf-8")
exc_out = open(sys.argv[3], "w", encoding="utf-8")
print("backend", mv.BACKEND)
BOX = "\\boxed{"


def last_box(sol):
    b = mv._boxes(sol)
    return b[-1][0] if b else None


t0 = time.time()
stats = collections.Counter()
types = collections.Counter()
exc = []


def check(src, key, raw, problem, locale, solution=None):
    if raw is None:
        stats[(src, "nobox")] += 1
        exc.append({"src": src, "key": key, "kind": "no boxed answer in the reference solution"})
        return
    g = mv.infer_gold(raw, problem=problem, locale=locale)
    types[(src, g["answer_type"])] += 1
    r = mv.grade(BOX + raw + "}", g)
    stats[(src, "n")] += 1
    stats[(src, "ok")] += r["correct"]
    if not r["correct"]:
        exc.append({"src": src, "key": key, "kind": "boxed-gold", "raw": raw, "gold": g, "span": r["span"],
                    "flag": r["flag"]})
    if solution is not None:
        r2 = mv.grade(solution, g)
        stats[(src, "sol_n")] += 1
        stats[(src, "sol_ok")] += r2["correct"]
        if not r2["correct"]:
            exc.append({"src": src, "key": key, "kind": "full-solution", "raw": raw, "gold": g,
                        "span": r2["span"], "method": r2["method"], "flag": r2["flag"]})


for f in sorted(glob.glob("data/test/math/*_vi.jsonl")):
    subj = os.path.basename(f)[:-len("_vi.jsonl")]
    vi = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    en = json.load(open(f"data/generate/MATH/test/{subj}.json", encoding="utf-8"))
    assert len(vi) == len(en), (subj, len(vi), len(en))
    for i, (v, e) in enumerate(zip(vi, en)):
        ge, gv = last_box(e["solution"]), last_box(v["solution"])
        check("math-en", f"{subj}#{i}", ge, e["problem"], "en", e["solution"])
        check("math-vi", f"{subj}#{i}", gv, v["problem"], "mixed", v["solution"])
        if ge is not None and gv is not None:
            stats[("cross", "n")] += 1
            ok = mv.grade(BOX + gv + "}", mv.infer_gold(ge, problem=e["problem"], locale="en"))["correct"]
            stats[("cross", "ok")] += ok
            if not ok:
                cross_out.write(json.dumps({"subj": subj, "i": i, "en": ge, "vi": gv}, ensure_ascii=False) + "\n")
for k, l in enumerate(open("data/test/gsm8k/gsm8k_test_vietnamese.jsonl", encoding="utf-8")):
    d = json.loads(l)
    check("gsm8k-vi", f"#{k}", d["ground_truth"], d["question"], "mixed", d["answer"])

for src in ("math-en", "math-vi", "gsm8k-vi", "cross"):
    n, ok = stats[(src, "n")], stats[(src, "ok")]
    sn, sok = stats[(src, "sol_n")], stats[(src, "sol_ok")]
    print(f"{src:9} boxed-gold self-match {ok}/{n} = {ok / max(n, 1):.4%}   full-solution {sok}/{sn}   "
          f"no-box {stats[(src, 'nobox')]}")
print("types", sorted(types.items()))
print("EXCEPTIONS", len(exc))
for x in exc:
    exc_out.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(" ", json.dumps(x, ensure_ascii=False)[:300])
print("time %.0fs" % (time.time() - t0))
