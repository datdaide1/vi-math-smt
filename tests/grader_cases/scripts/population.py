import json, glob, os, sys, collections
sys.path.insert(0, ".")
from vi_math_verified_aug.eval import math_verify as mv
print("backend", mv.BACKEND)
cnt = collections.Counter(); dis = []
for f in sorted(glob.glob("jobs/outputs/w2-pilot-v1/results/*/preds.jsonl")):
    model = os.path.basename(os.path.dirname(f))
    for l in open(f, encoding="utf-8"):
        d = json.loads(l)
        r = mv.grade(d["output"], {"answer_type": "int", "answer_canonical": d["gold"], "locale": "mixed"})
        cnt["n"] += 1; cnt["new"] += r["correct"]; cnt["old"] += d["correct"]
        cnt["method:" + r["method"]] += 1
        if r["flag"]: cnt["flag:" + r["flag"]] += 1
        if r["correct"] != d["correct"]:
            dis.append((model, d["test_index"], d["gold"], d["pred"][:30], d["correct"], r["correct"], r["method"], r["span"], r["flag"], d["output"][-160:]))
print(dict(cnt))
for x in dis: print("DIS", x[:9]); print("   ", repr(x[9]))
