"""FP probe: grade each MATH gold (EN) against the boxed gold of the NEXT item. Any True with a different value
is a false positive.  Pairs whose golds are mathematically equal are expected True; they are printed for review."""
import glob, json, os, sys
sys.path.insert(0, ".")
from vi_math_verified_aug.eval import math_verify as mv
mv.set_backend(sys.argv[1])
n = t = 0
for f in sorted(glob.glob("data/generate/MATH/test/*.json")):
    en = json.load(open(f, encoding="utf-8"))
    golds = [(mv._boxes(e["solution"])[-1][0], e["problem"]) for e in en]
    for k in range(len(golds) - 1):
        (gr, gp), (pr, _) = golds[k], golds[k + 1]
        g = mv.infer_gold(gr, problem=gp, locale="en")
        ok = mv.grade("\\boxed{" + pr + "}", g)["correct"]
        n += 1
        if ok:
            t += 1
            if gr.replace(" ", "") != pr.replace(" ", ""):
                print("TRUE-DIFFERENT-STRING", os.path.basename(f), k, repr(gr), "<-", repr(pr))
print(mv.BACKEND, "pairs", n, "graded True", t)
