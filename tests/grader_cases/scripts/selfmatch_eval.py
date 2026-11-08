"""Self-match over the frozen eval sets data/eval/{vi,en}_*.jsonl through the eval-runner API grade(resp, record["gold"]).
Per record, three predictions must grade correct:
  canonical           \\boxed{answer_canonical}
  gold_raw            \\boxed{gold_raw}            (skipped for VI text golds localised from English, e.g. east -> đông)
  gold_raw/no-locale  the same against the gold dict with its "locale" key removed (how X3's base-check passes it)
usage (repo root): python tests/grader_cases/scripts/selfmatch_eval.py [sympy|math-verify]"""
import glob
import json
import os
import sys

sys.path.insert(0, ".")
from vi_math_verified_aug.eval import math_verify as mv  # noqa: E402

if len(sys.argv) > 1:
    mv.set_backend(sys.argv[1])
print("grader", mv.GRADER_VERSION, "backend", mv.BACKEND)
B = "\\boxed{"
tot = ok_tot = localised = 0
for f in sorted(glob.glob("data/eval/*.jsonl")):
    n = ok = 0
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        g = r["gold"]
        loc_text = r["lang"] == "vi" and g["answer_type"] == "text" and \
            mv.infer_gold(r["gold_raw"])["answer_canonical"] != g["answer_canonical"]
        localised += loc_text
        checks = [("canonical", B + g["answer_canonical"] + "}", g)]
        checks += [(f"accept_also[{i}]", B + a["answer_canonical"] + "}", g)
                   for i, a in enumerate(g.get("accept_also") or [])]
        if not loc_text:
            checks += [("gold_raw", B + str(r["gold_raw"]) + "}", g),
                       ("gold_raw/no-locale", B + str(r["gold_raw"]) + "}",
                        {k: v for k, v in g.items() if k != "locale"})]
        n += 1
        fails = [(name, mv.grade(p, gg)) for name, p, gg in checks]
        fails = [(name, v) for name, v in fails if not v["correct"]]
        for name, v in fails:
            print(f"  FAIL {os.path.basename(f)} {r['id']} via={name} gold={json.dumps(g, ensure_ascii=False)} "
                  f"gold_raw={r['gold_raw']!r} span={v['span']!r} flag={v['flag']}")
        ok += not fails
    tot += n
    ok_tot += ok
    print(f"{os.path.basename(f):32} {ok}/{n}")
print(f"TOTAL {ok_tot}/{tot} = {ok_tot / tot:.4%}   (localised VI text golds: gold_raw check skipped on {localised})")
