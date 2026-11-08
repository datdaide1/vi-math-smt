"""QA-T: recompute eval metrics from preds.jsonl independently of evalrun.runner.summarize, and profile responses.

    python -m vi_math_verified_aug.qa.recompute_metrics PREDS.jsonl [--gold data/eval/vi_primary_1000.jsonl]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict

GREEDY_K = -1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("preds")
    ap.add_argument("--gold")
    ap.add_argument("--regrade", action="store_true")
    a = ap.parse_args(argv)
    rows = [json.loads(x) for x in open(a.preds, encoding="utf-8") if x.strip()]
    gk = sorted({r["sample_k"] for r in rows})
    print("sample_k values", gk, "rows", len(rows))
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["dataset"]][r["item_id"]].append(r)
    for ds, items in by.items():
        ks = Counter(len(v) for v in items.values())
        p1 = []; gr = []; majs = []; majeq = []
        for iid, rs in items.items():
            smp = [r for r in rs if r["sample_k"] >= 0]
            g = [r for r in rs if r["sample_k"] < 0]
            dup = Counter(r["sample_k"] for r in rs)
            assert max(dup.values()) == 1, (iid, dup)
            p1.append(sum(r["correct"] for r in smp) / len(smp))
            if g:
                gr.append(g[0]["correct"])
            spans = Counter(r.get("pred_span") for r in smp if r.get("pred_span"))
            if spans:
                top, c = spans.most_common(1)[0]
                ties = [s for s, n in spans.items() if n == c]
                majs.append(int(any(r["correct"] for r in smp if r.get("pred_span") == top)))
                # tie-robust: correct iff a correct answer is among the modal spans and ties broken in its favour
                majeq.append(len(ties))
            else:
                majs.append(0)
        n = len(items)
        print(f"{ds}: items {n}, rows/item {dict(ks)}, pass1 {sum(p1)/n:.5f}, greedy {sum(gr)/max(1,len(gr)):.5f}, "
              f"maj@K {sum(majs)/n:.5f}, items with tied modal span {sum(1 for t in majeq if t > 1)}")
    fr = Counter((r["decode"], r["finish_reason"]) for r in rows)
    print("finish reasons", dict(fr))
    flags = Counter(r.get("grader_flag") for r in rows)
    print("grader flags", flags.most_common(12))
    return 0


if __name__ == "__main__":
    sys.exit(main())
