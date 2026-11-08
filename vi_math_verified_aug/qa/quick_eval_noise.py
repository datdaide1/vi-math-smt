"""QA-T: false-abort rate of the 16-item quick-eval guard (guards.midrun_verdict QUICK_ACC_DROP, max_acc_drop 0.10).

Two decodes of the SAME model (sample_k 0 vs 1, T=0.6, base Qwen3-0.6B 4-shot, real preds) stand in for "step-0" and
"mid-run" models with identical true accuracy.  We draw 16 items (fixed per run, like dev_items[:16]) and count how
often acc_mid < acc_base - 0.10.  Also: the probability that at least one of 3 checks (25 %, 50 %, end) trips when
the three mid-run decodes are independent draws (k = 1, 2, 3 vs k = 0).

    python -m vi_math_verified_aug.qa.quick_eval_noise jobs/outputs/t1r-eval/ev-base-qwen3-0.6b-4shot-primary/preds.jsonl
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict


def main(path: str, n_items: int = 16, thr: float = 0.10, reps: int = 20000) -> None:
    by = defaultdict(dict)
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["dataset"] == "gsm8k_platinum":
            by[r["item_id"]][r["sample_k"]] = r["correct"]
    ids = [i for i, d in by.items() if all(k in d for k in (0, 1, 2, 3))]
    rng = random.Random(0)
    one, three = 0, 0
    for _ in range(reps):
        s = rng.sample(ids, n_items)
        base = sum(by[i][0] for i in s) / n_items
        trips = [sum(by[i][k] for i in s) / n_items < base - thr - 1e-9 for k in (1, 2, 3)]
        one += trips[0]
        three += any(trips)
    print(f"items {len(ids)}; P(single check trips | equal accuracy) = {one / reps:.3f}; "
          f"P(any of 3 checks trips) = {three / reps:.3f}  (n={n_items}, thr={thr})")


if __name__ == "__main__":
    main(sys.argv[1], *(int(x) for x in sys.argv[2:3]))
