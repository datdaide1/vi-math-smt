"""Stage ``align``: load the three Vietnamese train files, align each row to its English original, assign ids.

Alignment: GSM8K by ``index`` (R1 §1: VI and EN both 0..7472, verified); MATH algebra/prealgebra by ``index`` within
subject (R2: only precalculus and c&p 355-357 are out of alignment, neither is in the pool). Alignment is *checked*
here, not assumed: the VI ``ground_truth`` must equal the EN one after light normalisation, and every mismatch is
recorded for the detect stage (R1's misaligned GSM8K rows 220-225/2577/3077 show up this way only through the VI
``####`` line and the content, because their ``ground_truth`` field was copied from EN).
"""
from __future__ import annotations

import json
import re

from .common import EN_GSM8K, EN_MATH, SUBJECTS, VI_GSM8K, VI_MATH, WORK, is_dev, read_jsonl, seed_id, write_jsonl


def _norm_gt(s) -> str:
    s = str(s or "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\(?:text|mbox|textbf|mathrm)\{[^{}]*\}", "", s)
    s = re.sub(r"[\x00-\x1f]", "", s)
    return re.sub(r"\s+|\\!|\\,|\^\\circ|°", "", s)


def load_aligned() -> list[dict]:
    rows: list[dict] = []
    vi = read_jsonl(VI_GSM8K)
    en = read_jsonl(EN_GSM8K)
    assert len(vi) == len(en) == 7473, (len(vi), len(en))
    for i, (v, e) in enumerate(zip(vi, en)):
        assert int(v["index"]) == int(e["index"]) == i
        src = "vi_gsm8k_train"
        sid = seed_id(src, i)
        rows.append({
            "seed_id": sid, "source": src, "dataset": "gsm8k", "subject": None, "level": None, "index": i,
            "dev": is_dev(sid),
            "problem_vi": v["question"], "solution_vi": v["answer"], "gold_raw_vi": str(v.get("ground_truth")),
            "problem_en": e["question"], "solution_en": e["answer"], "gold_raw_en": str(e["ground_truth"]),
            "gt_field_match": _norm_gt(v.get("ground_truth")).replace(",", "") ==
                              _norm_gt(e["ground_truth"]).replace(",", ""),
        })
    for subj in SUBJECTS:
        vi = read_jsonl(str(VI_MATH).format(subject=subj))
        en = json.load(open(str(EN_MATH).format(subject=subj), encoding="utf-8"))
        assert len(vi) == len(en), (subj, len(vi), len(en))
        src = f"vi_math_train_{subj}"
        for i, (v, e) in enumerate(zip(vi, en)):
            assert int(v["index"]) == int(e["index"]) == i, (subj, i)
            sid = seed_id(src, i)
            rows.append({
                "seed_id": sid, "source": src, "dataset": "math", "subject": subj, "level": e.get("level"),
                "index": i, "dev": is_dev(sid),
                "problem_vi": v["problem"], "solution_vi": v["solution"], "gold_raw_vi": str(v.get("ground_truth")),
                "problem_en": e["problem"], "solution_en": e["solution"], "gold_raw_en": str(e["ground_truth"]),
                "level_vi": v.get("level"),
                "gt_field_match": _norm_gt(v.get("ground_truth")) == _norm_gt(e["ground_truth"]),
            })
    ids = [r["seed_id"] for r in rows]
    assert len(set(ids)) == len(ids), "seed_id collision"
    return rows


def run() -> dict:
    rows = load_aligned()
    write_jsonl(WORK / "aligned.jsonl", rows)
    from collections import Counter
    c = Counter((r["source"], r["dev"]) for r in rows)
    out = {"n": len(rows), "by_source_dev": {f"{s}|{'dev' if d else 'pool'}": n for (s, d), n in sorted(c.items())},
           "gt_field_mismatch": [(r["source"], r["index"]) for r in rows if not r["gt_field_match"]]}
    print(json.dumps(out, ensure_ascii=False)[:3000])
    return out
