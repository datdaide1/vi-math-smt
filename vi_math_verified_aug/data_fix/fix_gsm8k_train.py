"""Task W3-3a: Vi-GSM8K train -> v1.1.

1. deterministic: final line ``#### 9.000`` (VN thousands dot) -> ``#### 9000`` when it equals the EN gold;
   ``ground_truth`` := EN gold (canonical).
2. re-translate from English with MTA-vi (NIM translator + NIM solver G2 + Gemini judge G3, <=10 RPM):
   * the 14 corrupt rows of R1 §6 (220–225, 2577, 3077, 2345, 3539, 5085, 7454, 1801, 2911),
   * the 2 half-English rows (3424, 6465),
   * R1 hand-annotated translation errors (major/critical),
   * every row failing the deterministic G1 gate on the existing translation (numbers, unit/glossary
     forbid-list, CR-1/2/3/5, final line).
3. ACCEPT -> replace the row. DROP: rows known to be corrupt are excluded from the clean file (listed in the
   datacard); pattern-only rows keep the original text with flag ``mta_failed``.

Run: python -X utf8 -I vi_math_verified_aug/data_fix/fix_gsm8k_train.py [--dry]
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vi_math_verified_aug.data_fix import mta_vi, nim_client  # noqa: E402
from vi_math_verified_aug.data_fix.common import (  # noqa: E402
    CLEAN, DATA, canon_num, datacard, gsm8k_final, load_annotations, read_jsonl, rel, write_jsonl,
)

VERSION = "1.1"
OUT = CLEAN / "vi_gsm8k_train"
SCRIPT = "vi_math_verified_aug/data_fix/fix_gsm8k_train.py"
VI_PATH = DATA / "translation" / "gsm8k" / "gsm8k_train_vietnamese.jsonl"
EN_PATH = DATA / "generate" / "gsm8k" / "train.jsonl"
CORRUPT = [220, 221, 222, 223, 224, 225, 2577, 3077, 2345, 3539, 5085, 7454, 1801, 2911]
HALF_EN = [3424, 6465]
EN_STOP = re.compile(r"\b(the|and|of|is|are|how|many|much|what|each|with|has|have|will|does|she|he|they)\b", re.I)


def main(dry: bool = False) -> None:
    vi = read_jsonl(VI_PATH)
    en = read_jsonl(EN_PATH)
    assert len(vi) == len(en) == 7473
    assert all(str(v["index"]) == str(e["index"]) == str(i) for i, (v, e) in enumerate(zip(vi, en)))
    ann = load_annotations("vi_gsm8k_train")
    r1_bad = sorted(i for i, a in ann.items() if a.get("severity") in ("major", "critical")
                    and "translation" in (a.get("origin") or []))

    rows, changed, det_fixed = [], [], []
    for i, (v, e) in enumerate(zip(vi, en)):
        gold = canon_num(e["ground_truth"])
        ans = v["answer"]
        fin = gsm8k_final(ans)
        if fin is not None and canon_num(fin) != gold and re.fullmatch(r"\d{1,3}(\.\d{3})+", fin) \
                and fin.replace(".", "") == gold:
            ans = re.sub(r"####\s*.+?\s*$", "#### " + gold, ans)
            det_fixed.append(i)
            changed.append({"id": f"gsm8k/train/{i}", "fields": ["answer"],
                            "reason": f"final line '#### {fin}' (VN thousands dot) -> '#### {gold}'"})
        rows.append({"id": f"gsm8k/train/{i}", "index": i, "question": v["question"], "answer": ans,
                     "ground_truth": gold, "en_question": e["question"], "en_answer": e["answer"],
                     "provenance": "original", "flags": []})

    # ---- deterministic gate over the whole corpus (no API)
    g1_fail = {}
    for r in rows:
        it = mta_vi.Item(r["id"], "gsm8k", r["en_question"], r["en_answer"], r["ground_truth"])
        it.vi_problem, it.vi_solution = r["question"], r["answer"]
        f = mta_vi.g1(it)
        words = re.findall(r"[A-Za-zÀ-ỹ]+", r["question"])
        if words and len(EN_STOP.findall(r["question"])) / len(words) > 0.08:
            f.append("G0 English left in question")
        if f:
            g1_fail[r["index"]] = f
    must = set(CORRUPT) | set(HALF_EN) | set(r1_bad)
    target = sorted(must | set(g1_fail))
    print(f"targets: {len(target)} (corrupt {len(CORRUPT)}, half-EN {len(HALF_EN)}, R1 {len(r1_bad)}, "
          f"G1-fail {len(g1_fail)})")
    if dry:
        return

    items = [mta_vi.Item(f"gsm8k/train/{i}", "gsm8k", en[i]["question"], en[i]["answer"],
                         canon_num(en[i]["ground_truth"])) for i in target]
    res = mta_vi.run_batch(items, OUT / "_mta_cache.jsonl", workers=4)

    dropped, kept_flagged, retranslated = [], [], []
    for i in target:
        r = rows[i]
        rec = res[r["id"]]
        why = ("corrupt (R1 §5)" if i in CORRUPT else "half-English" if i in HALF_EN else
               "R1 annotation" if i in r1_bad else "G1 gate: " + "; ".join(g1_fail.get(i, []))[:300])
        if rec["status"].startswith("ACCEPT"):
            r["question"], r["answer"] = rec["vi_problem"], rec["vi_solution"]
            r["provenance"] = "mta_vi"
            r["mta"] = rec["provenance"]
            retranslated.append(i)
            changed.append({"id": r["id"], "fields": ["question", "answer"],
                            "reason": f"re-translated from EN with MTA-vi ({rec['status']}); trigger: {why}"})
        elif i in must:
            dropped.append({"id": r["id"], "reason": f"{why}; MTA-vi {rec['status']} after repair rounds",
                            "last_fails": (rec["history"][-1].get("fails") if rec["history"] else None)})
            r["_drop"] = True
        else:
            r["flags"].append("mta_failed: " + "; ".join(g1_fail.get(i, []))[:300])
            kept_flagged.append(i)
    clean = [r for r in rows if not r.pop("_drop", False)]
    out_p = OUT / f"vi_gsm8k_train_v{VERSION}.jsonl"
    write_jsonl(out_p, clean)
    stat = Counter(res[f"gsm8k/train/{i}"]["status"] for i in target)
    datacard(out_p, name="Vi-GSM8K-train", version=VERSION,
             description="Vietnamese GSM8K train with corrupt/misaligned rows re-translated from English via MTA-vi "
                         "(masking + triggered glossary + G0/G1/G2/G3 gates). ground_truth from English GSM8K.",
             sources=[{"path": rel(VI_PATH), "role": "Vietnamese translation (input, unmodified)"},
                      {"path": rel(EN_PATH), "role": "English GSM8K train (gold + source for re-translation)"}],
             transformations=[
                 "ground_truth := canonical EN gold",
                 "fix: '#### 9.000'-style final lines -> EN gold",
                 "deterministic G1 gate over all rows (numbers, glossary/units forbid-list, CR-1/2/3/5, final line)"
                 " + English-leak check",
                 "re-translate targets with MTA-vi: translator nvidia/nemotron-3-ultra-550b-a55b (NIM), "
                 "solver deepseek-ai/deepseek-v4.1-flash (NIM, G2), judge gemini-3.5-flash-lite (G3); <=2 post-edit "
                 "rounds; temperature 0, seed 42",
                 "DROP of a known-corrupt row -> row excluded; DROP of a pattern-only row -> original kept, flagged",
             ],
             counts={"rows_in": len(vi), "rows_out": len(clean), "targets": len(target),
                     "targets_by_reason": {"corrupt": len(CORRUPT), "half_english": len(HALF_EN),
                                           "r1_annotation": len(r1_bad), "g1_gate": len(g1_fail)},
                     "mta_status": dict(stat), "retranslated": len(retranslated),
                     "deterministic_final_line_fixes": len(det_fixed), "dropped": len(dropped),
                     "kept_with_flag": len(kept_flagged)},
             changed=changed, license="MIT (GSM8K); translation: project", script=SCRIPT,
             extra={"dropped_rows": dropped, "kept_flagged_ids": [f"gsm8k/train/{i}" for i in kept_flagged],
                    "api_calls_this_run": dict(nim_client.CALLS)})
    print("rows", len(clean), "retranslated", len(retranslated), "dropped", len(dropped), "flagged",
          len(kept_flagged), dict(stat), nim_client.CALLS)


if __name__ == "__main__":
    main(dry="--dry" in sys.argv)
