"""Step 2: base sets with English golds attached and alignment checked.

Run from the repo root with the project venv::

    .venv/Scripts/python.exe -m vi_math_verified_aug.eval_sets.build_base

Outputs (``data/eval/_work/``):
- ``gsm8k_platinum_base.jsonl``  1,209 parallel items (VI + EN text, Platinum gold typed ``int``)
- ``math_algprealg_base.jsonl``  2,058 parallel items (VI after deterministic repairs + EN, typed gold),
                                 ``in_math500`` marks the 206 MATH-500 items
- ``base_checks.json``           every alignment check with its counts (quoted in the report)
"""
from __future__ import annotations

import collections
import difflib
import json
import re

from vi_math_verified_aug.eval import math_verify as mv
from vi_math_verified_aug.eval_sets.common import (
    ROOT, SRC, WORK, VI_GSM8K_TEST, VI_MATH_TEST, all_boxed, deterministic_repair, digit_tokens, last_boxed,
    parity, read_jsonl, write_json, write_jsonl, ws,
)

SUBJ_NAME = {"algebra": "Algebra", "prealgebra": "Prealgebra"}

# Gold corrections for MATH's "last \boxed" convention (found by the multi-\boxed probe; each read by hand).
GOLD_OVERRIDES = {
    "math-test-algebra-0414": ("-2, 1", {"answer_canonical": "-2; 1", "answer_type": "set"},
                               "problem asks for all solutions; the solution boxes -2 and 1 and both check, but the "
                               "last \\boxed holds only 1"),
    "math-test-algebra-0333": ("-3", {"answer_canonical": "-3", "answer_type": "int",
                                      # MATH's own solution ends "Note: \boxed{3} is also accepted as an answer."
                                      "accept_also": [{"answer_canonical": "3", "answer_type": "int"}]},
                               "solution derives d = \\boxed{-3}; the last \\boxed is a note ('3 is also accepted'), "
                               "kept as gold.accept_also because the source accepts it (mathematically d = -3)"),
}
GOLD_LOCALE = "mixed"  # every gold dict stored under data/eval/_work carries this; freeze sets vi=mixed, en=en


def _gsm_gold(answer: str) -> str:
    g = answer.split("####")[-1].strip().replace(",", "")
    assert re.fullmatch(r"-?\d+", g), g
    return g


def build_gsm8k(checks: dict) -> list[dict]:
    plat = read_jsonl(SRC / "gsm8k_platinum_test.jsonl")
    en = read_jsonl(SRC / "gsm8k_test.jsonl")
    vi = read_jsonl(VI_GSM8K_TEST)
    local_en = json.loads((ROOT / "data" / "generate" / "gsm8k" / "test.json").read_text(encoding="utf-8"))
    c = checks.setdefault("gsm8k", {})
    c["rows"] = {"platinum": len(plat), "openai_test": len(en), "vi_test": len(vi), "local_en": len(local_en)}
    # (a) VI file is positionally aligned with the English GSM8K test
    c["local_en_equals_openai_question"] = sum(a["question"] == b["question"] for a, b in zip(en, local_en))
    c["vi_ground_truth_equals_en_gold"] = sum(
        str(v["ground_truth"]).replace(",", "") == _gsm_gold(e["answer"]) for v, e in zip(vi, en))
    c["vi_final_line_equals_en_gold"] = sum(
        _gsm_gold(v["answer"].replace(".", "")) == _gsm_gold(e["answer"]) if "####" in v["answer"] else False
        for v, e in zip(vi, en))
    # (b) Platinum -> GSM8K test index by exact question text
    pos = collections.defaultdict(list)
    for i, e in enumerate(en):
        pos[e["question"]].append(i)
    idx = []
    for p in plat:
        hits = pos.get(p["question"], [])
        assert len(hits) == 1, p["question"][:80]
        idx.append(hits[0])
    c["platinum_exact_question_matches"] = len(idx)
    c["platinum_unique_indices"] = len(set(idx))
    c["platinum_order_monotone"] = all(a < b for a, b in zip(idx, idx[1:]))
    c["removed_by_platinum"] = len(en) - len(set(idx))
    c["platinum_status"] = dict(collections.Counter(p["cleaning_status"] for p in plat))

    rows, revised, num_mismatch = [], [], 0
    for p, i in zip(plat, idx):
        gold = _gsm_gold(p["answer"])
        orig_gold = _gsm_gold(en[i]["answer"])
        status = p["cleaning_status"]
        if status == "revised":
            revised.append({"gsm8k_index": i, "orig_gold": orig_gold, "platinum_gold": gold})
        else:
            assert gold == orig_gold, (i, gold, orig_gold)
        par = parity(p["question"], vi[i]["question"])
        num_mismatch += not par["numbers_ok"]
        rows.append({
            "id": f"gsm8k-test-{i:04d}",
            "dataset": "gsm8k_platinum",
            "gsm8k_index": i,
            "platinum_status": status,
            "question_en": p["question"],
            "solution_en": p["answer"],
            "question_vi": vi[i]["question"],
            # VI solution text explains the *original* gold; for revised items it is stale, so it is dropped
            "solution_vi": None if status == "revised" else vi[i]["answer"],
            "gold_raw": gold,
            "gold_orig_gsm8k": orig_gold,
            "gold": {"answer_canonical": gold, "answer_type": "int", "locale": GOLD_LOCALE},
            "subject": None,
            "level": None,
            "n_steps": p["answer"].split("####")[0].strip().count("\n") + 1,
            "deterministic_fixes": [],
            "number_parity_ok": par["numbers_ok"],
            "number_parity_diff": par["numbers"],
        })
    c["revised"] = revised
    c["number_multiset_mismatch_vi_vs_en"] = num_mismatch
    c["gold_types"] = dict(collections.Counter(r["gold"]["answer_type"] for r in rows))
    return rows


def _norm_boxed(s: str) -> str:
    s = re.sub(r"\\[dt]frac", r"\\frac", s or "")
    s = re.sub(r"\\(?:text|mbox|textbf|mathrm)\{[^{}]*\}", "", s)
    s = re.sub(r"\\[!,;: ]|\s|\^\{?\\circ\}?|\\%|%|\\\$|\$", "", s)
    return s


def build_math(checks: dict) -> list[dict]:
    m500 = read_jsonl(SRC / "math500_test.jsonl")
    m500_by_text = {}
    for r in m500:
        if r["subject"] in ("Algebra", "Prealgebra"):
            m500_by_text[ws(r["problem"])] = r
    c = checks.setdefault("math", {})
    c["math500_algprealg"] = dict(collections.Counter(r["subject"] for r in m500_by_text.values()))
    rows = []
    matched_m500 = set()
    for subj in ("algebra", "prealgebra"):
        en = read_jsonl(SRC / f"hendrycks_math_{subj}_test.jsonl")
        vi = read_jsonl(VI_MATH_TEST[subj])
        local = json.loads((ROOT / "data" / "generate" / "MATH" / "test" / f"{subj}.json").read_text(encoding="utf-8"))
        cs = c.setdefault(subj, {})
        cs["rows"] = {"en_hf": len(en), "vi": len(vi), "local_en": len(local)}
        assert len(en) == len(vi) == len(local)
        cs["local_en_equals_hf_problem"] = sum(a["problem"] == b["problem"] for a, b in zip(en, local))
        cs["level_match"] = sum(a["level"] == b["level"] for a, b in zip(en, vi))
        cs["type_match"] = sum(a["type"] == b["type"] for a, b in zip(en, vi))
        boxed_eq, boxed_grader_eq, mism = 0, 0, []
        fixes_n = collections.Counter()
        for k, (e, v) in enumerate(zip(en, vi)):
            key = ws(e["problem"])
            m = m500_by_text.get(key)
            if m is not None:
                matched_m500.add(key)
            en_boxed = last_boxed(e["solution"])
            gold_raw = m["answer"] if m is not None else en_boxed
            gold = mv.infer_gold(gold_raw, problem=e["problem"], locale="mixed")
            rid = f"math-test-{subj}-{k:04d}"
            override_reason = None
            if rid in GOLD_OVERRIDES:
                gold_raw, gold, override_reason = GOLD_OVERRIDES[rid]
            # deterministic repairs (VI only; English is the reference)
            ref = e["problem"] + "\n" + e["solution"]
            pv, lp = deterministic_repair(v["problem"], ref)
            sv, ls = deterministic_repair(v["solution"], ref)
            fixes = [f"problem:{x}" for x in lp] + [f"solution:{x}" for x in ls]
            for f in fixes:
                fixes_n[f] += 1
            vi_boxed = last_boxed(sv)
            same = _norm_boxed(vi_boxed) == _norm_boxed(en_boxed)
            boxed_eq += same
            g_ok = vi_boxed is not None and mv.is_correct(f"\\boxed{{{vi_boxed}}}", gold)
            boxed_grader_eq += g_ok
            if not (same or g_ok):
                mism.append({"pos": k, "en": en_boxed, "vi": vi_boxed})
            par = parity(e["problem"], pv)
            n_en_box = len(set(_norm_boxed(b) for b in all_boxed(e["solution"])))
            # text answers localised by the translation ("Monday" -> "Thứ Hai"): the VI set keeps the VI wording
            gold_vi = None
            if gold["answer_type"] == "text" and vi_boxed is not None and not g_ok:
                gvi = mv.infer_gold(vi_boxed, locale="mixed")
                gold_vi = {"answer_canonical": gvi["answer_canonical"], "answer_type": gvi["answer_type"],
                           "locale": GOLD_LOCALE}
            rows.append({
                "id": rid,
                "dataset": "math_algprealg",
                "subject": subj,
                "level": e["level"],
                "type": e["type"],
                "position_in_subject": k,
                "in_math500": m is not None,
                "math500_unique_id": m["unique_id"] if m is not None else None,
                "question_en": e["problem"],
                "solution_en": e["solution"],
                "question_vi": pv,
                "solution_vi": sv,
                "gold_raw": gold_raw,
                "gold_source": "math500.answer" if m is not None else "hendrycks.last_boxed(en)",
                "gold": {k2: gold[k2] for k2 in ("answer_canonical", "answer_type") if k2 in gold}
                        | {k2: gold[k2] for k2 in ("percent", "vars", "target_var") if gold.get(k2)}
                        | {"locale": GOLD_LOCALE}
                        | ({"accept_also": [a | {"locale": GOLD_LOCALE} for a in gold["accept_also"]]}
                           if gold.get("accept_also") else {}),
                "gold_override_reason": override_reason,
                "gold_vi": gold_vi,
                "en_distinct_boxed": n_en_box,
                "deterministic_fixes": fixes,
                "number_parity_ok": par["numbers_ok"],
                "number_parity_diff": par["numbers"],
                "latex_parity_ok": par["latex_ok"],
                "latex_parity_diff": par["latex"],
            })
        cs["vi_last_boxed_equals_en_normalised"] = boxed_eq
        cs["vi_last_boxed_correct_under_grader_vs_en_gold"] = boxed_grader_eq
        cs["vi_vs_en_boxed_mismatch"] = mism
        cs["deterministic_fixes"] = dict(fixes_n)
    c["math500_matched_by_text"] = len(matched_m500)
    unmatched = [k for k in m500_by_text if k not in matched_m500]
    c["math500_unmatched"] = []
    for k in unmatched:  # report the nearest problem for any MATH-500 item that did not match exactly
        best = max(rows, key=lambda r: difflib.SequenceMatcher(None, ws(r["question_en"]), k).ratio())
        c["math500_unmatched"].append({"math500": k[:120], "nearest": best["id"]})
    # MATH-500 gold vs the hendrycks last \boxed for the matched items
    agree, disagree = 0, []
    for r in rows:
        if r["in_math500"]:
            lb = last_boxed(r["solution_en"])
            ok = mv.is_correct(f"\\boxed{{{lb}}}", mv.infer_gold(r["gold_raw"], problem=r["question_en"]))
            agree += ok
            if not ok:
                disagree.append({"id": r["id"], "math500_answer": r["gold_raw"], "hendrycks_last_boxed": lb})
    c["math500_answer_vs_hendrycks_last_boxed"] = {"agree": agree, "disagree": disagree}
    # gold self-consistency: the typed gold must accept its own raw string
    c["gold_self_match"] = sum(mv.is_correct(f"\\boxed{{{r['gold_raw']}}}", r["gold"]) for r in rows)
    c["gold_types"] = dict(collections.Counter(r["gold"]["answer_type"] for r in rows))
    c["multi_boxed_en_solutions"] = [
        {"id": r["id"], "in_math500": r["in_math500"], "boxed": all_boxed(r["solution_en"]), "gold_raw": r["gold_raw"]}
        for r in rows if r["en_distinct_boxed"] > 1]
    c["gold_overrides"] = {r["id"]: r["gold_override_reason"] for r in rows if r["gold_override_reason"]}
    c["gold_vi_localised_text"] = {r["id"]: [r["gold"]["answer_canonical"], r["gold_vi"]["answer_canonical"]]
                                   for r in rows if r["gold_vi"]}
    c["number_multiset_mismatch_vi_vs_en_problem"] = sum(not r["number_parity_ok"] for r in rows)
    c["latex_multiset_mismatch_vi_vs_en_problem"] = sum(not r["latex_parity_ok"] for r in rows)
    return rows


def main() -> None:
    checks: dict = {"grader": {"version": mv.GRADER_VERSION, "backend": mv.BACKEND}}
    g = build_gsm8k(checks)
    m = build_math(checks)
    write_jsonl(WORK / "gsm8k_platinum_base.jsonl", g)
    write_jsonl(WORK / "math_algprealg_base.jsonl", m)
    write_json(WORK / "base_checks.json", checks)
    short = json.loads(json.dumps(checks))
    for s in ("algebra", "prealgebra"):
        short["math"][s]["vi_vs_en_boxed_mismatch"] = len(short["math"][s]["vi_vs_en_boxed_mismatch"])
    short["math"]["multi_boxed_en_solutions"] = len(short["math"]["multi_boxed_en_solutions"])
    print(json.dumps(short, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
