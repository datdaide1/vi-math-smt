"""Step 5: apply auto-accepted fixes, draw the primary sample, write the frozen VI + EN sets.

Run from the repo root with the project venv (after ``opus_fix decide``)::

    .venv/Scripts/python.exe -m vi_math_verified_aug.eval_sets.freeze

Writes ``data/eval/<set>.jsonl`` + ``data/eval/<set>.datacard.json`` for every set and ``data/eval/SHA256SUMS``.

Sets (VI and EN share ids; EN = the English originals of exactly the same items):
- ``vi_gsm8k_platinum`` / ``en_gsm8k_platinum``          1,209  (co-primary #1 full set; EN = secondary in E1)
- ``vi_math_algprealg`` / ``en_math_algprealg``          2,058  MATH test algebra + prealgebra
- ``vi_math500_algprealg`` / ``en_math500_algprealg``    206    MATH-500 subset of the above
- ``vi_primary_1000`` / ``en_primary_1000``              1,000  600 GSM8K-Platinum + 400 MATH alg/prealg (⊇ the 206)
- ``en_math500_full``                                    500    English MATH-500, literature comparability only

Primary sample (T5 §2.1, E1 §5.6): seed 20261107; items still ``flagged`` by the review are not eligible (E1
exclusion rule, applied to both languages); GSM8K stratified by solution length (number of solution lines),
MATH by subject × level, proportional allocation with largest remainders; all 206 MATH-500 items forced in.
"""
from __future__ import annotations

import collections
import json
import random

from vi_math_verified_aug.eval import math_verify as mv
from vi_math_verified_aug.eval_sets.common import (
    BUILD_DATE, OUT, SEED, SRC, WORK, read_jsonl, sha256_file, write_json, write_jsonl,
)

VERSION = "v1.1"
OPUS_MODEL = "claude-opus-5-5"
N_GSM, N_MATH = 600, 400


def _largest_remainder(weights: dict, total: int, caps: dict | None = None) -> dict:
    """Integer allocation of ``total`` proportional to ``weights`` (ties broken by key order), respecting caps."""
    caps = caps or {}
    alloc = {k: 0 for k in weights}
    remaining, active = total, {k for k, w in weights.items() if w > 0}
    while remaining > 0 and active:
        wsum = sum(weights[k] for k in active)
        quotas = {k: remaining * weights[k] / wsum for k in active}
        add = {k: int(quotas[k]) for k in active}
        left = remaining - sum(add.values())
        for k in sorted(active, key=lambda k: (-(quotas[k] - int(quotas[k])), str(k)))[:left]:
            add[k] += 1
        progressed = False
        for k in list(active):
            cap = caps.get(k, float("inf"))
            take = min(add[k], cap - alloc[k])
            alloc[k] += take
            remaining -= take
            progressed |= take > 0
            if alloc[k] >= cap:
                active.discard(k)
        if not progressed:
            break
    return alloc


def _steps_bucket(n: int) -> str:
    return str(n) if n <= 6 else "7+"


def load_reviewed() -> tuple[list[dict], list[dict], dict]:
    decisions = {r["id"]: r for r in read_jsonl(WORK / "opus" / "decisions.jsonl")}
    solutions = {r["id"]: r for r in read_jsonl(WORK / "opus" / "solution.jsonl")}
    sets = []
    for f in ("gsm8k_platinum_base.jsonl", "math_algprealg_base.jsonl"):
        rows = read_jsonl(WORK / f)
        for r in rows:
            d = decisions.get(r["id"])
            r["question_vi_original"] = r["question_vi"]
            r["review_status"] = d["status"] if d else "not_queued"
            r["review_triggers"] = d["triggers"] if d else []
            r["review"] = None
            if d:
                r["review"] = {k: d.get(k) for k in ("status", "verdict", "reason", "why", "model", "parity_ok",
                                                    "solve_correct", "judge_equivalent", "judge_issues")}
            if d and d["status"] == "auto_fixed":
                r["question_vi"] = d["proposed_vi"]
            r["solution_note"] = None
            if r.get("platinum_status") == "revised":
                s = solutions.get(r["id"])
                if s and s.get("ok_response") and s["solution_checks"]["all_ok"]:
                    r["solution_vi"] = s["text"]
                    r["solution_note"] = (f"translated from the GSM8K-Platinum English solution by {s['model']} "
                                          "(checks: final #### = gold, <<>> annotations identical, numeral parity)")
                else:
                    r["solution_vi"] = None
                    r["solution_note"] = ("no VI solution: the original VI solution explains the pre-Platinum gold, "
                                          "and the re-translation of the Platinum solution failed numeral parity "
                                          "(it restated arithmetic); see datacard")
        sets.append(rows)
    return sets[0], sets[1], decisions


def _with_locale(gold: dict, locale: str) -> dict:
    """Copy of a gold dict with ``locale`` set on it and on every ``accept_also`` alternative."""
    g = dict(gold)
    g["locale"] = locale
    if g.get("accept_also"):
        g["accept_also"] = [dict(a, locale=locale) for a in g["accept_also"]]
    return g


def _vi_row(r: dict) -> dict:
    g = _with_locale(r.get("gold_vi") or r["gold"], "mixed")
    return {
        "id": r["id"], "lang": "vi", "dataset": r["dataset"], "question": r["question_vi"],
        "gold": g, "gold_raw": r["gold_raw"], "solution": r["solution_vi"], "solution_note": r["solution_note"],
        "subject": r.get("subject"), "level": r.get("level"), "in_math500": r.get("in_math500", False),
        "math500_unique_id": r.get("math500_unique_id"), "platinum_status": r.get("platinum_status"),
        "source_index": r.get("gsm8k_index", r.get("position_in_subject")),
        "review_status": r["review_status"], "review_triggers": r["review_triggers"],
        "deterministic_fixes": r["deterministic_fixes"],
    }


def _en_row(r: dict) -> dict:
    g = _with_locale(r["gold"], "en")
    return {
        "id": r["id"], "lang": "en", "dataset": r["dataset"], "question": r["question_en"],
        "gold": g, "gold_raw": r["gold_raw"], "solution": r["solution_en"],
        "subject": r.get("subject"), "level": r.get("level"), "in_math500": r.get("in_math500", False),
        "math500_unique_id": r.get("math500_unique_id"), "platinum_status": r.get("platinum_status"),
        "source_index": r.get("gsm8k_index", r.get("position_in_subject")),
        # VI review status is carried so that the E1 exclusion rule can be applied identically to both languages
        "vi_review_status": r["review_status"],
    }


def draw_primary(gsm: list[dict], math: list[dict]) -> tuple[list[str], dict]:
    rng = random.Random(SEED)
    info: dict = {"seed": SEED}
    # --- GSM8K-Platinum: 600 eligible items, stratified by solution length
    elig = [r for r in gsm if r["review_status"] != "flagged"]
    strata = collections.defaultdict(list)
    for r in elig:
        strata[_steps_bucket(r["n_steps"])].append(r["id"])
    alloc = _largest_remainder({k: len(v) for k, v in strata.items()}, N_GSM,
                               caps={k: len(v) for k, v in strata.items()})
    g_ids = []
    for k in sorted(strata):
        g_ids += rng.sample(sorted(strata[k]), alloc[k])
    info["gsm8k"] = {"eligible": len(elig), "excluded_flagged": len(gsm) - len(elig),
                     "stratum": "solution lines (1..6, 7+)",
                     "population": {k: len(strata[k]) for k in sorted(strata)},
                     "allocation": {k: alloc[k] for k in sorted(strata)}}
    # --- MATH alg/prealg: 400 = all 206 MATH-500 + 194 stratified by subject x level
    elig = [r for r in math if r["review_status"] != "flagged"]
    m500 = [r for r in math if r["in_math500"]]
    m500_flagged = [r["id"] for r in m500 if r["review_status"] == "flagged"]
    key = lambda r: f"{r['subject']}|{r['level']}"
    pop = collections.Counter(key(r) for r in elig)
    target = _largest_remainder(dict(pop), N_MATH)
    forced = collections.Counter(key(r) for r in m500)
    need = {k: max(0, target[k] - forced.get(k, 0)) for k in pop}
    pool = collections.defaultdict(list)
    for r in elig:
        if not r["in_math500"]:
            pool[key(r)].append(r["id"])
    n_rest = N_MATH - len(m500)
    rest_alloc = _largest_remainder(need, n_rest, caps={k: len(pool[k]) for k in pop})
    m_ids = [r["id"] for r in m500]
    for k in sorted(pool):
        m_ids += rng.sample(sorted(pool[k]), rest_alloc.get(k, 0))
    info["math"] = {"eligible": len(elig), "excluded_flagged": len(math) - len(elig),
                    "math500_forced": len(m500), "math500_flagged_but_forced": m500_flagged,
                    "stratum": "subject x level",
                    "population": dict(sorted(pop.items())), "proportional_target_400": dict(sorted(target.items())),
                    "math500_per_stratum": dict(sorted(forced.items())),
                    "sampled_non_math500": {k: rest_alloc.get(k, 0) for k in sorted(pop)},
                    "final_per_stratum": dict(sorted(collections.Counter(
                        key(r) for r in math if r["id"] in set(m_ids)).items()))}
    assert len(g_ids) == N_GSM and len(m_ids) == N_MATH, (len(g_ids), len(m_ids))
    return sorted(g_ids) + sorted(m_ids), info


def _sources() -> list[dict]:
    man = json.loads((SRC / "manifest.json").read_text(encoding="utf-8"))
    return man["files"]


def _local_sources() -> list[dict]:
    from vi_math_verified_aug.eval_sets.common import ROOT, VI_GSM8K_TEST, VI_MATH_TEST
    files = [VI_GSM8K_TEST, *VI_MATH_TEST.values(), ROOT / "data/generate/gsm8k/test.json",
             ROOT / "data/generate/MATH/test/algebra.json", ROOT / "data/generate/MATH/test/prealgebra.json"]
    return [{"path": str(p.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256_file(p)} for p in files]


def _changes(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        if r["review_status"] == "auto_fixed":
            out.append({"id": r["id"], "kind": "translation_fix", "reason": r["review"]["reason"],
                        "triggers": r["review_triggers"], "model": r["review"]["model"], "date": BUILD_DATE,
                        "checks": {"parity": True, "self_solve_matches_gold": True, "judge_equivalent": True},
                        "before": r["question_vi_original"], "after": r["question_vi"]})
        for f in r["deterministic_fixes"]:
            out.append({"id": r["id"], "kind": "deterministic_repair", "reason": f, "date": BUILD_DATE})
        if r.get("gold_override_reason"):
            out.append({"id": r["id"], "kind": "gold_override", "reason": r["gold_override_reason"],
                        "gold": r["gold"], "date": BUILD_DATE})
        if r.get("gold_vi"):
            out.append({"id": r["id"], "kind": "gold_vi_localised_text",
                        "reason": "text answer localised by the translation; VI gold keeps the VI wording",
                        "gold_en": r["gold"]["answer_canonical"], "gold_vi": r["gold_vi"]["answer_canonical"],
                        "date": BUILD_DATE})
        if r.get("platinum_status") == "revised":
            out.append({"id": r["id"], "kind": "platinum_gold_revision", "gold_orig_gsm8k": r["gold_orig_gsm8k"],
                        "gold_platinum": r["gold_raw"],
                        "vi_solution": ("original dropped (explained the old gold); "
                                        + (r["solution_note"] or "")),
                        "date": BUILD_DATE})
        if r.get("gold", {}).get("accept_also"):
            out.append({"id": r["id"], "kind": "gold_accept_also",
                        "reason": "the source solution explicitly accepts these alternative answers",
                        "accept_also": [a["answer_canonical"] for a in r["gold"]["accept_also"]], "date": BUILD_DATE})
    return out


CHANGELOG = [
    {"version": "v1.0", "date": "2026-11-07", "change": "first freeze"},
    {"version": "v1.1", "date": "2026-11-07", "change": (
        "(1) VI solutions for the 10 GSM8K-Platinum-revised items: 9 translated from the Platinum English solution by "
        "claude-opus-5-5 and accepted by checks (final #### = gold, <<>> annotations identical, numeral parity); "
        "gsm8k-test-1071 stays null (restated arithmetic broke numeral parity), explained in row.solution_note. "
        "(2) math-test-algebra-0333: gold.accept_also = ['3'] (the MATH solution says 3 is also accepted; gold "
        "stays -3). (3) every stored gold dict (and every accept_also entry) carries 'locale', also in "
        "data/eval/_work. New VI row field: solution_note.")},
]
GOLD_SCHEMA_NOTE = ("gold = {answer_canonical, answer_type, locale[, percent, vars, target_var][, accept_also]}; "
                    "accept_also is a list of gold dicts of the same shape (without accept_also); a prediction is "
                    "correct if math_verify.grade() accepts it against gold OR against any accept_also entry. "
                    "Grader 3.0.1 ignores accept_also, so the harness must take that OR until X1 supports it "
                    "(1 item: math-test-algebra-0333).")


def _review_summary(rows: list[dict]) -> dict:
    flagged = [{"id": r["id"], "why": r["review"]["why"] if r["review"] else None, "triggers": r["review_triggers"]}
               for r in rows if r["review_status"] == "flagged"]
    cleared = [r["id"] for r in rows if r["review_status"] == "cleared"]
    return {"status_counts": dict(collections.Counter(r["review_status"] for r in rows)),
            "still_flagged": flagged, "cleared_unchanged": cleared}


def _card(name: str, lang: str, rows: list[dict], out_rows: list[dict], sha: str, extra: dict) -> dict:
    ds = collections.Counter(r["dataset"] for r in rows)
    card = {
        "name": name, "version": VERSION, "language": lang, "build_date": BUILD_DATE,
        "file": f"{name}.jsonl", "sha256": sha, "rows": len(out_rows), "rows_per_dataset": dict(ds),
        "builder": "vi_math_verified_aug/eval_sets (fetch -> build_base -> review_queue -> opus_fix -> freeze)",
        "upstream_sources": _sources(), "local_sources": _local_sources(),
        "grader_for_gold_typing": {"module": "vi_math_verified_aug/eval/math_verify.py",
                                   "version": mv.GRADER_VERSION, "backend": mv.BACKEND},
        "gold": ("GSM8K: GSM8K-Platinum '#### n' (answer_type int). MATH: MATH-500 'answer' for the 206 MATH-500 "
                 "items, otherwise the last \\boxed{} of the English hendrycks_math solution, typed once with "
                 "math_verify.infer_gold; golds are never extracted from Vietnamese text (except gold_vi_localised_text)."),
        "id_scheme": "gsm8k-test-<openai/gsm8k test index>; math-test-<subject>-<row position in hendrycks_math test>",
        "gold_schema": GOLD_SCHEMA_NOTE,
        "changelog": CHANGELOG,
        "fields": sorted(out_rows[0].keys()) if out_rows else [],
        "gold_types": dict(collections.Counter(r["gold"]["answer_type"] for r in out_rows)),
    }
    if lang == "vi":
        card["transformations"] = [
            "Vietnamese text from data/test (project translation, origin documented in content_review R1/R2)",
            "MATH: deterministic repair of JSON-escape damage (control char + word -> \\right, \\times ... when the "
            "command occurs in the English) and literal two-character \\n -> newline",
            "GSM8K: original VI solution dropped for the 10 Platinum-revised items (it explains the old gold); "
            "replaced by an Opus translation of the Platinum English solution when its checks pass "
            "(9/10; gsm8k-test-1071 keeps solution=null, see solution_note)",
            f"translation-defect review: queue (R1/R2 patterns, numeral multiset EN<->VI, glossary) -> "
            f"{OPUS_MODEL} (effort medium, Batch API) proposal -> accepted only if numeral/LaTeX parity, "
            "Opus self-solve of the candidate text reproduces the gold, and an independent Opus judge call "
            "finds EN and VI equivalent; otherwise the original text is kept and the item stays flagged",
        ]
        card["review"] = _review_summary(rows)
        card["changes"] = _changes(rows)
        card["models_used"] = {"translation_review": OPUS_MODEL, "nim_or_gemini": "none (test items go only to Claude)"}
    else:
        card["transformations"] = ["none: English originals as published upstream (pinned revisions); "
                                   "GSM8K text = GSM8K-Platinum (identical questions to openai/gsm8k for these ids)"]
        card["gold_changes"] = [c for c in _changes(rows) if c["kind"] in ("gold_override", "platinum_gold_revision")]
        card["note_vi_review_status"] = ("vi_review_status mirrors the Vietnamese review so that the E1 exclusion "
                                         "rule (drop flagged items from both languages) can be applied")
    card.update(extra)
    return card


def main() -> None:
    gsm, math, _ = load_reviewed()
    primary_ids, pinfo = draw_primary(gsm, math)
    by_id = {r["id"]: r for r in gsm + math}
    plan = {
        "gsm8k_platinum": (gsm, {"role": "co-primary #1 (VI); secondary EN mirror (E1 §5.6)"}),
        "math_algprealg": (math, {"role": "secondary: MATH test algebra+prealgebra"}),
        "math500_algprealg": ([r for r in math if r["in_math500"]],
                              {"role": "MATH-500 algebra+prealgebra subset (124+82), matched by English problem text "
                                       "to HuggingFaceH4/MATH-500"}),
        "primary_1000": ([by_id[i] for i in primary_ids],
                         {"role": "frozen primary sample (T5 §2.1, E1 §5.6): 600 GSM8K-Platinum + 400 MATH "
                                  "alg/prealg incl. all 206 MATH-500 items", "sampling": pinfo}),
    }
    sums = []
    cards = {}
    for base, (rows, extra) in plan.items():
        for lang, fn in (("vi", _vi_row), ("en", _en_row)):
            name = f"{lang}_{base}"
            out_rows = [fn(r) for r in rows]
            sha = write_jsonl(OUT / f"{name}.jsonl", out_rows)
            card = _card(name, lang, rows, out_rows, sha, extra)
            write_json(OUT / f"{name}.datacard.json", card)
            cards[name] = card
            sums.append((sha, f"{name}.jsonl"))
    # English MATH-500 (all subjects), literature comparability only
    m500 = read_jsonl(SRC / "math500_test.jsonl")
    en_full = []
    for r in m500:
        g = mv.infer_gold(r["answer"], problem=r["problem"], locale="en")
        en_full.append({"id": "math500-" + r["unique_id"].replace("test/", "").replace(".json", "").replace("/", "-"),
                        "lang": "en", "dataset": "math500", "question": r["problem"],
                        "gold": {k: g[k] for k in ("answer_canonical", "answer_type", "locale")}
                                | {k: g[k] for k in ("percent", "vars", "target_var") if g.get(k)},
                        "gold_raw": r["answer"], "solution": r["solution"], "subject": r["subject"],
                        "level": r["level"], "math500_unique_id": r["unique_id"]})
    sha = write_jsonl(OUT / "en_math500_full.jsonl", en_full)
    write_json(OUT / "en_math500_full.datacard.json", {
        "name": "en_math500_full", "version": VERSION, "language": "en", "build_date": BUILD_DATE,
        "file": "en_math500_full.jsonl", "sha256": sha, "rows": len(en_full),
        "role": "secondary, literature comparability only (not parallel; E1 §5.6 uses only the alg/prealg 206)",
        "upstream_sources": [s for s in _sources() if s["name"] == "math500_test"],
        "transformations": ["gold typed with math_verify.infer_gold from MATH-500 'answer'"],
        "grader_for_gold_typing": {"version": mv.GRADER_VERSION, "backend": mv.BACKEND},
        "subjects": dict(collections.Counter(r["subject"] for r in en_full)),
        "gold_types": dict(collections.Counter(r["gold"]["answer_type"] for r in en_full)),
    })
    sums.append((sha, "en_math500_full.jsonl"))
    for p in sorted(OUT.glob("*.datacard.json")):
        sums.append((sha256_file(p), p.name))
    (OUT / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for h, n in sorted(sums, key=lambda x: x[1])),
                                    encoding="utf-8", newline="\n")
    print(json.dumps({n: {"rows": c["rows"], "sha256": c["sha256"][:16]} for n, c in cards.items()}, indent=1))
    print(json.dumps(pinfo, indent=1))


if __name__ == "__main__":
    main()
