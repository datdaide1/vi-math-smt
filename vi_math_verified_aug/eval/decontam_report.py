"""X5 report tables: combine the local lexical scans (``data/contam/scans/lex_*``) with the Modal bge-m3 scan
(``data/contam/embed_modal/{smoke,main}``) into the per-corpus × target tables and the benchmark review list.

    python -m vi_math_verified_aug.eval.decontam_report            # writes data/contam/scans/{final_table,review_flags}.json

Review rule for a benchmark item (used for Vi-ExamMath hand-off; see X5 report §5):
  13-gram coverage ≥ 0.5 (item ≥ 13 tokens)
  OR, for items ≥ 20 tokens, 13-gram coverage ≥ 0.3 or number-templated 13-gram coverage ≥ 0.5
  OR bge-m3 cosine ≥ 0.90.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from vi_math_verified_aug.eval.decontam import CONTAM, iter_vi_exam_raw, vi_exam_keys

CORPORA = ["hllj_crawl", "hllj_mcq", "vinhnq29", "luminarai", "metamathqa_vi"]
TAU_LOOSE, TAU_STRICT = 0.85, 0.90


def _emb_path(c: str) -> Path:
    sub = "smoke" if c == "vinhnq29" else "main"
    return CONTAM / "embed_modal" / sub / f"embed_{c}.jsonl"


def _jsonl(p: Path):
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def review_rule(r: dict, cos: float | None) -> bool:
    long_ = r["n_tokens"] >= 20
    return bool((r["ng13_exact"]["cov"] >= 0.5 and r["n_tokens"] >= 13)
                or (long_ and (r["ng13_exact"]["cov"] >= 0.3 or r["ng13_template"]["cov"] >= 0.5))
                or (cos is not None and cos >= TAU_STRICT))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    rows, flags = [], defaultdict(dict)
    for c in CORPORA:
        emb = {(r["set"], r["id"]): (r["cos"][0], r["doc"][0]) for r in _jsonl(_emb_path(c))}
        by = defaultdict(lambda: defaultdict(int))
        for r in _jsonl(CONTAM / "scans" / f"lex_{c}" / "items.jsonl"):
            t, b = r["target"], by[r["target"]]
            e, edoc = emb.get((t, r["id"]), (None, None))
            cov = r["ng13_exact"]["cov"] >= 0.5
            nod = r["ng13_nodiac"]["cov"] >= 0.5
            tmp = r["ng13_template"]["cov"] >= 0.5
            mh = r["minhash"]["j"] >= 0.5
            e85 = e is not None and e >= TAU_LOOSE
            lex = cov or nod or mh
            b["n"] += 1
            b["ng13_any(>=13tok)"] += r["ng13_exact"]["shared"] >= 1 and r["n_tokens"] >= 13
            b["ng13_cov>=.5"] += cov
            b["ng8_cov>=.5"] += r["ng8_exact"]["cov"] >= 0.5
            b["tmpl13_cov>=.5"] += tmp
            b["minhash>=.5"] += mh
            b["lexical_any"] += lex
            b["cos>=.85"] += e85
            b["cos>=.90"] += e is not None and e >= TAU_STRICT
            b["emb_only(cos>=.85,no lex)"] += e85 and not (lex or tmp)
            b["review_rule"] += review_rule(r, e)
            if review_rule(r, e):
                flags[t].setdefault(r["id"], []).append(
                    {"corpus": c, "ng13": r["ng13_exact"]["cov"], "tmpl": r["ng13_template"]["cov"], "cos": e,
                     "doc": r["ng13_exact"]["doc"] or edoc, "n_tokens": r["n_tokens"], "text": r["text"][:200]})
        for t, b in by.items():
            rows.append({"corpus": c, "target": t, **b})
    # exam year of flagged Vi-ExamMath items
    keys = {v["page"]: v for v in vi_exam_keys().values()}
    year_all, year_flag = defaultdict(int), defaultdict(int)
    for d in iter_vi_exam_raw():
        y = keys.get(d.meta["page"], {}).get("year")
        year_all[y] += 1
        year_flag[y] += d.id in flags.get("vi_exam_raw", {})
    cols = [k for k in rows[0] if k not in ("corpus", "target")]
    print("| corpus | target | " + " | ".join(cols) + " |")
    print("|" + "---|" * (len(cols) + 2))
    for r in rows:
        print(f"| {r['corpus']} | {r['target']} | " + " | ".join(str(r[k]) for k in cols) + " |")
    print("review-flagged items (any corpus):", {t: len(v) for t, v in flags.items()})
    print("Vi-ExamMath flagged by exam year:",
          {str(y): f"{year_flag[y]}/{year_all[y]}" for y in sorted(year_all, key=str)})
    (CONTAM / "scans" / "final_table.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    (CONTAM / "scans" / "review_flags.json").write_text(json.dumps(flags, indent=1, ensure_ascii=False),
                                                         encoding="utf-8")


# ---------------------------------------------------------------------------------------------------------------------
# Vi-ExamMath final items (X2 data/vi_exam/items.jsonl) — per-item flag file + post-cutoff subsets
# ---------------------------------------------------------------------------------------------------------------------
VI_CORPORA = ["hllj_crawl", "hllj_mcq", "vinhnq29", "luminarai", "metamathqa_vi"]
TAU_VI_VI, TAU_VI_EN = 0.85, 0.80     # X5 §5.1 (G4) thresholds


def _vex_signals(r: dict | None, cos: float | None, cdoc: str | None, *, en: bool = False) -> list[dict]:
    """Signals firing under the G4-recommended thresholds for one (item, corpus)."""
    out = []
    if r is not None:
        for key, sig in (("ng13_exact", "ng13"), ("ng13_nodiac", "ng13_nodiac"), ("ng13_template", "tmpl13")):
            if r[key]["cov"] >= 0.5:
                out.append({"signal": sig, "similarity": r[key]["cov"], "doc": r[key]["doc"]})
        if r["minhash"]["j"] >= 0.5:
            out.append({"signal": "minhash", "similarity": r["minhash"]["j"], "doc": r["minhash"]["doc"]})
    if cos is not None and cos >= (TAU_VI_EN if en else TAU_VI_VI):
        out.append({"signal": "cos_en" if en else "cos_vi", "similarity": cos, "doc": cdoc})
    return out


def vi_exam_flags() -> None:
    items = {}
    for r in _jsonl(Path(CONTAM).parent / "vi_exam" / "items.jsonl"):
        items[r["id"]] = {"core": bool(r.get("core_candidate")), "temporal": r.get("temporal") or {},
                          "exam_year": (r.get("provenance") or {}).get("exam_year")}
    emb_dir = CONTAM / "embed_modal" / "vex"
    rows = []
    ntok = {}
    review = set()
    for c in VI_CORPORA + ["metamathqa_en"]:
        lex = {}
        if c != "metamathqa_en":
            for r in _jsonl(CONTAM / "scans" / f"vex_{c}" / "items.jsonl"):
                lex[r["id"]] = r
                ntok[r["id"]] = r["n_tokens"]
        emb = {}
        ep = emb_dir / f"embed_{c}.jsonl"
        if ep.exists():
            emb = {r["id"]: (r["cos"][0], r["doc"][0]) for r in _jsonl(ep)}
        for iid in items:
            r = lex.get(iid)
            cos, cdoc = emb.get(iid, (None, None))
            for sgn in _vex_signals(r, cos, cdoc, en=(c == "metamathqa_en")):
                rows.append({"item_id": iid, "corpus": c, **sgn, "n_tokens": ntok.get(iid),
                             "core": items[iid]["core"]})
            if r is not None and review_rule(r, cos):
                review.add(iid)
            elif c == "metamathqa_en" and cos is not None and cos >= TAU_STRICT:
                review.add(iid)
    with open(CONTAM / "vi_exam_flags.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            out_row = {"item_id": row["item_id"], "corpus": row["corpus"], "signal": row["signal"],
                       "similarity": row["similarity"], "matched_id": row["doc"], "n_tokens": row["n_tokens"],
                       "core": row["core"], "review_rule": row["item_id"] in review}
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
    flagged = {r["item_id"] for r in rows}
    flagged_no_short_tmpl = {r["item_id"] for r in rows
                             if not (r["signal"] == "tmpl13" and (r["n_tokens"] or 0) < 20)}
    sig_count = defaultdict(set)
    for r in rows:
        if r["core"]:
            sig_count[(r["corpus"], r["signal"])].add(r["item_id"])
    subsets = {"Core (all years)": lambda m: True,
               "Core, exam 2026": lambda m: m["temporal"].get("exam_2026"),
               "Core, exam 2025+": lambda m: m["temporal"].get("exam_2025_plus"),
               "Core, first_seen >= 2026-06": lambda m: m["temporal"].get("first_seen_2026_06_plus")}
    print("| subset | n | flagged (G4 thresholds) | remaining | flagged w/o tmpl13 on items < 20 tok | remaining | "
          "flagged (review rule) | remaining |")
    print("|---|---|---|---|---|---|---|---|")
    out = {}
    for name, pred in subsets.items():
        ids = [i for i, m in items.items() if m["core"] and pred(m)]
        a = sum(i in flagged for i in ids)
        b = sum(i in flagged_no_short_tmpl for i in ids)
        c_ = sum(i in review for i in ids)
        out[name] = {"n": len(ids), "flagged_g4": a, "flagged_g4_no_short_tmpl": b, "flagged_review": c_}
        print(f"| {name} | {len(ids)} | {a} | {len(ids) - a} | {b} | {len(ids) - b} | {c_} | {len(ids) - c_} |")
    print("\nCore items per (corpus, signal):")
    for (c, sg), v in sorted(sig_count.items()):
        print(f"  {c:15s} {sg:12s} {len(v)}")
    allf = {r["item_id"] for r in rows}
    print("all items flagged:", len(allf), "of", len(items), "; flag rows:", len(rows))
    (CONTAM / "scans" / "vi_exam_subsets.json").write_text(
        json.dumps({"subsets": out, "per_signal_core": {f"{c}|{sg}": len(v) for (c, sg), v in sig_count.items()}},
                   indent=1), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "vex":
        sys.stdout.reconfigure(encoding="utf-8")
        vi_exam_flags()
    else:
        main()
