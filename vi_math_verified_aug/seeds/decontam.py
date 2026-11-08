"""Stage ``decontam``: X5's G4 gate and benchmark review rule, seed pool vs every eval set and Vi-ExamMath.

Targets: every ``data/eval/*.jsonl`` (VI and EN; ``question``) and ``data/vi_exam/items.jsonl`` (``problem``).
Each seed is checked with its Vietnamese text and with its English original.

Lexical (X5 ``decontam.py``: ``variant_tokens`` / ``NgramIndex(13)`` over the targets, exact / diacritic-stripped /
number-templated variants), embeddings (bge-m3, CLS, max_len 256, the X5-calibrated encoder; ``embed_modal.py`` on a
Modal L4, top-3 targets per seed text and target language).

Decision rules (X5 §5.1 / §5.3):
* **review rule -> DROP the seed** if, for some target item, any of
  1. 13-gram coverage (exact or diacritic-stripped) of the target item >= 0.5;
  2. the target item has >= 20 tokens and exact 13-gram coverage >= 0.3 or templated 13-gram coverage >= 0.5;
  3. bge-m3 cosine >= 0.90 (same language, or cross-lingual).
* **G4-only flags (recorded, not dropped)**: >= 1 shared exact/nodiac 13-gram; templated coverage >= 0.5 (any
  length); cosine >= 0.85 (same language) or >= 0.80 (VI<->EN).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from .common import DATA, ROOT, WORK, read_jsonl, write_json, write_jsonl

EMB_DIR = WORK / "embed"
EVAL = DATA / "eval"
VI_EXAM = DATA / "vi_exam" / "items.jsonl"
COS_REVIEW = 0.90
COS_G4_SAME = 0.85
COS_G4_X = 0.80


def targets() -> list[dict]:
    out = []
    for p in sorted(EVAL.glob("*.jsonl")):
        lang = p.name.split("_", 1)[0]
        for r in read_jsonl(p):
            out.append({"tid": f"{p.stem}/{r['id']}", "set": p.stem, "lang": lang, "text": r["question"]})
    for r in read_jsonl(VI_EXAM):
        out.append({"tid": f"vi_exam/{r['id']}", "set": "vi_exam", "lang": "vi", "text": r["problem"],
                    "core": bool(r.get("core_candidate"))})
    return out


def seed_texts() -> list[dict]:
    from .assemble import current_rows
    return [{"seed_id": r["seed_id"], "vi": r["problem_vi"], "en": r["problem_en"]} for r in current_rows()]


def write_embed_inputs() -> Path:
    T = targets()
    S = seed_texts()
    p = EMB_DIR / "inputs.json"
    write_json(p, {"seeds": S, "targets": [{"tid": t["tid"], "lang": t["lang"], "text": t["text"]} for t in T]})
    print(json.dumps({"seeds": len(S), "targets": len(T), "by_lang": dict(Counter(t["lang"] for t in T)),
                      "path": str(p)}))
    return p


def lexical(T: list[dict], S: list[dict]) -> dict[str, dict]:
    import numpy as np  # noqa: F401  (decontam needs numpy; this stage runs in the project .venv)

    from ..eval.decontam import NgramIndex, token_hashes, variant_tokens
    ix = {v: NgramIndex(13) for v in ("exact", "nodiac", "template")}
    ntok = []
    for i, t in enumerate(T):
        vt = variant_tokens(t["text"])
        ntok.append(len(vt["exact"]))
        for v in ix:
            ix[v].add(i, token_hashes(vt[v]))
    for v in ix.values():
        v.finalize()
    out = {}
    for s in S:
        best: dict[int, dict] = defaultdict(dict)        # target idx -> {variant: (cov, shared, side)}
        for side in ("vi", "en"):
            vt = variant_tokens(s[side])
            for v, x in ix.items():
                res = x.query(token_hashes(vt[v]))
                for ti, sh in res.items():
                    cov = sh / max(x.n_grams[ti], 1)
                    cur = best[ti].get(v)
                    if cur is None or cov > cur[0]:
                        best[ti][v] = [round(cov, 4), int(sh), side]
        flags = {"g4_ng13": [], "g4_tmpl": [], "rv_cov50": [], "rv_long": []}
        for ti, d in best.items():
            t = T[ti]
            ex, nd, tp = d.get("exact"), d.get("nodiac"), d.get("template")
            m = {"tid": t["tid"], "set": t["set"], "n_tok": ntok[ti],
                 "exact": ex, "nodiac": nd, "template": tp}
            if (ex and ex[1] >= 1) or (nd and nd[1] >= 1):
                flags["g4_ng13"].append(m)
            if tp and tp[0] >= 0.5:
                flags["g4_tmpl"].append(m)
            if max((ex or (0,))[0], (nd or (0,))[0]) >= 0.5:
                flags["rv_cov50"].append(m)
            if ntok[ti] >= 20 and (((ex or (0,))[0] >= 0.3) or ((tp or (0,))[0] >= 0.5)):
                flags["rv_long"].append(m)
        out[s["seed_id"]] = {k: sorted(v, key=lambda m: -max((m["exact"] or (0,))[0], (m["template"] or (0,))[0]))[:5]
                             for k, v in flags.items() if v}
    return out


def embedding(T: list[dict]) -> dict[str, dict]:
    p = EMB_DIR / "result.json"
    if not p.exists():
        print(f"[decontam] no embedding result at {p}: cosine rules not applied")
        return {}
    R = json.loads(p.read_text(encoding="utf-8"))
    tids = R["target_ids"]
    tset = {t["tid"]: t for t in T}
    out: dict[str, dict] = {}
    for pair, res in R["pairs"].items():                  # "vi>vi", "vi>en", "en>en", "en>vi"
        same = pair[:2] == pair[-2:]
        for sid, cos, idx in zip(R["seed_ids"], res["cos"], res["idx"]):
            for c, j in zip(cos, idx):
                tid = tids[pair[-2:]][j]
                rec = out.setdefault(sid, {"max_same": 0.0, "max_x": 0.0, "hits": []})
                if same:
                    rec["max_same"] = max(rec["max_same"], c)
                else:
                    rec["max_x"] = max(rec["max_x"], c)
                thr = COS_G4_SAME if same else COS_G4_X
                if c >= thr:
                    rec["hits"].append({"pair": pair, "tid": tid, "set": tset[tid]["set"], "cos": round(c, 4)})
    return out


def run() -> dict:
    T = targets()
    S = seed_texts()
    lex = lexical(T, S)
    emb = embedding(T)
    rows, c = [], Counter()
    by_set_drop: Counter = Counter()
    for s in S:
        L = lex.get(s["seed_id"], {})
        E = emb.get(s["seed_id"], {"max_same": None, "max_x": None, "hits": []})
        cos_rev = [h for h in E["hits"] if h["cos"] >= COS_REVIEW]
        review = {"cov50": L.get("rv_cov50", []), "long": L.get("rv_long", []), "cos90": cos_rev}
        g4 = {"ng13": L.get("g4_ng13", []), "tmpl50": L.get("g4_tmpl", []),
              "cos": [h for h in E["hits"] if h["cos"] < COS_REVIEW]}
        drop = any(review.values())
        g4_any = any(g4.values()) or drop
        rec = {"seed_id": s["seed_id"], "drop": drop,
               "review_rules": [k for k, v in review.items() if v],
               "g4_rules": [k for k, v in g4.items() if v],
               "g4_only": (not drop) and any(g4.values()),
               "max_cos_same": E["max_same"], "max_cos_x": E["max_x"], "review": review, "g4": g4}
        rows.append(rec)
        c["drop"] += drop
        c["g4_only"] += rec["g4_only"]
        c["g4_any"] += g4_any
        for k in rec["review_rules"]:
            c[f"review:{k}"] += 1
        for k in rec["g4_rules"]:
            c[f"g4:{k}"] += 1
        if drop:
            sets = {m["set"] for v in review.values() for m in v}
            for st in sets:
                by_set_drop[st] += 1
    write_jsonl(WORK / "decontam.jsonl", rows)
    summ = {"n_seeds": len(S), "n_targets": len(T), "targets_by_set": dict(Counter(t["set"] for t in T)),
            "embedding": bool(emb), **dict(c), "drop_by_target_set": dict(by_set_drop)}
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    write_json(WORK / "decontam_summary.json", summ)
    return summ
