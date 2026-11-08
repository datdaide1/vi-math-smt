"""Stage ``freeze``: data/seeds/{pool_v1,dev_v1}.jsonl + dropped_v1.jsonl + changelog_v1.jsonl + datacard + SHA256SUMS.

Drop rules (each dropped seed is listed in dropped_v1.jsonl with its reason and evidence):
  D1 decontamination: X5 review rule vs data/eval/* or Vi-ExamMath (``decontam`` stage);
  D2 a must-fix row (R1 corrupt / half-English / R1-R2 major-critical annotation) whose MTA-vi re-translation was
     not accepted;
  D3 gold not recoverable: the English solution does not support the English gold (inherited MATH defect) and no
     manual resolution is listed in ``gold.GOLD_FIXES``.
Everything else is kept; residual issues are listed in the seed's ``flags``.
"""
from __future__ import annotations

import json
import subprocess
import time
from collections import Counter

from .assemble import current_rows
from .common import (CLAUDE_CAP_USD, EN_GSM8K, EN_MATH, OUT, ROOT, SUBJECTS, VI_GSM8K, VI_MATH, WORK, read_jsonl, rel,
                     sha256_file, write_json, write_jsonl)

VERSION = "v1"          # frozen versions are write-once: a new freeze must use a new VERSION (v1.1, ...)
FROZEN = ("v1",)        # never rewritten in place (lead, 2026-11-08)
SCHEMA = "vi-math-seed-pool/1 (S3, 2026-11-07)"




def _git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def _changelog_md(card: dict, changelog: list[dict], dropped: list[dict]) -> str:
    c = card["counts"]
    kinds = Counter(x.get("kind") for x in changelog)
    L = [f"# Seed pool {VERSION} — changelog", "",
         f"Created {card['created']} (git {card['git_head']}). Machine-readable: `changelog_{VERSION}.jsonl` "
         f"(one row per change, with its reason; re-translations carry the old and new problem text) and "
         f"`dropped_{VERSION}.jsonl` (one row per dropped seed, with the rule and its evidence).", "",
         "## v1 (initial freeze)", "",
         f"- pool: {c['pool']['n']} seeds; dev: {c['dev']['n']} seeds; dropped: {c['dropped']} "
         f"({', '.join(f'{k}: {v}' for k, v in sorted(c['dropped_by_rule'].items()))}).",
         f"- changes kept in the frozen files: {len(changelog)} "
         f"({', '.join(f'{k}: {v}' for k, v in sorted(kinds.items()))}).", "",
         "| seed_id | source#index | kind | reason |", "|---|---|---|---|"]
    for x in changelog:
        L.append(f"| {x['seed_id']} | {x.get('source')}#{x.get('index')} | {x.get('kind')} | "
                 f"{str(x.get('reason', '')).replace('|', '/')[:220]} |")
    L += ["", "## Dropped seeds", "", "| seed_id | source#index | split | rule |", "|---|---|---|---|"]
    for x in dropped:
        L.append(f"| {x['seed_id']} | {x['source']}#{x['index']} | {'dev' if x['dev'] else 'pool'} | "
                 f"{'; '.join(r['rule'] for r in x['reasons'])} |")
    return "\n".join(L) + "\n"


def run(version: str | None = None) -> dict:
    global VERSION
    VERSION = version or VERSION
    if VERSION in FROZEN or (OUT / f"pool_{VERSION}.jsonl").exists():
        raise SystemExit(f"pool_{VERSION} is frozen and write-once; run freeze with a new --version (e.g. v1.1)")
    rows = current_rows()
    gold = {g["seed_id"]: g for g in read_jsonl(WORK / "gold.jsonl")}
    dec = {d["seed_id"]: d for d in read_jsonl(WORK / "decontam.jsonl")}
    pool, dev, dropped, changelog = [], [], [], []
    for r in rows:
        sid = r["seed_id"]
        g = gold[sid]
        d = dec[sid]
        typed = g["gold"]
        flags = []
        reasons = []
        if d["drop"]:
            ev = {k: [{"tid": m["tid"], **({"cos": m["cos"]} if "cos" in m else
                                            {"exact": m["exact"], "nodiac": m.get("nodiac"),
                                             "template": m.get("template"), "n_tok": m.get("n_tok")})}
                      for m in v[:3]] for k, v in d["review"].items() if v}
            reasons.append({"rule": "D1 decontamination (X5 review rule)", "evidence": ev})
        if r["must"] and r["provenance"] != "mta_vi":
            reasons.append({"rule": "D2 must-fix row not repaired", "evidence": {"must": r["must"],
                                                                                  "mta_status": r["mta_status"]}})
        fix = g.get("gold_fix")
        if fix and fix["action"] == "drop":
            reasons.append({"rule": "D3 gold not representable / not recoverable", "evidence": {"reason": fix["reason"],
                                                                                              "en_boxes": g.get("en_boxes")}})
        elif fix:
            changelog.append({"seed_id": sid, "source": r["source"], "index": r["index"], "stage": "gold",
                              "kind": f"gold_{fix['action']}", "field": "gold", "reason": fix["reason"],
                              "old": fix["inferred"], "new": typed})
        elif not g["agree_en_solution"]:
            reasons.append({"rule": "D3 English solution does not support the English gold",
                            "evidence": {"gold": g["gold"], "en_boxes": g.get("en_boxes")}})
        if reasons:
            dropped.append({"seed_id": sid, "source": r["source"], "index": r["index"], "dev": r["dev"],
                            "reasons": reasons, "problem_vi": r["problem_vi"], "problem_en": r["problem_en"]})
            continue
        if r["provenance"] != "mta_vi" and r["gate_fails"]:
            flags += ["gate:" + f[:160] for f in r["gate_fails"]]
        if r["solvecheck"] == "FAIL" and r["provenance"] != "mta_vi":
            flags.append("solvecheck:FAIL (Gemini solved the English, not the Vietnamese; MTA-vi re-translation "
                         "not accepted -> original kept)")
        if not g["agree_vi_solution"] and r["provenance"] != "mta_vi":
            flags.append("vi_solution_final_answer_differs_from_gold (reference solution only; gold is from EN)")
        if d["g4_only"]:
            flags.append("decontam:G4-only " + ",".join(d["g4_rules"]))
        for c in r["changes"]:
            changelog.append({"seed_id": sid, "source": r["source"], "index": r["index"], **c,
                              **({"new_problem_vi": r["problem_vi"]} if c.get("kind") == "retranslated" else {})})
        rec = {
            "seed_id": sid, "source": r["source"], "dataset": r["dataset"], "subject": r["subject"],
            "level": r["level"], "index": r["index"], "split": "dev" if r["dev"] else "pool",
            "problem": r["problem_vi"], "solution": r["solution_vi"], "gold": typed,
            "gold_raw": r["gold_raw_en"],
            "en": {"problem": r["problem_en"], "solution": r["solution_en"], "gold_raw": r["gold_raw_en"]},
            "provenance": r["provenance"], "n_changes": len(r["changes"]), "flags": flags,
            "screens": {"solvecheck": r["solvecheck"] or "NOT_RUN", "solvecheck_models": r.get("solvecheck_models"),
                        "solvecheck_coverage": "checked" if r["solvecheck"] else "not_checked", "gold_agree_vi_solution": g["agree_vi_solution"],
                        "gold_agree_en_solution": g["agree_en_solution"], "decontam_g4_only": d["g4_only"],
                        "decontam_g4_rules": d["g4_rules"], "max_cos_same_lang": d["max_cos_same"],
                        "max_cos_cross_lang": d["max_cos_x"]},
            "schema": SCHEMA,
        }
        (dev if r["dev"] else pool).append(rec)
    OUT.mkdir(parents=True, exist_ok=True)
    files = {"pool": OUT / f"pool_{VERSION}.jsonl", "dev": OUT / f"dev_{VERSION}.jsonl",
             "dropped": OUT / f"dropped_{VERSION}.jsonl", "changelog": OUT / f"changelog_{VERSION}.jsonl"}
    write_jsonl(files["pool"], pool)
    write_jsonl(files["dev"], dev)
    write_jsonl(files["dropped"], dropped)
    write_jsonl(files["changelog"], changelog)

    def cnt(rs):
        return {"n": len(rs), "by_source": dict(Counter(x["source"] for x in rs)),
                "by_provenance": dict(Counter(x["provenance"] for x in rs)),
                "by_gold_type": dict(Counter(x["gold"]["answer_type"] for x in rs)),
                "flagged": sum(bool(x["flags"]) for x in rs)}
    drop_c = Counter(rr["rule"].split(" ")[0] for x in dropped for rr in x["reasons"])
    tr = json.loads((WORK / "translationese_summary.json").read_text(encoding="utf-8")) \
        if (WORK / "translationese_summary.json").exists() else None
    dsum = json.loads((WORK / "decontam_summary.json").read_text(encoding="utf-8"))
    inputs = [{"path": rel(VI_GSM8K), "role": "Vietnamese GSM8K train (unmodified input)"},
              {"path": rel(EN_GSM8K), "role": "English GSM8K train (alignment, gold, English mirror)"}]
    for s in SUBJECTS:
        inputs += [{"path": rel(str(VI_MATH).format(subject=s)), "role": f"Vietnamese MATH train {s}"},
                   {"path": rel(str(EN_MATH).format(subject=s)), "role": f"English MATH train {s}"}]
    for i in inputs:
        i["sha256"] = sha256_file(ROOT / i["path"])
    from ..eval.math_verify import GRADER_VERSION
    from .common import client
    card = {
        "name": "Vi-Math seed pool", "version": VERSION, "schema": SCHEMA,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "git_head": _git_head(),
        "script": "python -m vi_math_verified_aug.seeds.build {align,detect,mta,solvecheck,gold,embed-inputs,"
                  "decontam,translationese,freeze}",
        "description": "Shared, frozen seed pool for every generation arm (T1 §2.0): Vietnamese GSM8K train + Vietnamese "
                       "MATH train algebra/prealgebra, each aligned to its English original (en.* fields = the English "
                       "mirror, E1). Faithful translations (NOT rewritten for fluency); damaged rows repaired "
                       "deterministically or re-translated with MTA-vi; typed gold (grader "
                       f"{GRADER_VERSION}); decontaminated against data/eval/* and Vi-ExamMath (X5 review rule).",
        "seed_id": "sha1(f'{source}||{index}')[:12]; DEV = int(seed_id,16) % 20 == 0 (5 %, pilot/calibration only)",
        "sources": inputs,
        "counts": {"pool": cnt(pool), "dev": cnt(dev), "dropped": len(dropped), "dropped_by_rule": dict(drop_c),
                   "changelog_entries": len(changelog)},
        "decontamination": dsum, "translationese": tr,
        "glossary": {"path": "vi_math_verified_aug/seeds/glossary_vi_math_s3.json",
                     "sha256": sha256_file(ROOT / "vi_math_verified_aug/seeds/glossary_vi_math_s3.json")},
        "prompts": {p: sha256_file(ROOT / "prompts" / f"{p}.v1.md")[:16] for p in
                    ("seed_translate", "seed_postedit", "seed_solve", "seed_solve_en", "seed_mqm")},
        "grader_version": GRADER_VERSION,
        "claude_spend_usd": client().costs.summary(), "claude_cap_usd": CLAUDE_CAP_USD,
        "license": "GSM8K: MIT; MATH: MIT; Vietnamese translations: project",
        "report": "docs/research_2026-11/exec/S3_seed_pool.md",
    }
    for k, p in files.items():
        card.setdefault("files", {})[k] = {"path": rel(p), "sha256": sha256_file(p), "rows": sum(1 for _ in open(
            p, encoding="utf-8"))}
    card_p = OUT / f"pool_{VERSION}.datacard.json"
    write_json(card_p, card)
    md_p = OUT / ("CHANGELOG.md" if VERSION == "v1" else f"CHANGELOG_{VERSION}.md")
    md_p.write_text(_changelog_md(card, changelog, dropped), encoding="utf-8", newline="\n")
    sums = [f"{sha256_file(p)}  {p.name}" for p in list(files.values()) + [card_p, md_p]]
    # append-only: earlier versions' entries are kept (v1 lines are never touched by a v1.1 freeze)
    old = (OUT / "SHA256SUMS").read_text(encoding="utf-8").splitlines() if (OUT / "SHA256SUMS").exists() else []
    names = {x.split()[-1] for x in sums}
    keep = [x for x in old if x.strip() and x.split()[-1] not in names]
    (OUT / "SHA256SUMS").write_text("\n".join(keep + sums) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(card["counts"], indent=1))
    return card
