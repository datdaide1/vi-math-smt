"""D5 joint quality-gate audit: 200 items per arm x {plus, minus} x language, ONE judge for both arms.

    E:\\venv_tools\\Scripts\\python -m vi_math_verified_aug.analysis.c3_d5_audit status
    E:\\venv_tools\\Scripts\\python -m vi_math_verified_aug.analysis.c3_d5_audit sample  vi        # draw + freeze samples
    E:\\venv_tools\\Scripts\\python -m vi_math_verified_aug.analysis.c3_d5_audit judge   vi [--max-usd X]
    .venv\\Scripts\\python        -m vi_math_verified_aug.analysis.c3_d5_audit report               # metrics + tables

Inputs: data/train/{Y,N}/{vi,en}/train_10k_{plus,minus}.jsonl (the items that enter training; DELIVER_BRIEF layout).
Fallback when a train file is missing: pool.jsonl filtered on the ``plus`` flag (minus = the whole pool).
Sampling: the final training set (select_v1; before it exists, the minus cell uses select_v1 part 1), the 200 seeds
with the lowest sha1("D5aud|<lang>|<variant>|<seed_id>") -- paired across Y and N, stable under re-selection; the drawn ids
are frozen in data/audit/d5/sample_<arm>_<lang>_<variant>.jsonl (re-running ``sample`` never redraws a frozen cell).
Judge: Opus 5.5 + validity_judge.v2 (vi) / validity_judge_en.v1 (en, same rubric), effort medium, Batches API;
an item that appears in both the plus and the minus sample is judged once (key = hash of lang, seed, op, problem).
Metrics: the T1 §2.5 gate (teacher >= 70%, new M2-M6 >= 60%, leak 0%, absurd <= 3%) + every error class, Wilson 95%.
E1 manipulation check (R8): error(minus) - error(plus) per arm and language (Newcombe), and Y- vs N- per language.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

from .c3_data import ROOT, last_by_key, load_jsonl, outcomes, wellposed

TRAIN = ROOT / "data" / "train"
OUT = ROOT / "data" / "audit" / "d5"
SEEDS = ROOT / "data" / "seeds" / "pool_v1.jsonl"
N_PER_CELL = 100        # lead 2026-11-08 03:30: 100 per cell (budget)
N_TRAIN = 3000            # lead 2026-11-08: n_train = 3,000 per arm x language (select_v1 final files hold exactly n)
ARMS, VARIANTS, LANGS = ("Y", "N"), ("plus", "minus"), ("vi", "en")
GATE = {"teacher": (">=", 0.70), "new": (">=", 0.60), "leak": ("==", 0.0), "absurd": ("<=", 0.03)}


# ------------------------------------------------------------------------------------------------ inputs
def cell_source(arm: str, lang: str, variant: str) -> tuple[Path | None, str]:
    f = TRAIN / "final" / arm / lang / f"train_{variant}.jsonl"      # D5 select_v1 final training sets (primary)
    if f.exists() and f.stat().st_size > 0:
        return f, "final"
    d = TRAIN / arm / lang
    f = d / f"train_10k_{variant}.jsonl"
    if f.exists() and f.stat().st_size > 0:
        return f, "train_10k"
    p = d / "pool.jsonl"
    if p.exists() and p.stat().st_size > 0:
        return p, "pool"
    return None, "missing"


def _text(r: dict) -> str | None:
    for k in ("question", "problem", "text"):
        if isinstance(r.get(k), str) and r[k].strip():
            return r[k]
    return None


def _is_plus(r: dict) -> bool:
    for k in ("plus", "verified"):
        if k in r:
            return bool(r[k])
    return False


def part1_rows(arm: str, lang: str) -> tuple[list[dict], Path] | None:
    """Before the final files exist: the minus cell = the S- first items of select_v1 part 1 (selection_vi_v1.json)."""
    sp = TRAIN / "final" / "selection_vi_v1.json"
    pp = TRAIN / arm / lang / "pool.jsonl"
    if not (sp.exists() and pp.exists()):
        return None
    p1 = json.load(open(sp, encoding="utf-8"))
    want = {p1["slots"][arm][s] for s in p1["seeds_minus"] if s in p1["slots"][arm]}
    rows = [r for r in load_jsonl(pp) if (r.get("slot_id") or r.get("slot")) in want and r.get("minus_eligible", True)]
    return rows, sp


def load_cell_rows(arm: str, lang: str, variant: str) -> tuple[list[dict], str, Path | None]:
    path, kind = cell_source(arm, lang, variant)
    if kind != "final" and variant == "minus":
        pr = part1_rows(arm, lang)
        if pr is not None:
            return [r for r in pr[0] if _text(r)], "select_v1_part1", pr[1]
    if path is None:
        return [], kind, None
    rows = load_jsonl(path)
    if kind == "pool" and variant == "plus":
        rows = [r for r in rows if _is_plus(r)]
    rows = [r for r in rows[:N_TRAIN] if _text(r)]           # the training subsample = first 4,000 rows, file order
    return rows, kind, path


def seeds() -> dict[str, dict]:
    out = {}
    for r in load_jsonl(SEEDS):
        sid = r.get("seed_id") or r.get("id")
        if sid:
            out[sid] = r
    return out


def seed_text(seed: dict | None, lang: str) -> str:
    if not seed:
        return ""
    if lang == "vi":
        return seed.get("problem") or seed.get("text") or ""
    en = seed.get("en")
    if isinstance(en, dict):
        return en.get("problem") or en.get("text") or ""
    return seed.get("en_text") or seed.get("problem_en") or ""


def item_key(lang: str, seed: str, op: str, problem: str) -> str:
    return hashlib.sha256(f"{lang}\x1f{seed}\x1f{op}\x1f{problem}".encode()).hexdigest()[:24]


# ------------------------------------------------------------------------------------------------ sample
def sample_path(arm, lang, variant) -> Path:
    return OUT / f"sample_{arm}_{lang}_{variant}.jsonl"


def draw(arm: str, lang: str, variant: str, n: int = N_PER_CELL, force: bool = False) -> dict:
    sp = sample_path(arm, lang, variant)
    if sp.exists() and not force:
        return {"cell": f"{arm}{'+' if variant == 'plus' else '-'} {lang}", "frozen": True, "n": len(load_jsonl(sp))}
    rows, kind, path = load_cell_rows(arm, lang, variant)
    if not rows:
        return {"cell": f"{arm} {variant} {lang}", "missing": True}
    # paired, stable sampling: the 200 lowest sha1("D5aud|<lang>|<variant>|<seed>") -> Y and N cells of one sign
    # audit the SAME seeds, and a later redraw from the final files keeps every seed that is still selected
    key = lambda r: hashlib.sha1(f"D5aud|{lang}|{variant}|{r.get('seed_id') or r.get('id')}".encode()).hexdigest()  # noqa: E731
    pick = sorted(rows, key=key)[:min(n, len(rows))]
    S = seeds()
    OUT.mkdir(parents=True, exist_ok=True)
    with open(sp, "w", encoding="utf-8") as f:
        for r in pick:
            sd = S.get(r.get("seed_id"))
            st = seed_text(sd, lang)
            op = r.get("op") or ""
            prob = _text(r)
            rec = {"id": r.get("id"), "arm": arm, "lang": lang, "variant": variant, "seed_id": r.get("seed_id"),
                   "op": op, "partner": r.get("partner"), "problem": prob, "seed": st, "answer": r.get("answer"),
                   "plus_flag": r.get("plus"), "source_kind": kind, "source": str(path.relative_to(ROOT)).replace("\\", "/"),
                   "jkey": item_key(lang, st, op, prob), "seed_found": bool(st)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    meta = {"arm": arm, "lang": lang, "variant": variant, "n_population": len(rows), "n_sample": len(pick),
            "sampling": f"{n} lowest sha1('D5aud|<lang>|<variant>|<seed_id>') (paired across arms, stable)",
            "source": str(path.relative_to(ROOT)).replace("\\", "/"), "source_kind": kind,
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rng": f"D5|{arm}|{lang}|{variant}|20261108"}
    (OUT / f"sample_{arm}_{lang}_{variant}.meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def judge_items(lang: str, cells: list[tuple[str, str]] | None = None) -> list[dict]:
    """Unique items to judge for one language (dedup across plus/minus/arms by jkey)."""
    from .c3_judge import OP_INSTR_EN
    from ..pilot_gen.util import OP_INSTR
    S = seeds()
    out, seen = [], set()
    for arm in ARMS:
        for variant in VARIANTS:
            if cells and (arm, variant) not in cells:
                continue
            for r in load_jsonl(sample_path(arm, lang, variant)):
                if r["jkey"] in seen:
                    continue
                seen.add(r["jkey"])
                op = r["op"]
                instr_extra = ""
                if op == "M6" and r.get("partner"):
                    instr_extra = seed_text(S.get(r["partner"]), lang)
                out.append({"key": r["jkey"], "seed": r["seed"], "op": op, "problem": r["problem"], "lang": lang,
                            "op_extra": instr_extra})
    return out


def judge_store():
    from .c3_judge import Store
    return Store(OUT / "judge_opus.jsonl")


# ------------------------------------------------------------------------------------------------ metrics
def _leak_regex(text: str, lang: str) -> bool:
    from ..scale_gen.checks import EN_LEAK, VI_LEAK
    return bool((VI_LEAK if lang == "vi" else EN_LEAK).search(text or ""))


def cell_metrics(arm: str, lang: str, variant: str, J: dict) -> dict | None:
    from .c3_stats import rate
    rows = load_jsonl(sample_path(arm, lang, variant))
    if not rows:
        return None
    ys, nat, new, wp, sev = [], [], [], [], []
    unjudged = 0
    for r in rows:
        j = J.get(r["jkey"])
        if not (j and j.get("parsed") and j.get("labels")):
            unjudged += 1
            continue
        lab = j["labels"]
        y = outcomes(lab, leak_regex=_leak_regex(r["problem"], lang))
        y["teacher"] = 1 - y["not_teacher"]
        y["valid"] = 1 - y["not_valid"]
        y["accept"] = int(y["teacher"] and y["valid"] and not (y["absurd"] or y["leak"] or y["tail"]))
        y["critical"] = int(lab.get("severity") == "critical")
        ys.append((r["op"], y))
        nat.append(lab.get("natural") or 0)
        w = wellposed(j.get("verdict"))
        if w is not None:
            wp.append(w)
        if r["op"] != "M1":
            new.append(int(bool(lab.get("new"))))
    res = {"n_sample": len(rows), "n_judged": len(ys), "unjudged": unjudged}
    keys = ("teacher", "valid", "accept", "absurd", "tail", "leak", "unansw", "fluency", "critical", "not_teacher",
            "not_valid")
    for k in keys:
        res[k] = rate([y[k] for _, y in ys])
    res["neardup"] = rate([y["neardup"] for op, y in ys if op != "M1"])
    res["new"] = rate(new)
    res["wellposed"] = rate(wp)
    res["natural_mean"] = sum(nat) / len(nat) if nat else None
    res["op_mix"] = {op: sum(1 for o, _ in ys if o == op) for op in sorted({o for o, _ in ys})}
    gate = {}
    for k, (cmp_, thr) in GATE.items():
        r = res[k]
        if not r["n"]:
            continue
        point = {">=": r["p"] >= thr, "<=": r["p"] <= thr, "==": r["p"] == thr}[cmp_]
        ci = {">=": r["lo"] >= thr, "<=": r["hi"] <= thr, "==": r["k"] == 0 and r["hi"] <= 0.03}[cmp_]
        gate[k] = {"threshold": f"{cmp_} {thr:.0%}", "pass_point": point, "pass_ci": ci}
    res["gate"] = gate
    res["gate_pass_point"] = all(g["pass_point"] for g in gate.values()) if gate else None
    return res


def manipulation(M: dict) -> dict:
    """E1 manipulation check (R8): verification must lower the error rates: minus - plus per arm and language
    (independent samples; Newcombe); and Y- vs N-, VI vs EN."""
    from .c3_stats import newcombe_unpaired
    out = {}
    for lang in LANGS:
        for arm in ARMS:
            mp, mm = M.get(f"{arm}|{lang}|plus"), M.get(f"{arm}|{lang}|minus")
            if not (mp and mm):
                continue
            d = {}
            for k in ("not_valid", "not_teacher", "absurd", "unansw", "leak", "tail", "critical"):
                a, b = mp[k], mm[k]
                if a["n"] and b["n"]:
                    d[k] = newcombe_unpaired(a["k"], a["n"], b["k"], b["n"])    # minus - plus
            out[f"{arm}|{lang}|minus-plus"] = d
        ym, nm = M.get(f"Y|{lang}|minus"), M.get(f"N|{lang}|minus")
        if ym and nm:
            out[f"Y-N|{lang}|minus"] = {k: newcombe_unpaired(nm[k]["k"], nm[k]["n"], ym[k]["k"], ym[k]["n"])
                                        for k in ("not_valid", "not_teacher", "absurd", "unansw", "critical")
                                        if ym[k]["n"] and nm[k]["n"]}
    for arm in ARMS:
        a, b = M.get(f"{arm}|en|minus"), M.get(f"{arm}|vi|minus")
        if a and b:
            out[f"{arm}|minus|vi-en"] = {k: newcombe_unpaired(a[k]["k"], a[k]["n"], b[k]["k"], b[k]["n"])
                                         for k in ("not_valid", "not_teacher", "absurd", "unansw", "critical")
                                         if a[k]["n"] and b[k]["n"]}
    return out


def report() -> dict:
    J = last_by_key(load_jsonl(OUT / "judge_opus.jsonl"))
    M = {}
    for arm in ARMS:
        for lang in LANGS:
            for variant in VARIANTS:
                m = cell_metrics(arm, lang, variant, J)
                if m:
                    M[f"{arm}|{lang}|{variant}"] = m
    R = {"cells": M, "manipulation": manipulation(M),
         "spend": sum((j.get("cost") or 0) for j in J.values() if not j.get("cache_hit"))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(R, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    (OUT / "tables.md").write_text(tables(R), encoding="utf-8")
    return R


def tables(R: dict) -> str:
    from .c3_stats import fmt_pct
    L = ["# D5 joint gate audit tables (`python -m vi_math_verified_aug.analysis.c3_d5_audit report`)\n",
         "Judge: Opus 5.5, validity_judge.v2 (vi) / validity_judge_en.v1 (en), effort medium, Batches API. Wilson 95%.\n",
         "| cell | n judged | teacher (≥70%) | new M2–M6 (≥60%) | leak (0%) | absurd (≤3%) | gate (point) | valid | accept* | "
         "tail | unanswerable | fluency | near-dup | critical | natural |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, m in R["cells"].items():
        arm, lang, v = key.split("|")
        L.append(f"| {arm}{'+' if v == 'plus' else '−'} {lang} | {m['n_judged']}/{m['n_sample']} | {fmt_pct(m['teacher'])} | "
                 f"{fmt_pct(m['new'])} | {fmt_pct(m['leak'])} | {fmt_pct(m['absurd'])} | "
                 f"{'PASS' if m['gate_pass_point'] else 'FAIL'} ({sum(g['pass_point'] for g in m['gate'].values())}/4) | "
                 f"{fmt_pct(m['valid'])} | {fmt_pct(m['accept'])} | {fmt_pct(m['tail'])} | {fmt_pct(m['unansw'])} | "
                 f"{fmt_pct(m['fluency'])} | {fmt_pct(m['neardup'])} | {fmt_pct(m['critical'])} | "
                 f"{(m['natural_mean'] or 0):.2f} |")
    L.append("\n\\*accept = teacher ∧ valid ∧ ¬absurd ∧ ¬leak ∧ ¬tail (G4 definition).\n")
    L.append("## E1 manipulation check (R8): error rate difference, pp [Newcombe 95% CI]\n")
    L.append("| contrast | not valid | not teacher | absurd | unanswerable | leak | tail | critical |")
    L.append("|---|---|---|---|---|---|---|---|")
    for k, d in R["manipulation"].items():
        cells = []
        for c in ("not_valid", "not_teacher", "absurd", "unansw", "leak", "tail", "critical"):
            x = d.get(c)
            cells.append(f"{100 * x['d']:+.1f} [{100 * x['lo']:+.1f}, {100 * x['hi']:+.1f}]" if x else "–")
        L.append(f"| {k} | " + " | ".join(cells) + " |")
    L.append("\n`minus-plus` = error(−) − error(+); `Y-N|lang|minus` = error(Y−) − error(N−); "
             "`arm|minus|vi-en` = error(VI −) − error(EN −).\n")
    return "\n".join(L)


# ------------------------------------------------------------------------------------------------ CLI
def status() -> list[str]:
    out = []
    for arm in ARMS:
        for lang in LANGS:
            for variant in VARIANTS:
                path, kind = cell_source(arm, lang, variant)
                n = sum(1 for _ in open(path, encoding="utf-8")) if path else 0
                out.append(f"{arm} {lang} {variant}: {kind} {n} rows; sample frozen={sample_path(arm, lang, variant).exists()}")
    return out


def main(argv):
    cmd = argv[0] if argv else "status"
    if cmd == "status":
        print("\n".join(status()))
    elif cmd == "sample":
        lang = argv[1]
        n = int(argv[argv.index("--n") + 1]) if "--n" in argv else N_PER_CELL
        for arm in ARMS:
            for variant in VARIANTS:
                print(draw(arm, lang, variant, n=n))
    elif cmd == "judge":
        from .c3_judge import estimate, judge, run
        lang = argv[1]
        max_usd = float(argv[argv.index("--max-usd") + 1]) if "--max-usd" in argv else None
        cells = None
        if "--cells" in argv:                      # e.g. --cells Y:plus,N:plus
            cells = [tuple(c.split(":")) for c in argv[argv.index("--cells") + 1].split(",")]
        its = judge_items(lang, cells)
        print(f"{lang}: {len(its)} unique items; worst-case ${estimate(its):.2f}", flush=True)
        if "--go" in argv:
            print(run(judge(its, judge_store(), max_usd=max_usd)), flush=True)
    elif cmd == "report":
        R = report()
        print(json.dumps({k: {"n": v["n_judged"], "gate": v["gate_pass_point"]} for k, v in R["cells"].items()}))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
