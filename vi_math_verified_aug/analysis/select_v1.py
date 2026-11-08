"""D5 final training-set selection (T1 §2.6, lead decisions 2026-11-07/08): n_train = 3,000 per arm x language.

    .venv\\Scripts\\python -m vi_math_verified_aug.analysis.select_v1 status
    .venv\\Scripts\\python -m vi_math_verified_aug.analysis.select_v1 select vi            [--n 3000] [--minus-seeds independent|shared] [--force]
    .venv\\Scripts\\python -m vi_math_verified_aug.analysis.select_v1 select vi,en         # joint: same seeds in both languages
    .venv\\Scripts\\python -m vi_math_verified_aug.analysis.select_v1 checks vi            # SMD / KS balance report
    modal run -m vi_math_verified_aug.analysis.select_v1_embed_modal --lang vi          # bge-m3 embeddings (<= $0.3)
    .venv\\Scripts\\python -m vi_math_verified_aug.analysis.select_v1 diversity vi         # Vendi, pairwise cos, chrF/cos to seed

Inputs: data/train/{Y,N}/<lang>/pool.jsonl (G5 arm_y.export / D1 scale_gen.finalize), each row with seed_id, slot, op,
``plus`` (the arm's verification outcome; both "+" definitions include the common Haiku-v3 ∧ lite screen).
Seed metadata: data/seeds/pool_v1.jsonl; difficulty proxy per T1 §2.6 = MATH level for MATH seeds, and for GSM8K
seeds the op count of the seed IR formalized by arm Y (data/train/Y/_work/*/formalize.jsonl), fallback the GSM8K
``<<..>>`` calculator-step count.

Design (deviations from T1 §2.6 are documented in exec/D5_selection.md):
* Arms: Y+, Y-, N+, N-. "+" = pool rows with plus=True. "-" = ALL generated pool rows of the arm, drawn at random:
  the item choice for a "-" arm never reads the ``plus`` flag.
* Operator matching: T1 pairs arms on the same (seed, slot), but Y (G5 slots, pi without M6) and N (D1 slots_v1,
  pi with M6) use different slot schedules, so the arms are matched on SEED + operator class; M6 is excluded from
  the matched set (lead decision).
* Seed sets (``--minus-seeds``):
    independent (default) S+ = seeds where Y+ and N+ both have an item -> feeds Y+, N+;
                          S- = seeds where Y- and N- both have an item, chosen with NO verification information
                          -> feeds Y-, N-. Same strata, same stratum quotas (taken from the "+" allocation), same n.
    shared                one S = S+ for all four arms (lead's original wording; the "-" arms are then partly
                          verified, which is measured and reported).
* Strata = source (gsm8k / algebra / prealgebra / other) x seed step-count tercile (cut points per source over the
  candidate seeds). Proportional allocation (largest remainder) to the "+" candidates; <= 1 item per seed per arm
  while seeds remain, then a 2nd item per seed (round 2), and so on. Seed order inside a stratum =
  sha1("D5sel_v1|<seed_id>") (no salt per group: S+ and S- overlap as much as chance allows).
* Item choice inside a seed: "+" arms take the operator class least represented so far in that arm x stratum
  (balances the op mix across strata), ties by sha1("D5item_v1|<arm>|<item id>"); "-" arms draw UNIFORMLY over the
  seed's generated items across all generation waves (pure hash order: no plus flag, no wave number, no op).
  For EN, the VI-chosen slot is preferred.
* G5 generates adaptively (waves J=2/3 only on seeds without a Y+ yet); the ``wave`` field is never read here.
* Reserve: the next 10% of seeds per stratum, in the same order, are listed so EN attrition can be refilled.
* Difficulty p-hat is NOT available yet: ``add_difficulty`` is the hook to re-stratify on (source, tercile, p-hat
  bucket) when a reference-solver run exists.
* Natural-coverage set (T1 secondary, queued for the successor): each arm independently, all its accepted items
  (M6 included), coverage-maximising round-robin over seeds, same n for every arm.
Outputs (never silently overwritten: a previous version is moved to data/train/final/_history/<ts>/):
  data/train/final/{Y,N}/<lang>/train_{plus,minus}.jsonl + SHA256SUMS; data/train/final/manifest_<lang>.json;
  data/train/final/selection_v1_<lang>.json (seed sets, reserve, chosen slots, every slot of the selected seeds);
  data/train/final_natural/{Y,N}/<lang>/train_{plus,minus}.jsonl + SHA256SUMS.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "data" / "train"
FINAL = TRAIN / "final"
NATURAL = TRAIN / "final_natural"
SHARED = TRAIN / "final_shared"          # secondary: one seed set S = S+ for all four arms (sensitivity)
SEED_POOL = ROOT / "data" / "seeds" / "pool_v1.jsonl"
SEED_ALIGNED = ROOT / "data" / "seeds" / "_work" / "aligned.jsonl"
EXCLUDE_SOLVECHECK = {"FAIL", "UNINF"}
SEED_DEV = ROOT / "data" / "seeds" / "dev_v1.jsonl"            # S3 DEV seeds: never selected
VERSION = "select_v1"
N_DEFAULT = 3000
RESERVE_FRAC = 0.10
ARMS = ("Y+", "Y-", "N+", "N-")
EDIT_CLASS = {"M1": "value", "M2": "premise", "M4": "premise", "M3": "structural", "M5": "structural",
              "M6": "compose"}


def h(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def read_jsonl(p: Path):
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def sha256_file(p: Path) -> str:
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def words(t: str | None) -> int:
    return len(re.findall(r"\S+", t or ""))


# ------------------------------------------------------------------------------------------------ inputs
def is_retry(r: dict) -> bool:
    """A row produced by a retry round (D1 EN-N next-slot retry, G5 EN-Y re-render, G5 retry waves): any of the
    retry markers the producers write. First-attempt rows are the R8 manipulation-check population."""
    prov = r.get("provenance") if isinstance(r.get("provenance"), dict) else {}
    for k in ("retry", "rerender", "is_retry", "plus_only"):
        if r.get(k) or prov.get(k):
            return True
    for k in ("attempt", "attempt_no", "round", "retry_round"):
        v = r.get(k, prov.get(k))
        if isinstance(v, (int, float)) and v > 1:
            return True
        if isinstance(v, str) and ("retry" in v.lower() or "rerender" in v.lower()):
            return True
    return False


class Pool:
    """Light index of one arm's pool: per row only the fields the selector needs + the byte offset of the row."""

    def __init__(self, path: Path):
        self.path = path
        self.rows: list[dict] = []
        if not path.exists():
            return
        with open(path, "rb") as f:
            off = 0
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    q = r.get("question") or r.get("problem") or ""
                    slot = r.get("slot_id") or r.get("slot")
                    ver = r["verified"] if "verified" in r else r.get("plus")     # True / False / None (unscreened)
                    self.rows.append({"id": r.get("id") or slot, "seed_id": r.get("seed_id"),
                                      "slot": slot, "op": r.get("op"),
                                      "plus": None if ver is None else bool(ver),
                                      # retry rows are "+"-only (D1 plus_only, attempt_no 2): never minus
                                      "minus_ok": bool(r.get("minus_eligible", True)) and not r.get("plus_only"),
                                      "q_words": words(q), "s_words": words(r.get("solution")), "off": off,
                                      "dataset": r.get("dataset"), "wave": r.get("wave"),
                                      "retry": is_retry(r)})
                off += len(line)

    def full(self, rows: list[dict]) -> list[dict]:
        out = []
        with open(self.path, "rb") as f:
            for r in rows:
                f.seek(r["off"])
                line = f.readline()
                try:
                    full = json.loads(line)
                except ValueError:
                    full = None
                if not isinstance(full, dict) or (full.get("id") or full.get("slot_id") or full.get("slot")) != r["id"]:
                    raise SelectionInvalid(f"{self.path} changed since it was indexed (row {r['id']}); re-run select")
                out.append(full)
        return out


def pool_path(arm: str, lang: str) -> Path:
    return TRAIN / arm / lang / "pool.jsonl"


def load_pools(lang: str) -> dict[str, Pool]:
    return {"Y": Pool(pool_path("Y", lang)), "N": Pool(pool_path("N", lang))}


def allowed_seeds() -> tuple[set[str], dict]:
    """S3 frozen pool_v1 seeds minus dev_v1 seeds, minus seeds whose screens.solvecheck is FAIL or UNINF (doubtful
    gold/text); PENDING2 seeds are allowed and flagged (lead 2026-11-08)."""
    if not SEED_POOL.exists():
        raise SystemExit(f"missing {SEED_POOL}: the selector only runs on S3's frozen pool_v1")
    pool, sc = set(), {}
    for r in read_jsonl(SEED_POOL):
        if r.get("seed_id"):
            pool.add(r["seed_id"])
            sc[r["seed_id"]] = (r.get("screens") or {}).get("solvecheck")
    bad = {s for s, v in sc.items() if v in EXCLUDE_SOLVECHECK}
    dev = {r["seed_id"] for r in read_jsonl(SEED_DEV) if r.get("seed_id")} if SEED_DEV.exists() else set()
    info = {"pool_v1": str(SEED_POOL.relative_to(ROOT)).replace("\\", "/"), "pool_v1_sha256": sha256_file(SEED_POOL),
            "n_pool_v1": len(pool), "dev_v1_sha256": sha256_file(SEED_DEV) if SEED_DEV.exists() else None,
            "n_dev_v1": len(dev), "n_overlap_pool_dev": len(pool & dev),
            "solvecheck_counts": dict(Counter(sc.values())), "excluded_solvecheck": sorted(EXCLUDE_SOLVECHECK),
            "n_excluded_solvecheck": len(bad), "pending2_seeds": sorted(s for s, v in sc.items() if v == "PENDING2")}
    return pool - dev - bad, info


def candidates(pools: dict[str, Pool], *, exclude_ops=("M6",), minus_waves: str = "all",
               allowed: set[str] | None = None) -> dict[str, dict[str, list[dict]]]:
    """arm -> seed -> rows. '-' arms take every generated row (``minus_waves="1"``: only wave-1 / non-adaptive rows,
    a sensitivity option); '+' arms the rows with plus=True."""
    out: dict[str, dict[str, list[dict]]] = {a: defaultdict(list) for a in ARMS}
    for x in ("Y", "N"):
        for r in pools[x].rows:
            if r["op"] in exclude_ops or not r["seed_id"] or (allowed is not None and r["seed_id"] not in allowed):
                continue
            # minus: generated & minus-eligible rows only; plus: any verified row, including "+"-only retry rows
            # (D1 attempt_no 2 / G5 EN2 carry minus_eligible False by design)
            if r["minus_ok"] and (minus_waves == "all" or r.get("wave") in (None, 1, "1")):
                out[f"{x}-"][r["seed_id"]].append(r)
            if r["plus"] is True:                     # None (unscreened) is NOT False and NOT True: not usable yet
                out[f"{x}+"][r["seed_id"]].append(r)
    return out


def seed_meta(seed_ids: set[str] | None = None) -> dict[str, dict]:
    """seed -> {source, steps, steps_src}."""
    src = SEED_POOL if SEED_POOL.exists() else SEED_ALIGNED
    meta: dict[str, dict] = {}
    for r in read_jsonl(src):
        sid = r.get("seed_id")
        if not sid or (seed_ids is not None and sid not in seed_ids):
            continue
        ds = (r.get("dataset") or "").lower()
        sub = (r.get("subject") or "").lower()
        source = "gsm8k" if ds == "gsm8k" else ("prealgebra" if "prealgebra" in sub else
                                                ("algebra" if "algebra" in sub else (sub or ds or "other")))
        sol = r.get("solution_en") or r.get("solution_vi") or r.get("solution") or ""
        calc = len(re.findall(r"<<[^>]*>>", sol))
        lvl = r.get("level")
        lvl_n = int(re.sub(r"\D", "", str(lvl)) or 0) if lvl else None
        meta[sid] = {"source": source, "steps_calc": calc if ds == "gsm8k" else None, "level": lvl_n}
    ir_ops = {}
    for f in sorted((TRAIN / "Y" / "_work").glob("*/formalize.jsonl")):
        for r in read_jsonl(f):
            ir = r.get("ir")
            if r.get("accepted") and isinstance(ir, dict) and isinstance(ir.get("ops"), list):
                ir_ops[r.get("seed_id") or r.get("key")] = len(ir["ops"])
    for sid, m in meta.items():
        if m["source"] != "gsm8k" and m["level"]:              # T1 §2.6: MATH level for MATH seeds
            m["steps"], m["steps_src"] = m["level"], "math_level"
        elif sid in ir_ops:
            m["steps"], m["steps_src"] = ir_ops[sid], "seed_ir_ops"
        elif m["steps_calc"]:
            m["steps"], m["steps_src"] = m["steps_calc"], "gsm8k_calc"
        elif m["level"]:
            m["steps"], m["steps_src"] = m["level"], "math_level"
        else:
            m["steps"], m["steps_src"] = 0, "none"
    return meta


def tercile_cuts(values: list[int]) -> tuple[float, float]:
    v = sorted(values)
    if not v:
        return (0.0, 0.0)
    q = lambda p: v[min(len(v) - 1, int(math.floor(p * (len(v) - 1) + 0.5)))]  # noqa: E731
    return (q(1 / 3), q(2 / 3))


def strata_for(seeds: set[str], meta: dict[str, dict]) -> tuple[dict[str, str], dict]:
    by_src = defaultdict(list)
    for s in seeds:
        m = meta.get(s, {"source": "other", "steps": 0})
        by_src[m["source"]].append(m["steps"])
    cuts = {src: tercile_cuts(v) for src, v in by_src.items()}
    out = {}
    for s in seeds:
        m = meta.get(s, {"source": "other", "steps": 0})
        a, b = cuts[m["source"]]
        t = 0 if m["steps"] <= a else (1 if m["steps"] <= b else 2)
        out[s] = f"{m['source']}|t{t}"
    return out, {k: list(v) for k, v in cuts.items()}


# ------------------------------------------------------------------------------------------------ allocation
def largest_remainder(weights: dict[str, int], n: int) -> dict[str, int]:
    tot = sum(weights.values())
    if tot == 0 or n <= 0:
        return {k: 0 for k in weights}
    raw = {k: n * w / tot for k, w in weights.items()}
    q = {k: int(math.floor(v)) for k, v in raw.items()}
    rest = n - sum(q.values())
    for k in sorted(raw, key=lambda k: (-(raw[k] - q[k]), k))[:rest]:
        q[k] += 1
    return q


def allocate(cap: dict[str, int], strata: dict[str, str], n: int, quota: dict[str, int] | None = None,
             respread: bool = True) -> dict:
    """Seed -> number of items (<= cap), filling ``n`` round by round (round r uses seeds with cap >= r).

    quota None: proportional to the stratum sizes of the round's eligible seeds. quota given: per-stratum item
    targets (from another group); shortfalls are re-spread proportionally over strata with spare capacity."""
    order = defaultdict(list)
    for s in sorted(cap, key=lambda s: h(f"D5sel_v1|{s}")):
        if cap[s] > 0:
            order[strata[s]].append(s)
    counts: dict[str, int] = defaultdict(int)
    if quota is not None:
        need = dict(quota)
    left = n
    r = 1
    while left > 0:
        elig = {k: [s for s in v if cap[s] >= r] for k, v in order.items()}
        elig = {k: v for k, v in elig.items() if v}
        if not elig:
            break
        if quota is None:
            sizes = {k: len(v) for k, v in elig.items()}
            take = largest_remainder(sizes, min(left, sum(sizes.values())))
        else:
            take = {k: min(len(elig.get(k, [])), max(0, need.get(k, 0))) for k in set(need) | set(elig)}
            short = left - sum(take.values())
            spare = {k: len(elig.get(k, [])) - take.get(k, 0) for k in elig}
            spare = {k: v for k, v in spare.items() if v > 0}
            if respread and short > 0 and spare and sum(take.values()) < sum(len(v) for v in elig.values()):
                extra = largest_remainder(spare, min(short, sum(spare.values())))
                for k, e in extra.items():
                    take[k] = take.get(k, 0) + min(e, spare[k])
        got = 0
        for k, t in take.items():
            for s in elig.get(k, [])[:t]:
                counts[s] += 1
                got += 1
            if quota is not None:
                need[k] = need.get(k, 0) - min(t, len(elig.get(k, [])))
        left -= got
        r += 1
        if got == 0:
            break
    return dict(counts)


def reserve_seeds(cap: dict[str, int], strata: dict[str, str], chosen: set[str], k: int) -> list[str]:
    order = defaultdict(list)
    for s in sorted(cap, key=lambda s: h(f"D5sel_v1|{s}")):
        if cap[s] > 0 and s not in chosen:
            order[strata[s]].append(s)
    take = largest_remainder({kk: len(v) for kk, v in order.items()}, min(k, sum(len(v) for v in order.values())))
    return [s for kk, t in take.items() for s in order[kk][:t]]


def pick_items(arm: str, counts: dict[str, int], cand: dict[str, list[dict]], strata: dict[str, str],
               prefer_slots: dict[str, list[str]] | None = None, balance_ops: bool | None = None) -> list[dict]:
    """Per seed choose counts[s] rows: preferred slots first (EN mirrors VI); then
    '+' arms: the op class least used so far in arm x stratum, ties by sha1(arm, item id);
    '-' arms: UNIFORM over the seed's generated items across all generation waves (pure sha1(arm, item id) order;
    lead 2026-11-08), so neither the plus flag nor the wave number nor the op is used.
    Never reads ``plus`` or ``wave``."""
    if balance_ops is None:
        balance_ops = arm.endswith("+")
    used = defaultdict(Counter)
    out = []
    for s in sorted(counts, key=lambda s: (strata[s], h(f"D5sel_v1|{s}"))):
        rows = sorted(cand.get(s, []), key=lambda r: h(f"D5item_v1|{arm}|{r['id']}"))
        st = strata[s]
        pref = set((prefer_slots or {}).get(s, []))
        chosen = []
        for _ in range(counts[s]):
            rest = [r for r in rows if r not in chosen]
            if not rest:
                break
            if balance_ops:
                best = min(rest, key=lambda r: (r["slot"] not in pref, used[st][EDIT_CLASS.get(r["op"], r["op"])],
                                                used[st][r["op"]]))
            else:
                best = min(rest, key=lambda r: r["slot"] not in pref)      # stable: keeps the hash order
            chosen.append(best)
            used[st][EDIT_CLASS.get(best["op"], best["op"])] += 1
            used[st][best["op"]] += 1
        for r in chosen:
            out.append({**r, "_stratum": st})
    return out


# ------------------------------------------------------------------------------------------------ selection
def select(langs: list[str], n: int = N_DEFAULT, minus_seeds: str = "independent", pools_by_lang=None,
           meta: dict | None = None, prefer: dict | None = None, minus_waves: str = "all") -> dict:
    """Joint selection over ``langs`` (capacity = min over languages, so every language gets the same seeds)."""
    pools_by_lang = pools_by_lang or {l: load_pools(l) for l in langs}
    allowed, seed_info = allowed_seeds()
    cands = {l: candidates(pools_by_lang[l], minus_waves=minus_waves, allowed=allowed) for l in langs}
    seed_info["unscreened"] = {f"{x}/{l}": {
        "rows_verified_none": sum(1 for r in pools_by_lang[l][x].rows if r["minus_ok"] and r["plus"] is None),
        "seeds_with_no_true_but_some_none": len({r["seed_id"] for r in pools_by_lang[l][x].rows if r["minus_ok"]
                                                 and r["plus"] is None} - set(cands[l][f"{x}+"]))}
        for l in langs for x in ("Y", "N")}
    seed_info["pool_rows_outside_allowed_seeds"] = {
        f"{x}/{l}": sum(1 for r in pools_by_lang[l][x].rows if r["seed_id"] not in allowed) for l in langs for x in ("Y", "N")}
    all_seeds = set()
    for l in langs:
        for a in ARMS:
            all_seeds |= set(cands[l][a])
    meta = meta if meta is not None else seed_meta(all_seeds)

    def cap_of(arms):
        cap = {}
        for s in all_seeds:
            cap[s] = min(len(cands[l][a].get(s, [])) for l in langs for a in arms)
        return {s: c for s, c in cap.items() if c > 0}

    cap_plus = cap_of(("Y+", "N+"))
    cap_minus = cap_of(("Y-", "N-"))
    strata, cuts = strata_for(set(cap_plus) | set(cap_minus), meta)
    alloc_plus = allocate(cap_plus, strata, n)
    n_eff = sum(alloc_plus.values())
    q_plus = Counter()
    for s, c in alloc_plus.items():
        q_plus[strata[s]] += c
    if minus_seeds == "shared":
        alloc_minus = dict(alloc_plus)
    else:
        alloc_minus = allocate(cap_minus, strata, n_eff, quota=dict(q_plus))
    q_minus = Counter()
    for s, c in alloc_minus.items():
        q_minus[strata[s]] += c
    k_res = int(math.ceil(RESERVE_FRAC * n))
    res_plus = reserve_seeds(cap_plus, strata, set(alloc_plus), k_res)
    res_minus = reserve_seeds(cap_minus, strata, set(alloc_minus), k_res) if minus_seeds != "shared" else []
    # verification selection effect (lead, P3): S+ vs an UNCONSTRAINED random S- (proportional to S- candidates)
    alloc_minus_free = allocate(cap_minus, strata, n_eff) if minus_seeds != "shared" else dict(alloc_plus)
    sel_effect = selection_effect(cap_plus, cap_minus, alloc_plus, alloc_minus_free, alloc_minus, strata, meta)
    picks = {}
    for l in langs:
        picks[l] = {}
        for a in ARMS:
            alloc = alloc_plus if a.endswith("+") else alloc_minus
            pref = (prefer or {}).get(a)
            picks[l][a] = pick_items(a, alloc, cands[l][a], strata, prefer_slots=pref)
    return {"langs": langs, "n_target": n, "n_effective": n_eff, "minus_seeds": minus_seeds, "minus_waves": minus_waves,
            "alloc": {"plus": alloc_plus, "minus": alloc_minus}, "quota": {"plus": dict(q_plus), "minus": dict(q_minus)},
            "strata": {s: strata[s] for s in set(alloc_plus) | set(alloc_minus) | set(res_plus) | set(res_minus)},
            "tercile_cuts": cuts, "reserve": {"plus": res_plus, "minus": res_minus},
            "capacity": {"seeds_plus": len(cap_plus), "items_plus": sum(cap_plus.values()),
                         "seeds_minus": len(cap_minus), "items_minus": sum(cap_minus.values())},
            "steps_src": dict(Counter(meta.get(s, {}).get("steps_src", "none") for s in alloc_plus)),
            "selection_effect": sel_effect, "seed_universe": seed_info, "picks": picks, "pools": pools_by_lang, "meta": meta}


def selection_effect(cap_plus, cap_minus, alloc_plus, alloc_free, alloc_matched, strata, meta) -> dict:
    """Stratum distribution of S+ vs S- (candidates and allocations): the coverage/difficulty shift that conditioning
    on verification induces, and that the matched quotas remove. Total-variation distance + mean seed steps."""
    def dist(seeds):
        c = Counter(strata[s] for s in seeds)
        tot = sum(c.values()) or 1
        return {k: round(v / tot, 4) for k, v in sorted(c.items())}

    def tv(a, b):
        return round(0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b)), 4)

    def steps(seeds):
        v = [meta.get(s, {}).get("steps", 0) for s in seeds]
        return round(sum(v) / len(v), 3) if v else None

    def src(seeds):
        c = Counter(meta.get(s, {}).get("source") for s in seeds)
        tot = sum(c.values()) or 1
        return {k: round(v / tot, 4) for k, v in sorted(c.items(), key=lambda kv: str(kv[0]))}

    sets = {"candidates_S+ (verified in Y and N)": set(cap_plus), "candidates_S- (generated in Y and N)": set(cap_minus),
            "selected_S+": set(alloc_plus), "unconstrained_random_S-": set(alloc_free),
            "selected_S- (quota-matched)": set(alloc_matched)}
    D = {k: dist(v) for k, v in sets.items()}
    return {"strata_share": D, "source_share": {k: src(v) for k, v in sets.items()},
            "mean_seed_steps": {k: steps(v) for k, v in sets.items()},
            "tv_selectedS+_vs_unconstrainedS-": tv(D["selected_S+"], D["unconstrained_random_S-"]),
            "tv_selectedS+_vs_matchedS-": tv(D["selected_S+"], D["selected_S- (quota-matched)"]),
            "verification_rate_by_stratum": {k: round(sum(1 for s in cap_plus if strata[s] == k) /
                                                      max(1, sum(1 for s in cap_minus if strata[s] == k)), 4)
                                             for k in sorted({strata[s] for s in cap_minus})}}


def natural(lang: str, n: int, pools: dict[str, Pool] | None = None) -> dict:
    """T1 secondary set: each arm independently from all its accepted items (M6 included), coverage-maximising."""
    pools = pools or load_pools(lang)
    cand = candidates(pools, exclude_ops=(), allowed=allowed_seeds()[0])
    for a in ("Y-", "N-"):                       # minus = the non-adaptive first item only (lead 2026-11-08 03:15)
        cand[a] = {s: [r for r in v if _is_first(r)] for s, v in cand[a].items()}
        cand[a] = {s: v for s, v in cand[a].items() if v}
    n_nat = min([n] + [sum(len(v) for v in cand[a].values()) for a in ARMS])
    out = {}
    for a in ARMS:
        seeds = sorted(cand[a], key=lambda s: h(f"D5nat_v1|{s}"))
        rows_by = {s: sorted(cand[a][s], key=lambda r: h(f"D5item_v1|{a}|{r['id']}")) for s in seeds}
        sel, r = [], 0
        while len(sel) < n_nat:
            added = 0
            for s in seeds:
                if r < len(rows_by[s]) and len(sel) < n_nat:
                    sel.append({**rows_by[s][r], "_stratum": None})
                    added += 1
            r += 1
            if not added:
                break
        out[a] = sel
    return {"n": n_nat, "picks": out}


# ------------------------------------------------------------------------------------------------ write
def _archive(paths: list[Path]):
    ex = [p for p in paths if p.exists()]
    if not ex:
        return None
    dst = FINAL / "_history" / time.strftime("%Y%m%dT%H%M%S")
    for p in ex:
        rel = p.relative_to(TRAIN)
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst / rel)
    return dst


class SelectionInvalid(RuntimeError):
    pass


def validate_set(lang: str, rows_by_arm: dict[str, list[dict]], allowed: set[str] | None) -> None:
    """All four arms present, one n everywhere, ids unique, every row a real item (non-empty question, a seed in the
    allowed pool_v1 universe, the arm's sign respected: '+' rows verified). Raises SelectionInvalid."""
    ns = {a: len(rows_by_arm.get(a, [])) for a in ARMS}
    if set(rows_by_arm) != set(ARMS) or len(set(ns.values())) != 1 or min(ns.values()) == 0:
        raise SelectionInvalid(f"{lang}: arms/n invalid {ns}")
    for a, rows in rows_by_arm.items():
        ids = [r.get("id") for r in rows]
        if len(set(ids)) != len(ids):
            raise SelectionInvalid(f"{lang} {a}: duplicate ids")
        for r in rows:
            q = r.get("question") or r.get("problem") or ""
            if len(q.strip()) < 20 or len(set(q.split())) < 4:
                raise SelectionInvalid(f"{lang} {a}: placeholder/empty question in {r.get('id')}")
            if allowed is not None and r.get("seed_id") not in allowed:
                raise SelectionInvalid(f"{lang} {a}: seed {r.get('seed_id')} outside pool_v1 universe")
            if a.endswith("+"):
                v = r["verified"] if "verified" in r else r.get("plus")
                if v is not True:
                    raise SelectionInvalid(f"{lang} {a}: unverified row {r.get('id')} in a plus arm")


def _write_set(base: Path, lang: str, picks: dict[str, list[dict]], pools: dict[str, Pool], tag: dict,
               allowed: set[str] | None = None) -> dict:
    """Stage every file (4 arms + 2 SHA256SUMS) as *.staging, validate the whole set, then rename atomically.
    Nothing is written to the final paths unless the complete set validates."""
    base = Path(base)
    if TRAIN.resolve() not in base.resolve().parents:
        raise SelectionInvalid(f"refusing to write outside {TRAIN}: {base}")
    full_by_arm = {}
    for x in ("Y", "N"):
        for sign in ("+", "-"):
            sel = picks[f"{x}{sign}"]
            full = pools[x].full(sel)
            full_by_arm[f"{x}{sign}"] = [{**r, "selection": {**tag, "arm": f"{x}{sign}", "stratum": s["_stratum"],
                                                              "rank": k}} for k, (r, s) in enumerate(zip(full, sel))]
    validate_set(lang, full_by_arm, allowed)
    staged, files = [], {}
    try:
        for x in ("Y", "N"):
            d = base / x / lang
            d.mkdir(parents=True, exist_ok=True)
            sums = {}
            for sign, name in (("+", "plus"), ("-", "minus")):
                p = d / f"train_{name}.jsonl"
                tmp = d / f"train_{name}.jsonl.staging"
                with open(tmp, "w", encoding="utf-8", newline="\n") as f:
                    for r in full_by_arm[f"{x}{sign}"]:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                sums[p.name] = sha256_file(tmp)
                staged.append((tmp, p))
                files[str(p.relative_to(ROOT)).replace("\\", "/")] = {"rows": len(full_by_arm[f"{x}{sign}"]),
                                                                        "sha256": sums[p.name]}
            tmp = d / "SHA256SUMS.staging"
            tmp.write_text("".join(f"{v}  {k}\n" for k, v in sums.items()), encoding="utf-8", newline="\n")
            staged.append((tmp, d / "SHA256SUMS"))
        for tmp, _ in staged:                                    # re-verify the staged bytes before publishing
            if tmp.name.startswith("train_"):
                n = sum(1 for _ in open(tmp, encoding="utf-8"))
                if n != len(full_by_arm[next(iter(full_by_arm))]):
                    raise SelectionInvalid(f"staged {tmp} has {n} rows")
        for tmp, p in staged:
            os.replace(tmp, p)
    finally:
        for tmp, _ in staged:
            if tmp.exists():
                tmp.unlink()
    return files


def slots_of(seeds: set[str]) -> dict[str, list[str]]:
    """Every slot key of the given seeds in the Y and N slot schedules (for the EN generation runs)."""
    out = {"Y": [], "N": []}
    ysl = [p for p in (TRAIN / "Y" / "_work").glob("*/slots.jsonl")]
    for p in ysl:
        out["Y"] += [r["key"] for r in read_jsonl(p) if r.get("seed_id") in seeds]
    nsl = TRAIN / "slots_v1.jsonl"
    if nsl.exists():
        out["N"] = [r["key"] for r in read_jsonl(nsl) if r.get("seed_id") in seeds]
    return {k: sorted(set(v)) for k, v in out.items()}


def write(sel: dict, force: bool = False, base: Path | None = None) -> dict:
    base = base or FINAL
    tag = {"version": VERSION, "langs": sel["langs"], "minus_seeds": sel["minus_seeds"]}
    # S0 slice (lead 06:00): S+ seeds whose "+" items are first attempts in EVERY arm x language (no retry item);
    # the first-attempt verification rates are the R8 manipulation-check population
    retry_seeds = {r["seed_id"] for P in sel["picks"].values() for a in ("Y+", "N+") for r in P[a] if r.get("retry")}
    s0 = sorted(set(sel["alloc"]["plus"]) - retry_seeds)
    out = {}
    for lang in sel["langs"]:
        targets = [base / x / lang / f"train_{s}.jsonl" for x in ("Y", "N") for s in ("plus", "minus")]
        if any(p.exists() for p in targets) and not force:
            raise SystemExit(f"{lang}: final files exist; re-run with --force (the old version is archived)")
        arch = _archive(targets + [base / f"manifest_{lang}.json", base / f"selection_v1_{lang}.json"])
        files = _write_set(base, lang, sel["picks"][lang], sel["pools"][lang], tag, allowed=allowed_seeds()[0])
        chosen = {a: sorted(r["slot"] for r in sel["picks"][lang][a]) for a in ARMS}
        seeds_all = set(sel["alloc"]["plus"]) | set(sel["alloc"]["minus"]) | set(sel["reserve"]["plus"]) | \
            set(sel["reserve"]["minus"])
        selection = {"version": VERSION, "lang": lang, "joint_langs": sel["langs"], "n": sel["n_effective"],
                     "seeds_plus": sorted(sel["alloc"]["plus"]), "seeds_minus": sorted(sel["alloc"]["minus"]),
                     "items_per_seed_plus": sel["alloc"]["plus"], "items_per_seed_minus": sel["alloc"]["minus"],
                     "reserve_plus": sel["reserve"]["plus"], "reserve_minus": sel["reserve"]["minus"],
                     "chosen_slots": chosen, "all_slots_of_selected_and_reserve_seeds": slots_of(seeds_all)}
        (base / f"selection_v1_{lang}.json").write_text(json.dumps(selection, indent=1), encoding="utf-8")
        manifest = {"version": VERSION, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "workstream": "D5",
                    "lang": lang, "joint_langs": sel["langs"], "n_target": sel["n_target"],
                    "n_effective": sel["n_effective"], "minus_seeds": sel["minus_seeds"],
                    "minus_waves": sel.get("minus_waves", "all"), "excluded_ops": ["M6"],
                    "strata": "source x seed step-count tercile", "tercile_cuts": sel["tercile_cuts"],
                    "steps_source_counts": sel["steps_src"], "quota": sel["quota"], "capacity": sel["capacity"],
                    "reserve_frac": RESERVE_FRAC, "hashes": {"seed_order": "sha1('D5sel_v1|<seed>')",
                                                             "item_tiebreak": "sha1('D5item_v1|<arm>|<id>')"},
                    "inputs": {f"{x}/{lang}/pool.jsonl": sha256_file(pool_path(x, lang)) for x in ("Y", "N")},
                    "seed_meta_source": str((SEED_POOL if SEED_POOL.exists() else SEED_ALIGNED).relative_to(ROOT)),
                    "seed_universe": sel["seed_universe"],
                    "files": files, "archived_previous": str(arch) if arch else None,
                    "difficulty": "p-hat pending (no reference-solver run yet); see add_difficulty()",
                    "summary": summarize(sel, lang), "selection_effect": sel["selection_effect"],
                    "first_attempt_verification_rate": {
                        x: (lambda rows: {"verified": sum(r["plus"] is True for r in rows),
                                          "decided": sum(r["plus"] is not None for r in rows),
                                          "rate": round(sum(r["plus"] is True for r in rows) /
                                                        max(1, sum(r["plus"] is not None for r in rows)), 4)})(
                            [r for r in sel["pools"][lang][x].rows if r["minus_ok"] and not r.get("retry")
                             and r["op"] != "M6"]) for x in ("Y", "N")},
                    "s0_slice": {"n": len(s0), "n_plus_seeds_with_a_retry_item": len(retry_seeds),
                                 "rule": "S+ seeds whose + items are first attempts in all 4 arm x language cells; "
                                         "seed list in selection_vi_v1.json['s0_plus_seeds']"},
                    "outcome_information_used": "none from student models or eval sets: only the generation pipelines' "
                                                "own verification flags (plus), decontamination already applied in the "
                                                "pools, operator, and seed metadata (source, IR/solution step count)"}
        (base / f"manifest_{lang}.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False),
                                                       encoding="utf-8")
        out[lang] = manifest
    return out


def write_natural(lang: str, nat: dict, pools: dict[str, Pool]) -> dict:
    targets = [NATURAL / x / lang / f"train_{s}.jsonl" for x in ("Y", "N") for s in ("plus", "minus")]
    _archive(targets)
    files = _write_set(NATURAL, lang, nat["picks"], pools, {"version": VERSION + "_natural", "lang": lang})
    man = {"version": VERSION + "_natural", "lang": lang, "n": nat["n"], "files": files,
           "rule": "each arm independently, all accepted items (M6 included), round-robin over seeds in "
                   "sha1('D5nat_v1|<seed>') order, item order sha1('D5item_v1|<arm>|<id>')",
           "unique_seeds": {a: len({r["seed_id"] for r in nat["picks"][a]}) for a in ARMS},
           "ops": {a: dict(Counter(r["op"] for r in nat["picks"][a])) for a in ARMS}}
    (NATURAL / f"manifest_{lang}.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    return man


# ------------------------------------------------------------------------------------------------ checks
def smd(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    sd = math.sqrt((va + vb) / 2)
    return 0.0 if sd == 0 else (ma - mb) / sd


def ks(a: list[float], b: list[float]) -> dict:
    """Two-sample KS statistic and asymptotic p (Kolmogorov distribution)."""
    a, b = sorted(a), sorted(b)
    i = j = 0
    d = 0.0
    while i < len(a) and j < len(b):
        x = min(a[i], b[j])
        while i < len(a) and a[i] <= x:
            i += 1
        while j < len(b) and b[j] <= x:
            j += 1
        d = max(d, abs(i / len(a) - j / len(b)))
    ne = len(a) * len(b) / (len(a) + len(b))
    lam = (math.sqrt(ne) + 0.12 + 0.11 / math.sqrt(ne)) * d
    p = 2 * sum((-1) ** (k - 1) * math.exp(-2 * k * k * lam * lam) for k in range(1, 101))
    return {"D": d, "p": min(1.0, max(0.0, p))}


def summarize(sel: dict, lang: str) -> dict:
    P = sel["picks"][lang]
    vi_slot = {a: {r["seed_id"]: r["slot"] for r in sel["picks"]["vi"][a]} for a in ("Y+", "N+")}         if "vi" in sel["picks"] else {}
    meta = sel["meta"]
    res = {"n": {a: len(P[a]) for a in ARMS}, "unique_seeds": {a: len({r["seed_id"] for r in P[a]}) for a in ARMS},
           "ops": {a: dict(sorted(Counter(r["op"] for r in P[a]).items())) for a in ARMS},
           "edit_class": {a: dict(sorted(Counter(EDIT_CLASS.get(r["op"], r["op"]) for r in P[a]).items())) for a in ARMS},
           "source": {a: dict(sorted(Counter(meta.get(r["seed_id"], {}).get("source") for r in P[a]).items()))
                      for a in ARMS},
           "minus_items_that_are_plus": {a: sum(r["plus"] is True for r in P[a]) for a in ("Y-", "N-")},
           "minus_items_unscreened": {a: sum(r["plus"] is None for r in P[a]) for a in ("Y-", "N-")},
           "minus_share_plus": {a: round(sum(r["plus"] is True for r in P[a]) /
                                         max(1, sum(r["plus"] is not None for r in P[a])), 4) for a in ("Y-", "N-")},
           "waves": {a: dict(sorted(Counter(str(r.get("wave")) for r in P[a]).items())) for a in ARMS},
           "retry_items": {a: sum(bool(r.get("retry")) for r in P[a]) for a in ARMS},
           "plus_slot_differs_from_vi": ({a: sum(1 for r in P[a] if r["slot"] != vi_slot[a].get(r["seed_id"]))
                                          for a in ("Y+", "N+")} if lang != "vi" and "vi" in sel["picks"] else None),
           "seed_overlap_plus_minus": len(set(sel["alloc"]["plus"]) & set(sel["alloc"]["minus"])),
           "selected_pending2_seeds": {a: sum(1 for r in P[a] if r["seed_id"] in
                                               set(sel.get("seed_universe", {}).get("pending2_seeds", [])))
                                       for a in ARMS}}
    feats = {"seed_steps": lambda r: meta.get(r["seed_id"], {}).get("steps", 0),
             "question_words": lambda r: r["q_words"], "solution_words": lambda r: r["s_words"]}
    for src in sorted({m.get("source") for m in meta.values()}):
        feats[f"source={src}"] = (lambda src_: (lambda r: float(meta.get(r["seed_id"], {}).get("source") == src_)))(src)
    pairs = [("Y+", "N+"), ("Y-", "N-"), ("Y+", "Y-"), ("N+", "N-"), ("Y+", "N-"), ("Y-", "N+")]
    bal = {}
    for f, fn in feats.items():
        bal[f] = {f"{a} vs {b}": round(smd([fn(r) for r in P[a]], [fn(r) for r in P[b]]), 4) for a, b in pairs}
    res["smd"] = bal
    res["smd_max_abs"] = max(abs(v) for d in bal.values() for v in d.values() if not math.isnan(v)) if bal else None
    res["ks"] = {f"{a} vs {b}": {k: ks([float(r[k]) for r in P[a]], [float(r[k]) for r in P[b]])
                                 for k in ("q_words", "s_words")} for a, b in pairs}
    return res


def add_difficulty(sel: dict, p_hat: dict[str, float], buckets=(0, .25, .5, .75, 1.0)) -> dict:
    """HOOK (successor): re-stratify on (source, tercile, p-hat bucket) once a reference-solver pass rate exists per
    item id. Returns {stratum -> per-arm counts} for the chosen items; re-running ``select`` with a strata function
    that appends the bucket gives the T1 §2.6 difficulty-matched set."""
    def bucket(p):
        return min(buckets, key=lambda b: abs(b - p))
    out = defaultdict(Counter)
    for lang, P in sel["picks"].items():
        for a, rows in P.items():
            for r in rows:
                if r["id"] in p_hat:
                    out[f"{lang}|{r['_stratum']}|p{bucket(p_hat[r['id']])}"][a] += 1
    return {k: dict(v) for k, v in out.items()}


# ------------------------------------------------------------------------------------------------ CLI
def status():
    for lang in ("vi", "en"):
        for x in ("Y", "N"):
            p = pool_path(x, lang)
            n = sum(1 for _ in open(p, encoding="utf-8")) if p.exists() else 0
            m = (TRAIN / x / lang / "manifest.json").exists()
            print(f"{x} {lang}: pool rows {n}; manifest {m}")


def main(argv):
    cmd = argv[0] if argv else "status"
    if cmd == "status":
        status()
        return
    langs = argv[1].split(",")
    n = int(argv[argv.index("--n") + 1]) if "--n" in argv else N_DEFAULT
    mode = argv[argv.index("--minus-seeds") + 1] if "--minus-seeds" in argv else "independent"
    if cmd in ("select", "dry"):
        prefer = None
        if "--prefer-vi" in argv and "vi" not in langs:             # EN-only: mirror the VI chosen slots
            vs = json.load(open(FINAL / "selection_v1_vi.json", encoding="utf-8"))
            prefer = {}
            for a, slots in vs["chosen_slots"].items():
                d = defaultdict(list)
                for k in slots:
                    d[k.split(":")[0]].append(k)
                prefer[a] = d
        sel = select(langs, n=n, minus_seeds=mode, prefer=prefer)
        for lang in langs:
            s = summarize(sel, lang)
            print(lang, json.dumps({"n_effective": sel["n_effective"], "capacity": sel["capacity"],
                                    "n": s["n"], "unique_seeds": s["unique_seeds"],
                                    "smd_max_abs": s["smd_max_abs"], "minus_items_that_are_plus":
                                        s["minus_items_that_are_plus"], "overlap": s["seed_overlap_plus_minus"]}))
        if cmd == "select":
            man = write(sel, force="--force" in argv)
            if mode == "independent":                                  # secondary: shared S (sensitivity)
                sel_sh = select(langs, n=n, minus_seeds="shared", pools_by_lang=sel["pools"], meta=sel["meta"],
                                prefer=prefer)
                write(sel_sh, force=True, base=SHARED)
            for lang in langs:
                nat = natural(lang, n, sel["pools"][lang])
                write_natural(lang, nat, sel["pools"][lang])
            print("wrote", {l: man[l]["files"] for l in man})
    elif cmd == "part1":
        for lang in langs:
            p1 = part1_minus(lang, n)
            print(write_part1(p1), p1["n"], p1["counts"])
    elif cmd in ("part2", "part2dry"):
        p1 = json.load(open(FINAL / "selection_vi_v1.json", encoding="utf-8"))
        p1["_sha256"] = sha256_file(FINAL / "selection_vi_v1.json")
        sel = select_part2(langs, p1, n)
        for lang in langs:
            s_ = summarize(sel, lang)
            print(lang, json.dumps({"n_effective": sel["n_effective"], "n": s_["n"], "capacity": sel["capacity"],
                                    "smd_max_abs": s_["smd_max_abs"], "subst": len(sel["seed_universe"]["minus_substitutions"]),
                                    "dropped": len(sel["seed_universe"]["minus_dropped_no_reserve"])}))
        if cmd == "part2":
            man = write(sel, force="--force" in argv)
            sh = shared_from(sel)
            write(sh, force=True, base=SHARED)
            for lang in langs:
                write_natural(lang, natural(lang, n, sel["pools"][lang]), sel["pools"][lang])
                write_opmatched(sel, lang)
            # selection file: add the S+ part
            p1["seeds_plus"] = sorted(sel["alloc"]["plus"])
            p1["reserve_plus"] = sel["reserve"]["plus"]
            p1["slots_plus"] = {a: sorted(r["slot"] for r in sel["picks"][langs[0]][a]) for a in ("Y+", "N+")}
            p1["part"] = 2
            p1["final_minus_seeds"] = sorted(sel["alloc"]["minus"])
            rs = {r["seed_id"] for P in sel["picks"].values() for a in ("Y+", "N+") for r in P[a] if r.get("retry")}
            p1["s0_plus_seeds"] = sorted(set(sel["alloc"]["plus"]) - rs)
            p1["slots_plus_by_lang"] = {l: {a: {r["seed_id"]: r["slot"] for r in sel["picks"][l][a]} for a in ("Y+", "N+")}
                                        for l in langs}
            p1.pop("_sha256", None)
            write_part1(p1)
            print("wrote", {l: man[l]["files"] for l in man})
    elif cmd == "project":
        print(json.dumps(project_n(ROOT / "data" / "audit" / "d5" / "projection.jsonl"), indent=1))
    elif cmd == "funnel":
        print(json.dumps(funnel(), indent=1))
    elif cmd == "coverage":
        print(json.dumps(coverage_report(), indent=1))
    elif cmd == "checks":
        for lang in langs:
            print(json.dumps(json.load(open(FINAL / f"manifest_{lang}.json", encoding="utf-8"))["summary"], indent=1))
    elif cmd == "diversity":
        from .select_v1_diversity import run as div
        for lang in langs:
            print(json.dumps(div(lang), indent=1))
    else:
        raise SystemExit(__doc__)




# ================================================================================================ two-part selection
# Lead decision 2026-11-08 ~03:15: the minus arms draw ONLY from the non-adaptive first items (Y wave-1 = G5 slot 1,
# N slot j=1), one per seed, chosen before any verification -> S- depends on no verification outcome and is
# selected first (part 1, from the eligible population's own strata). S+ (any wave) is selected second (part 2) with
# quotas matched to S-'s strata.
Y_WORK = TRAIN / "Y" / "_work"


def y_wave1_eligible(lang: str, run: str | None = None) -> dict[str, dict]:
    """seed -> wave-1 Y item info that passes the Y- checks available now (from G5's stores, read-only):
    tag_ok ∧ leak_ok ∧ ¬answer_untagged ∧ solver trace with \boxed{}. Decontamination is applied later by G5's export;
    seeds that fail it are refilled from the reserve."""
    run_dir = Y_WORK / (run or ("vi_scale" if lang == "vi" else f"{lang}_scale"))
    cfg = json.load(open(run_dir / "config.json", encoding="utf-8"))
    var = (cfg.get("variants") or ["R2b"])[0]
    gates = {r.get("slot"): r for r in read_jsonl(run_dir / "gates.jsonl")} if (run_dir / "gates.jsonl").exists() else {}
    sol = {r["key"]: r for r in read_jsonl(run_dir / f"solve_{var}.jsonl")} if (run_dir / f"solve_{var}.jsonl").exists() else {}
    out = {}
    for r in read_jsonl(run_dir / f"render_{var}.jsonl"):
        if int(r.get("slot", 0)) != 1 or not r.get("tag_ok"):
            continue
        g = gates.get(r["key"])
        if g is None or not g.get("leak_ok") or g.get("answer_untagged"):
            continue
        tr = (sol.get(r["key"]) or {}).get("trace") or ""
        if r"\boxed" not in tr:
            continue
        out[r["seed_id"]] = {"slot": r["key"], "op": r.get("op"), "variant": var}
    return out


def n_slot1_eligible(lang: str) -> dict[str, dict]:
    out = {}
    for r in read_jsonl(pool_path("N", lang)):
        slot = r.get("slot_id") or r.get("slot") or ""
        if slot.endswith(":1") and r.get("minus_eligible", True) and r.get("op") != "M6":   # M6 excluded (lead)
            out[r["seed_id"]] = {"slot": slot, "op": r.get("op")}
    return out


def part1_minus(lang: str = "vi", n: int = N_DEFAULT, reserve_frac: float = RESERVE_FRAC) -> dict:
    allowed, info = allowed_seeds()
    ye, ne = y_wave1_eligible(lang), n_slot1_eligible(lang)
    elig = set(ye) & set(ne) & allowed
    meta = seed_meta(elig)
    strata, cuts = strata_for(elig, meta)
    cap = {s: 1 for s in elig}
    alloc = allocate(cap, strata, n)
    quota = Counter(strata[s] for s in alloc)
    res = reserve_seeds(cap, strata, set(alloc), int(math.ceil(reserve_frac * n)))
    pop = Counter(strata[s] for s in elig)
    return {"part": 1, "lang": lang, "n_target": n, "n": len(alloc), "seeds_minus": sorted(alloc),
            "reserve_minus": res, "quota_minus": dict(sorted(quota.items())),
            "eligible_population_strata": dict(sorted(pop.items())), "tercile_cuts": cuts,
            "strata": {s: strata[s] for s in list(alloc) + res},
            "slots": {"Y": {s: ye[s]["slot"] for s in list(alloc) + res}, "N": {s: ne[s]["slot"] for s in list(alloc) + res}},
            "ops": {"Y": dict(Counter(ye[s]["op"] for s in alloc)), "N": dict(Counter(ne[s]["op"] for s in alloc))},
            "counts": {"y_wave1_eligible": len(ye), "n_slot1_eligible": len(ne), "allowed_seeds": len(allowed),
                       "eligible_both": len(elig)},
            "seed_universe": {k: v for k, v in info.items() if k != "pending2_seeds"},
            "pending2_selected": sum(1 for s in alloc if s in set(info["pending2_seeds"])),
            "inputs": {"N_pool_sha256": sha256_file(pool_path("N", lang)),
                       "Y_run": str((Y_WORK / ("vi_scale" if lang == "vi" else f"{lang}_scale")).relative_to(ROOT)).replace("\\", "/")},
            "rule": "S- = seeds (pool_v1 \ dev_v1 \ solvecheck FAIL/UNINF) with an eligible non-adaptive first item in "
                    "BOTH arms (Y wave-1 slot 1: tag_ok, leak_ok, no untagged answer, boxed solver trace; N slot j=1: "
                    "minus_eligible, op != M6 -- M6 is excluded from the matched sets because Y has no M6). Strata = source x seed step tercile over this eligible population; proportional "
                    "quotas (largest remainder); seed order sha1('D5sel_v1|<seed>'); reserve = next 10% per stratum. "
                    "No verification outcome is read.",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "version": VERSION}


def write_part1(sel: dict) -> Path:
    FINAL.mkdir(parents=True, exist_ok=True)
    p = FINAL / f"selection_{sel['lang']}_v1.json"
    _archive([p])
    p.write_text(json.dumps(sel, indent=1), encoding="utf-8")
    return p


def assign_strata(seeds, meta: dict, cuts: dict) -> dict[str, str]:
    out = {}
    for s in seeds:
        m = meta.get(s, {"source": "other", "steps": 0})
        a, b = cuts.get(m["source"], (0, 0))
        out[s] = f"{m['source']}|t{0 if m['steps'] <= a else (1 if m['steps'] <= b else 2)}"
    return out


def _is_first(r: dict) -> bool:
    return str(r.get("slot") or "").endswith(":1") and r.get("wave") in (None, 1, "1")


def select_part2(langs: list[str], part1: dict, n: int | None = None, pools_by_lang=None,
                 check_langs: list[str] | None = None) -> dict:
    """S+ (any wave, verified in Y and N in every language) with quotas = part-1 S- quotas, and the S- items of
    part 1 (non-adaptive first items), with reserve substitution for seeds whose first item is missing from a final
    pool (e.g. failed export decontam) in any arm x language. Returns the same structure as ``select``."""
    n = n or part1["n_target"]
    pools_by_lang = pools_by_lang or {l: load_pools(l) for l in langs}
    allowed, seed_info = allowed_seeds()
    cands = {l: candidates(pools_by_lang[l], allowed=allowed) for l in langs}
    idx = {l: {x: {r["slot"]: r for r in pools_by_lang[l][x].rows if r["minus_ok"]} for x in ("Y", "N")} for l in langs}
    # S- must stay ONE cross-language set: the first items must also be eligible in every other language whose pool
    # already exists (lead 04:00: N-EN frozen before Y-EN) -> refills happen once, for VI and EN together
    check_langs = check_langs if check_langs is not None else ["vi", "en"]
    chk = {}
    for l in check_langs:
        for x in ("Y", "N"):
            if l in langs:
                chk[(l, x)] = set(idx[l][x])
            elif pool_path(x, l).exists() and pool_path(x, l).stat().st_size > 0:
                chk[(l, x)] = {r["slot"] for r in Pool(pool_path(x, l)).rows if r["minus_ok"]}
    cuts = {k: tuple(v) for k, v in part1["tercile_cuts"].items()}
    all_seeds = set().union(*[set(cands[l][a]) for l in langs for a in ARMS]) | set(part1["seeds_minus"]) | \
        set(part1["reserve_minus"])
    meta = seed_meta(all_seeds)
    strata = assign_strata(all_seeds, meta, cuts)
    # ---- S- from part 1 (+ reserve substitution within stratum)
    def first_ok(s):
        return all(part1["slots"][x].get(s) in sl for (l, x), sl in chk.items())
    reserve = defaultdict(list)
    for s in part1["reserve_minus"]:
        reserve[part1["strata"][s]].append(s)
    minus, subst, dropped = [], [], []
    for s in part1["seeds_minus"]:
        if first_ok(s):
            minus.append(s)
            continue
        st = part1["strata"][s]
        rep = None
        while reserve[st]:
            c = reserve[st].pop(0)
            if first_ok(c):
                rep = c
                break
        if rep:
            minus.append(rep)
            subst.append((s, rep))
        else:
            dropped.append(s)
    alloc_minus = {s: 1 for s in minus}
    q_minus = Counter(strata[s] if s in strata else part1["strata"][s] for s in minus)
    # ---- S+ with S- quotas
    cap_plus = {}
    for s in all_seeds:
        c = min(len(cands[l][a].get(s, [])) for l in langs for a in ("Y+", "N+"))
        if c > 0:
            cap_plus[s] = 1                 # one item per seed, as in S- (symmetric coverage; lead 04:10)
    n_eff = min(n, len(minus))
    # strict per-stratum quotas (no re-spread): S+ takes min(S- quota, available) in every stratum, and S- is then
    # trimmed to exactly the same per-stratum counts -> one n and identical strata quotas in all four arms
    alloc_plus = allocate(cap_plus, strata, n_eff, quota=dict(q_minus), respread=False)
    n_plus = sum(alloc_plus.values())
    q_plus = Counter()
    for s, c in alloc_plus.items():
        q_plus[strata[s]] += c
    res_plus = reserve_seeds(cap_plus, strata, set(alloc_plus), int(math.ceil(RESERVE_FRAC * n)))
    # equal n in every arm: trim S- (last seeds in hash order within over-quota strata) if S+ fell short
    if n_plus < len(minus):
        keep, need = [], dict(q_plus)
        for s in sorted(minus, key=lambda s: h(f"D5sel_v1|{s}")):
            st = strata.get(s, part1["strata"].get(s))
            if need.get(st, 0) > 0:
                keep.append(s)
                need[st] -= 1
        short = n_plus - len(keep)
        if short > 0:
            rest = [s for s in sorted(minus, key=lambda s: h(f"D5sel_v1|{s}")) if s not in set(keep)]
            keep += rest[:short]
        alloc_minus = {s: 1 for s in keep}
    picks = {}
    for l in langs:
        picks[l] = {}
        for x in ("Y", "N"):
            pref = None
            if l != langs[0]:                       # EN mirrors the VI-chosen slot of the same seed when verified
                pref = defaultdict(list)
                for r in picks[langs[0]][f"{x}+"]:
                    pref[r["seed_id"]].append(r["slot"])
            picks[l][f"{x}+"] = pick_items(f"{x}+", alloc_plus, cands[l][f"{x}+"], strata, prefer_slots=pref)
            picks[l][f"{x}-"] = [{**idx[l][x][part1["slots"][x][s]], "_stratum": strata.get(s)} for s in
                                 sorted(alloc_minus, key=lambda s: (strata.get(s), h(f"D5sel_v1|{s}")))]
    cap_minus_pop = {s: 1 for s in part1["strata"]}                     # part-1 selected + reserve (proxy)
    sel_effect = selection_effect(cap_plus, {s: 1 for s in part1.get("eligible_seeds", part1["seeds_minus"] +
                                                                      part1["reserve_minus"])},
                                  alloc_plus, {x: 1 for x in part1["seeds_minus"]}, alloc_minus,
                                  {**strata, **part1["strata"]}, meta)
    seed_info["minus_substitutions"] = subst
    seed_info["minus_eligibility_checked_in"] = sorted(f"{x}/{l}" for (l, x) in chk)
    seed_info["minus_dropped_no_reserve"] = dropped
    return {"langs": langs, "n_target": n, "n_effective": min(n_plus, len(alloc_minus)), "minus_seeds": "independent",
            "minus_waves": "first-item-only (Y wave-1 slot 1 / N j=1)", "alloc": {"plus": alloc_plus, "minus": alloc_minus},
            "quota": {"plus": dict(q_plus), "minus": dict(Counter(strata.get(s) for s in alloc_minus))},
            "strata": {s: strata.get(s, part1["strata"].get(s)) for s in set(alloc_plus) | set(alloc_minus) | set(res_plus)},
            "tercile_cuts": {k: list(v) for k, v in cuts.items()},
            "reserve": {"plus": res_plus, "minus": [s for v in reserve.values() for s in v]},
            "capacity": {"seeds_plus": len(cap_plus), "items_plus": sum(cap_plus.values()),
                         "seeds_minus_part1": len(part1["seeds_minus"])},
            "steps_src": dict(Counter(meta.get(s, {}).get("steps_src", "none") for s in alloc_plus)),
            "selection_effect": sel_effect, "seed_universe": seed_info, "picks": picks, "pools": pools_by_lang,
            "meta": meta, "part1_sha256": part1.get("_sha256")}


def shared_from(sel: dict) -> dict:
    """Secondary (sensitivity): one seed set S = S+ for all four arms; the minus arms take each S+ seed's
    non-adaptive first item when it is minus-eligible in every arm x language (S+ seeds without one are dropped from
    all four arms, so n stays equal). The share of those minus items that are verified is reported."""
    firsts = {}
    for l in sel["picks"]:
        for x in ("Y", "N"):
            firsts[(l, x)] = {r["seed_id"]: r for r in sel["pools"][l][x].rows
                              if r["minus_ok"] and _is_first(r) and r["op"] != "M6"}
    keep = [s for s in sel["alloc"]["plus"] if all(s in f for f in firsts.values())]
    ks = set(keep)
    out = {**sel, "minus_seeds": "shared", "picks": {}, "alloc": {"plus": {s: 1 for s in keep},
                                                                   "minus": {s: 1 for s in keep}},
           "n_effective": len(keep)}
    for l, P in sel["picks"].items():
        out["picks"][l] = {a: [r for r in P[a] if r["seed_id"] in ks] for a in ("Y+", "N+")}
        for x in ("Y", "N"):
            out["picks"][l][f"{x}-"] = [{**firsts[(l, x)][s], "_stratum": sel["strata"].get(s)} for s in keep]
    return out


# ================================================================================================ op-matched sensitivity
OPMATCH = TRAIN / "final_opmatched"
CLASSES3 = ("value", "premise", "structural")


def _max_circulation(nm: dict[tuple[str, str], int], classes) -> dict[tuple[str, str], int]:
    """Largest f <= nm on (Y class a, N class b) cells with equal Y and N marginals (a circulation on 3 nodes):
    all diagonal cells + the best mix of 3-cycles (both orientations) and 2-cycles, by exhaustive search."""
    c = list(classes)
    f = {k: 0 for k in nm}
    for a in c:
        f[(a, a)] = nm.get((a, a), 0)
    if len(c) < 2:
        return f
    best, best_f = -1, None
    if len(c) == 3:
        a, b, d = c
        cyc1 = [(a, b), (b, d), (d, a)]
        cyc2 = [(a, d), (d, b), (b, a)]
        m1 = min(nm.get(e, 0) for e in cyc1)
        m2 = min(nm.get(e, 0) for e in cyc2)
    else:
        cyc1 = cyc2 = []
        m1 = m2 = 0
    for t1 in range(m1 + 1):
        for t2 in range(m2 + 1):
            rem = dict(nm)
            for e in cyc1:
                rem[e] = rem.get(e, 0) - t1
            for e in cyc2:
                rem[e] = rem.get(e, 0) - t2
            if any(v < 0 for v in rem.values()):
                continue
            two = {}
            tot = 3 * t1 + 3 * t2
            for i, x in enumerate(c):
                for y in c[i + 1:]:
                    k = min(rem.get((x, y), 0), rem.get((y, x), 0))
                    two[(x, y)] = two[(y, x)] = k
                    tot += 2 * k
            if tot > best:
                best = tot
                g = {k: 0 for k in nm}
                for e in cyc1:
                    g[e] = g.get(e, 0) + t1
                for e in cyc2:
                    g[e] = g.get(e, 0) + t2
                for e, k in two.items():
                    g[e] = g.get(e, 0) + k
                best_f = g
    for k, v in (best_f or {}).items():
        if k[0] != k[1]:
            f[k] = v
    return f


def opmatch_sign(seeds: list[str], ycls: dict[str, str], ncls: dict[str, str]) -> tuple[list[str], dict]:
    """Largest subset of the (seed-matched) S with equal Y and N edit-class histograms; classes one arm lacks are
    dropped from both arms. Seeds inside a (Y class, N class) cell are taken in sha1('D5opm_v1|<seed>') order."""
    present = [c for c in CLASSES3 if any(ycls[s] == c for s in seeds) and any(ncls[s] == c for s in seeds)]
    cell = defaultdict(list)
    for s in sorted(seeds, key=lambda s: h(f"D5opm_v1|{s}")):
        if ycls[s] in present and ncls[s] in present:
            cell[(ycls[s], ncls[s])].append(s)
    f = _max_circulation({k: len(v) for k, v in cell.items()}, present)
    keep = [s for k, v in cell.items() for s in v[:f.get(k, 0)]]
    return keep, {"classes_kept": present, "cells": {f"{a}->{b}": f.get((a, b), 0) for (a, b) in cell}}


def _trim_balanced(keep: list[str], ycls, ncls, n: int) -> list[str]:
    """Drop seeds until len == n while keeping the Y/N class marginals equal: diagonal cells first (hash order from
    the end), then matched 2-cycles (one seed from each direction)."""
    keep = list(keep)
    order = {s: i for i, s in enumerate(keep)}
    diag = sorted([s for s in keep if ycls[s] == ncls[s]], key=lambda s: -order[s])
    while len(keep) > n and diag:
        keep.remove(diag.pop(0))
    while len(keep) > n:
        off = [s for s in keep if ycls[s] != ncls[s]]
        done = False
        for s in sorted(off, key=lambda s: -order[s]):
            mate = next((t for t in sorted(off, key=lambda t: -order[t]) if t != s and ycls[t] == ncls[s] and
                         ncls[t] == ycls[s]), None)
            if mate and len(keep) - 2 >= n - 1:
                keep.remove(s)
                keep.remove(mate)
                done = True
                break
        if not done:
            break
    return keep


def opmatched_from(sel: dict, lang: str) -> dict:
    """T1 §2.6 sensitivity subset (lead 2026-11-08): S+ and S- subsampled so the Y and N edit-class histograms are
    equal inside each sign, then trimmed to one n for all four arms."""
    P = sel["picks"][lang]
    by = {a: {r["seed_id"]: r for r in P[a]} for a in ARMS}
    res, keep = {}, {}
    for sign in ("+", "-"):
        seeds = sorted(set(by[f"Y{sign}"]) & set(by[f"N{sign}"]))
        yc = {s: EDIT_CLASS.get(by[f"Y{sign}"][s]["op"], "other") for s in seeds}
        nc = {s: EDIT_CLASS.get(by[f"N{sign}"][s]["op"], "other") for s in seeds}
        k, info = opmatch_sign(seeds, yc, nc)
        keep[sign], res[sign] = (k, yc, nc), info
    n_op = min(len(keep["+"][0]), len(keep["-"][0]))
    picks, hist = {}, {}
    for sign in ("+", "-"):
        k, yc, nc = keep[sign]
        k = _trim_balanced(k, yc, nc, n_op)
        n_op = min(n_op, len(k))
        keep[sign] = (k, yc, nc)
    for sign in ("+", "-"):
        k, yc, nc = keep[sign]
        k = k[:n_op] if len(k) > n_op else k
        for x in ("Y", "N"):
            picks[f"{x}{sign}"] = [by[f"{x}{sign}"][s] for s in k]
        hist[sign] = {"Y": dict(Counter(yc[s] for s in k)), "N": dict(Counter(nc[s] for s in k))}
    return {"lang": lang, "n": min(len(v) for v in picks.values()), "picks": picks, "hist": hist, "info": res}


def write_opmatched(sel: dict, lang: str) -> dict:
    om = opmatched_from(sel, lang)
    targets = [OPMATCH / x / lang / f"train_{s}.jsonl" for x in ("Y", "N") for s in ("plus", "minus")]
    _archive(targets)
    files = _write_set(OPMATCH, lang, om["picks"], sel["pools"][lang], {"version": VERSION + "_opmatched", "lang": lang})
    man = {"version": VERSION + "_opmatched", "lang": lang, "n": om["n"], "edit_class_histograms": om["hist"],
           "circulation": om["info"], "files": files,
           "rule": "inside each sign, the largest seed subset of the primary matched set whose Y and N edit-class "
                   "histograms (value=M1 / premise=M2,M4 / structural=M3,M5) are equal (max circulation over "
                   "(Y class, N class) cells; classes one arm lacks dropped); both signs trimmed to one n while "
                   "keeping the marginals equal. Successor runs it; not part of the 11-08 runs unless time allows."}
    OPMATCH.mkdir(parents=True, exist_ok=True)
    (OPMATCH / f"manifest_{lang}.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    return man


def coverage_report() -> dict:
    """P3: share of pool_v1 (allowed) MATH seeds the Y arm cannot reach (no accepted Y formalization / no Y item)."""
    allowed, _ = allowed_seeds()
    ds = {r["seed_id"]: (r.get("dataset"), (r.get("subject") or "")) for r in read_jsonl(SEED_POOL)}
    formal = set()
    attempted = set()
    for f in Y_WORK.glob("*/formalize.jsonl"):
        for r in read_jsonl(f):
            attempted.add(r.get("seed_id") or r.get("key"))
            if r.get("accepted"):
                formal.add(r.get("seed_id") or r.get("key"))
    out = {}
    for grp in ("gsm8k", "math"):
        ss = {s for s in allowed if (ds.get(s, ("", ""))[0] == "gsm8k") == (grp == "gsm8k")}
        out[grp] = {"allowed_seeds": len(ss), "attempted_by_Y": len(ss & attempted), "formalized_by_Y": len(ss & formal),
                    "unreachable_share": round(1 - len(ss & formal) / max(1, len(ss)), 4)}
    return out


# ================================================================================================ projection (hourly)
def y_vi_provisional(allowed: set[str]) -> dict[str, str]:
    """Seed -> T/U/F for VI Y+ from G5's work stores while the pool release is partial (read-only)."""
    W = Y_WORK / "vi_scale"
    if not (W / "vgate.jsonl").exists():
        return {}
    vg = {r["slot"]: r for r in read_jsonl(W / "vgate.jsonl")}
    gates = {r.get("slot"): r for r in read_jsonl(W / "gates.jsonl")} if (W / "gates.jsonl").exists() else {}
    cyc = {}
    for f in sorted(W.glob("cycle*_R2b.jsonl")):
        for r in read_jsonl(f):
            if "cycle_ok" in r:
                cyc[r["key"]] = r["cycle_ok"]
    out = {}
    for r in read_jsonl(TRAIN / "Y" / "vi" / "pool.jsonl"):
        sid, slot = r["seed_id"], r.get("slot")
        if sid not in allowed or r.get("op") == "M6":
            continue
        v, g, c = vg.get(slot), gates.get(slot, {}), cyc.get(slot)
        if v is None:
            val = "U"
        elif not (v.get("pass") and v.get("trace_match")) or g.get("chrf_ok") is False or g.get("dedup_ok") is False                 or c is False:
            val = "F"
        else:
            val = "T" if c is True else "U"
        if out.get(sid) != "T":
            out[sid] = val if out.get(sid) is None or val != "F" else out[sid]
    return out


def project_n(log: Path | None = None) -> dict:
    """Projected final n = |S+| over {Y,N} x {vi,en} from the current pools (verified True / False / None).
    p_EN[arm] = P(EN item verified | VI verified on the same seed), measured on seeds where both are decided.
    Projection = certain + sum over undecided seeds of the product of the per-cell pass probabilities."""
    allowed, _ = allowed_seeds()
    st: dict[tuple[str, str], dict[str, str]] = {}
    for l in ("vi", "en"):
        for x in ("Y", "N"):
            d: dict[str, str] = {}
            p = pool_path(x, l)
            if p.exists():
                for r in Pool(p).rows:
                    s = r["seed_id"]
                    if s not in allowed or r["op"] == "M6" or not (r["minus_ok"] or r["plus"] is True):
                        continue
                    v = "T" if r["plus"] is True else ("U" if r["plus"] is None else "F")
                    if d.get(s) != "T":
                        d[s] = "T" if v == "T" else ("U" if (v == "U" or d.get(s) == "U") else "F")
            st[(l, x)] = d
    # Y pools flag "not yet evaluated" as False in partial (r0) releases: treat them as unknown, and for VI use the
    # provisional status from G5's work stores (screen / trace / chrF / dedup / cycle)
    for l in ("vi", "en"):
        rel = TRAIN / "Y" / l / "RELEASE"
        if rel.exists() and re.search(r"partial|pending", rel.read_text(encoding="utf-8"), re.I):
            st[("_partial", l)] = {}
            st[(l, "Y")] = {k: ("U" if v == "F" else v) for k, v in st[(l, "Y")].items()}
            if l == "vi":
                for k, v in y_vi_provisional(allowed).items():
                    st[(l, "Y")][k] = v
    p_en = {}
    for x in ("Y", "N"):
        vi, en = st[("vi", x)], st[("en", x)]
        dec = [s for s in en if vi.get(s) == "T" and en[s] in "TF"]
        n_t = sum(en[s] == "T" for s in dec)
        # an EN screen in progress decides failures first (agree-fail before the screen): no estimate until a pass
        p_en[x] = (n_t / len(dec), len(dec)) if dec and n_t > 0 else (None, len(dec))
    seeds = set().union(*[set(d) for k, d in st.items() if k[0] != "_partial"])
    cert = proj = 0.0
    bind = Counter()
    for s in seeds:
        cells = {(l, x): st[(l, x)].get(s) for l in ("vi", "en") for x in ("Y", "N")}
        if any(v == "F" for v in cells.values()):
            continue
        if all(v == "T" for v in cells.values()):
            cert += 1
            proj += 1
            continue
        pr = 1.0
        for (l, x), v in cells.items():
            if v == "T":
                continue
            if v is None:
                pr = 0.0
                bind[f"no {x}-{l} item"] += 1
                break
            q = p_en[x][0] if l == "en" and p_en[x][0] is not None else 0.85
            pr *= q
            bind[f"{x}-{l} undecided"] += 1
        proj += pr
    out = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "certain_all4": int(cert), "projected_n": round(proj),
           "p_EN": {x: {"p": None if v[0] is None else round(v[0], 3), "n_decided": v[1]} for x, v in p_en.items()},
           "cells": {f"{x}-{l}": dict(Counter(d.values())) for (l, x), d in st.items() if l != "_partial"},
           "partial_releases": [k[1] for k in st if k[0] == "_partial"], "blocking_counts": dict(bind.most_common(8))}
    if log:
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps(out) + "\n")
    return out




def funnel() -> dict:
    """Lost-seed breakdown for S+ (D5_selection.md): sequential seed counts from the allowed universe to 'verified in
    all 4 arm x language cells', each step attributing the seeds it loses to one reason."""
    allowed, info = allowed_seeds()
    ds = {r["seed_id"]: r.get("dataset") for r in read_jsonl(SEED_POOL)}
    gsm = {s for s in allowed if ds.get(s) == "gsm8k"}
    formal = set()
    for f in Y_WORK.glob("vi_scale/formalize.jsonl"):
        for r in read_jsonl(f):
            if r.get("accepted"):
                formal.add(r.get("seed_id") or r.get("key"))

    def cell(l, x, first_only=False):
        d = {}
        p = pool_path(x, l)
        if not p.exists():
            return d
        for r in Pool(p).rows:
            if r["seed_id"] not in allowed or r["op"] == "M6" or not (r["minus_ok"] or r["plus"] is True):
                continue
            if first_only and r.get("retry"):
                continue
            v = "T" if r["plus"] is True else ("U" if r["plus"] is None else "F")
            cur = d.get(r["seed_id"])
            d[r["seed_id"]] = "T" if "T" in (v, cur) else ("U" if "U" in (v, cur) else "F")
        return d
    steps = []
    cur = set(allowed)
    def step(name, keep):
        nonlocal cur
        new = cur & keep
        steps.append({"step": name, "seeds": len(new), "lost": len(cur) - len(new)})
        cur = new
    steps.append({"step": "allowed (pool_v1 \ dev_v1 \ FAIL/UNINF)", "seeds": len(cur), "lost": None})
    step("GSM8K only (Y formalizer scope; MATH unreachable)", gsm)
    step("Y formalization accepted", formal)
    for l, x, lab in (("vi", "Y", "VI Y item exists (wave-1 tag/leak/boxed)"), ):
        d = cell(l, x)
        step(lab, set(d))
        step("VI Y+ (cycle, chrF/dedup, trace match, Haiku-v3 screen)", {s for s, v in d.items() if v == "T"})
    d = cell("vi", "N")
    step("VI N item exists (j-slot, minus_eligible, op != M6)", set(d))
    step("VI N+ (agree, Haiku-v3 screen)", {s for s, v in d.items() if v == "T"})
    for x in ("Y", "N"):
        d = cell("en", x)
        step(f"EN {x} item exists", set(d))
        step(f"EN {x}+ (incl. retries)", {s for s, v in d.items() if v == "T"})
    return {"steps": steps, "final_seeds": len(cur)}


if __name__ == "__main__":
    main(sys.argv[1:])
