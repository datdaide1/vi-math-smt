"""C3 attribution study (workstream D5): load every already-judged pilot item into one long table.

Read-only over other workstreams' outputs:
  G1  data/pilot/generator/   4 generator models x 30 paired seeds (native, arm N) and 4 editor models on the same
                              edit lists (judge_render), Opus 5.5 validity_judge.v2 (judge_v2, verbatim the same rubric)
  G2  data/pilot/g2/          4 native prompts (v1..v4) x 40 paired seeds, Haiku 5.5 generator, Opus v2 judge
  G3  data/pilot/symbolic/    60 seeds, 180 slots, X4 operators as fixed by G3, renderer = Haiku editor.v3 (+ M1
                              deterministic); Opus v2 judge on all 139 tag-passing renders and on the 60 seed texts
  G4  data/pilot/symbolic_v2/ same 60 seeds / 180 slots, operators fixed (M6 dropped, story-role filters), renderers
                              E (G3 editor) / R1 (E + fluency pass, USD->dong x1000) / R2 (free render from IR)
                              and arm N (Haiku native_generator.v3) on the SAME slots; Opus v2 + Sonnet v3 judges

Every row: one judged item under one judge. ``labels`` uses the lab_v2 / lab_v3 label set of G1-G4.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "data" / "pilot"

# error classes of the brief, mapped to PROTOCOL codes / lab_v2 label keys
CLASSES = ("absurd", "tail", "leak", "unansw", "fluency", "neardup")
CLASS_CODE = {"absurd": "NL-ABSURD", "tail": "NL-TACKED", "leak": "NL-LEAK", "unansw": "NL-UNANSW",
              "fluency": "NL-FLUENCY", "neardup": "NL-NEARDUP"}
SUMMARY = ("not_teacher", "not_valid")
ALL_OUTCOMES = CLASSES + SUMMARY


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def last_by_key(rows: Iterable[dict], key: str = "key") -> dict[str, dict]:
    """Stores are append-only; the last row for a key wins (G3/G4 Store semantics)."""
    d: dict[str, dict] = {}
    for r in rows:
        d[r[key]] = r
    return d


@dataclass
class Item:
    source: str            # G1n | G1e | G2 | G3 | G4
    variant: str           # e.g. G3, E, R1, R2, N, haiku, v3 ...
    arm: str               # Y (symbolic) | N (LLM-native)
    renderer: str          # editor | edit+fluency | ir_render | native | deterministic
    gen_model: str         # model that wrote the text
    prompt: str
    op: str
    seed_id: str
    slot: str              # paired unit within a source (slot key or seed id)
    judge: str             # opus_v2 | sonnet_v3
    y: dict                # outcome name -> 0/1
    natural: float | None = None
    survivor: bool | None = None
    cycle_ok: bool | None = None
    solve_agree: bool | None = None
    note: str = ""
    text: str = ""
    extra: dict = field(default_factory=dict)


def outcomes(labels: dict, leak_regex: bool = False) -> dict[str, int]:
    """lab_v2/lab_v3 labels -> the six error classes + two summaries (1 = error present)."""
    return {"absurd": int(bool(labels.get("absurd"))), "tail": int(bool(labels.get("tail"))),
            "leak": int(bool(labels.get("leakj")) or bool(leak_regex)), "unansw": int(bool(labels.get("unansw"))),
            "fluency": int(bool(labels.get("fluency"))), "neardup": int(bool(labels.get("neardupj"))),
            "not_teacher": int(not labels.get("teacher")), "not_valid": int(not labels.get("valid"))}


def wellposed(v: dict | None) -> int | None:
    """GSM-Plus-style 'qualified variation': the 5 MathQ-Verify booleans hold and the operator was applied
    (no naturalness requirement). None when the verdict is not a validity_judge.v2 verdict."""
    if not isinstance(v, dict) or "atomic" not in v:
        return None
    return int(all(v.get(b) is True for b in _BOOLS) and v.get("op_followed") is True)


def _note(row: dict) -> str:
    v = row.get("verdict") or {}
    return v.get("note", "") if isinstance(v, dict) else ""


# ------------------------------------------------------------------------------------------------ seeds (G3)
def seed_labels(base: Path = PILOT) -> dict[str, dict]:
    """Opus v2 verdicts on the 60 untransformed seed texts (G3 'M0' baseline)."""
    out = {}
    for r in last_by_key(load_jsonl(base / "symbolic" / "run_b" / "judge_opus_seeds.jsonl")).values():
        if r.get("parsed") and r.get("labels"):
            y = outcomes(r["labels"])
            y.pop("neardup")                   # meaningless for an untransformed text
            out[r["key"]] = {"y": y, "natural": r["labels"].get("natural"), "note": _note(r)}
    return out


def seed_texts(base: Path = PILOT) -> dict[str, dict]:
    return {r["seed_id"]: r for r in load_jsonl(base / "symbolic" / "seeds.jsonl")}


# ------------------------------------------------------------------------------------------------ G3
def load_g3(base: Path = PILOT) -> list[Item]:
    run = base / "symbolic" / "run_b"
    rend = last_by_key(load_jsonl(run / "render.jsonl"))
    gates = last_by_key(load_jsonl(run / "gates.jsonl"))
    solve = last_by_key(load_jsonl(run / "solve_opus.jsonl"))
    out = []
    for k, j in last_by_key(load_jsonl(run / "judge_opus.jsonl")).items():
        if not (j.get("parsed") and j.get("labels")):
            continue
        r, g = rend.get(k, {}), gates.get(k, {})
        out.append(Item("G3", "G3", "Y", "deterministic" if r.get("renderer") == "deterministic" else "editor",
                        "claude-haiku-5-5", "editor.v3", r.get("op") or g.get("op"), r.get("seed_id") or k.split(":")[0],
                        k, "opus_v2", outcomes(j["labels"], leak_regex=bool(g.get("leak_regex"))),
                        natural=j["labels"].get("natural"), survivor=g.get("survivor"), cycle_ok=g.get("cycle_ok"),
                        solve_agree=(solve.get(k) or {}).get("agree"), note=_note(j), text=r.get("text", ""),
                        extra={"wellposed": wellposed(j.get("verdict"))}))
    return out


# ------------------------------------------------------------------------------------------------ G4
G4_RENDERER = {"E": "editor", "R1": "edit+fluency", "R2": "ir_render", "R1b": "edit+fluency", "N": "native"}
G4_PROMPT = {"E": "editor.v3", "R1": "editor.v3+fluency.v1", "R2": "ir_render.v1", "R1b": "editor.v3+fluency.v2",
             "N": "native_generator.v3"}


def load_g4(base: Path = PILOT, judge: str = "opus") -> list[Item]:
    d = base / "symbolic_v2"
    fn = {"opus": "judge_opus.jsonl", "sonnet": "judge_sonnet.jsonl"}[judge]
    gates = last_by_key(load_jsonl(d / "gates.jsonl"))
    solve = last_by_key(load_jsonl(d / "solve_opus.jsonl"))
    mut = last_by_key(load_jsonl(d / "mutate.jsonl"))
    renders: dict[str, dict] = {}
    for v in ("E", "R1", "R2", "R1b"):
        for r in last_by_key(load_jsonl(d / f"render_{v}.jsonl")).values():
            renders[f"{v}:{r['key']}"] = r
    for r in last_by_key(load_jsonl(d / "native_N.jsonl")).values():
        renders[f"N:{r['key']}"] = r
    out = []
    for k, j in last_by_key(load_jsonl(d / fn)).items():
        if not (j.get("parsed") and j.get("labels")):
            continue
        v, slot = k.split(":", 1)
        r, g = renders.get(k, {}), gates.get(k, {})
        m = mut.get(slot, {})
        renderer = G4_RENDERER[v]
        if v == "E" and r.get("renderer") == "deterministic":
            renderer = "deterministic"
        out.append(Item("G4", v, "N" if v == "N" else "Y", renderer, "claude-haiku-5-5", G4_PROMPT[v],
                        r.get("op") or m.get("op"), slot.split(":")[0], slot, f"{judge}_v{2 if judge == 'opus' else 3}",
                        outcomes(j["labels"], leak_regex=bool(g.get("leak_regex"))),
                        natural=j["labels"].get("natural"), survivor=g.get("survivor") if v != "N" else None,
                        cycle_ok=g.get("cycle_ok") if v != "N" else None,
                        solve_agree=(solve.get(k) or {}).get("agree") if judge == "opus" else None,
                        note=_note(j), text=r.get("text", ""),
                        extra={"same_as_g3": m.get("same_as_g3"), "localized": r.get("localized"),
                               "warning": r.get("warning"),
                               "wellposed": wellposed(j.get("verdict")) if judge == "opus" else None}))
    return out


def load_r1b(base: Path = PILOT) -> list[Item]:
    """R1b (G4 fluency.v2, 1 USD = 10.000 dong, no imperial relabel): rendered by G4, judged by D5 with the same
    Opus v2 request builder (data/audit/c3/judge_opus_R1b.jsonl). Tag-passing renders only, no cycle check."""
    store = base.parent / "audit" / "c3" / "judge_opus_R1b.jsonl"
    rend = last_by_key(load_jsonl(base / "symbolic_v2" / "render_R1b.jsonl"))
    out = []
    for k, j in last_by_key(load_jsonl(store)).items():
        if not (j.get("parsed") and j.get("labels")):
            continue
        slot = k.split(":", 1)[1]
        r = rend.get(slot, {})
        out.append(Item("G4", "R1b", "Y", "edit+fluency", "claude-haiku-5-5", G4_PROMPT["R1b"], r.get("op") or j.get("op"),
                        slot.split(":")[0], slot, "opus_v2", outcomes(j["labels"]), natural=j["labels"].get("natural"),
                        note=_note(j), text=r.get("text", ""),
                        extra={"localized": r.get("localized"), "wellposed": wellposed(j.get("verdict"))}))
    return out


def g4_mutations(base: Path = PILOT) -> dict[str, dict]:
    return last_by_key(load_jsonl(base / "symbolic_v2" / "mutate.jsonl"))


def g3_mutations(base: Path = PILOT) -> dict[str, dict]:
    return last_by_key(load_jsonl(base / "symbolic" / "run_b" / "mutate.jsonl"))


# ------------------------------------------------------------------------------------------------ G1
G1_MODELS = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5", "gemini": "gemini-3.5-flash-lite",
             "gemini31": "gemini-3.1-flash-lite"}


def load_g1(base: Path = PILOT) -> list[Item]:
    d = base / "generator"
    seeds = {r["seed_id"]: r for r in load_jsonl(d / "seeds.jsonl")}
    scored = {(r["gen"], r["seed_id"]): r for r in load_jsonl(d / "items_scored.jsonl")}
    out = []
    for g, model in G1_MODELS.items():
        solve = {r["seed_id"]: r for r in load_jsonl(d / "opus_solve" / f"{g}.jsonl")}
        for r in load_jsonl(d / "judge_opus" / f"{g}.jsonl"):
            v = r.get("verdict")
            if not (r.get("parsed") and isinstance(v, dict)):
                continue
            sid = r["seed_id"]
            sc = scored.get((g, sid), {})
            out.append(Item("G1n", g, "N", "native", model, "native_v1", seeds[sid]["op"], sid, sid, "opus_v2",
                            outcomes(lab_v2(v), leak_regex=bool(sc.get("leak_regex"))), natural=lab_v2(v)["natural"],
                            solve_agree=(solve.get(sid) or {}).get("agrees"), note=v.get("note", ""),
                            text=sc.get("problem") or ""))
        for r in load_jsonl(d / "judge_render" / f"{g}.jsonl"):
            v = r.get("verdict")
            if not isinstance(v, dict):
                continue
            out.append(Item("G1e", g, "Y", "editor", model, "render_v1", r.get("op"), r["seed_id"], r["seed_id"],
                            "opus_v2", outcomes(lab_v2(v)), natural=lab_v2(v)["natural"], note=v.get("note", "")))
    return out


# ------------------------------------------------------------------------------------------------ G2
def load_g2(base: Path = PILOT) -> list[Item]:
    out = []
    for r in load_jsonl(base / "g2" / "items_task3.jsonl"):
        lab = (r.get("L") or {}).get("opus")
        if not (r.get("parsed") and lab):
            continue
        out.append(Item("G2", r["arm"], "N", "native", "claude-haiku-5-5", f"native_generator.{r['arm']}", r["op"],
                        r["seed_id"], r["seed_id"], "opus_v2", outcomes(lab, leak_regex=bool(r.get("leak_regex"))),
                        natural=lab.get("natural"), solve_agree=r.get("opus_agrees"), text=r.get("problem") or ""))
    return out


# lab_v2 is duplicated here (2 lines) so this module needs no import from another workstream's package at load time;
# test_c3 checks that it equals pilot_symbolic.judge.lab_v2 on real verdicts.
_BOOLS = ("atomic", "no_conflict", "complete", "single_question", "realistic")


def _nat(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def lab_v2(v: dict) -> dict:
    cs = set(v.get("codes") or [])
    valid = all(v.get(b) is True for b in _BOOLS) and _nat(v.get("natural")) >= 4
    return {"valid": valid, "teacher": v.get("teacher_would_write") is True, "new": v.get("new_vs_seed") is True,
            "op": v.get("op_followed") is True, "natural": _nat(v.get("natural")), "leakj": "NL-LEAK" in cs,
            "absurd": "NL-ABSURD" in cs, "tail": "NL-TACKED" in cs, "neardupj": "NL-NEARDUP" in cs,
            "fluency": "NL-FLUENCY" in cs, "unansw": "NL-UNANSW" in cs, "severity": v.get("severity")}


def load_all(base: Path = PILOT) -> dict[str, list[Item]]:
    return {"G1": load_g1(base), "G2": load_g2(base), "G3": load_g3(base), "G4": load_g4(base, "opus"),
            "G4s": load_g4(base, "sonnet")}
