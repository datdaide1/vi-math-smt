"""MTA-vi: the math-translation algorithm of T4 §3 (training-side items only).

mask -> triggered glossary + construction rules -> translate (one item per call, tagged plain text)
-> unmask -> gate G0 (structure) + G1 (numbers, glossary, CR, units) + G2 (solve-check, NIM solver)
+ G3 (MQM-span judge, Gemini = different family) -> up to 2 post-edit rounds with explicit feedback
-> ACCEPT or DROP.  (G4 MetricX is not run here: no GPU in W3.)

Roles (each a different family): translator ``nvidia/nemotron-3-ultra-550b-a55b`` (NIM),
solver ``deepseek-ai/deepseek-v4.1-flash`` (NIM), judge ``gemini-3.5-flash-lite`` (Gemini key pool).
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from vi_math_verified_aug.data_fix import nim_client
from vi_math_verified_aug.data_fix.common import (
    CTRL_RE, brace_balance, canon_num, gsm8k_final, last_boxed, norm_boxed, number_multiset_diff,
)

HERE = Path(__file__).resolve().parent
PROMPTS = HERE / "prompts"
GLOSSARY = json.loads((HERE / "glossary_vi_math_v0.json").read_text(encoding="utf-8"))
TRANSLATOR = "nvidia/nemotron-3-ultra-550b-a55b"
SOLVER = "deepseek-ai/deepseek-v4.1-flash"
JUDGE = "gemini-3.5-flash-lite"
JUDGE_FALLBACK = "gemini-3.1-flash-lite"

MASK_ORDER = [
    ("A", re.compile(r"\[asy\].*?\[/asy\]", re.S)),
    ("E", re.compile(r"\\begin\{(\w+\*?)\}.*?\\end\{\1\}", re.S)),
    ("D", re.compile(r"\$\$.*?\$\$|\\\[.*?\\\]", re.S)),
    ("M", re.compile(r"\\\(.*?\\\)|(?<!\\)\$(?:\\\$|[^$])+?(?<!\\)\$", re.S)),
    ("C", re.compile(r"<<[^<>]*>>")),
    ("U", re.compile(r"https?://\S+")),
]
TAG_RE = re.compile(r"<([A-Z])(\d+)/>")
EN_STOP = re.compile(r"\b(the|and|of|is|are|that|this|how|many|much|what|each|with|has|have|will|does)\b", re.I)


def _prompt(name: str) -> tuple[str, str]:
    t = (PROMPTS / name).read_text(encoding="utf-8")
    if "### SYSTEM" in t:
        sys_part, user = t.split("### USER", 1)
        return sys_part.replace("### SYSTEM", "").strip(), user.strip()
    return "", t.split("### USER", 1)[1].strip()


def _boxed_spans(text: str) -> list[tuple[int, int]]:
    out = []
    for m in re.finditer(r"\\boxed\s*\{", text):
        i, d = m.end(), 1
        while i < len(text) and d:
            if text[i] == "\\":
                i += 2
                continue
            d += (text[i] == "{") - (text[i] == "}")
            i += 1
        if d == 0:
            out.append((m.start(), i))
    return out


def mask(text: str, kind: str, spans: dict[str, str]) -> str:
    """Replace math/code/annotations with <Xn/> tags (shared numbering across problem+solution)."""
    order = MASK_ORDER if kind == "math" else [m for m in MASK_ORDER if m[0] in ("C", "U")]
    # \boxed{..} outside math delimiters (balanced parser)
    if kind == "math":
        for s, e in reversed(_boxed_spans(text)):
            # skip if already inside $..$: crude check by dollar parity before s
            if len(re.findall(r"(?<!\\)\$", text[:s])) % 2 == 0:
                k = f"B{len(spans) + 1}"
                spans[k] = text[s:e]
                text = text[:s] + f"<{k}/>" + text[e:]
    for tag, rx in order:
        def rep(m):
            k = f"{tag}{len(spans) + 1}"
            spans[k] = m.group(0)
            return f"<{k}/>"
        text = rx.sub(rep, text)
    return text


def unmask(text: str, spans: dict[str, str]) -> str:
    return TAG_RE.sub(lambda m: spans.get(m.group(1) + m.group(2), m.group(0)), text)


def fired(en: str) -> tuple[list[dict], list[dict]]:
    g = [e for e in GLOSSARY["entries"] if re.search(e["en"], en, re.I)]
    r = [c for c in GLOSSARY["rules"] if re.search(c["en"], en, re.I)]
    return g, r


def _fmt(gl: list[dict], rules: list[dict]) -> tuple[str, str]:
    gs = "\n".join(f"   - {e['line']}" for e in gl[:15]) or "   (không có)"
    rs = "\n".join(f"   - {c['id']}: {c['line']}" for c in rules) or "   (không có)"
    return gs, rs


def parse_blocks(out: str) -> tuple[str | None, str | None]:
    out = re.sub(r"<think>.*?</think>", "", out or "", flags=re.S)
    p = re.search(r"<problem>\s*(.*?)\s*</problem>", out, re.S)
    s = re.search(r"<solution>\s*(.*?)\s*</solution>", out, re.S)
    return (p.group(1) if p else None), (s.group(1) if s else None)


def prose(s: str, kind: str = "math") -> str:
    if kind == "gsm8k":  # GSM8K '$' are dollar amounts, not math delimiters
        return re.sub(r"<<[^<>]*>>", " ", s or "")
    return re.sub(r"\$\$.*?\$\$|\\\[.*?\\\]|(?<!\\)\$(?:\\\$|[^$])*?(?<!\\)\$|\[asy\].*?\[/asy\]|<<[^<>]*>>", " ",
                  s or "", flags=re.S)


@dataclass
class Item:
    id: str
    kind: str                      # "gsm8k" | "math"
    problem_en: str
    solution_en: str
    gold: str
    spans: dict = field(default_factory=dict)
    masked_p: str = ""
    masked_s: str = ""
    vi_problem: str = ""
    vi_solution: str = ""
    checks: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    status: str = "NEW"
    rounds: int = 0


# ----------------------------------------------------------------------------- gates
def g0(it: Item, vm_p: str, vm_s: str) -> list[str]:
    fails = []
    tags_out = TAG_RE.findall(vm_p + "\n" + vm_s)
    keys_out = [a + b for a, b in tags_out]
    if sorted(keys_out) != sorted(it.spans):
        missing = sorted(set(it.spans) - set(keys_out))
        extra = sorted(set(keys_out) - set(it.spans))
        dup = sorted({k for k in keys_out if keys_out.count(k) > 1})
        fails.append(f"G0 placeholder mismatch: missing {missing} extra {extra} duplicated {dup}")
    for fld, vi, en in (("problem", it.vi_problem, it.problem_en), ("solution", it.vi_solution, it.solution_en)):
        if CTRL_RE.search(vi):
            fails.append(f"G0 control characters in {fld}")
        if brace_balance(vi) != brace_balance(en):
            fails.append(f"G0 brace balance differs in {fld}")
        t = prose(vi, it.kind)
        words = re.findall(r"[A-Za-zÀ-ỹ]+", t)
        if words and len(EN_STOP.findall(t)) / len(words) > 0.06:
            fails.append(f"G0 English left in {fld} ({len(EN_STOP.findall(t))} EN stopwords)")
        lr = len(vi) / max(1, len(en))
        if not (0.6 <= lr <= 2.5):
            fails.append(f"G0 length ratio {lr:.2f} in {fld}")
    new_cmds = set(re.findall(r"\\[A-Za-z]+", it.vi_problem + it.vi_solution)) - \
        set(re.findall(r"\\[A-Za-z]+", it.problem_en + it.solution_en))
    if new_cmds:
        fails.append(f"G0 new LaTeX commands {sorted(new_cmds)}")
    return fails


def g1(it: Item) -> list[str]:
    fails = []
    en_p, vi_p = it.problem_en, it.vi_problem
    nd = number_multiset_diff(prose(en_p, it.kind), prose(vi_p, it.kind))
    if nd["missing_set"] or nd["added_set"]:
        fails.append(f"G1 numbers in problem: EN values missing in VI {nd['missing_set']}, "
                     f"VI values not in EN {nd['added_set']}")
    if it.kind == "gsm8k":
        fin = gsm8k_final(it.vi_solution)
        if fin is None or canon_num(fin) != canon_num(it.gold):
            fails.append(f"G1 final line must be '#### {canon_num(it.gold)}' (got {fin!r})")
    else:
        b = last_boxed(it.vi_solution)
        if norm_boxed(b) != norm_boxed(it.gold) and not (b and norm_boxed(it.gold) in norm_boxed(b)):
            fails.append(f"G1 last \\boxed{{}} must be {it.gold!r} (got {b!r})")
    gl, rules = fired(en_p + "\n" + it.solution_en)
    for e in gl:
        for fld, en, vi in (("problem", en_p, vi_p), ("solution", it.solution_en, it.vi_solution)):
            if not re.search(e["en"], en, re.I):
                continue
            if e.get("forbid") and re.search(e["forbid"], prose(vi, it.kind), re.I):
                fails.append(f"G1 glossary {e['id']} in {fld}: forbidden rendering /{e['forbid']}/ — {e['line']}")
    for c in rules:
        if not re.search(c["en"], en_p, re.I):
            continue
        if c.get("vi_forbid") and re.search(c["vi_forbid"], vi_p, re.I):
            m = re.search(c["vi_forbid"], vi_p, re.I)
            fails.append(f"G1 {c['id']} in problem: '{m.group(0)}' — {c['line']}")
        if c.get("vi_require") and c["vi_require"] not in vi_p:
            fails.append(f"G1 {c['id']} in problem: required '{c['vi_require']}' missing — {c['line']}")
    return fails


def _equal(pred: str | None, gold: str, kind: str) -> bool:
    if pred is None:
        return False
    if kind == "gsm8k":
        p = re.sub(r"[^\d.,\-]", "", pred)
        try:
            return abs(float(canon_num(p.replace(".", "") if re.fullmatch(r"-?\d{1,3}(\.\d{3})+", p) else
                                       p.replace(",", "."))) - float(canon_num(gold))) < 1e-6
        except ValueError:
            return False
    try:
        from vi_math_verified_aug.eval.math_verify import is_correct  # W1's grader, read-only use
        return bool(is_correct(pred, gold))
    except Exception:
        return norm_boxed(pred) == norm_boxed(gold)


def g2(it: Item) -> tuple[str, list[str]]:
    _, tmpl = _prompt("g2_solve_v1.txt")
    out = nim_client.nim_chat(it.id, SOLVER, [{"role": "user", "content": tmpl.format(problem=it.vi_problem)}],
                              temperature=0.0, max_tokens=6000, role="G2-solve-vi", prompt_file="g2_solve_v1.txt")
    pred = last_boxed(out)
    if _equal(pred, it.gold, it.kind):
        return "PASS", []
    out_en = nim_client.nim_chat(it.id, SOLVER, [{"role": "user", "content": tmpl.format(problem=it.problem_en)}],
                                 temperature=0.0, max_tokens=6000, role="G2-solve-en",
                                 prompt_file="g2_solve_v1.txt")
    if not _equal(last_boxed(out_en), it.gold, it.kind):
        return "UNINFORMATIVE", []
    return "FAIL", [f"G2 solve-check: solver got {pred!r} on the Vietnamese problem but the gold is {it.gold!r} "
                    f"(the English problem is solved correctly) — the Vietnamese wording likely changes the math"]


def g3(it: Item) -> tuple[bool, list[str], list]:
    _, tmpl = _prompt("g3_mqm_judge_v1.txt")
    prompt = tmpl.format(gold=it.gold, problem_en=it.problem_en, problem_vi=it.vi_problem,
                         solution_en=it.solution_en, solution_vi=it.vi_solution)
    raw = nim_client.gemini(it.id, JUDGE, prompt, role="G3-mqm", prompt_file="g3_mqm_judge_v1.txt")
    if not raw:
        raw = nim_client.gemini(it.id, JUDGE_FALLBACK, prompt, role="G3-mqm", prompt_file="g3_mqm_judge_v1.txt")
    try:
        errs = json.loads(re.sub(r"^```(json)?|```$", "", raw.strip()))["errors"]
    except Exception:
        return True, ["G3 judge unavailable/unparsable (not blocking)"], []
    bad = [e for e in errs if str(e.get("severity", "")).lower() in ("major", "critical")]
    fb = [f"G3 judge ({e.get('severity')}, {e.get('category')}): '{e.get('span_vi')}' — {e.get('explanation')}"
          for e in bad]
    return not bad, fb, errs


def gate(it: Item, vm_p: str, vm_s: str, use_g2: bool = True, use_g3: bool = True) -> list[str]:
    fails = g0(it, vm_p, vm_s) + g1(it)
    it.checks = {"G0G1": list(fails)}
    if fails:  # cheap gates first; do not spend solver/judge calls on a structurally broken draft
        return fails
    if use_g2:
        st, fb = g2(it)
        it.checks["G2"] = st
        fails += fb
    if use_g3 and not fails:
        ok, fb, errs = g3(it)
        it.checks["G3"] = {"pass": ok, "errors": errs}
        if not ok:
            fails += fb
    return fails


# ----------------------------------------------------------------------------- driver
def translate_item(it: Item, max_rounds: int = 2, use_g2: bool = True, use_g3: bool = True) -> Item:
    nim_client.guard(it.id)
    it.spans = {}
    it.masked_p = mask(it.problem_en, it.kind, it.spans)
    it.masked_s = mask(it.solution_en, it.kind, it.spans)
    gl, rules = fired(it.problem_en + "\n" + it.solution_en)
    gs, rs = _fmt(gl, rules)
    sys_t, user_t = _prompt("mta_vi_translate_v1.txt")
    msgs = [{"role": "system", "content": sys_t},
            {"role": "user", "content": user_t.format(glossary=gs, rules=rs, problem=it.masked_p,
                                                      solution=it.masked_s)}]
    out = nim_client.nim_chat(it.id, TRANSLATOR, msgs, temperature=0.0, max_tokens=8000, role="T-translate",
                              prompt_file="mta_vi_translate_v1.txt")
    vm_p, vm_s = parse_blocks(out)
    fails = []
    for r in range(max_rounds + 1):
        if vm_p is None or vm_s is None:
            fails = ["output must contain exactly one <problem>...</problem> and one <solution>...</solution> block"]
        else:
            it.vi_problem, it.vi_solution = unmask(vm_p, it.spans), unmask(vm_s, it.spans)
            fails = gate(it, vm_p, vm_s, use_g2, use_g3)
        it.history.append({"round": r, "fails": fails, "checks": it.checks,
                           "vi_problem": it.vi_problem, "vi_solution": it.vi_solution})
        if not fails:
            it.status = "ACCEPT" if r == 0 else f"ACCEPT_REPAIR_{r}"
            it.rounds = r
            return it
        if r == max_rounds:
            break
        sys_t2, pe = _prompt("mta_vi_postedit_v1.txt")
        msgs = [{"role": "system", "content": sys_t2},
                {"role": "user", "content": pe.format(glossary=gs, rules=rs, feedback="\n".join(f"- {f}" for f in fails),
                                                      problem=it.masked_p, solution=it.masked_s,
                                                      vi_problem=vm_p or "", vi_solution=vm_s or "")}]
        out = nim_client.nim_chat(it.id, TRANSLATOR, msgs, temperature=0.0, max_tokens=8000, role=f"T-postedit-{r + 1}",
                                  prompt_file="mta_vi_postedit_v1.txt")
        p2, s2 = parse_blocks(out)
        vm_p, vm_s = (p2, s2) if p2 is not None and s2 is not None else (vm_p, vm_s)
    it.status = "DROP"
    it.rounds = max_rounds
    return it


def provenance(it: Item) -> dict:
    return {"method": "MTA-vi (T4 §3)", "translator": TRANSLATOR, "solver_G2": SOLVER, "judge_G3": JUDGE,
            "glossary_version": GLOSSARY["version"], "prompts": ["mta_vi_translate_v1.txt", "mta_vi_postedit_v1.txt",
                                                                 "g2_solve_v1.txt", "g3_mqm_judge_v1.txt"],
            "status": it.status, "repair_rounds": it.rounds, "checks": it.checks,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}


def run_batch(items: list[Item], cache_path: Path, workers: int = 4, **kw) -> dict[str, dict]:
    """Translate items concurrently (the shared NIM bucket keeps the process <= 10 RPM). Resumable:
    results are appended to ``cache_path`` (JSONL keyed by id) and cached ids are not re-sent."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    cache: dict[str, dict] = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                cache[r["id"]] = r
    todo = [it for it in items if it.id not in cache]
    lock = threading.Lock()
    print(f"[mta] {len(items)} items, {len(cache)} cached, {len(todo)} to translate", flush=True)

    def one(it: Item) -> dict:
        try:
            translate_item(it, **kw)
        except nim_client.ForbiddenItem:
            raise
        except Exception as e:  # keep going; record the failure
            it.status = "ERROR"
            it.history.append({"error": f"{type(e).__name__}: {e}"})
        rec = {"id": it.id, "status": it.status, "vi_problem": it.vi_problem, "vi_solution": it.vi_solution,
               "provenance": provenance(it), "history": it.history}
        with lock:
            with open(cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[mta] {it.id}: {it.status} (rounds {it.rounds})", flush=True)
        return rec

    if todo:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(workers) as ex:
            for fut in as_completed([ex.submit(one, it) for it in todo]):
                r = fut.result()
                cache[r["id"]] = r
    return cache
