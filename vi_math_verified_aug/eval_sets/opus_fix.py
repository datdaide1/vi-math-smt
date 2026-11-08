"""Step 4: Claude Opus proposals for queued translation defects, with three acceptance checks.

Test items go ONLY to Claude (``claude-opus-5-5``, effort ``medium``, Message Batches API, 50 % price).
Run with the tools venv (``anthropic`` SDK) from the repo root::

    E:/venv_tools/Scripts/python.exe -m vi_math_verified_aug.eval_sets.opus_fix run --stage propose [--limit N]
    E:/venv_tools/Scripts/python.exe -m vi_math_verified_aug.eval_sets.opus_fix run --stage solve
    E:/venv_tools/Scripts/python.exe -m vi_math_verified_aug.eval_sets.opus_fix run --stage judge
    E:/venv_tools/Scripts/python.exe -m vi_math_verified_aug.eval_sets.opus_fix decide

Stages (each only for the items that survived the previous one, so no money is spent on a lost cause):

1. ``propose``  sees EN (source of truth) + current VI + the triggers; returns ``{verdict: ok|fix, corrected_vi,
   reason}``.  The candidate text is ``corrected_vi`` for ``fix`` and the current VI for ``ok``.
   Check 1 (local, free): numeral multiset and LaTeX command multiset of the candidate equal the English ones.
2. ``solve``    a fresh call sees ONLY the candidate VI text and solves it; check 2: ``math_verify`` grades the
   answer correct against the (English-derived) gold.
3. ``judge``    a second, independent call sees EN + candidate VI (not the proposer's output, flags or reason)
   and says whether they state the same problem; check 3: ``equivalent == true``.

``decide`` writes ``data/eval/_work/opus/decisions.jsonl``: ``auto_fixed`` (fix + 3 checks), ``cleared``
(ok + 3 checks, text unchanged) or ``flagged`` (anything else; the original text is kept).
Every request/response is logged under ``logs/api/X3/`` (no keys).  Cost is computed from returned usage.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from vi_math_verified_aug.eval import math_verify as mv
from vi_math_verified_aug.eval_sets.common import ROOT, WORK, parity, read_jsonl, write_jsonl

MODEL = "claude-opus-5-5"
EFFORT = "medium"
MAX_TOKENS = 16000
# Batch API = 50 % of list price ($4 / $20 per MTok for claude-opus-5-5, cached 2026-10-06 in the claude-api skill)
PRICE_IN, PRICE_OUT = 2.0 / 1e6, 10.0 / 1e6
BUDGET_USD = 4.0

OPUS = WORK / "opus"
LOG = ROOT / "logs" / "api" / "X3"

PROPOSE_SYSTEM = """You review Vietnamese translations of English school-mathematics test problems.
The English problem is the source of truth; its official answer was computed from the English text.
You get the English problem, the current Vietnamese translation, and the automatic checks that flagged it
(flags are only suspicions; many flagged translations are fine).

Decide whether the Vietnamese problem states exactly the same mathematical problem: the same quantities,
relations (e.g. "N more than X" is additive, "k times" is multiplicative, "N of the remaining" takes N from what
is left), conditions, units and the same asked quantity, so that a careful Vietnamese student would compute the
English answer. Also treat these as defects: calques such as "nhiều hơn N X so với Y" that read as "more than N";
wrong mathematical terms (e.g. "congruent" must be "bằng nhau", never "đồng dạng"); untranslated English
sentences; English thousands commas such as "54,321" (in Vietnamese the comma is the decimal mark); broken LaTeX.

If the Vietnamese is faithful, answer verdict "ok" and copy the current Vietnamese unchanged into corrected_vi.
Otherwise answer verdict "fix" and give the full corrected Vietnamese problem with minimal edits.

Rules for corrected_vi:
- Every number written with digits in the English must appear with the same digits (a number written as an
  English word may stay a Vietnamese word). Write numbers of 1000 or more without any thousands separator
  (54321, not 54,321 or 54.321). Keep decimals exactly as the English writes them (1.5 stays 1.5).
- Keep every LaTeX command, every $...$ / \\[...\\] span, and every [asy]...[/asy] block identical to the English.
- Natural Vietnamese school-mathematics terminology; keep person names.
- Do not add hints, units or information that the English does not contain; do not solve the problem.
reason: one short English sentence naming the defect fixed, or why the translation is fine."""

RETRANSLATE_NOTE = ("This item's official answer was revised by GSM8K-Platinum, so the problem must be re-translated "
                    "afresh from the English: answer verdict \"fix\" with a complete new faithful translation (the "
                    "current Vietnamese is shown only for terminology).")

SOLVE_SYSTEM = ("Giải bài toán sau một cách cẩn thận. Trình bày lời giải ngắn gọn bằng tiếng Việt và kết thúc bằng "
                "đáp án cuối cùng trong \\boxed{}.")

JUDGE_SYSTEM = """You check translations of mathematics problems. You get an English problem and a Vietnamese
problem. Decide whether the Vietnamese states exactly the same mathematical problem: the same numbers, units,
relations (additive vs multiplicative comparisons, what "remaining" refers to), conditions and the same asked
quantity, so that a careful reader of the Vietnamese alone would compute the same answer as a reader of the
English. Ignore style, fluency and number formatting conventions that do not change values. Answer equivalent=true
only if you find no meaning difference; otherwise list the differences in issues."""

SOLUTION_SYSTEM = """Translate the following English GSM8K solution into natural Vietnamese for Vietnamese students.
Keep every calculator annotation <<...>> exactly as written, keep every number with the same digits, and keep the
final line "#### n" exactly as in the English. Output only the translated solution text, nothing else."""

PROPOSE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["ok", "fix"]},
        "corrected_vi": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "corrected_vi", "reason"],
    "additionalProperties": False,
}
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"equivalent": {"type": "boolean"}, "issues": {"type": "string"}},
    "required": ["equivalent", "issues"],
    "additionalProperties": False,
}


def _client():
    from dotenv import load_dotenv
    import anthropic

    load_dotenv(ROOT / ".env")
    ws = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    headers = {"anthropic-workspace-id": ws} if ws else {}
    return anthropic.Anthropic(default_headers=headers, max_retries=4)


def _items() -> dict:
    rows = {}
    for f in ("gsm8k_platinum_base.jsonl", "math_algprealg_base.jsonl"):
        for r in read_jsonl(WORK / f):
            rows[r["id"]] = r
    return rows


def _queue() -> list[dict]:
    return read_jsonl(WORK / "review_queue.jsonl")


def _load(stage: str) -> dict:
    p = OPUS / f"{stage}.jsonl"
    return {r["id"]: r for r in read_jsonl(p)} if p.exists() else {}


def _append(stage: str, rows: list[dict]) -> None:
    OPUS.mkdir(parents=True, exist_ok=True)
    with open(OPUS / f"{stage}.jsonl", "a", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _gold_for_vi(r: dict) -> dict:
    return r.get("gold_vi") or r["gold"]


def candidate_text(item: dict, prop: dict) -> str:
    return prop["corrected_vi"] if prop["verdict"] == "fix" else item["question_vi"]


def solution_checks(item: dict, vi_sol: str) -> dict:
    """Acceptance checks for a translated GSM8K solution (all must hold).

    final ``#### n`` equals the Platinum gold; the ``<<...>>`` calculator annotations are identical (as a multiset)
    to the English Platinum solution; numeral parity with the English solution (number words tolerated).
    """
    import collections
    import re

    final = vi_sol.split("####")[-1].strip().replace(",", "") if "####" in vi_sol else None
    ann = lambda t: collections.Counter(re.findall(r"<<[^<>]*>>", t))
    par = parity(item["solution_en"], vi_sol, allow_en_number_words=True)
    c = {"final_equals_gold": final == item["gold_raw"],
         "annotations_identical": ann(item["solution_en"]) == ann(vi_sol),
         "numbers_ok": par["numbers_ok"], "numbers_diff": par["numbers"]}
    c["all_ok"] = c["final_equals_gold"] and c["annotations_identical"] and c["numbers_ok"]
    return c


def candidate_parity(item: dict, prop: dict) -> dict:
    """Check 1, always recomputed from the candidate text (never trusted from a stored record)."""
    return parity(item["question_en"], candidate_text(item, prop), allow_en_number_words=True)


def _params(stage: str, item: dict, q: dict, props: dict) -> dict:
    base = {"model": MODEL, "max_tokens": MAX_TOKENS, "thinking": {"type": "adaptive"}}
    if stage == "propose":
        note = RETRANSLATE_NOTE + "\n\n" if q["forced"] else ""
        user = (f"{note}Automatic flags: {', '.join(q['triggers'])}\n\n"
                f"<english>\n{item['question_en']}\n</english>\n\n<vietnamese>\n{item['question_vi']}\n</vietnamese>")
        return base | {"system": PROPOSE_SYSTEM, "messages": [{"role": "user", "content": user}],
                       "output_config": {"effort": EFFORT, "format": {"type": "json_schema", "schema": PROPOSE_SCHEMA}}}
    if stage == "solution":  # VI solution for the Platinum-revised items (their old VI solution explains the old gold)
        return base | {"system": SOLUTION_SYSTEM, "messages": [{"role": "user", "content": item["solution_en"]}],
                       "output_config": {"effort": EFFORT}}
    if stage == "judge_orig":  # negative control: the same judge on the ORIGINAL text of an auto-fixed item
        user = f"<english>\n{item['question_en']}\n</english>\n\n<vietnamese>\n{item['question_vi']}\n</vietnamese>"
        return base | {"system": JUDGE_SYSTEM, "messages": [{"role": "user", "content": user}],
                       "output_config": {"effort": EFFORT, "format": {"type": "json_schema", "schema": JUDGE_SCHEMA}}}
    cand = candidate_text(item, props[item["id"]])
    if stage == "solve":
        return base | {"system": SOLVE_SYSTEM, "messages": [{"role": "user", "content": cand}],
                       "output_config": {"effort": EFFORT}}
    if stage == "judge":
        user = f"<english>\n{item['question_en']}\n</english>\n\n<vietnamese>\n{cand}\n</vietnamese>"
        return base | {"system": JUDGE_SYSTEM, "messages": [{"role": "user", "content": user}],
                       "output_config": {"effort": EFFORT, "format": {"type": "json_schema", "schema": JUDGE_SCHEMA}}}
    raise ValueError(stage)


def _eligible(stage: str, limit: int | None, ids: list[str] | None) -> list[dict]:
    q = _queue()
    if ids:
        q = [x for x in q if x["id"] in set(ids)]
    done = _load(stage)
    props, solves = _load("propose"), _load("solve")
    items = _items()
    out = []
    for x in q:
        if x["id"] in done:
            continue
        if stage in ("solve", "judge"):
            p = props.get(x["id"])
            if not p or not p.get("ok_response"):
                continue
            par = candidate_parity(items[x["id"]], p)
            if not (par["numbers_ok"] and par["latex_ok"]):
                continue
            if x["forced"] and p["verdict"] != "fix":
                continue
        if stage == "judge":
            s = solves.get(x["id"])
            if not s or not s.get("correct"):
                continue
        if stage == "solution" and not x["forced"]:
            continue
        if stage == "judge_orig":
            dec = {r["id"]: r for r in read_jsonl(OPUS / "decisions.jsonl")}
            if dec.get(x["id"], {}).get("status") != "auto_fixed":
                continue
        out.append(x)
    del items
    return out[:limit] if limit else out


def spent_usd() -> float:
    tot = 0.0
    for st in ("propose", "solve", "judge", "judge_orig", "solution"):
        for r in _load(st).values():
            u = r.get("usage") or {}
            tot += u.get("input_tokens", 0) * PRICE_IN + u.get("output_tokens", 0) * PRICE_OUT
    return tot


def run(stage: str, limit: int | None, ids: list[str] | None) -> None:
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    todo = _eligible(stage, limit, ids)
    if not todo:
        print(f"{stage}: nothing to do")
        return
    spent = spent_usd()
    if spent >= BUDGET_USD:
        sys.exit(f"budget exhausted: ${spent:.2f} spent")
    items, props = _items(), _load("propose")
    reqs, meta = [], {}
    for k, x in enumerate(todo):
        cid = f"{stage}-{k:04d}"
        meta[cid] = x["id"]
        reqs.append(Request(custom_id=cid,
                            params=MessageCreateParamsNonStreaming(**_params(stage, items[x["id"]], x, props))))
    client = _client()
    LOG.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    batch = client.messages.batches.create(requests=reqs)
    print(f"{stage}: batch {batch.id} with {len(reqs)} requests (spent so far ${spent:.3f})", flush=True)
    with open(LOG / f"{stamp}_{stage}_requests.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in reqs:
            f.write(json.dumps({"batch_id": batch.id, "item_id": meta[r["custom_id"]], **r}, ensure_ascii=False) + "\n")
    while True:
        b = client.messages.batches.retrieve(batch.id)
        if b.processing_status == "ended":
            break
        print(f"  {b.processing_status} {b.request_counts}", flush=True)
        time.sleep(30)
    out, raw_log = [], []
    for res in client.messages.batches.results(batch.id):
        item_id = meta[res.custom_id]
        rec = {"id": item_id, "stage": stage, "batch_id": batch.id, "custom_id": res.custom_id,
               "result_type": res.result.type, "model": None, "ok_response": False, "usage": None}
        if res.result.type == "succeeded":
            msg = res.result.message
            rec["model"] = msg.model
            rec["stop_reason"] = msg.stop_reason
            rec["usage"] = {"input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens}
            text = "".join(b.text for b in msg.content if b.type == "text")
            rec["text"] = text
            if msg.stop_reason in ("end_turn", "stop_sequence"):
                _postprocess(stage, rec, items[item_id], props)
        else:
            rec["error"] = str(getattr(res.result, "error", ""))
        out.append(rec)
        raw_log.append(rec)
    _append(stage, out)
    with open(LOG / f"{stamp}_{stage}_results.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in raw_log:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{stage}: {sum(r['ok_response'] for r in out)}/{len(out)} usable; total spent ${spent_usd():.3f}")


def _postprocess(stage: str, rec: dict, item: dict, props: dict) -> None:
    text = rec["text"]
    if stage == "propose":
        try:
            d = json.loads(text)
        except json.JSONDecodeError:
            return
        rec.update(verdict=d["verdict"], corrected_vi=d["corrected_vi"], reason=d["reason"], ok_response=True)
        cand = d["corrected_vi"] if d["verdict"] == "fix" else item["question_vi"]
        rec["parity"] = parity(item["question_en"], cand, allow_en_number_words=True)
    elif stage == "solve":
        g = mv.grade(text, _gold_for_vi(item))
        rec.update(correct=bool(g["correct"]), pred_span=g.get("span"), grader_version=g.get("grader_version"),
                   ok_response=True)
    elif stage == "solution":
        rec.update(solution_checks=solution_checks(item, text), ok_response=True)
    elif stage in ("judge", "judge_orig"):
        try:
            d = json.loads(text)
        except json.JSONDecodeError:
            return
        rec.update(equivalent=bool(d["equivalent"]), issues=d["issues"], ok_response=True)


def decide() -> None:
    items, q = _items(), _queue()
    props, solves, judges = _load("propose"), _load("solve"), _load("judge")
    out = []
    for x in q:
        i = x["id"]
        p, s, j = props.get(i), solves.get(i), judges.get(i)
        rec = {"id": i, "dataset": x["dataset"], "triggers": x["triggers"], "forced": x["forced"],
               "in_math500": x["in_math500"]}
        if not p:
            rec.update(status="flagged", why="not reviewed (no proposal)")
            out.append(rec)
            continue
        par = candidate_parity(items[i], p) if p.get("ok_response") else {}
        rec.update(verdict=p.get("verdict"), reason=p.get("reason"), model=p.get("model"),
                   parity_ok=bool(par.get("numbers_ok") and par.get("latex_ok")), parity=par or None,
                   solve_correct=s.get("correct") if s else None, solve_span=s.get("pred_span") if s else None,
                   judge_equivalent=j.get("equivalent") if j else None, judge_issues=j.get("issues") if j else None)
        if p.get("verdict") == "fix":
            rec["proposed_vi"] = p["corrected_vi"]
        checks = rec["parity_ok"] and rec["solve_correct"] is True and rec["judge_equivalent"] is True
        if not p.get("ok_response"):
            rec.update(status="flagged", why="proposal unusable")
        elif x["forced"] and p.get("verdict") != "fix":
            rec.update(status="flagged", why="forced re-translation not delivered")
        elif checks:
            rec["status"] = "auto_fixed" if p["verdict"] == "fix" else "cleared"
        else:
            if not rec["parity_ok"]:
                why = "failed parity (solve/judge not run)"
            elif rec["solve_correct"] is not True:
                why = "failed solve (judge not run)"
            else:
                why = "failed judge"
            rec.update(status="flagged", why=why)
        if p.get("verdict") == "fix" and p.get("corrected_vi", "").strip() == items[i]["question_vi"].strip():
            rec["note"] = "verdict fix but text identical"
        out.append(rec)
    write_jsonl(OPUS / "decisions.jsonl", out)
    import collections
    print(json.dumps({ds: dict(collections.Counter(r["status"] for r in out if r["dataset"] == ds))
                      for ds in ("gsm8k_platinum", "math_algprealg")}, indent=1))
    print(f"spent ${spent_usd():.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--stage", required=True, choices=["propose", "solve", "judge", "judge_orig", "solution"])
    r.add_argument("--limit", type=int)
    r.add_argument("--ids", nargs="*")
    sub.add_parser("decide")
    sub.add_parser("cost")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.stage, a.limit, a.ids)
    elif a.cmd == "decide":
        decide()
    else:
        print(f"${spent_usd():.4f}")


if __name__ == "__main__":
    main()
