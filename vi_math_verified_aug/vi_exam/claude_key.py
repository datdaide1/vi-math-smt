"""Key #3 of the 3-key protocol (T5 §3.1 stage 2): Claude solves Core candidates *blind* (no page solution, no
provisional answer, no SymPy program). Message Batches API (50 % price), ``claude-opus-5-5``, effort ``low``.

  python -m vi_math_verified_aug.vi_exam.claude_key submit --n 20 --tag pilot      # pilot, measures tokens
  python -m vi_math_verified_aug.vi_exam.claude_key submit --budget 2.4 --tag main # sized from the pilot
  python -m vi_math_verified_aug.vi_exam.claude_key collect <batch_id>             # -> data/vi_exam/claude_keys.jsonl

Only Vi-ExamMath test items go to Claude (EXEC_BRIEF2: never to NIM/Gemini). Every request and result is logged
under ``logs/api/X2/``. The key is never printed. Prices (claude-api skill table, cached 2026-10-06): Opus 5.5
$4 / $20 per MTok; batch = 50 % -> $2 / $10.

Selection order (deterministic): post-cutoff Core items first (exam_year >= 2025), then items whose SymPy key
disagrees with / cannot parse the provisional answer, then the rest by a hash of the id.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import time
from datetime import datetime, timezone

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "vi_exam"
LOG = REPO / "logs" / "api" / "X2"
ITEMS = DATA / "items.jsonl"
KEYS = DATA / "claude_keys.jsonl"
MODEL = "claude-opus-5-5"
PRICE_IN, PRICE_OUT = 2.0 / 1e6, 10.0 / 1e6  # batch prices, $/token

SYSTEM = (
    "Bạn là giáo viên Toán THCS. Giải bài toán thi tuyển sinh vào lớp 10 sau một cách ngắn gọn, chính xác. "
    "Nếu bài yêu cầu giải phương trình/hệ phương trình, nêu đầy đủ tập nghiệm thực. Nếu yêu cầu rút gọn, nêu biểu "
    "thức đã rút gọn. Nếu yêu cầu tìm tham số, nêu điều kiện/giá trị cuối cùng. Dòng CUỐI CÙNG của câu trả lời phải "
    "có dạng:\nĐÁP ÁN: <đáp án cuối cùng, dùng LaTeX nếu cần, không giải thích>")


def _env() -> dict:
    vals = {}
    for ln in (REPO / ".env").read_text(encoding="utf-8").splitlines():
        if "=" in ln and not ln.lstrip().startswith("#"):
            k, v = ln.split("=", 1)
            vals[k.strip()] = v.strip().strip('"').strip("'")
    return vals


def client():
    import anthropic
    e = _env()
    return anthropic.Anthropic(api_key=e["ANTHROPIC_API_KEY"],
                               default_headers={"anthropic-workspace-id": e["ANTHROPIC_WORKSPACE_ID"]})


def cid(item_id: str) -> str:
    """Batch custom_id ([A-Za-z0-9_-]{1,64}): '.' -> '_', '~' (repeated sub-label suffix) -> '-'."""
    out = item_id.replace(".", "_").replace("~", "-")
    if len(out) > 64:
        raise ValueError(f"custom_id too long: {item_id}")
    return out


def problem_sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def select(n: int | None, exclude: set[str]) -> list[dict]:
    core = []
    with ITEMS.open(encoding="utf-8") as fh:
        for ln in fh:
            it = json.loads(ln)
            if it["core_candidate"] and it["id"] not in exclude:
                core.append({"id": it["id"], "problem": it["problem"], "year": it["provenance"]["exam_year"] or 0,
                             "agree": it["key_agreement_sympy_vs_provisional"]})

    def rank(r):
        h = int(hashlib.sha1(r["id"].encode()).hexdigest()[:8], 16)
        return (0 if r["year"] >= 2025 else 1, 0 if r["agree"] in ("disagree", "unparsed") else 1, h)
    core.sort(key=rank)
    return core if n is None else core[:n]


def request(it: dict) -> dict:
    return {"custom_id": cid(it["id"]),
            "params": {"model": MODEL, "max_tokens": 6000, "system": SYSTEM,
                       "output_config": {"effort": "low"},
                       "messages": [{"role": "user", "content": it["problem"]}]}}


def done_ids() -> set[str]:
    out = set()
    if KEYS.exists():
        for ln in KEYS.open(encoding="utf-8"):
            out.add(json.loads(ln)["id"])
    return out


def submit(n: int | None, budget: float | None, per_item: float | None, tag: str):
    exclude = done_ids()
    pending = LOG / "pending_batches.jsonl"
    collected = set()
    ledger = LOG / "cost_ledger.jsonl"
    if ledger.exists():
        collected = {json.loads(ln)["batch_id"] for ln in ledger.open(encoding="utf-8")}
    if pending.exists():  # only batches still in flight; collected ones are represented by claude_keys.jsonl
        for ln in pending.open(encoding="utf-8"):
            r = json.loads(ln)
            if r["batch_id"] not in collected:
                exclude |= set(r["ids"])
    if budget is not None:
        n = int(budget / per_item)
    items = select(n, exclude)
    reqs = [request(it) for it in items]
    LOG.mkdir(parents=True, exist_ok=True)
    st = _stamp()
    (LOG / f"{st}_{tag}_requests.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in reqs),
                                                     encoding="utf-8")
    b = client().messages.batches.create(requests=reqs)
    with pending.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"batch_id": b.id, "tag": tag, "submitted_at": st, "n": len(reqs),
                             "ids": [it["id"] for it in items]}) + "\n")
    print(f"submitted batch {b.id} with {len(reqs)} requests ({tag})")


RE_FINAL = re.compile(r"ĐÁP\s*ÁN\s*:\s*(.+)$", re.I | re.M)


def collect(batch_id: str, wait: bool = True):
    c = client()
    while True:
        b = c.messages.batches.retrieve(batch_id)
        if b.processing_status == "ended" or not wait:
            break
        time.sleep(30)
    if b.processing_status != "ended":
        print(f"batch {batch_id}: {b.processing_status} {b.request_counts}")
        return
    ids = {}
    sha = {}
    pend = LOG / "pending_batches.jsonl"
    for ln in pend.open(encoding="utf-8"):
        r = json.loads(ln)
        if r["batch_id"] == batch_id:
            ids = {cid(i): i for i in r["ids"]}
            tag = r["tag"]
            reqf = LOG / f"{r['submitted_at']}_{tag}_requests.jsonl"
            if reqf.exists():  # content address: the build only accepts a key whose problem text still matches
                for rl in reqf.open(encoding="utf-8"):
                    q = json.loads(rl)
                    sha[q["custom_id"]] = problem_sha1(q["params"]["messages"][0]["content"])
    st = _stamp()
    raw_log = (LOG / f"{st}_{tag}_results.jsonl").open("w", encoding="utf-8")
    tin = tout = 0
    n_ok = 0
    with KEYS.open("a", encoding="utf-8") as out:
        for res in c.messages.batches.results(batch_id):
            rec = {"custom_id": res.custom_id, "type": res.result.type}
            row = {"id": ids.get(res.custom_id, res.custom_id), "model": MODEL, "batch_id": batch_id,
                   "status": res.result.type, "answer": None, "stop_reason": None,
                   "problem_sha1": sha.get(res.custom_id)}
            if res.result.type == "succeeded":
                msg = res.result.message
                text = "".join(getattr(blk, "text", "") for blk in msg.content if blk.type == "text")
                u = msg.usage
                tin += u.input_tokens
                tout += u.output_tokens
                m = RE_FINAL.findall(text)
                row.update(answer=m[-1].strip() if m else None, stop_reason=msg.stop_reason,
                           input_tokens=u.input_tokens, output_tokens=u.output_tokens, response_tail=text[-600:])
                rec.update(stop_reason=msg.stop_reason, usage={"in": u.input_tokens, "out": u.output_tokens},
                           text=text)
                n_ok += 1
            raw_log.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    raw_log.close()
    cost = tin * PRICE_IN + tout * PRICE_OUT
    summ = {"batch_id": batch_id, "tag": tag, "succeeded": n_ok, "input_tokens": tin, "output_tokens": tout,
            "cost_usd": round(cost, 4), "per_item_usd": round(cost / max(1, n_ok), 5), "collected_at": st}
    with (LOG / "cost_ledger.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(summ) + "\n")
    print(json.dumps(summ))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--n", type=int)
    s.add_argument("--budget", type=float)
    s.add_argument("--per-item", type=float)
    s.add_argument("--tag", default="main")
    c = sub.add_parser("collect")
    c.add_argument("batch_id")
    c.add_argument("--no-wait", action="store_true")
    a = ap.parse_args()
    if a.cmd == "submit":
        submit(a.n, a.budget, a.per_item, a.tag)
    else:
        collect(a.batch_id, wait=not a.no_wait)


if __name__ == "__main__":
    main()
