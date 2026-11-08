"""Resumable, budget-guarded Message Batches runner (Claude only, -50%).

* Requests are submitted in chunks; every submitted batch is appended to ``<WORK>/batches.jsonl`` with its
  ``request_hash -> item key`` map and an *expected-cost* estimate, BEFORE results exist. A crashed or restarted
  process re-attaches to every un-collected batch (no double submission, no lost results).
* The batch manifest is also written in ``common.llm.ClaudeBatches``'s format (``<state>/batches/<id>.json``), so
  collection, API logging (``logs/api/D1/claude.jsonl``), the response cache and the idempotent spend ledger are
  the shared implementation (``ClaudeBatches.collect``).
* Budget: before each submission, ``spent (ledger, scope D1) + expected cost of un-collected batches + this chunk``
  must stay <= the cap; otherwise ``BudgetStop`` is raised and nothing is sent. The estimate per item is supplied by
  the caller (measured on the pilot, with a safety factor) instead of ``max_tokens`` worst case, which would
  reserve ~10x the real cost and block the pipeline.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import time
from pathlib import Path
from typing import Awaitable, Callable

from ..common.apilog import utc_now
from ..common.llm import ClaudeProvider, LLMRequest, LLMResponse
from .common import CAP_USD, WORK, client, iter_jsonl


class BudgetStop(RuntimeError):
    pass


class BatchManager:
    def __init__(self, state_path: Path = WORK / "batches.jsonl", cap_usd: float = CAP_USD):
        self.path = state_path
        self.cap = cap_usd
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.batches: dict[str, dict] = {}
        if self.path.exists():
            for r in iter_jsonl(self.path):
                self.batches[r["batch_id"]] = {**self.batches.get(r["batch_id"], {}), **r}

    def _append(self, row: dict) -> None:
        self.batches[row["batch_id"]] = {**self.batches.get(row["batch_id"], {}), **row}
        with open(self.path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------ queries
    def outstanding(self, stage: str | None = None) -> list[dict]:
        return [b for b in self.batches.values() if b.get("status") == "submitted"
                and (stage is None or b["stage"] == stage)]

    def outstanding_keys(self, stage: str) -> set[str]:
        return {k for b in self.outstanding(stage) for k in b["keys"].values()}

    def committed_usd(self) -> float:
        return sum(b["est_usd"] for b in self.outstanding())

    def spent_usd(self) -> float:
        return client().costs.spent()

    # ------------------------------------------------------------ submit
    async def submit(self, stage: str, lang: str, jobs: list[tuple[str, LLMRequest]], est_per_item: float,
                     on_cached: Callable[[str, LLMResponse], Awaitable[None]] | None = None) -> dict | None:
        c = client()
        todo: dict[str, LLMRequest] = {}
        keys: dict[str, str] = {}
        for key, r in jobs:
            h = r.request_hash("claude")
            if c.cache is not None and h in c.cache:          # identical request answered before: free
                if on_cached is not None:
                    await on_cached(key, await c.call(r))
                continue
            if h not in todo:
                todo[h] = r
                keys[h] = key
        if not todo:
            return None
        est = est_per_item * len(todo)
        spent, committed = self.spent_usd(), self.committed_usd()
        if spent + committed + est > self.cap:
            raise BudgetStop(f"[D1] spent ${spent:.3f} + un-collected ${committed:.3f} + this {stage}/{lang} chunk "
                             f"(n={len(todo)}) ${est:.3f} > cap ${self.cap:.2f}")
        prov = c.provider("claude")
        b = await prov.batch_create([{"custom_id": h, "params": ClaudeProvider.params(r)} for h, r in todo.items()])
        man = {"batch_id": b["id"], "created_utc": utc_now(), "n_requests": len(todo), "n_cached": 0,
               "items": {h: dataclasses.asdict(r) for h, r in todo.items()}, "reserved_usd": est}
        (c.batches._dir() / f"{b['id']}.json").write_text(json.dumps(man, ensure_ascii=False), encoding="utf-8")
        c.logger.log({"provider": "claude", "event": "batch_submit", "batch_id": b["id"], "stage": stage,
                      "lang": lang, "n_items": len(todo), "est_usd": round(est, 6),
                      "client_lib_version": prov.lib_version()})
        row = {"batch_id": b["id"], "stage": stage, "lang": lang, "n": len(todo), "est_usd": est, "keys": keys,
               "status": "submitted", "submitted_utc": utc_now()}
        self._append(row)
        return row

    # ------------------------------------------------------------ poll / collect
    async def poll(self, on_result: Callable[[str, str, LLMResponse], Awaitable[None]]) -> int:
        """Collect every ended batch; ``on_result(stage, key, resp)``. Returns the number of batches collected."""
        c = client()
        n = 0
        for b in list(self.outstanding()):
            try:
                st = await c.batches.status(b["batch_id"])
            except Exception as e:                           # transient API error: try next tick
                print(f"[batch] status {b['batch_id']} failed: {type(e).__name__}: {e}", flush=True)
                continue
            if st.get("processing_status") != "ended":
                continue
            got = await c.batches.collect(b["batch_id"])
            for h, resp in got.items():
                key = b["keys"].get(h)
                if key is not None:
                    await on_result(b["stage"], key, resp)
            cost = sum(r.cost_usd or 0.0 for r in got.values())
            self._append({"batch_id": b["batch_id"], "status": "collected", "collected_utc": utc_now(),
                          "n_results": len(got), "n_ok": sum(r.ok for r in got.values()), "cost_usd": cost,
                          "request_counts": st.get("request_counts")})
            print(f"[batch] collected {b['stage']}/{b['lang']} {b['batch_id']} n={len(got)} "
                  f"ok={sum(r.ok for r in got.values())} ${cost:.4f}", flush=True)
            n += 1
        return n

    def summary(self) -> dict:
        by: dict[str, dict] = {}
        for b in self.batches.values():
            k = f"{b['stage']}/{b['lang']}"
            d = by.setdefault(k, {"batches": 0, "n": 0, "outstanding": 0, "cost_usd": 0.0})
            d["batches"] += 1
            d["n"] += b["n"]
            d["outstanding"] += b.get("status") == "submitted"
            d["cost_usd"] += b.get("cost_usd") or 0.0
        return by


async def sleep_until(t: float) -> None:
    d = t - time.time()
    if d > 0:
        await asyncio.sleep(d)
