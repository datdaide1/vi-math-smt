"""SFT data path: read -> tokenise -> DROP over-long items (never truncate) -> seeded subsample -> step plan.

Batching is **padded and length-grouped** by default (:func:`collate`).  :func:`collate_packed` concatenates a
micro-batch into one row (``position_ids`` reset per item, ``packed_seq_lengths``, no attention mask) for the
Unsloth engine only; it is used only after the start-of-job logits gate passes.  (The W2 v1 "leak" was the
DynamicCache, not SDPA: S2 §3 and O1 §0.3; plain-HF packing is exact with ``use_cache=False`` but slower on T4.)

* Every optimiser step sees exactly ``seq_per_step`` sequences (32).  The step's items come from a seeded
  per-epoch permutation; *inside* the step they are sorted by length and cut into micro-batches, so the set of
  items per step is the same as with random micro-batches but padding waste is small.
* The loss of a step is the token mean over all completion tokens of its 32 sequences (``n_completion_tokens``
  is computed per step and every micro-batch divides by it).
* The trailing ``n_train % seq_per_step`` items of each epoch are not used, so every arm with the same
  ``n_train`` has the identical step count (matched items, T3 §3.2 option A).

Torch-free on purpose (tests run without torch); ``collate`` returns plain lists.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Iterable

from . import template as T

DEFAULT_FIELDS = {
    "id": ("id", "item_id", "uid"),
    "question": ("question", "problem"),
    "solution": ("solution", "completion"),
    "answer": ("answer_canonical", "ground_truth", "final_answer"),
    "gold": ("gold",),
}


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_field(row: dict, name: str, mapping: dict | None = None):
    """Look ``name`` up through the config mapping (``data.fields``) or the default aliases."""
    keys = (mapping or {}).get(name) or DEFAULT_FIELDS.get(name, (name,))
    if isinstance(keys, str):
        keys = (keys,)
    for k in keys:
        cur = row
        for part in k.split("."):
            cur = cur.get(part) if isinstance(cur, dict) else None
        if cur not in (None, ""):
            return cur
    return None


def item_answer(row: dict, mapping: dict | None = None):
    a = get_field(row, "answer", mapping)
    if a is None:
        g = get_field(row, "gold", mapping)
        if isinstance(g, dict):
            a = g.get("answer_canonical", g.get("value"))
    if a is None and not (mapping or {}).get("answer"):
        # generator outputs (D1/G4) carry a short final answer in `answer`; raw GSM8K's `answer` is the whole
        # solution ("... #### 72"), which must never become the boxed answer
        b = row.get("answer")
        if isinstance(b, (str, int, float)) and not isinstance(b, bool):
            b = str(b).strip()
            if b and "\n" not in b and "####" not in b and len(b) <= 64:
                a = b
    return a


def encode(tok, question: str, solution: str, answer, max_length: int, lang: str = "vi") -> dict | None:
    """Token ids + labels (prompt masked with -100, completion + EOS trained).  None if longer than max_length."""
    from .modeling import bos_prefix
    bos = bos_prefix(tok)
    p = tok(T.prompt(question, lang), add_special_tokens=False)["input_ids"]
    c = tok(T.completion(solution, None if answer is None else str(answer), lang),
            add_special_tokens=False)["input_ids"]
    c = c + [tok.eos_token_id]
    ids = bos + p + c
    if len(ids) > max_length:
        return None
    return {"ids": ids, "labels": [-100] * (len(bos) + len(p)) + c, "n_completion": len(c)}


def _pct(xs: list[int], q: float) -> int:
    if not xs:
        return 0
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))]


def prepare_pool(rows: Iterable[dict], tok, max_length: int, fields: dict | None = None,
                 lang: str = "vi") -> tuple[list[dict], dict]:
    """Tokenise the whole arm pool and drop (never truncate) items longer than ``max_length``.

    The drop happens on the pool BEFORE the seeded subsample, so every arm still gets exactly ``n_train`` items.
    Returns (eligible items in pool order, stats).  Stats carry the per-arm drop rate that must be reported.
    """
    items, n_pool, n_long, n_bad, lens_all = [], 0, 0, 0, []
    for idx, r in enumerate(rows):
        n_pool += 1
        q, sol, ans = get_field(r, "question", fields), get_field(r, "solution", fields), item_answer(r, fields)
        if not q or not sol:
            n_bad += 1
            continue
        try:
            enc = encode(tok, q, sol, ans, max_length=10 ** 9, lang=lang)
        except ValueError:  # no answer anywhere
            n_bad += 1
            continue
        lens_all.append(len(enc["ids"]))
        if len(enc["ids"]) > max_length:
            n_long += 1
            continue
        enc["id"] = str(get_field(r, "id", fields) or f"row{idx}")
        enc["pool_index"] = idx
        enc.update(q=q, sol=sol, ans=None if ans is None else str(ans))  # kept for the pre-flight guards
        items.append(enc)
    n_ok = n_pool - n_bad
    if n_pool and n_bad / n_pool > 0.01:  # a field-name mismatch must not silently empty or thin the pool
        raise ValueError(f"{n_bad}/{n_pool} rows have no question/solution/answer under the field mapping "
                         f"{fields or DEFAULT_FIELDS}")
    stats = {"n_pool": n_pool, "n_malformed": n_bad, "n_dropped_too_long": n_long, "max_length": max_length,
             "drop_rate": round(n_long / n_ok, 6) if n_ok else None, "n_eligible": len(items),
             "pool_len_p50": _pct(lens_all, 0.50), "pool_len_p99": _pct(lens_all, 0.99),
             "pool_len_max": max(lens_all) if lens_all else 0}
    return items, stats


def suggest_max_length(lengths: list[int], q: float = 0.99, multiple: int = 64) -> int:
    """T3: max_length = p99 of the pooled training lengths, rounded up to a multiple of 64."""
    return int(math.ceil(_pct(lengths, q) / multiple) * multiple)


def subsample(items: list[dict], n: int | None, seed: int) -> list[dict]:
    """Seeded draw of n items (the seed's own subsample, T5 §2.1); kept in pool order.

    QA-T B3: n_eff = min(n, eligible) -- a few dropped rows never crash a run; the caller records the shortfall and
    aborts only when it exceeds 1 % (``sft.SFTRun.setup``)."""
    if n is None or n >= len(items):
        return list(items)
    rng = random.Random(f"subsample:{seed}")
    picked = sorted(rng.sample(range(len(items)), n))
    return [items[i] for i in picked]


def fingerprint(items: list[dict]) -> str:
    h = hashlib.sha1()
    for it in items:
        h.update(it["id"].encode("utf-8"))
        h.update(b"\0")
        h.update(str(len(it["ids"])).encode())
        h.update(b"\n")
    return h.hexdigest()[:16]


def length_stats(items: list[dict]) -> dict:
    comp = [it["n_completion"] for it in items]
    tot = [len(it["ids"]) for it in items]
    return {"n_items": len(items), "completion_tokens_mean": round(sum(comp) / max(1, len(comp)), 2),
            "completion_tokens_p99": _pct(comp, 0.99), "completion_tokens_total": sum(comp),
            "sequence_tokens_total": sum(tot), "sequence_tokens_p99": _pct(tot, 0.99)}


class StepPlan:
    """Which items each optimiser step sees.  Deterministic in (seed, epoch); resumable by global step."""

    def __init__(self, n_items: int, seq_per_step: int, seed: int):
        if n_items < seq_per_step:
            raise ValueError(f"{n_items} items < seq_per_step={seq_per_step}")
        self.n, self.k, self.seed = n_items, seq_per_step, seed
        self.steps_per_epoch = n_items // seq_per_step
        self._cache: dict[int, list[int]] = {}

    def order(self, epoch: int) -> list[int]:
        if epoch not in self._cache:
            idx = list(range(self.n))
            random.Random(f"order:{self.seed}:{epoch}").shuffle(idx)
            self._cache = {epoch: idx}
        return self._cache[epoch]

    def step_indices(self, global_step: int) -> list[int]:
        e, s = divmod(global_step, self.steps_per_epoch)
        return self.order(e)[s * self.k:(s + 1) * self.k]


def micro_batches(step_items: list[dict], micro: int, grouping: str = "within_step") -> list[list[dict]]:
    its = sorted(step_items, key=lambda it: -len(it["ids"])) if grouping == "within_step" else list(step_items)
    return [its[i:i + micro] for i in range(0, len(its), micro)]


def collate(batch: list[dict], pad_id: int) -> dict:
    """Right-padded rows, one sequence per row, explicit attention mask.  Never packs.

    Returns lists: input_ids, attention_mask, labels; plus real/padded token counts.
    """
    L = max(len(it["ids"]) for it in batch)
    ids, att, lab = [], [], []
    for it in batch:
        n = len(it["ids"])
        assert len(it["labels"]) == n, "labels/ids length mismatch"
        ids.append(it["ids"] + [pad_id] * (L - n))
        att.append([1] * n + [0] * (L - n))
        lab.append(it["labels"] + [-100] * (L - n))
    return {"input_ids": ids, "attention_mask": att, "labels": lab,
            "real_tokens": sum(len(it["ids"]) for it in batch), "padded_tokens": L * len(batch)}


def collate_packed(batch: list[dict]) -> dict:
    """One row: the micro-batch's items concatenated.  Each item's own labels already start with -100 (its prompt),
    so no target ever crosses an item boundary; positions restart at 0 for every item."""
    ids, lab, pos, lens = [], [], [], []
    for it in batch:
        n = len(it["ids"])
        assert len(it["labels"]) == n and it["labels"][0] == -100, "packed items must start with a masked token"
        ids += it["ids"]
        lab += it["labels"]
        pos += list(range(n))
        lens.append(n)
    return {"input_ids": [ids], "labels": [lab], "position_ids": [pos], "packed_seq_lengths": lens,
            "real_tokens": len(ids), "padded_tokens": len(ids)}


def n_completion_tokens(step_items: list[dict]) -> int:
    """Number of trained (label != -100) next-token targets in a step = the token-mean denominator."""
    return sum(sum(1 for x in it["labels"][1:] if x != -100) for it in step_items)
