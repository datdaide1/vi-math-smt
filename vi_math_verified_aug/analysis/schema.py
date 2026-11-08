"""Input schema for item-level predictions and its aggregation to a dense run x item table.

One JSONL row = one decoded sample of one run on one item:

==========  =======  =====================================================================================
field       type     meaning
==========  =======  =====================================================================================
run_id      str      unique per trained run (family, size, arm, language, training seed)
arm         str      ``S0``, ``T``, ``T-off``, ``M``, ``Y+``, ``Y-``, ``N+``, ``N-`` (``N+V2`` etc. allowed)
model       str      model id, e.g. ``qwen3-0.6b`` (family x size); the GLMM "model" factor
size        str      ``0.6B`` / ``1.7B`` / ``4B`` ...
seed        int      training-seed index; with CRN the VI and EN runs of one (size, arm, seed) share it
lang        str      ``vi`` or ``en`` -- the training *and* evaluation language of the run
dataset     str      language-neutral set name (``gsm8k_platinum``, ``math_algprealg``, ``vi_exam_core``)
item_id     str      item id inside ``dataset``; **shared by an EN original and its VI translation**
sample_k    int      sample index 0..K-1 (``-1`` = greedy decode, unless ``decode`` says otherwise)
correct     0/1      ``math_verify`` verdict (after judge adjudication)
tokens      int      generated tokens
==========  =======  =====================================================================================

Optional row fields: ``decode`` (``sample``/``greedy``), ``family``, ``evaltype`` (``native``/``parallel``),
``certifiable`` (bool, item has a symbolic certificate type -- P3), ``period`` (``pre``/``post`` cutoff),
``pair_id`` (temporal slot-matched pair), ``truncated``, ``judged``.

The item key used for every item-level random effect and for item resampling is ``dataset/item_id``, so the EN
and VI versions of a parallel item are one cluster.  Each run is in one language, so (run, item key) is unique.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

REQUIRED = ("run_id", "arm", "model", "size", "seed", "lang", "dataset", "item_id", "sample_k", "correct",
            "tokens")
RUN_FIELDS = ("arm", "model", "size", "seed", "lang")
ITEM_OPTIONAL = ("evaltype", "certifiable", "period", "pair_id")
LANGS = ("vi", "en")
_ARM_RE = re.compile(r"^([YN])([+-])")


def normalize_arm(arm: str) -> str:
    return str(arm).strip().replace("−", "-").replace("–", "-")


def arm_method_verif(arm: str) -> tuple[str | None, str | None]:
    """``Y+`` -> (``Y``, ``+``); ``N-`` -> (``N``, ``-``); anything else -> (None, None)."""
    m = _ARM_RE.match(normalize_arm(arm))
    return (m.group(1), m.group(2)) if m else (None, None)


def default_evaltype(dataset: str) -> str:
    """Native Vietnamese exam items vs the parallel (translated in VI / original in EN) sets."""
    d = dataset.lower()
    return "native" if ("exam" in d or "core" in d or "hard" in d) else "parallel"


@dataclass
class EvalData:
    """Dense run x item representation (R runs, I items). ``n == 0`` marks an unobserved cell."""
    runs: pd.DataFrame            # one row per run, positional index r
    items: pd.DataFrame           # one row per item key, positional index i
    y: np.ndarray                 # (R, I) float32: correct samples (sampled decodes only)
    n: np.ndarray                 # (R, I) int16: number of sampled decodes
    tok: np.ndarray               # (R, I) float32: mean generated tokens over sampled decodes (nan if n == 0)
    greedy: np.ndarray            # (R, I) float32: greedy correctness (nan if absent)
    report: dict = field(default_factory=dict)

    @property
    def acc(self) -> np.ndarray:
        """Per run-item mean pass@1 (nan where unobserved)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.n > 0, self.y / np.maximum(self.n, 1), np.nan)

    def subset_runs(self, mask: np.ndarray) -> "EvalData":
        mask = np.asarray(mask, bool)
        return EvalData(self.runs[mask].reset_index(drop=True), self.items, self.y[mask], self.n[mask],
                        self.tok[mask], self.greedy[mask], dict(self.report))

    def subset_items(self, mask: np.ndarray) -> "EvalData":
        mask = np.asarray(mask, bool)
        return EvalData(self.runs, self.items[mask].reset_index(drop=True), self.y[:, mask], self.n[:, mask],
                        self.tok[:, mask], self.greedy[:, mask], dict(self.report))

    def run_mask(self, **flt) -> np.ndarray:
        return filter_mask(self.runs, flt)

    def item_mask(self, **flt) -> np.ndarray:
        return filter_mask(self.items, flt)


def filter_mask(df: pd.DataFrame, flt: dict) -> np.ndarray:
    """AND of column filters; a value may be a scalar or a list/tuple/set of allowed values."""
    m = np.ones(len(df), bool)
    for k, v in flt.items():
        if v is None:
            continue
        col = df[k].to_numpy()
        if isinstance(v, (list, tuple, set, frozenset)):
            m &= np.isin(col, list(v))
        else:
            m &= col == v
    return m


class SchemaError(ValueError):
    pass


def iter_jsonl(path: str) -> Iterator[dict]:
    with open(path, encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise SchemaError(f"{path}:{ln}: bad JSON ({e})") from None


def _is_greedy(row: dict) -> bool:
    dec = row.get("decode")
    if dec is not None:
        return str(dec).lower() == "greedy"
    return int(row["sample_k"]) < 0


def build(rows: Iterable[dict], *, evaltype_fn=default_evaltype, item_meta: dict | None = None,
          strict: bool = True) -> EvalData:
    """Validate rows (streamed) and aggregate them to an :class:`EvalData`.

    Memory is O(runs x items), never O(rows): rows are folded into per-cell accumulators as they stream in.
    """
    runs: dict[str, dict] = {}
    items: dict[str, dict] = {}
    acc: dict[tuple[str, str], list] = {}      # (run_id, item_key) -> [y, n, tok_sum, greedy, seen bitmask]
    errors: list[str] = []
    n_rows = n_greedy = n_dup = n_trunc = n_judged = 0

    def err(msg):
        if strict and len(errors) >= 20:
            raise SchemaError("; ".join(errors))
        errors.append(msg)

    for row in rows:
        n_rows += 1
        miss = [f for f in REQUIRED if f not in row]
        if miss:
            err(f"row {n_rows}: missing {miss}")
            continue
        rid = str(row["run_id"])
        meta = {"arm": normalize_arm(row["arm"]), "model": str(row["model"]), "size": str(row["size"]),
                "seed": int(row["seed"]), "lang": str(row["lang"]).lower()}
        if meta["lang"] not in LANGS:
            err(f"row {n_rows}: lang={meta['lang']!r} not in {LANGS}")
            continue
        c = row["correct"]
        if c not in (0, 1, True, False):
            err(f"row {n_rows}: correct={c!r} not 0/1")
            continue
        if rid in runs:
            old = runs[rid]
            bad = [k for k in RUN_FIELDS if old[k] != meta[k]]
            if bad:
                err(f"run {rid}: inconsistent {bad} ({[old[k] for k in bad]} vs {[meta[k] for k in bad]})")
                continue
        else:
            meta["family"] = str(row.get("family", meta["model"].split("-")[0]))
            runs[rid] = meta
        ds = str(row["dataset"])
        key = f"{ds}/{row['item_id']}"
        if key not in items:
            items[key] = {"item_key": key, "dataset": ds, "item_id": str(row["item_id"]),
                          "evaltype": str(row.get("evaltype") or evaltype_fn(ds))}
        it = items[key]
        for f in ("certifiable", "period", "pair_id"):
            if f in row and row[f] is not None and f not in it:
                it[f] = row[f]
        cell = acc.get((rid, key))
        if cell is None:
            cell = acc[(rid, key)] = [0, 0, 0.0, np.nan, 0]
        greedy = _is_greedy(row)
        sk = -1 if greedy else int(row["sample_k"])
        bit = 1 << (sk + 1)                     # bit 0 = greedy, bit k+1 = sample k
        if cell[4] & bit:
            n_dup += 1
            err(f"duplicate sample: run {rid} item {key} sample {sk}")
            continue
        cell[4] |= bit
        n_trunc += int(bool(row.get("truncated", False)))
        n_judged += int(bool(row.get("judged", False)))
        if greedy:
            n_greedy += 1
            cell[3] = float(bool(c))
        else:
            cell[0] += int(bool(c))
            cell[1] += 1
            cell[2] += float(row["tokens"])
    if errors and strict:
        raise SchemaError("; ".join(errors[:20]))

    run_df = pd.DataFrame([{"run_id": r, **m} for r, m in runs.items()])
    if len(run_df) == 0:
        raise SchemaError("no valid rows")
    mv = [arm_method_verif(a) for a in run_df["arm"]]
    run_df["method"] = [m for m, _ in mv]
    run_df["verif"] = [v for _, v in mv]
    run_df = run_df.sort_values(["lang", "arm", "model", "seed", "run_id"]).reset_index(drop=True)
    item_df = pd.DataFrame(list(items.values()))
    if item_meta:
        for f in ITEM_OPTIONAL:
            vals = [item_meta.get(k, {}).get(f) for k in item_df["item_key"]]
            if any(v is not None for v in vals):
                item_df[f] = [v if v is not None else (item_df.at[j, f] if f in item_df else None)
                              for j, v in enumerate(vals)]
    item_df = item_df.sort_values(["dataset", "item_id"]).reset_index(drop=True)
    ridx = {r: j for j, r in enumerate(run_df["run_id"])}
    iidx = {k: j for j, k in enumerate(item_df["item_key"])}
    R, I = len(run_df), len(item_df)
    y = np.zeros((R, I), np.float32)
    n = np.zeros((R, I), np.int16)
    tok = np.full((R, I), np.nan, np.float32)
    g = np.full((R, I), np.nan, np.float32)
    for (rid, key), (cy, cn, ct, cg, _) in acc.items():
        r, i = ridx[rid], iidx[key]
        y[r, i], n[r, i], g[r, i] = cy, cn, cg
        if cn:
            tok[r, i] = ct / cn
    k_per_run = [sorted(set(n[r][n[r] > 0].tolist())) for r in range(R)]
    report = {"rows": n_rows, "runs": R, "items": I, "greedy_rows": n_greedy, "duplicates": n_dup,
              "truncated_rows": n_trunc, "judged_rows": n_judged, "errors": errors[:50],
              "K_values": sorted({k for ks in k_per_run for k in ks}),
              "runs_with_mixed_K": int(sum(len(ks) > 1 for ks in k_per_run))}
    return EvalData(run_df, item_df, y, n, tok, g, report)


def load_predictions(path: str | list[str], **kw) -> EvalData:
    paths = [path] if isinstance(path, str) else list(path)

    def gen():
        for p in paths:
            yield from iter_jsonl(p)
    return build(gen(), **kw)


def to_long(d: EvalData) -> pd.DataFrame:
    """One row per observed (run, item): the binomial table used by the GLMMs (R export and Python VB)."""
    r_idx, i_idx = np.nonzero(d.n > 0)
    runs, items = d.runs.iloc[r_idx].reset_index(drop=True), d.items.iloc[i_idx].reset_index(drop=True)
    out = pd.concat([runs, items], axis=1)
    out["correct"] = d.y[r_idx, i_idx].astype(int)
    out["K"] = d.n[r_idx, i_idx].astype(int)
    return out


def run_table(d: EvalData) -> pd.DataFrame:
    """Per run x dataset: mean pass@1 over K, greedy accuracy, mean tokens, n_items (T5 §2.6)."""
    rows = []
    acc = d.acc
    for ds in sorted(d.items["dataset"].unique()):
        cols = d.items["dataset"].to_numpy() == ds
        for r in range(len(d.runs)):
            obs = d.n[r, cols] > 0
            if not obs.any():
                continue
            a = acc[r, cols][obs]
            gr = d.greedy[r, cols][obs]
            rows.append({**d.runs.iloc[r][["run_id", "lang", "arm", "model", "size", "seed"]].to_dict(),
                         "dataset": ds, "n_items": int(obs.sum()), "pass1_mean": float(a.mean()),
                         "greedy_acc": float(np.nanmean(gr)) if np.isfinite(gr).any() else None,
                         "tokens_mean": float(np.nanmean(d.tok[r, cols][obs]))})
    return pd.DataFrame(rows)
