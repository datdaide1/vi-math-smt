"""D5: per-arm diversity of the selected training sets (T1 §2.6 "report, do not force-match"; P3 evidence).

Per arm x set (final = matched, final_natural, final_shared): unique seeds, Vendi score (bge-m3, cosine kernel),
mean pairwise cosine distance, mean cosine and chrF to the item's own seed, edit-class histogram.
Vendi = exp(-sum lambda log lambda) over the eigenvalues of K/n, K = X X^T (unit rows); computed from the d x d
matrix X^T X / n (same non-zero spectrum), so 3,000 x 1,024 costs ~12 MB. Embeddings: select_v1_embed_modal.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .select_v1 import EDIT_CLASS, FINAL, NATURAL, ROOT, SEED_ALIGNED, SEED_POOL, SHARED, read_jsonl

EMB = ROOT / "data" / "audit" / "d5"


def seed_texts(lang: str, seeds: set) -> dict[str, str]:
    src = SEED_POOL if SEED_POOL.exists() else SEED_ALIGNED
    out = {}
    for r in read_jsonl(src):
        sid = r.get("seed_id")
        if sid in seeds:
            if lang == "vi":
                t = r.get("problem_vi") or r.get("problem") or r.get("text")
            else:
                en = r.get("en")
                t = (en.get("problem") if isinstance(en, dict) else None) or r.get("problem_en") or r.get("en_text")
            if t:
                out[sid] = t
    return out


def collect(lang: str) -> tuple[list[str], list[str]]:
    """Texts to embed: every selected training item (final, final_natural, final_shared, final_opmatched) keyed
    'item:<sha1(question)>' (deduplicated) + every seed of those items 'seed:<seed_id>'. Training-split text only."""
    from .select_v1 import OPMATCH
    keys, texts, seen, sids = [], [], set(), set()
    for base in (FINAL, NATURAL, SHARED, OPMATCH):
        for x in ("Y", "N"):
            for s in ("plus", "minus"):
                p = base / x / lang / f"train_{s}.jsonl"
                if not p.exists():
                    continue
                for r in read_jsonl(p):
                    q = r.get("question") or ""
                    k = "item:" + hashlib.sha1(q.encode("utf-8")).hexdigest()
                    sids.add(r.get("seed_id"))
                    if k not in seen:
                        seen.add(k)
                        keys.append(k)
                        texts.append(q)
    st = seed_texts(lang, sids)
    for sid in sorted(st):
        keys.append("seed:" + sid)
        texts.append(st[sid])
    return keys, texts


def vendi(X: np.ndarray) -> float:
    X = X.astype(np.float64)
    n = X.shape[0]
    if n == 0:
        return float("nan")
    C = X.T @ X / n
    lam = np.linalg.eigvalsh(C)
    lam = lam[lam > 1e-12]
    lam = lam / lam.sum()
    return float(np.exp(-(lam * np.log(lam)).sum()))


def mean_pairwise_cos_distance(X: np.ndarray) -> float:
    n = X.shape[0]
    if n < 2:
        return float("nan")
    s = X.astype(np.float64).sum(0)
    tot = float(s @ s)
    diag = float((X.astype(np.float64) ** 2).sum())
    return 1.0 - (tot - diag) / (n * (n - 1))


def run(lang: str) -> dict:
    from ..pilot_gen.util import chrf
    keys = json.load(open(EMB / f"select_emb_{lang}.keys.json", encoding="utf-8"))["keys"]
    E = np.load(EMB / f"select_emb_{lang}.npy")
    idx = {k: i for i, k in enumerate(keys)}
    out = {}
    from .select_v1 import OPMATCH
    for name, base in (("matched", FINAL), ("natural", NATURAL), ("shared", SHARED), ("opmatched", OPMATCH)):
        for x in ("Y", "N"):
            for s in ("plus", "minus"):
                p = base / x / lang / f"train_{s}.jsonl"
                if not p.exists():
                    continue
                rows = list(read_jsonl(p))
                st = seed_texts(lang, {r.get("seed_id") for r in rows})
                ii = [idx.get("item:" + hashlib.sha1((r.get("question") or "").encode("utf-8")).hexdigest())
                      for r in rows]
                ok = [i for i in ii if i is not None]
                X = E[ok].astype(np.float32)
                cos_seed, chrf_seed = [], []
                for r, i in zip(rows, ii):
                    j = idx.get("seed:" + str(r.get("seed_id")))
                    if i is not None and j is not None:
                        cos_seed.append(float(E[i].astype(np.float32) @ E[j].astype(np.float32)))
                    if r.get("seed_id") in st:
                        chrf_seed.append(chrf(r.get("question") or "", st[r["seed_id"]]))
                out[f"{name}|{x}{'+' if s == 'plus' else '-'}"] = {
                    "n": len(rows), "embedded": len(ok), "unique_seeds": len({r.get("seed_id") for r in rows}),
                    "vendi": round(vendi(X), 2), "mean_pairwise_cos_dist": round(mean_pairwise_cos_distance(X), 4),
                    "mean_cos_to_seed": round(float(np.mean(cos_seed)), 4) if cos_seed else None,
                    "mean_chrf_to_seed": round(float(np.mean(chrf_seed)), 4) if chrf_seed else None,
                    "edit_class": dict(Counter(EDIT_CLASS.get(r.get("op"), r.get("op")) for r in rows)),
                    "ops": dict(sorted(Counter(r.get("op") for r in rows).items()))}
    (EMB / f"select_diversity_{lang}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out
