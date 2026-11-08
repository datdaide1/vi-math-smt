"""Decontamination + contamination-scan tooling (workstream X5).

Design sources: ``tech/T5_evaluation.md`` §3.4 (C2 training-corpus overlap), ``04_data_eval.md`` §5 / H7 (which
corpora), ``tech/T1_synthesis.md`` §2.3 gates G4 (13-gram = 0, cosine < τ vs every test set) and G5 (MinHash
Jaccard < 0.8 within an arm). Report: ``docs/research_2026-11/exec/X5_decontam.md``.

Four detectors:

1. **Normalisation** (:func:`normalize`, :func:`tokenize`) — Vietnamese-aware: HTML-unescape, literal ``\\n``
   unescape, NFC, lowercase, LaTeX canonicalisation (delimiters dropped, ``\\dfrac``→``frac``, ``\\le``→``≤``,
   ``\\cdot``→``*`` …), Vietnamese number canonicalisation (``2,5``/``2{,}5``→``2.5``; ``1.000.000``/``10 000``→
   digits). Three token *variants*: ``exact``; ``nodiac`` (diacritics stripped, ``đ``→``d``: unaccented copies);
   ``template`` (every number → ``#``: value-perturbed copies, GSM-Symbolic style).
2. **n-gram overlap** (:class:`NgramIndex`) — token n-grams hashed to 64 bit (one rolling sweep for all lengths),
   stored as sorted ``uint64`` keys + ``uint32`` postings in shards (``np.save``d and memory-mapped when a workdir is
   given) behind a one-hash bit filter. ~12 bytes/gram, no Python sets. Scans index the *small* side (test sets,
   ~9k items) and stream the big side (corpora), so RAM is O(test set); ``bench-index`` indexes a 395k-doc corpus on
   disk instead. Docs shorter than n are indexed as one whole-sequence gram.
3. **MinHash LSH** (:class:`MinHasher`, :class:`MinHashLSH`) — own numpy implementation: word 5-gram shingles,
   128 permutations ``splitmix64(x XOR seed_i)``, 32 bands × 4 rows, incremental insert/query (G5 needs that).
4. **Cross-lingual embedding** (:class:`Encoder`, :class:`EmbeddingIndex`) — HF ``transformers`` encoder
   (``BAAI/bge-m3`` = the calibrated choice; ``intfloat/multilingual-e5-small`` also calibrated), torch imported
   lazily; parts 1–3 need only numpy and run in the project ``.venv``. The corpus-scale embedding run lives in
   ``decontam_modal.py`` (Modal GPU).

CLI (``python -m vi_math_verified_aug.eval.decontam``):

* ``scan --corpus SRC [--corpus SRC …] --target SRC [--target SRC …] --out DIR [--embed-model M]`` — per-target-item
  JSONL report (``items.jsonl``), ``summary.json`` (counts, runtime, peak RSS), ``examples.md``.
* ``calibrate`` — embedding threshold on known translation pairs (local GPU/CPU; the Modal job does the same).
* ``gate --target SRC … --candidates FILE`` — G4/G5 check of generated items (see :class:`Gate`).
* ``bench-index``, ``toanmath-titles``, ``resummarize`` — see the X5 report.

Source specs (see :func:`iter_source`): registered names (``vi_gsm8k_test``, ``vi_math_test``, ``vi_exam_raw``,
``luminarai``, ``luminarai_segments``, ``toanmath_labels``, ``hllj_crawl``, ``hllj_mcq``, ``vinhnq29:<split>``,
``metamathqa_vi``, ``gsm8k_en_test``, ``math_en_test``) or generic ``jsonl:<path>:<field>[+<field>…]``.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
CONTAM = DATA / "contam"
CACHE = CONTAM / "_cache"

# =====================================================================================================================
# 1. Normalisation
# =====================================================================================================================

# LaTeX → plain tokens. Order matters: longer commands first.
_LATEX_SUBS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\\(?:begin|end)\{[a-zA-Z*]*\}(?:\{[^{}]*\})?"), " "),
    (re.compile(r"\\(?:text|textbf|textit|mathrm|mathbf|mathit|operatorname|mbox|hbox)\s*\{([^{}]*)\}"), r" \1 "),
    (re.compile(r"\\[dt]?frac"), r" frac "),
    (re.compile(r"\\(?:leqslant|leq|le)(?![a-zA-Z])"), " ≤ "),
    (re.compile(r"\\(?:geqslant|geq|ge)(?![a-zA-Z])"), " ≥ "),
    (re.compile(r"\\(?:neq|ne)(?![a-zA-Z])"), " ≠ "),
    (re.compile(r"\\(?:cdot|times|ast)(?![a-zA-Z])"), " * "),
    (re.compile(r"\\div(?![a-zA-Z])"), " / "),
    (re.compile(r"\\(?:left|right|bigl|bigr|Bigl|Bigr|big|Big|displaystyle|limits|nolimits)(?![a-zA-Z])"), " "),
    (re.compile(r"\\[,;:! ]|\\quad|\\qquad|~"), " "),
    (re.compile(r"\\%"), "%"),
    (re.compile(r"\\\$"), " $ "),
    (re.compile(r"\\[()\[\]]"), " "),          # \( \) \[ \]
    (re.compile(r"\$+"), " "),                 # $ / $$ delimiters
    (re.compile(r"\\([a-zA-Z]+)"), r" \1 "),   # remaining commands → bare word (\sqrt → sqrt)
]
_UNICODE_MATH = str.maketrans({
    "−": "-", "–": "-", "—": "-", "×": "*", "·": "*", "÷": "/", "⩽": "≤", "⩾": "≥",
    "\u00a0": " ", "\u200b": " ", "“": '"', "”": '"', "‘": "'", "’": "'",
})
_DEC_COMMA = re.compile(r"(?<=\d)\s*\{,\}\s*(?=\d)|(?<=\d),(?=\d)")
_THOUSANDS = re.compile(r"(?<![\d.])[1-9]\d{0,2}(?:\.\d{3})+(?![\d.])")
_THOUSANDS_SP = re.compile(r"(?<![\d.])[1-9]\d{0,2}(?:[ \u202f]\d{3})+(?![\d.])")
_TOKEN = re.compile(r"\d+(?:\.\d+)?|[^\W\d_]+|[=+\-*/^<>%≤≥≠]", re.UNICODE)
_NUM = re.compile(r"\d+(?:\.\d+)?$")
_TAG = re.compile(r"<[^>]+>")

VARIANTS = ("exact", "nodiac", "template")


def strip_diacritics(s: str) -> str:
    """Remove Vietnamese diacritics (``đ``→``d``)."""
    s = s.replace("đ", "d").replace("Đ", "D")
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", s)


def canon_numbers(s: str) -> str:
    """Vietnamese number conventions → canonical: ``2,5``→``2.5``, ``1.000.000``→``1000000``, ``10 000``→``10000``.

    Thousands-dot grouping is resolved before the decimal comma, so ``1.234,5`` → ``1234.5``. Ambiguity note: an
    English ``2.500`` (= 2.5) becomes ``2500``; acceptable for overlap detection, never used for grading.
    """
    s = _THOUSANDS_SP.sub(lambda m: re.sub(r"[ \u202f]", "", m.group(0)), s)
    s = _THOUSANDS.sub(lambda m: m.group(0).replace(".", ""), s)
    s = _DEC_COMMA.sub(".", s)
    return s


def canon_latex(s: str) -> str:
    for pat, rep in _LATEX_SUBS:
        s = pat.sub(rep, s)
    return s


def normalize(text: str, *, diacritics: bool = True, template: bool = False) -> str:
    """Normalised text, tokens joined by single spaces (see :func:`tokenize`)."""
    return " ".join(tokenize(text, diacritics=diacritics, template=template))


# A literal two-character "\n" (escaped JSON left in scraped text, e.g. 217k of 306k LuminarAI rows) is turned into
# a newline unless it starts a LaTeX command (\neq, \ne, \nu, \notin, \nabla, \not, \neg, \ni, \newline, …).
_LITERAL_NL = re.compile(r"\\n(?!(?:eq|e|u|otin|abla|ot|eg|i|ewline|onumber|leq|geq|mid|cong|subseteq|supseteq"
                         r"|parallel|exists|less|gtr|rightarrow|leftarrow)(?![a-zA-Z]))")


def unescape_newlines(s: str) -> str:
    return _LITERAL_NL.sub("\n", s) if "\\n" in s else s


def _base_tokens(text: str) -> list[str]:
    s = unescape_newlines(html.unescape(text or ""))
    s = _TAG.sub(" ", s)
    s = unicodedata.normalize("NFC", s).translate(_UNICODE_MATH)
    s = canon_latex(s)
    s = canon_numbers(s)
    s = s.lower()
    return _TOKEN.findall(s)


def tokenize(text: str, *, diacritics: bool = True, template: bool = False) -> list[str]:
    toks = _base_tokens(text)
    if not diacritics:
        toks = [strip_diacritics(t) for t in toks]
    if template:
        toks = ["#" if _NUM.match(t) else t for t in toks]
    return toks


def variant_tokens(text: str) -> dict[str, list[str]]:
    """All three variants from one pass of the expensive normalisation."""
    base = _base_tokens(text)
    return {
        "exact": base,
        "nodiac": [_strip_cached(t) for t in base],
        "template": ["#" if _NUM.match(t) else t for t in base],
    }


_STRIP_CACHE: dict[str, str] = {}
_VH_CACHE: dict[str, tuple[int, int, int]] = {}
_NUM_HASH: int | None = None


def variant_hashes(text: str) -> dict[str, np.ndarray]:
    """Token-hash arrays for the three variants in one pass (one dict lookup per token; used on the streamed side).

    Equivalent to ``{v: token_hashes(variant_tokens(text)[v])}`` (tested)."""
    global _NUM_HASH
    if _NUM_HASH is None:
        _NUM_HASH = _tok_hash("#")
    base = _base_tokens(text)
    if not base:
        e = np.empty(0, dtype=np.uint64)
        return {"exact": e, "nodiac": e, "template": e}
    cache = _VH_CACHE
    rows = []
    for t in base:
        r = cache.get(t)
        if r is None:
            he = _tok_hash(t)
            hn = he if t.isascii() else _tok_hash(_strip_cached(t))
            ht = _NUM_HASH if _NUM.match(t) else he
            r = (he, hn, ht)
            if len(cache) > 300_000:
                cache.clear()
            cache[t] = r
        rows.append(r)
    a = np.array(rows, dtype=np.uint64)
    return {"exact": a[:, 0], "nodiac": a[:, 1], "template": a[:, 2]}


def _strip_cached(t: str) -> str:
    r = _STRIP_CACHE.get(t)
    if r is None:
        if t.isascii():
            r = t
        else:
            r = strip_diacritics(t)
        if len(_STRIP_CACHE) > 200_000:
            _STRIP_CACHE.clear()
        _STRIP_CACHE[t] = r
    return r


# =====================================================================================================================
# 2. Hashing + n-gram index
# =====================================================================================================================

_TOK_CACHE: dict[str, int] = {}
_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)
_ROLL = np.uint64(0x9E3779B97F4A7C15)


def _tok_hash(t: str) -> int:
    h = _TOK_CACHE.get(t)
    if h is None:
        h = int.from_bytes(hashlib.blake2b(t.encode("utf-8"), digest_size=8).digest(), "little")
        if len(_TOK_CACHE) > 300_000:
            _TOK_CACHE.clear()
        _TOK_CACHE[t] = h
    return h


def token_hashes(toks: Sequence[str]) -> np.ndarray:
    return np.fromiter((_tok_hash(t) for t in toks), dtype=np.uint64, count=len(toks))


def _mix64(h: np.ndarray) -> np.ndarray:
    """splitmix64 finaliser (vectorised, wraps mod 2^64)."""
    with np.errstate(over="ignore"):
        h = h ^ (h >> np.uint64(30))
        h = h * np.uint64(0xBF58476D1CE4E5B9)
        h = h ^ (h >> np.uint64(27))
        h = h * np.uint64(0x94D049BB133111EB)
        h = h ^ (h >> np.uint64(31))
    return h


def ngram_hashes(th: np.ndarray, n: int) -> np.ndarray:
    """64-bit hashes of all contiguous n-grams of the token-hash array ``th`` (empty if len < n).

    The length n is mixed in, so a 5-gram never collides with a 13-gram by construction.
    """
    if n <= 0 or len(th) < n:
        return np.empty(0, dtype=np.uint64)
    return multi_ngram_hashes(th, (n,))[n]


_LEN_SALT = np.uint64(0xD6E8FEB86659FD93)


def multi_ngram_hashes(th: np.ndarray, lengths: Iterable[int]) -> dict[int, np.ndarray]:
    """n-gram hashes for several n in ONE rolling sweep (raw polynomial hash, then ``mix64(raw XOR n·salt)``).

    Cost is that of the longest n, so querying 13-grams plus all short-item lengths 6..12 is no dearer than 13
    alone. Lengths longer than the sequence are omitted from the result."""
    L = len(th)
    want = sorted({n for n in lengths if 0 < n <= L})
    out: dict[int, np.ndarray] = {}
    if not want:
        return out
    with np.errstate(over="ignore"):
        raw = th.copy()
        if 1 in want:
            out[1] = _mix64(raw ^ _LEN_SALT)
        for k in range(1, want[-1]):
            m = L - k
            raw = raw[:m] * _ROLL + th[k:k + m]
            if k + 1 in want:
                out[k + 1] = _mix64(raw ^ (np.uint64(k + 1) * _LEN_SALT))
    return out


@dataclass
class _Shard:
    keys: np.ndarray       # uint64, sorted
    docs: np.ndarray       # uint32, aligned with keys


class NgramIndex:
    """Sorted-hash n-gram index over a set of documents (doc ids are dense ints 0..N-1).

    * ``add(doc_id, token_hashes)`` buffers the doc's *unique* n-gram hashes.
    * When the buffer passes ``shard_grams``, it is sorted into a shard (and written to ``workdir`` as ``.npy`` and
      re-opened memory-mapped, if a workdir is given) — so building over ~1M docs stays bounded in RAM.
    * ``query(token_hashes)`` returns ``{doc_id: #shared unique n-grams}``; ``n_grams[doc_id]`` gives the
      denominator, so coverage = shared / n_grams[doc] (fraction of the indexed doc's n-grams found in the query).

    Docs shorter than ``n`` but at least ``min_len`` tokens are indexed as one whole-sequence gram of their own
    length L; queries then also hash their L-grams for every such L, so a short item contained verbatim is caught.
    """

    def __init__(self, n: int, *, min_len: int = 6, shard_grams: int = 20_000_000, workdir: str | Path | None = None):
        self.n = n
        self.min_len = min_len
        self.shard_grams = shard_grams
        self.workdir = Path(workdir) if workdir else None
        self.short_lens: set[int] = set()
        self.n_grams: list[int] = []
        self._buf_k: list[np.ndarray] = []
        self._buf_d: list[np.ndarray] = []
        self._buf_size = 0
        self.shards: list[_Shard] = []

    # -- build --------------------------------------------------------------------------------------------------
    def _grams_for_index(self, th: np.ndarray) -> np.ndarray:
        if len(th) >= self.n:
            return np.unique(ngram_hashes(th, self.n))
        if len(th) >= self.min_len:
            self.short_lens.add(len(th))
            return ngram_hashes(th, len(th))
        return np.empty(0, dtype=np.uint64)

    def add(self, doc_id: int, th: np.ndarray) -> int:
        g = self._grams_for_index(th)
        while len(self.n_grams) <= doc_id:
            self.n_grams.append(0)
        self.n_grams[doc_id] = len(g)
        if len(g):
            self._buf_k.append(g)
            self._buf_d.append(np.full(len(g), doc_id, dtype=np.uint32))
            self._buf_size += len(g)
            if self._buf_size >= self.shard_grams:
                self._flush()
        return len(g)

    def _flush(self) -> None:
        if not self._buf_k:
            return
        k = np.concatenate(self._buf_k)
        d = np.concatenate(self._buf_d)
        o = np.argsort(k, kind="stable")
        k, d = k[o], d[o]
        if self.workdir:
            self.workdir.mkdir(parents=True, exist_ok=True)
            i = len(self.shards)
            np.save(self.workdir / f"shard{i:04d}_k.npy", k)
            np.save(self.workdir / f"shard{i:04d}_d.npy", d)
            k = np.load(self.workdir / f"shard{i:04d}_k.npy", mmap_mode="r")
            d = np.load(self.workdir / f"shard{i:04d}_d.npy", mmap_mode="r")
        self.shards.append(_Shard(k, d))
        self._buf_k, self._buf_d, self._buf_size = [], [], 0

    def finalize(self) -> "NgramIndex":
        self._flush()
        self._bloom = None
        return self

    @property
    def total_grams(self) -> int:
        return int(sum(len(s.keys) for s in self.shards))

    # -- query --------------------------------------------------------------------------------------------------
    def _bloom_build(self) -> None:
        """One-hash bit filter over all keys (top bits of the hash; ≥ 16 bits per key, 2^20..2^31 bits).

        Queries test the filter first, so only true hits + ~1/64 false positives reach ``searchsorted`` — the
        profiler showed ``searchsorted`` + ``unique`` over every corpus gram was 75 % of the scan time.
        """
        n = max(self.total_grams, 1)
        bits = int(min(max(1 << 20, 1 << int(np.ceil(np.log2(64 * n)))), 1 << 31))
        self._bloom_shift = np.uint64(64 - int(np.log2(bits)))
        self._bloom = np.zeros(bits // 8, dtype=np.uint8)
        for sh in self.shards:
            for a in range(0, len(sh.keys), 5_000_000):
                k = np.asarray(sh.keys[a:a + 5_000_000]) >> self._bloom_shift
                np.bitwise_or.at(self._bloom, (k >> np.uint64(3)).astype(np.int64),
                                 (np.uint8(1) << (k & np.uint64(7)).astype(np.uint8)))

    def _bloom_test(self, q: np.ndarray) -> np.ndarray:
        if getattr(self, "_bloom", None) is None:
            self._bloom_build()
        k = q >> self._bloom_shift
        return (self._bloom[(k >> np.uint64(3)).astype(np.int64)] >> (k & np.uint64(7)).astype(np.uint8)) & 1 == 1

    @property
    def lengths(self) -> set[int]:
        """All gram lengths a query must hash: n plus the whole-sequence lengths of short indexed docs."""
        return {self.n} | self.short_lens

    def query_grams(self, th: np.ndarray, prefilter: bool = True, pre: dict[int, np.ndarray] | None = None
                    ) -> np.ndarray:
        g = pre if pre is not None else multi_ngram_hashes(th, self.lengths)
        parts = [g[L] for L in self.lengths if L in g]
        if not parts:
            return np.empty(0, dtype=np.uint64)
        q = np.concatenate(parts) if len(parts) > 1 else parts[0]
        if prefilter and len(q):
            q = q[self._bloom_test(q)]
        return np.unique(q)

    def query(self, th: np.ndarray | None = None, pre: dict[int, np.ndarray] | None = None) -> dict[int, int]:
        """``pre``: precomputed ``{length: hashes}`` (from :func:`multi_ngram_hashes`) covering ``self.lengths``."""
        q = self.query_grams(th, pre=pre)
        if not len(q):
            return {}
        out: Counter = Counter()
        for s in self.shards:
            nk = len(s.keys)
            lo = np.searchsorted(s.keys, q, "left")
            ok = lo < nk
            ok[ok] = np.asarray(s.keys[lo[ok]]) == q[ok]
            if not ok.any():
                continue
            lo = lo[ok]
            hi = np.searchsorted(s.keys, q[ok], "right")
            cnt = hi - lo
            if cnt.max() == 1:
                docs = np.asarray(s.docs[lo])
            else:
                idx = np.repeat(lo - np.concatenate(([0], np.cumsum(cnt)[:-1])), cnt) + np.arange(cnt.sum())
                docs = np.asarray(s.docs[idx])
            u, c = np.unique(docs, return_counts=True)
            for a, b in zip(u.tolist(), c.tolist()):
                out[a] += b
        return dict(out)

    def contains_any(self, th: np.ndarray) -> bool:
        q = self.query_grams(th)
        if not len(q):
            return False
        for s in self.shards:
            i = np.searchsorted(s.keys, q)
            i = np.minimum(i, len(s.keys) - 1)
            if len(s.keys) and (np.asarray(s.keys[i]) == q).any():
                return True
        return False

    # -- persistence --------------------------------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.finalize()
        for i, s in enumerate(self.shards):
            if not (self.workdir and self.workdir.resolve() == path.resolve()):
                np.save(path / f"shard{i:04d}_k.npy", np.asarray(s.keys))
                np.save(path / f"shard{i:04d}_d.npy", np.asarray(s.docs))
        meta = {"n": self.n, "min_len": self.min_len, "short_lens": sorted(self.short_lens),
                "n_shards": len(self.shards), "n_grams": self.n_grams}
        (path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, mmap: bool = True) -> "NgramIndex":
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        ix = cls(meta["n"], min_len=meta["min_len"])
        ix.short_lens = set(meta["short_lens"])
        ix.n_grams = meta["n_grams"]
        mm = "r" if mmap else None
        for i in range(meta["n_shards"]):
            ix.shards.append(_Shard(np.load(path / f"shard{i:04d}_k.npy", mmap_mode=mm),
                                    np.load(path / f"shard{i:04d}_d.npy", mmap_mode=mm)))
        return ix


# =====================================================================================================================
# 3. MinHash LSH
# =====================================================================================================================

_M32 = np.uint64(0xFFFFFFFF)


class MinHasher:
    """MinHash over word ``shingle``-grams (64-bit shingle hashes).

    Permutation family: ``h_i(x) = splitmix64(x XOR s_i)`` with 64-bit random seeds ``s_i``. (A first version used
    ``(a·x+b) mod (2^61−1)`` with 31-bit a,b to avoid uint64 overflow; with a·x < 4·P that map is nearly monotone in
    x, so all permutations picked the same shingles and the estimate collapsed — caught by
    ``test_minhash_estimate_close_to_exact``.) Signatures keep the low 32 bits of each minimum.
    """

    def __init__(self, num_perm: int = 128, shingle: int = 5, seed: int = 1):
        rng = np.random.default_rng(seed)
        self.num_perm = num_perm
        self.shingle = shingle
        self.seeds = rng.integers(0, np.iinfo(np.uint64).max, size=num_perm, dtype=np.uint64, endpoint=True)

    def shingles(self, th: np.ndarray) -> np.ndarray:
        if len(th) == 0:
            return np.empty(0, dtype=np.uint64)
        n = min(self.shingle, len(th))
        return np.unique(ngram_hashes(th, n))

    def signature(self, th: np.ndarray) -> np.ndarray | None:
        x = self.shingles(th)
        if not len(x):
            return None
        ph = _mix64(x[:, None] ^ self.seeds[None, :])
        return (ph.min(axis=0) & _M32).astype(np.uint32)

    @staticmethod
    def jaccard(s1: np.ndarray, s2: np.ndarray) -> float:
        return float(np.mean(s1 == s2))


def exact_jaccard(t1: Sequence[str], t2: Sequence[str], shingle: int = 5) -> float:
    def sh(t):
        n = min(shingle, len(t))
        return {tuple(t[i:i + n]) for i in range(len(t) - n + 1)} if t else set()
    a, b = sh(t1), sh(t2)
    return len(a & b) / len(a | b) if (a or b) else 0.0


class MinHashLSH:
    """Banded LSH over MinHash signatures; incremental ``insert``/``query``. Candidate pairs are verified with the
    signature-estimated Jaccard (SE ≈ √(J(1−J)/num_perm) ≈ 0.04 at 128 perms)."""

    def __init__(self, num_perm: int = 128, bands: int = 32):
        assert num_perm % bands == 0
        self.bands, self.rows = bands, num_perm // bands
        self.tables: list[dict[bytes, list[int]]] = [dict() for _ in range(bands)]
        self.sigs: list[np.ndarray] = []
        self.ids: list[int] = []

    @property
    def approx_threshold(self) -> float:
        return (1.0 / self.bands) ** (1.0 / self.rows)

    def _keys(self, sig: np.ndarray) -> list[bytes]:
        r = self.rows
        return [sig[i * r:(i + 1) * r].tobytes() for i in range(self.bands)]

    def insert(self, doc_id: int, sig: np.ndarray) -> None:
        slot = len(self.sigs)
        self.sigs.append(sig)
        self.ids.append(doc_id)
        for t, k in zip(self.tables, self._keys(sig)):
            t.setdefault(k, []).append(slot)

    def candidates(self, sig: np.ndarray) -> set[int]:
        out: set[int] = set()
        for t, k in zip(self.tables, self._keys(sig)):
            v = t.get(k)
            if v:
                out.update(v)
        return out

    def query(self, sig: np.ndarray, threshold: float = 0.0) -> list[tuple[int, float]]:
        """``[(doc_id, est_jaccard)]`` for LSH candidates with est. Jaccard ≥ threshold, best first."""
        slots = self.candidates(sig)
        if not slots:
            return []
        sl = np.fromiter(slots, dtype=np.int64)
        S = np.stack([self.sigs[i] for i in sl])
        j = (S == sig[None, :]).mean(axis=1)
        keep = j >= threshold
        res = sorted(zip((self.ids[i] for i in sl[keep]), j[keep].tolist()), key=lambda x: -x[1])
        return res


# =====================================================================================================================
# 4. Embeddings (lazy torch)
# =====================================================================================================================

MODEL_SPECS = {
    # name: (pooling, query prefix, passage prefix)
    "intfloat/multilingual-e5-small": ("mean", "query: ", "query: "),
    "intfloat/multilingual-e5-base": ("mean", "query: ", "query: "),
    "BAAI/bge-m3": ("cls", "", ""),
}


class Encoder:
    """Sentence encoder via ``transformers``; L2-normalised float16 numpy output. Symmetric tasks use the same
    prefix on both sides (e5: ``query: ``)."""

    def __init__(self, model: str = "intfloat/multilingual-e5-small", device: str | None = None,
                 max_length: int = 256, batch_size: int = 32, fp16: bool = True, dtype: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.name = model
        self.pool, self.prefix, _ = MODEL_SPECS.get(model, ("mean", "", ""))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model)
        dtype = dtype or ("fp16" if fp16 else "fp32")
        if self.device == "cuda" and dtype == "bf16":
            self.model = self.model.to(torch.bfloat16)
        elif self.device == "cuda" and dtype == "fp16":
            self.model = self.model.half()
        self.model.to(self.device).eval()
        self.max_length, self.batch_size = max_length, batch_size

    def encode(self, texts: Sequence[str], progress: bool = False) -> np.ndarray:
        torch = self.torch
        texts = [self.prefix + (t or "")[:4000] for t in texts]
        order = np.argsort([-len(t) for t in texts], kind="stable")
        out = np.zeros((len(texts), self.model.config.hidden_size), dtype=np.float16)
        bs = self.batch_size
        with torch.inference_mode():
            for s in range(0, len(texts), bs):
                idx = order[s:s + bs]
                enc = self.tok([texts[i] for i in idx], padding=True, truncation=True,
                               max_length=self.max_length, return_tensors="pt").to(self.device)
                h = self.model(**enc).last_hidden_state
                if self.pool == "cls":
                    v = h[:, 0]
                else:
                    m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
                    v = (h * m).sum(1) / m.sum(1).clamp(min=1)
                v = torch.nn.functional.normalize(v.float(), dim=-1)
                out[idx] = v.cpu().numpy().astype(np.float16)
                if progress and (s // bs) % 200 == 0:
                    print(f"  encoded {s + len(idx)}/{len(texts)}", file=sys.stderr, flush=True)
        return out


class EmbeddingIndex:
    """Dense targets (small) × streamed corpus chunks; keeps top-k corpus hits per target. Torch matmul on GPU when
    available, numpy otherwise."""

    def __init__(self, target_emb: np.ndarray, k: int = 3, device: str | None = None):
        self.k = k
        self.n = len(target_emb)
        self.best_val = np.full((self.n, k), -1.0, dtype=np.float32)
        self.best_id = np.full((self.n, k), -1, dtype=np.int64)
        try:
            import torch
            self.torch = torch
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            dt = torch.float16 if self.device == "cuda" else torch.float32
            self.T = torch.from_numpy(target_emb.astype(np.float32)).to(self.device, dt)
        except ImportError:
            self.torch = None
            self.T = target_emb.astype(np.float32)

    def update(self, emb: np.ndarray, ids: np.ndarray) -> np.ndarray:
        """Score a corpus chunk; returns the max cosine of each corpus row vs any target."""
        if self.torch is not None:
            t = self.torch
            C = t.from_numpy(emb.astype(np.float32)).to(self.device, self.T.dtype)
            S = (self.T @ C.T).float()                         # n_t × n_c
            kk = min(self.k, S.shape[1])
            v, i = t.topk(S, kk, dim=1)
            v, i = v.cpu().numpy(), i.cpu().numpy()
            row_max = S.max(dim=0).values.cpu().numpy()
        else:
            S = self.T @ emb.astype(np.float32).T
            kk = min(self.k, S.shape[1])
            i = np.argsort(-S, axis=1)[:, :kk]
            v = np.take_along_axis(S, i, axis=1)
            row_max = S.max(axis=0)
        cand_v = np.concatenate([self.best_val, v], axis=1)
        cand_i = np.concatenate([self.best_id, np.asarray(ids)[i]], axis=1)
        o = np.argsort(-cand_v, axis=1)[:, :self.k]
        self.best_val = np.take_along_axis(cand_v, o, axis=1)
        self.best_id = np.take_along_axis(cand_i, o, axis=1)
        return row_max


# =====================================================================================================================
# 5. Sources
# =====================================================================================================================

@dataclass
class Doc:
    id: str
    text: str                       # full text used for n-gram / MinHash
    embed_text: str | None = None   # shorter text for the encoder (question only); defaults to text
    meta: dict = field(default_factory=dict)


def _jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _parquet_rows(path: Path, columns: list[str] | None = None, batch: int = 2000) -> Iterator[dict]:
    """Stream rows in small record batches (never materialise the whole table as Python objects)."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    for rb in pf.iter_batches(batch_size=batch, columns=columns):
        yield from rb.to_pylist()


def _hf_parquet_cached(repo: str, path_in_repo: str, local_name: str) -> Path:
    """Small (< ~60 MB) HF parquet conversions are cached under data/contam/_cache/; big ones are streamed."""
    p = CACHE / local_name
    if not p.exists():
        import urllib.request
        CACHE.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/{repo}/resolve/refs%2Fconvert%2Fparquet/{path_in_repo}"
        urllib.request.urlretrieve(url, p)
    return p


# ---- Vi-ExamMath raw pages (X2's cache, read-only) ------------------------------------------------------------------

_BLOCK_END = re.compile(r"</(?:p|div|li|h\d|tr|table|br)\s*>|<br\s*/?>", re.I)
_SCRIPT = re.compile(r"<(script|style|noscript)[^>]*>.*?</\1>", re.I | re.S)
_Q_HEAD = re.compile(r"^\s*(?:câu|bài)\s*(\d+|[ivx]+)\b", re.I)
_SOL_HEAD = re.compile(r"^\s*(?:lời giải|đáp án|hướng dẫn giải|hướng dẫn chấm|phương pháp giải)\b", re.I)


def html_to_lines(h: str) -> list[str]:
    h = _SCRIPT.sub(" ", h)
    h = _BLOCK_END.sub("\n", h)
    h = _TAG.sub(" ", h)
    h = html.unescape(h)
    return [re.sub(r"[ \t\r\f\v]+", " ", ln).strip() for ln in h.split("\n") if ln.strip()]


def extract_exam_statement(page_html: str) -> list[str]:
    """Problem-statement lines of a cached exam page (loigiaihay / archive.org).

    loigiaihay pages put the statement in ``id="sub-question-1"`` ("Đề bài") and the solution in later
    ``sub-question-*`` blocks; we take the statement block and stop at the first solution heading. Fallback: the
    ``box-content`` / ``main-content`` / body text up to the first solution heading.
    """
    h = page_html
    i = h.find('id="sub-question-1"')
    statement_block = i >= 0
    if statement_block:
        j = h.find('id="sub-question-2"', i)
        block = h[h.rfind("<", 0, i):j if j > 0 else len(h)]
    else:
        block = h
        for marker in ('id="box-content"', 'id="main-content"', "<body"):
            k = h.find(marker)
            if k >= 0:
                block = h[h.rfind("<", 0, k):]
                break
    out, started = [], statement_block
    for ln in html_to_lines(block):
        if _SOL_HEAD.match(ln) and (started and (out or not statement_block)):
            break
        if _Q_HEAD.match(ln):
            started = True
        if started:
            out.append(ln)
    return out


def segment_questions(lines: list[str]) -> list[str]:
    """Split statement lines into question units at "Câu N"/"Bài N" heads (whole statement if none)."""
    units, cur = [], []
    for ln in lines:
        if _Q_HEAD.match(ln) and cur:
            units.append(" ".join(cur))
            cur = []
        cur.append(ln)
    if cur:
        units.append(" ".join(cur))
    units = [u for u in units if len(u) >= 30]
    return units


def iter_vi_exam_raw(root: Path | None = None, unit: str = "question") -> Iterator[Doc]:
    root = root or DATA / "vi_exam" / "raw"
    for site in sorted(p for p in root.iterdir() if p.is_dir()):
        for hp in sorted(site.glob("*.html")):
            meta = {}
            mp = hp.with_suffix(".json")
            if mp.exists():
                try:
                    meta = json.loads(mp.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}
            url = meta.get("final_url") or meta.get("url") or ""
            lines = extract_exam_statement(hp.read_text(encoding="utf-8", errors="replace"))
            if not lines:
                continue
            units = segment_questions(lines) if unit == "question" else [" ".join(lines)]
            for k, u in enumerate(units):
                body = strip_question_header(u)
                if len(body) < 15:
                    continue
                yield Doc(id=f"{site.name}/{hp.stem}#q{k + 1}", text=body, meta={"url": url, "page": hp.stem})


_Q_HEADER = re.compile(r"^\s*(?:câu|bài)\s*(?:\d+|[ivx]+)\s*[.:)]?\s*(?:\(\s*[\d.,]+\s*điểm\s*\)\s*)?[.:]?\s*",
                       re.I)


def strip_question_header(u: str) -> str:
    """"Câu 3 (1,5 điểm): Cho …" → "Cho …" (the exam-paper header is not part of the problem text)."""
    return _Q_HEADER.sub("", u, count=1).strip()


def _src_vi_exam_items(path: Path | None = None) -> Iterator[Doc]:
    """X2's final Vi-ExamMath items (read-only). Text = ``problem`` (context + own text + MCQ options)."""
    path = path or DATA / "vi_exam" / "items.jsonl"
    for r in _jsonl(path):
        yield Doc(r["id"], r.get("problem") or "", meta={"core": bool(r.get("core_candidate")),
                                                        "temporal": r.get("temporal") or {},
                                                        "exam_year": (r.get("provenance") or {}).get("exam_year")})


# ---- registered sources ---------------------------------------------------------------------------------------------

def _src_vi_gsm8k_test() -> Iterator[Doc]:
    for r in _jsonl(DATA / "clean" / "vi_gsm8k_test" / "vi_gsm8k_test_full_v1.1.jsonl"):
        yield Doc(r["id"], r["question"], meta={"en": r.get("en_question")})


def _src_vi_math_test() -> Iterator[Doc]:
    for r in _jsonl(DATA / "clean" / "vi_math_test" / "vi_math_test_full_v1.1.jsonl"):
        yield Doc(r["id"], r["problem"], meta={"en": r.get("en_problem"), "subject": r.get("subject")})


def _src_gsm8k_en_test() -> Iterator[Doc]:
    for r in _jsonl(DATA / "clean" / "vi_gsm8k_test" / "vi_gsm8k_test_full_v1.1.jsonl"):
        yield Doc(r["id"], r["en_question"])


def _src_math_en_test() -> Iterator[Doc]:
    for r in _jsonl(DATA / "clean" / "vi_math_test" / "vi_math_test_full_v1.1.jsonl"):
        yield Doc(r["id"], r["en_problem"])


_PROB_HEAD = re.compile(r"^\s*(?:đề bài|bài\s*\d+|câu\s*\d+|ví dụ\s*\d*|bài tập\s*\d+|luyện tập\s*\d*|"
                        r"vận dụng\s*\d*|hoạt động\s*\d+)\b", re.I)
_PROB_CUE = re.compile(r"\b(?:tính|tìm|giải|chứng minh|cho|rút gọn|hỏi|bao nhiêu|xác định|so sánh|viết)\b", re.I)
_BOILER = re.compile(r"xin giới thiệu|chúng tôi|montoan|loigiaihay|bài viết|sách giáo khoa|sgk", re.I)
_SOL_STOP = re.compile(r"^\s*(?:phương pháp giải|lời giải|đáp án|hướng dẫn giải|xem lời giải|giải chi tiết)\b", re.I)


def problem_segments(page: str, min_len: int = 40, max_len: int = 1200) -> list[str]:
    """Problem-statement-like segments of a scraped textbook/solution page (for the embedding stage, where a whole
    9 kB page truncated to 256 tokens would only embed its marketing intro).

    A segment starts at a problem head ("Đề bài", "Bài 3", "Câu 2", "Ví dụ 1", …) and runs until a solution head
    or the next problem head; a paragraph without a head is kept if it has a task cue (tính/tìm/giải/cho/…) and
    contains a digit or LaTeX. Segments are de-duplicated within the page.
    """
    lines = [ln.strip() for ln in unescape_newlines(page).split("\n")]
    segs, cur, in_sol = [], [], False

    def close():
        if cur:
            t = " ".join(cur).strip()
            if t.lower().startswith("đề bài"):
                t = t[6:].strip(" :")
            if min_len <= len(t):
                segs.append(t[:max_len])
        cur.clear()

    for ln in lines:
        if not ln:
            continue
        if _PROB_HEAD.match(ln):
            close()
            in_sol = False
            cur.append(ln)
        elif _SOL_STOP.match(ln):
            close()
            in_sol = True
        elif cur and not in_sol:
            cur.append(ln)
        elif not in_sol and _PROB_CUE.search(ln) and not _BOILER.search(ln) and (re.search(r"\d|\\\(|\$", ln)) and len(ln) >= min_len:
            segs.append(ln[:max_len])
    close()
    seen, out = set(), []
    for t in segs:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _src_luminarai() -> Iterator[Doc]:
    p = _hf_parquet_cached("LuminarAI/vietnamese-ms-hs-textbook-math-300K", "default/train/0000.parquet",
                           "luminarai.parquet")
    for i, r in enumerate(_parquet_rows(p)):
        q, a = r.get("question") or "", r.get("answer") or ""
        yield Doc(f"luminarai/{i}", q + "\n" + a, embed_text=q)


def _src_luminarai_segments() -> Iterator[Doc]:
    """One doc per problem-like segment (see :func:`problem_segments`); used by the embedding stage."""
    p = _hf_parquet_cached("LuminarAI/vietnamese-ms-hs-textbook-math-300K", "default/train/0000.parquet",
                           "luminarai.parquet")
    for i, r in enumerate(_parquet_rows(p, columns=["question"])):
        for k, seg in enumerate(problem_segments(r.get("question") or "")):
            yield Doc(f"luminarai/{i}#s{k}", seg)


def _src_hllj_crawl() -> Iterator[Doc]:
    p = _hf_parquet_cached("hllj/vi_math_problem_crawl", "default/train/0000.parquet", "hllj_crawl.parquet")
    for i, r in enumerate(_parquet_rows(p)):
        c = r.get("contents") or ""
        c = "\n".join(map(str, c)) if isinstance(c, (list, tuple)) else str(c)
        t = (r.get("title") or "") + "\n" + c
        yield Doc(f"hllj_crawl/{r.get('id', i)}", t, embed_text=c[:1500] or t, meta={"url": r.get("url")})


def _src_hllj_mcq() -> Iterator[Doc]:
    p = _hf_parquet_cached("hllj/vi_grade_school_math_mcq", "default/train/0000.parquet", "hllj_mcq.parquet")
    for i, r in enumerate(_parquet_rows(p)):
        probs = r.get("problems")
        if isinstance(probs, str):
            try:
                probs = json.loads(probs)
            except Exception:
                probs = [probs]
        probs = probs or []
        for k, pr in enumerate(probs):
            if isinstance(pr, dict):
                t = " ".join(" ".join(map(str, v)) if isinstance(v, (list, tuple)) else str(v or "")
                             for v in (pr.get(x) for x in ("question", "choices", "explanation", "solution")))
                q = str(pr.get("question") or t)
            else:
                t = q = str(pr)
            yield Doc(f"hllj_mcq/{r.get('id', i)}/{k}", t, embed_text=q, meta={"url": r.get("url")})


def _src_vinhnq29(split: str) -> Iterator[Doc]:
    p = _hf_parquet_cached("vinhnq29/vietnamese-elementary-math", f"default/{split}/0000.parquet",
                           f"vinhnq29_{split}.parquet")
    for i, r in enumerate(_parquet_rows(p)):
        q = r.get("question") or ""
        ch = r.get("choices")
        t = q + "\n" + (json.dumps(ch, ensure_ascii=False) if ch is not None else "") + "\n" + (r.get("answer") or "")
        yield Doc(f"vinhnq29/{split}/{i}", t, embed_text=q)


def _src_metamathqa_vi(path: Path | None = None) -> Iterator[Doc]:
    """Local 824 MB JSONL, streamed line by line. Doc = query_vi + response_vi; embed on query_vi."""
    path = path or DATA / "test" / "metaMATH" / "metaMathQA-395K_json_vi.jsonl"
    for i, r in enumerate(_jsonl(path)):
        q = r.get("query_vi") or ""
        yield Doc(f"metamathqa_vi/{i}", q + "\n" + (r.get("response_vi") or ""), embed_text=q,
                  meta={"type": r.get("type")})


def _src_toanmath_labels(max_files: int | None = None) -> Iterator[Doc]:
    """daominhwysi/toanmath.com-alpha: 315 × ~1 GB parquet (images + label). We read ONLY the ``label`` and ``path``
    column chunks with HTTP range requests (pyarrow + fsspec over the HF hub), never the image bytes. The label text
    is cached locally (data/contam/_cache/toanmath_labels/<file>.jsonl) so re-scans do not touch the network."""
    out_dir = CACHE / "toanmath_labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    files = toanmath_label_files(max_files)
    for fn in files:
        cp = out_dir / (Path(fn).stem + ".jsonl")
        if not cp.exists():
            fetch_toanmath_labels(fn, cp)
        for r in _jsonl(cp):
            yield Doc(f"toanmath/{r['path']}", r["label"] or "")


def toanmath_label_files(max_files: int | None = None) -> list[str]:
    from huggingface_hub import HfApi
    files = [f for f in HfApi().list_repo_files("daominhwysi/toanmath.com-alpha", repo_type="dataset")
             if f.endswith(".parquet")]
    files.sort(key=lambda f: int(re.findall(r"\d+", f)[-1]) if re.findall(r"\d+", f) else 0)
    return files[:max_files] if max_files else files


def fetch_toanmath_labels(fn: str, cache_path: Path) -> int:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    tmp = cache_path.with_suffix(".part")
    n = 0
    with fs.open(f"datasets/daominhwysi/toanmath.com-alpha/{fn}", "rb", block_size=2 ** 20, cache_type="none") as fh:
        pf = pq.ParquetFile(fh)
        with open(tmp, "w", encoding="utf-8") as out:
            for rb in pf.iter_batches(batch_size=512, columns=["label", "path"]):
                for r in rb.to_pylist():
                    out.write(json.dumps({"path": r["path"], "label": r["label"]}, ensure_ascii=False) + "\n")
                    n += 1
    os.replace(tmp, cache_path)
    return n


# ---- toanmath.com-alpha: document-title (slug) overlap ---------------------------------------------------------------
# The dataset's `label` column is NOT page text: it is the toanmath document slug (34–86 chars, e.g.
# "de-cuoi-ky-1-toan-10-nam-2023-2024-truong-thpt-phu-loc-tt-hue"); the content exists only as page images. Text
# overlap therefore cannot be measured without OCR; what we can measure is which Vi-ExamMath exams (province × year
# entrance exams) exist as documents in the corpus.

_LGH_EXAM = re.compile(r"(?:de-(?:thi-)?(?:minh-hoa-|chinh-thuc-)?(?:thi-)?vao-(?:lop-)?10-mon-toan-)(?P<prov>.+?)"
                       r"(?:-nam)?-(?P<year>20\d\d)\b")
_ENTRANCE = re.compile(r"tuyen-sinh|vao-lop-10|vao-10|lop-10-thpt|thi-lop-10|ts10|tuyen-sinh-10")


def _slug_exam_years(slug: str) -> set[int]:
    """Exam year(s) named by a slug; a school year "2021-2022" means the June-2021 entrance exam."""
    ys = [int(y) for y in re.findall(r"(?<!\d)(20\d\d)(?!\d)", slug)]
    out, i = set(), 0
    while i < len(ys):
        out.add(ys[i])
        i += 2 if (i + 1 < len(ys) and ys[i + 1] == ys[i] + 1) else 1
    return out


def vi_exam_keys(root: Path | None = None) -> dict[str, dict]:
    """(province-slug, year, kind) for every cached loigiaihay grade-10 entrance-exam article."""
    root = root or DATA / "vi_exam" / "raw" / "loigiaihay.com"
    out = {}
    for mp in sorted(root.glob("*.json")):
        try:
            u = json.loads(mp.read_text(encoding="utf-8")).get("final_url") or ""
        except Exception:
            continue
        slug = u.rsplit("/", 1)[-1]
        m = _LGH_EXAM.search(slug)
        if not m or not re.search(r"-a\d+\.html$", slug):
            continue
        prov = re.sub(r"^(?:tp-|thanh-pho-)", "", m.group("prov"))
        kind = "minh_hoa" if "minh-hoa" in slug else "official"
        out[slug] = {"province": prov, "year": int(m.group("year")), "kind": kind, "page": mp.stem}
    return out


def toanmath_title_overlap(label_dir: Path | None = None) -> dict:
    """Match Vi-ExamMath exams to toanmath-alpha document slugs: same province slug + same year (as one of the
    years in the slug) + an entrance-exam cue, and not a specialised-school ("chuyen") paper."""
    label_dir = label_dir or CACHE / "toanmath_labels"
    slugs: Counter = Counter()
    for fp in sorted(label_dir.glob("*.jsonl")):
        for r in _jsonl(fp):
            slugs[r["label"]] += 1
    entr = [sl for sl in slugs if _ENTRANCE.search(sl) and "toan" in sl]
    keys = vi_exam_keys()
    matches = {}
    for slug, k in keys.items():
        prov, yr = k["province"], str(k["year"])
        hits = [t for t in entr if f"-{prov}" in t + "-" and int(yr) in _slug_exam_years(t) and "chuyen" not in t]
        hits = [t for t in hits if re.search(rf"(?:^|-){re.escape(prov)}(?:-|$)", t)]
        matches[slug] = {**k, "toanmath_docs": hits, "pages": sum(slugs[h] for h in hits)}
    n_exam = len(keys)
    hit = sum(1 for v in matches.values() if v["toanmath_docs"])
    by_year: dict[int, list[int]] = {}
    for v in matches.values():
        a = by_year.setdefault(v["year"], [0, 0])
        a[0] += 1
        a[1] += bool(v["toanmath_docs"])
    return {"n_toanmath_docs": len(slugs), "n_toanmath_pages": int(sum(slugs.values())),
            "n_toanmath_entrance_docs": len(entr), "n_vi_exam_exams": n_exam, "n_exams_in_toanmath": hit,
            "by_year": {y: {"exams": a[0], "in_toanmath": a[1]} for y, a in sorted(by_year.items())},
            "matches": matches}


def _jsonl_spec(spec: str) -> Iterator[Doc]:
    """``jsonl:<path>:<field>[+<field>…]`` (path may contain a drive letter)."""
    path, fields = spec[len("jsonl:"):].rsplit(":", 1)
    fl = fields.split("+")
    for i, r in enumerate(_jsonl(Path(path))):
        t = "\n".join(str(r.get(f) or "") for f in fl)
        yield Doc(str(r.get("id", r.get("item_id", i))), t, embed_text=str(r.get(fl[0]) or t))


SOURCES: dict[str, Callable[[], Iterator[Doc]]] = {
    "vi_gsm8k_test": _src_vi_gsm8k_test,
    "vi_math_test": _src_vi_math_test,
    "gsm8k_en_test": _src_gsm8k_en_test,
    "math_en_test": _src_math_en_test,
    "vi_exam_raw": lambda: iter_vi_exam_raw(unit="question"),
    "vi_exam_raw_page": lambda: iter_vi_exam_raw(unit="page"),
    "vi_exam_items": lambda: _src_vi_exam_items(),
    "luminarai": _src_luminarai,
    "luminarai_segments": _src_luminarai_segments,
    "hllj_crawl": _src_hllj_crawl,
    "hllj_mcq": _src_hllj_mcq,
    "metamathqa_vi": _src_metamathqa_vi,
    "toanmath_labels": _src_toanmath_labels,
}


def iter_source(spec: str, limit: int | None = None) -> Iterator[Doc]:
    if spec.startswith("vinhnq29:"):
        it = _src_vinhnq29(spec.split(":", 1)[1])
    elif spec.startswith("jsonl:"):
        it = _jsonl_spec(spec)
    elif spec.startswith("toanmath_labels:"):
        it = _src_toanmath_labels(int(spec.split(":", 1)[1]))
    else:
        it = SOURCES[spec]()
    for k, d in enumerate(it):
        if limit is not None and k >= limit:
            return
        yield d


# =====================================================================================================================
# 6. Scanner
# =====================================================================================================================

NGRAM_NS = (8, 13)
MINHASH_REPORT_J = 0.5     # T5 §3.4 C2(ii)
MINHASH_FLOOR = 0.3        # LSH candidates below this estimated Jaccard are not recorded
# which n-gram sizes are scanned per normalisation variant (8-grams only on the exact variant: the 8-gram
# template/nodiac variants add false positives and cost without adding a decision the 13-gram ones don't make)
SCAN_PLAN: dict[str, tuple[int, ...]] = {"exact": (8, 13), "nodiac": (13,), "template": (13,)}


@dataclass
class _TargetSet:
    docs: list[Doc]
    tags: list[str]                                         # target-set name per doc
    ngram: dict[tuple[int, str], NgramIndex]               # (n, variant) → index
    lsh: MinHashLSH
    sigs: list[np.ndarray | None]
    toks: list[dict[str, list[str]]]


class Scanner:
    """Index the target items (test sets), stream a corpus, keep the best corpus match per target item for every
    detector. Results are per target item (:meth:`results`)."""

    def __init__(self, targets: dict[str, list[Doc]], *, plan: dict[str, Sequence[int]] | None = None,
                 num_perm: int = 128, bands: int = 32, encoder: Encoder | None = None, embed_k: int = 3):
        self.plan = {v: tuple(ns) for v, ns in (plan or SCAN_PLAN).items()}
        self.variants = tuple(self.plan)
        self.mh = MinHasher(num_perm=num_perm, shingle=5)
        docs, tags = [], []
        for name, ds in targets.items():
            docs += ds
            tags += [name] * len(ds)
        self.T = _TargetSet(docs, tags, {}, MinHashLSH(num_perm, bands), [], [])
        for v, ns in self.plan.items():
            for n in ns:
                self.T.ngram[(n, v)] = NgramIndex(n)
        for i, d in enumerate(docs):
            vt = variant_tokens(d.text)
            self.T.toks.append(vt)
            for v, ns in self.plan.items():
                th = token_hashes(vt[v])
                for n in ns:
                    self.T.ngram[(n, v)].add(i, th)
            sig = self.mh.signature(token_hashes(vt["exact"]))
            self.T.sigs.append(sig)
            if sig is not None:
                self.T.lsh.insert(i, sig)
        for ix in self.T.ngram.values():
            ix.finalize()
        # shingle filter for the MinHash bound (5-gram shingles of all targets)
        self._shingle_ix = NgramIndex(self.mh.shingle, min_len=1)
        self._max_target_shingles = 1
        for i, vt in enumerate(self.T.toks):
            th = token_hashes(vt["exact"])
            ns = self._shingle_ix.add(i, th)
            self._max_target_shingles = max(self._max_target_shingles, ns)
        self._shingle_ix.finalize()
        N = len(docs)
        # best-so-far per target: (n, variant) → arrays
        self.ng_best = {k: np.zeros(N, dtype=np.float32) for k in self.T.ngram}
        self.ng_shared = {k: np.zeros(N, dtype=np.int32) for k in self.T.ngram}
        self.ng_doc = {k: np.full(N, -1, dtype=np.int64) for k in self.T.ngram}
        self.ng_ndocs = {k: np.zeros(N, dtype=np.int32) for k in self.T.ngram}   # #corpus docs sharing ≥1 gram
        self.mh_best = np.zeros(N, dtype=np.float32)
        self.mh_doc = np.full(N, -1, dtype=np.int64)
        self.encoder = encoder
        self.emb = None
        if encoder is not None:
            self.emb = EmbeddingIndex(encoder.encode([d.embed_text or d.text for d in docs]), k=embed_k)
        self.id_of: dict[int, str] = {}       # corpus position -> doc id, only for docs that are/were a best match
        self.corpus_snip: dict[int, str] = {}
        self.n_corpus = 0
        self.corpus_hit_docs = 0          # corpus docs sharing >= 1 exact 13-gram with any target (removal count)

    def _maybe_signature(self, th: np.ndarray) -> np.ndarray | None:
        """MinHash signature, or None when J ≥ MINHASH_FLOOR with any target is impossible.

        Bound: J(A,B) = |A∩B| / |A∪B| ≤ |A ∩ ⋃targets| / |A|; the numerator is over-estimated by the shingle bloom
        filter, so the bound is safe. It also skips long corpus docs vs short targets (J ≤ |B|/|A|)."""
        sh = self.mh.shingles(th)
        if not len(sh):
            return None
        if len(sh) * MINHASH_FLOOR > self._max_target_shingles:
            return None
        if self._shingle_ix._bloom_test(sh).sum() < MINHASH_FLOOR * len(sh):
            return None
        ph = _mix64(sh[:, None] ^ self.mh.seeds[None, :])
        return (ph.min(axis=0) & _M32).astype(np.uint32)

    def _remember(self, ci: int, d_id: str, text: str) -> None:
        self.id_of[ci] = d_id
        self.corpus_snip[ci] = text[:20000]

    def feed(self, docs: Iterable[Doc], *, embed_batch: int = 4096, progress_every: int = 50000) -> None:
        buf_e, buf_i, buf_id = [], [], []
        t0 = time.time()
        for d in docs:
            ci = self.n_corpus
            self.n_corpus += 1
            vh = variant_hashes(d.text)
            hit_any = False
            for v, ns in self.plan.items():
                lens = set().union(*(self.T.ngram[(n, v)].lengths for n in ns))
                pre = multi_ngram_hashes(vh[v], lens)
                for n in ns:
                    key = (n, v)
                    res = self.T.ngram[key].query(pre=pre)
                    if not res:
                        continue
                    ids = np.fromiter(res.keys(), dtype=np.int64, count=len(res))
                    sh = np.fromiter(res.values(), dtype=np.int32, count=len(res))
                    den = np.maximum(np.asarray([self.T.ngram[key].n_grams[i] for i in ids]), 1)
                    cov = sh / den
                    self.ng_ndocs[key][ids] += 1
                    better = cov > self.ng_best[key][ids]
                    if better.any():
                        b = ids[better]
                        self.ng_best[key][b] = cov[better]
                        self.ng_shared[key][b] = sh[better]
                        self.ng_doc[key][b] = ci
                        hit_any = True
                    if n == 13 and v == "exact":
                        self.corpus_hit_docs += 1
            sig = self._maybe_signature(vh["exact"])
            if sig is not None:
                for ti, j in self.T.lsh.query(sig, threshold=MINHASH_FLOOR):
                    if j > self.mh_best[ti]:
                        self.mh_best[ti] = j
                        self.mh_doc[ti] = ci
                        hit_any = True
            if hit_any:
                self._remember(ci, d.id, d.text)
            if self.emb is not None:
                buf_e.append(d.embed_text or d.text)
                buf_i.append(ci)
                buf_id.append(d.id)
                if len(buf_e) >= embed_batch:
                    self._flush_emb(buf_e, buf_i, buf_id)
                    buf_e, buf_i, buf_id = [], [], []
            if progress_every and self.n_corpus % progress_every == 0:
                print(f"  [{self.n_corpus:,} docs, {time.time() - t0:,.0f}s, peak {peak_rss_mb():,.0f} MB]",
                      file=sys.stderr, flush=True)
        if buf_e:
            self._flush_emb(buf_e, buf_i, buf_id)

    def _flush_emb(self, texts, ids, doc_ids) -> None:
        E = self.encoder.encode(texts)
        self.emb.update(E, np.asarray(ids))
        top = set(self.emb.best_id.ravel().tolist())
        for t, i, di in zip(texts, ids, doc_ids):
            if i in top and i not in self.id_of:
                self._remember(i, di, t)

    def _cid(self, ci: int) -> str | None:
        return self.id_of.get(int(ci)) if ci >= 0 else None

    def results(self) -> list[dict]:
        out = []
        for i, d in enumerate(self.T.docs):
            r = {"target": self.T.tags[i], "id": d.id, "n_tokens": len(self.T.toks[i]["exact"]),
                 "text": d.text[:300]}
            for (n, v) in self.T.ngram:
                k = (n, v)
                ci = int(self.ng_doc[k][i])
                r[f"ng{n}_{v}"] = {"cov": round(float(self.ng_best[k][i]), 4), "shared": int(self.ng_shared[k][i]),
                                   "n_grams": int(self.T.ngram[k].n_grams[i]), "n_docs": int(self.ng_ndocs[k][i]),
                                   "doc": self._cid(ci)}
            r["minhash"] = {"j": round(float(self.mh_best[i]), 4), "doc": self._cid(int(self.mh_doc[i]))}
            if self.emb is not None:
                r["embed"] = {"cos": [round(float(x), 4) for x in self.emb.best_val[i]],
                              "doc": [self._cid(int(x)) for x in self.emb.best_id[i]]}
            out.append(r)
        return out


# ---- hit rules ------------------------------------------------------------------------------------------------------

@dataclass
class Thresholds:
    ng13_any: int = 1             # GPT-3 / G4 rule: ≥ 1 shared 13-gram
    ng_cov: float = 0.5           # "substantial" coverage of the target item's n-grams by one corpus doc
    minhash: float = 0.5          # T5 C2(ii)
    embed: float | None = None    # calibrated (see `calibrate`)
    min_tokens: int = 13          # items shorter than this are only counted via whole-sequence match


def hit_flags(r: dict, th: Thresholds) -> dict[str, bool]:
    f = {
        "ng13_any": r["ng13_exact"]["shared"] >= th.ng13_any,
        "ng13_cov50": r["ng13_exact"]["cov"] >= th.ng_cov,
        "ng8_cov50": r["ng8_exact"]["cov"] >= th.ng_cov,
        "ng13_nodiac_cov50": r.get("ng13_nodiac", {}).get("cov", 0) >= th.ng_cov,
        "ng13_template_cov50": r.get("ng13_template", {}).get("cov", 0) >= th.ng_cov,
        "minhash50": r["minhash"]["j"] >= th.minhash,
    }
    if th.embed is not None and "embed" in r:
        f["embed"] = r["embed"]["cos"][0] >= th.embed
    long_ = r["n_tokens"] >= th.min_tokens
    f["ng13_any_long"] = f["ng13_any"] and long_          # GPT-3 rule restricted to items with >= 13 tokens
    f["ng13_cov50_long"] = f["ng13_cov50"] and long_
    f["short_whole_match"] = f["ng13_cov50"] and not long_  # item < 13 tokens contained verbatim
    f["any_lexical"] = f["ng13_cov50"] or f["ng13_nodiac_cov50"] or f["minhash50"]
    f["any_strict"] = f["any_lexical"] or f.get("embed", False)
    return f


def summarize(results: list[dict], th: Thresholds) -> dict:
    by: dict[str, Counter] = {}
    n_by: Counter = Counter()
    for r in results:
        n_by[r["target"]] += 1
        c = by.setdefault(r["target"], Counter())
        for k, v in hit_flags(r, th).items():
            c[k] += int(v)
    return {t: {"n": n_by[t], **{k: int(v) for k, v in by[t].items()}} for t in n_by}


# =====================================================================================================================
# 7. Calibration (embedding threshold on known translation pairs)
# =====================================================================================================================

def calibrate_pairs(enc: Encoder, a: list[str], b: list[str], fpr: float = 0.01, seed: int = 0) -> dict:
    """a[i] ↔ b[i] are the same item (translation / alternative translation).

    Returns: positive-pair cosines, random-pair cosines, hard negatives (max cosine of a[i] to any b[j≠i]), top-1
    retrieval accuracy, and the threshold τ at the requested FPR on (i) random pairs and (ii) hard negatives, with
    the recall of positives at each τ.
    """
    A = enc.encode(a).astype(np.float32)
    B = enc.encode(b).astype(np.float32)
    S = A @ B.T
    n = len(a)
    pos = np.diag(S).copy()
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, 20000)
    j = rng.integers(0, n, 20000)
    m = i != j
    rnd = S[i[m], j[m]]
    S2 = S.copy()
    np.fill_diagonal(S2, -1)
    hard = S2.max(axis=1)
    top1 = float((S.argmax(axis=1) == np.arange(n)).mean())
    tau_rnd = float(np.quantile(rnd, 1 - fpr))
    tau_hard = float(np.quantile(hard, 1 - fpr))

    def q(x):
        return {p: round(float(np.quantile(x, p)), 4) for p in (0.01, 0.05, 0.1, 0.5, 0.9, 0.99)}
    res = {"n": n, "top1_acc": round(top1, 4), "pos": q(pos), "random": q(rnd), "hard_neg": q(hard),
           "tau_fpr_random": round(tau_rnd, 4), "recall_at_tau_random": round(float((pos >= tau_rnd).mean()), 4),
           "tau_fpr_hard": round(tau_hard, 4), "recall_at_tau_hard": round(float((pos >= tau_hard).mean()), 4)}
    for t in (0.80, 0.85, 0.88, 0.90, 0.92, 0.95):
        res[f"at_{t:.2f}"] = {"recall": round(float((pos >= t).mean()), 4),
                              "fpr_random": round(float((rnd >= t).mean()), 5),
                              "fpr_hard": round(float((hard >= t).mean()), 4)}
    return res


# =====================================================================================================================
# 8. Gate API (T1 §2.3 G4 / G5) for the generation pipeline
# =====================================================================================================================

class Gate:
    """Generation-side gates (T1 §2.3), with the thresholds recommended in the X5 report.

    G4 (vs the test sets): reject if the item shares ≥ 1 13-gram with a test item (exact or diacritic-stripped
    variant), OR ≥ ``template_cov`` of a test item's number-templated 13-grams (a value-perturbed copy), OR its max
    bge-m3 cosine vs the test items is ≥ ``cos_threshold`` (X5 calibration: 0.85 for Vietnamese↔Vietnamese).
    G5 (within an arm): reject if est. MinHash Jaccard (5-gram shingles) ≥ ``dedup_j`` vs an accepted item; call
    :meth:`accept` after an item passes."""

    def __init__(self, test_docs: list[Doc], *, encoder: Encoder | None = None, cos_threshold: float | None = None,
                 dedup_j: float = 0.8, template_cov: float | None = 0.5):
        self.template_cov = template_cov
        self.ix = {v: NgramIndex(13) for v in ("exact", "nodiac", "template")}
        for i, d in enumerate(test_docs):
            vt = variant_tokens(d.text)
            for v in self.ix:
                self.ix[v].add(i, token_hashes(vt[v]))
        for ix in self.ix.values():
            ix.finalize()
        self.encoder, self.cos_threshold = encoder, cos_threshold
        self.E = encoder.encode([d.embed_text or d.text for d in test_docs]).astype(np.float32) if encoder else None
        self.mh = MinHasher()
        self.lsh = MinHashLSH(bands=16)       # threshold ≈ 0.71, below the 0.8 decision point
        self.dedup_j = dedup_j
        self.test_ids = [d.id for d in test_docs]

    def check(self, text: str) -> dict:
        vt = variant_tokens(text)
        r = {"g4_ngram": None, "g4_cos": None, "g5_dup": None}
        for v, ix in self.ix.items():
            res = ix.query(token_hashes(vt[v]))
            if v == "template":
                if self.template_cov is None:
                    continue
                res = {i: c for i, c in res.items() if c / max(ix.n_grams[i], 1) >= self.template_cov}
            if res:
                i = max(res, key=res.get)
                r["g4_ngram"] = {"variant": v, "test_id": self.test_ids[i], "shared": res[i]}
                break
        if self.encoder is not None and self.cos_threshold is not None:
            e = self.encoder.encode([text]).astype(np.float32)[0]
            s = self.E @ e
            k = int(s.argmax())
            if s[k] >= self.cos_threshold:
                r["g4_cos"] = {"test_id": self.test_ids[k], "cos": float(s[k])}
        sig = self.mh.signature(token_hashes(vt["exact"]))
        if sig is not None:
            q = self.lsh.query(sig, self.dedup_j)
            if q:
                r["g5_dup"] = {"accepted_id": q[0][0], "j": q[0][1]}
        r["pass"] = not (r["g4_ngram"] or r["g4_cos"] or r["g5_dup"])
        r["_sig"] = sig
        return r

    def accept(self, item_id: int, check_result: dict) -> None:
        if check_result.get("_sig") is not None:
            self.lsh.insert(item_id, check_result["_sig"])


# =====================================================================================================================
# 9. CLI
# =====================================================================================================================

def _window(corpus_text: str, target_text: str, width: int = 260) -> str:
    """The part of a corpus doc around the first occurrence of a 4-word piece of the target (else its start)."""
    ct = unescape_newlines(corpus_text)
    low = ct.lower()
    w = target_text.split()
    for a in range(0, max(1, len(w) - 4), 2):
        piece = " ".join(w[a:a + 4]).lower()
        i = low.find(piece)
        if i >= 0:
            return ct[max(0, i - 40):i - 40 + width]
    return ct[:width]


def _write_examples(path: Path, results: list[dict], snippets: dict[str, str], k: int = 8) -> None:
    lines = ["# Top hits per target set (sorted by 13-gram coverage, then MinHash, then cosine)\n"]
    for tgt in sorted({r["target"] for r in results}):
        rs = [r for r in results if r["target"] == tgt and (r["ng13_exact"]["cov"] > 0 or r["minhash"]["j"] > 0
                                                           or r.get("ng13_template", {}).get("cov", 0) >= 0.5
                                                           or "embed" in r)]
        rs.sort(key=lambda r: (-r["ng13_exact"]["cov"], -r["minhash"]["j"],
                               -(r.get("embed", {}).get("cos", [0])[0])))
        lines.append(f"\n## {tgt}\n")
        for r in rs[:k]:
            doc = r["ng13_exact"]["doc"] or r["minhash"]["doc"] or (r.get("embed", {}).get("doc", [None])[0])
            snip = _window(snippets.get(doc, "") if doc else "", r["text"])
            lines.append(f"- **{r['id']}** ng13 cov={r['ng13_exact']['cov']} shared={r['ng13_exact']['shared']} "
                         f"tmpl={r['ng13_template']['cov'] if 'ng13_template' in r else '-'} "
                         f"mh={r['minhash']['j']} emb={r.get('embed', {}).get('cos', ['-'])[0]} -> `{doc}`\n"
                         f"  - target: {r['text'][:220]!r}\n  - corpus: {snip[:220]!r}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def peak_rss_mb() -> float:
    """Peak resident set size of this process in MB (Windows: PeakWorkingSetSize; POSIX: ru_maxrss)."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            k32 = ctypes.WinDLL("kernel32")
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi = ctypes.WinDLL("psapi")
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
            return pmc.PeakWorkingSetSize / 2 ** 20
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        return float("nan")


def cmd_scan(a: argparse.Namespace) -> None:
    t0 = time.time()
    targets = {t: list(iter_source(t, a.target_limit)) for t in a.target}
    for t, ds in targets.items():
        print(f"target {t}: {len(ds)} items", file=sys.stderr)
    enc = Encoder(a.embed_model, batch_size=a.embed_bs, max_length=a.embed_maxlen) if a.embed_model else None
    sc = Scanner(targets, encoder=enc)
    t1 = time.time()
    for c in a.corpus:
        print(f"corpus {c} …", file=sys.stderr)
        sc.feed(iter_source(c, a.corpus_limit), embed_batch=a.embed_chunk)
    t2 = time.time()
    res = sc.results()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "items.jsonl", "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    th = Thresholds(embed=a.embed_threshold)
    summ = {"corpus": a.corpus, "targets": a.target, "n_corpus_docs": sc.n_corpus,
            "corpus_docs_with_13gram_hit": sc.corpus_hit_docs, "embed_model": a.embed_model,
            "thresholds": th.__dict__, "counts": summarize(res, th),
            "runtime_s": {"index": round(t1 - t0, 1), "stream": round(t2 - t1, 1), "total": round(time.time() - t0, 1)},
            "peak_rss_mb": round(peak_rss_mb(), 1)}
    if enc is not None:
        import torch
        summ["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1) if torch.cuda.is_available() else 0
    (out / "summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False), encoding="utf-8")
    snippets = {sc.id_of[i]: t for i, t in sc.corpus_snip.items()}
    with open(out / "corpus_snippets.jsonl", "w", encoding="utf-8") as f:
        for cid, t in snippets.items():
            f.write(json.dumps({"id": cid, "text": t}, ensure_ascii=False) + "\n")
    _write_examples(out / "examples.md", res, snippets)
    print(json.dumps(summ, indent=1, ensure_ascii=False))


def cmd_calibrate(a: argparse.Namespace) -> None:
    enc = Encoder(a.embed_model, batch_size=a.embed_bs, max_length=a.embed_maxlen)
    out = {"model": a.embed_model}
    t0 = time.time()
    g = list(_jsonl(DATA / "clean" / "vi_gsm8k_test" / "vi_gsm8k_test_full_v1.1.jsonl"))
    out["gsm8k_en_vi"] = calibrate_pairs(enc, [r["en_question"] for r in g], [r["question"] for r in g])
    m = list(_jsonl(DATA / "clean" / "vi_math_test" / "vi_math_test_full_v1.1.jsonl"))
    rng = np.random.default_rng(0)
    sel = rng.choice(len(m), size=min(1500, len(m)), replace=False)
    out["math_en_vi"] = calibrate_pairs(enc, [m[i]["en_problem"] for i in sel], [m[i]["problem"] for i in sel])
    # Vi ↔ Vi: our Vi-GSM8K vs an independent Vietnamese translation of the same GSM8K test items (vinhnq29)
    alt = [d.embed_text for d in iter_source("vinhnq29:vigsm8k_test")]
    if len(alt) == len(g):
        out["gsm8k_vi_vi_alt_translation"] = calibrate_pairs(enc, [r["question"] for r in g], alt)
    # value-perturbed copies: numbers resampled in our Vi-GSM8K (template-identical, values differ)
    pert = [perturb_numbers(r["question"], seed=k) for k, r in enumerate(g)]
    out["gsm8k_vi_number_perturbed"] = calibrate_pairs(enc, [r["question"] for r in g], pert)
    out["runtime_s"] = round(time.time() - t0, 1)
    p = Path(a.out)
    p.mkdir(parents=True, exist_ok=True)
    (p / f"calibration_{a.embed_model.replace('/', '__')}.json").write_text(json.dumps(out, indent=1),
                                                                         encoding="utf-8")
    print(json.dumps(out, indent=1))


def perturb_numbers(text: str, seed: int = 0) -> str:
    """Resample every integer in ``text`` (same digit count, ≠ original). For value-perturbation tests."""
    rng = np.random.default_rng(seed)

    def rep(m):
        s = m.group(0)
        lo, hi = (10 ** (len(s) - 1) if len(s) > 1 else 1), 10 ** len(s)
        v = s
        while v == s:
            v = str(int(rng.integers(lo, hi)))
        return v
    return re.sub(r"(?<![\d.,])\d+(?![\d.,]\d)", rep, text)


def cmd_gate(a: argparse.Namespace) -> None:
    tests = [d for t in a.target for d in iter_source(t)]
    enc = Encoder(a.embed_model) if a.embed_model else None
    g = Gate(tests, encoder=enc, cos_threshold=a.embed_threshold, dedup_j=a.dedup_j)
    n = acc = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for k, r in enumerate(_jsonl(Path(a.candidates))):
            res = g.check(r[a.field])
            if res["pass"]:
                g.accept(k, res)
                acc += 1
            n += 1
            res.pop("_sig", None)
            f.write(json.dumps({"id": r.get("id", k), **res}, ensure_ascii=False) + "\n")
    print(json.dumps({"n": n, "accepted": acc}))


def cmd_bench_index(a: argparse.Namespace) -> None:
    """Index the BIG side (a corpus) on disk, then query the test items against it: a scaling benchmark for the
    disk-backed NgramIndex and a cross-check of the streaming direction (same coverage numbers expected)."""
    wd = Path(a.workdir)
    t0 = time.time()
    ix = NgramIndex(a.n, shard_grams=a.shard_grams, workdir=wd)
    ndocs = 0
    for i, d in enumerate(iter_source(a.corpus, a.corpus_limit)):
        ix.add(i, token_hashes(tokenize(d.text)))
        ndocs += 1
        if ndocs % 100000 == 0:
            print(f"  indexed {ndocs:,} docs, {ix.total_grams + ix._buf_size:,} grams, "
                  f"{time.time() - t0:,.0f}s, peak {peak_rss_mb():,.0f} MB", file=sys.stderr, flush=True)
    ix.finalize()
    t1 = time.time()
    ix._bloom_build()
    t2 = time.time()
    disk = sum(f.stat().st_size for f in wd.glob("shard*.npy"))
    hits: Counter = Counter()
    nq = 0
    examples = []
    for t in a.target:
        for d in iter_source(t):
            th = token_hashes(tokenize(d.text))
            res = ix.query(th)
            nq += 1
            if res:
                hits[t] += 1
                if len(examples) < 20:
                    examples.append({"target": t, "id": d.id, "n_corpus_docs": len(res)})
    t3 = time.time()
    out = {"corpus": a.corpus, "n": a.n, "docs": ndocs, "grams": ix.total_grams, "shards": len(ix.shards),
           "disk_mb": round(disk / 2 ** 20, 1), "bloom_mb": round(len(ix._bloom) / 2 ** 20, 1),
           "build_s": round(t1 - t0, 1), "bloom_s": round(t2 - t1, 1), "queries": nq,
           "query_s": round(t3 - t2, 1), "query_ms_each": round(1000 * (t3 - t2) / max(nq, 1), 2),
           "targets_with_any_shared_gram": dict(hits), "peak_rss_mb": round(peak_rss_mb(), 1),
           "examples": examples}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))


def cmd_toanmath_titles(a: argparse.Namespace) -> None:
    res = toanmath_title_overlap()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "toanmath_title_overlap.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "matches"}, indent=1))


def cmd_resummarize(a: argparse.Namespace) -> None:
    """Recompute summary counts (and examples) of an existing scan dir with other thresholds / an embed file."""
    d = Path(a.scan)
    res = list(_jsonl(d / "items.jsonl"))
    if a.embed_jsonl:
        emb = {(r["set"], r["id"]): r for r in _jsonl(Path(a.embed_jsonl))}
        for r in res:
            e = emb.get((r["target"], r["id"]))
            if e:
                r["embed"] = {"cos": e["cos"], "doc": e["doc"]}
    th = Thresholds(embed=a.embed_threshold)
    print(json.dumps(summarize(res, th), indent=1))


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="decontam")
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("scan")
    s.add_argument("--corpus", action="append", required=True)
    s.add_argument("--target", action="append", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--corpus-limit", type=int)
    s.add_argument("--target-limit", type=int)
    s.add_argument("--embed-model")
    s.add_argument("--embed-threshold", type=float)
    s.add_argument("--embed-bs", type=int, default=32)
    s.add_argument("--embed-maxlen", type=int, default=256)
    s.add_argument("--embed-chunk", type=int, default=8192)
    s.set_defaults(fn=cmd_scan)
    c = sp.add_parser("calibrate")
    c.add_argument("--embed-model", default="intfloat/multilingual-e5-small")
    c.add_argument("--embed-bs", type=int, default=32)
    c.add_argument("--embed-maxlen", type=int, default=256)
    c.add_argument("--out", default=str(CONTAM / "calibration"))
    c.set_defaults(fn=cmd_calibrate)
    g = sp.add_parser("gate")
    g.add_argument("--target", action="append", required=True)
    g.add_argument("--candidates", required=True)
    g.add_argument("--field", default="question")
    g.add_argument("--out", required=True)
    g.add_argument("--embed-model")
    g.add_argument("--embed-threshold", type=float)
    g.add_argument("--dedup-j", type=float, default=0.8)
    g.set_defaults(fn=cmd_gate)
    b = sp.add_parser("bench-index")
    b.add_argument("--corpus", required=True)
    b.add_argument("--target", action="append", required=True)
    b.add_argument("--n", type=int, default=13)
    b.add_argument("--shard-grams", type=int, default=20_000_000)
    b.add_argument("--corpus-limit", type=int)
    b.add_argument("--workdir", default=str(CONTAM / "index" / "bench"))
    b.add_argument("--out", default=str(CONTAM / "scans" / "bench_index.json"))
    b.set_defaults(fn=cmd_bench_index)
    t = sp.add_parser("toanmath-titles")
    t.add_argument("--out", default=str(CONTAM / "scans" / "toanmath_titles"))
    t.set_defaults(fn=cmd_toanmath_titles)
    r = sp.add_parser("resummarize")
    r.add_argument("--scan", required=True)
    r.add_argument("--embed-jsonl")
    r.add_argument("--embed-threshold", type=float)
    r.set_defaults(fn=cmd_resummarize)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
