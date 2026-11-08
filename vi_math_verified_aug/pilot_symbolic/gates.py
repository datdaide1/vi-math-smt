"""Common gates for arm Y (T1 §2.2 step 4, §2.3 G3/G5; thresholds §2.5).

* leak       ``pilot_gen.util.LEAK_RE`` (the R5-derived regex used by G1) — zero tolerance
* answer     the new answer's value written anywhere in the text (``symbolic.contains_value``, digits and number
             words, all Vietnamese numeric surface forms) — zero tolerance (T1 §2.5 "Leak")
* near_dup   chrF(variant, seed) > 0.90 for M2-M6 (M6: against the first seed)
* dedup      character 5-gram Jaccard >= 0.8 against an earlier accepted item of the arm, or > 3 accepted items per
             seed (T1 §2.3 G5; exact Jaccard instead of MinHash at pilot size)
"""
from __future__ import annotations

import re
from fractions import Fraction

from ..pilot_gen.util import chrf, leak_regex
from ..symbolic import contains_value

CHRF_MAX = 0.90
JACCARD_MAX = 0.8
PER_SEED_MAX = 3


def shingles(text: str, n: int = 5) -> set[str]:
    s = re.sub(r"\s+", " ", (text or "").lower()).strip()
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


def item_gates(text: str, seed_text: str, answer: Fraction, op: str) -> dict:
    leaks = leak_regex(text)
    c = chrf(text, seed_text)
    return {"leak_regex": leaks, "leak_ok": not leaks, "answer_in_text": contains_value(text, answer),
            "chrf": round(c, 4), "chrf_ok": op == "M1" or c <= CHRF_MAX,
            "len_ratio": round(len(text) / max(1, len(seed_text)), 3)}


class Deduper:
    def __init__(self):
        self.kept: list[tuple[str, set]] = []
        self.per_seed: dict[str, int] = {}

    def check_and_add(self, key: str, seed_id: str, text: str) -> tuple[bool, float, str | None]:
        sh = shingles(text)
        best, who = 0.0, None
        for k, s in self.kept:
            j = jaccard(sh, s)
            if j > best:
                best, who = j, k
        ok = best < JACCARD_MAX and self.per_seed.get(seed_id, 0) < PER_SEED_MAX
        if ok:
            self.kept.append((key, sh))
            self.per_seed[seed_id] = self.per_seed.get(seed_id, 0) + 1
        return ok, round(best, 4), who


__all__ = ["item_gates", "Deduper", "shingles", "jaccard", "CHRF_MAX", "JACCARD_MAX"]
