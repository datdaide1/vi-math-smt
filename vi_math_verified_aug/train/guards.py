"""Fail-fast guards for every SFT run (T2, lead's rule 2026-11-07): nothing runs on full data and is found wrong later.

Torch-free (unit-tested on mocks); ``sft.SFTRun`` calls them.

1. Pre-flight, before step 1 (any failure -> ``GuardAbort("PREFLIGHT_<what>")``):
   * :func:`preflight_items` decodes 3 items' trained tokens (label != -100) and asserts they are exactly the
     completion text (+ EOS): no prompt, no padding, the answer present; the trained-token fraction per item is in
     ``label_frac_range``; the template round-trips and EOS is the last token.
   * :func:`check_dataset` -- sha256 / row count against the expected manifest values.
   * :func:`overlap_with_eval` -- no training question may equal (normalised) an eval question.
   * step-0 loss inside ``step0_loss_band`` (measured once on S0, T1r report).
2. In-loop (:class:`LossGuard`): non-finite loss (or gnorm when no GradScaler), no progress (mean loss of steps
   50-100 not below steps 1-20 by ``min_progress``), spike (> ``spike_x`` x running median), collapse
   (< ``collapse_below`` in the first 25 % of steps), throughput < ``min_tok_frac`` x expected.
3. Mid-run (25 %, 50 %; :func:`midrun_verdict`): dev loss must not diverge from train loss; greedy quick-eval on
   64 fixed dev items.  QA-T #2 (P(false abort) 0.25/check at n=16, `qa/quick_eval_noise.py`): an accuracy drop is a
   WARNING only; hard aborts only on format collapse (repetition > 0.5, no-answer > 0.5, accuracy == 0 after step
   0).  The end-of-run check records and never aborts.
"""
from __future__ import annotations

import hashlib
import math
import re
import statistics
import unicodedata

DEFAULTS = {
    "enabled": True,
    "label_frac_range": [0.15, 0.95],   # trained tokens / all tokens, per item
    "n_decode": 3,
    "expected_sha256": None,            # of the training file, from the data manifest
    "expected_rows": None,
    "eval_hashes_path": None,           # JSON list of sha1(normalised eval question)
    "step0_loss_band": None,            # [lo, hi]; T1r S0 measured
    "min_progress": 0.03,               # mean(50-100) must be <= mean(1-20) * (1 - min_progress)
    "spike_x": 3.0,
    "spike_after": 20,
    "collapse_below": 0.05,
    "expected_tok_per_s": None,
    "min_tok_frac": 0.5,
    "tok_window": 20,
    "midrun_fracs": [0.25, 0.5],        # + the end
    "quick_n": 64,
    "quick_max_new_tokens": 384,
    "max_acc_drop": 0.10,
    "max_repetition": 0.20,            # warning level
    "abort_repetition": 0.5,           # hard abort (format collapse)
    "abort_no_answer": 0.5,
    "max_shortfall": 0.01,             # QA-T B3: n_eff < n_train by more than 1 % -> abort (data/setup error)
    "max_dev_train_ratio": 2.0,        # WARNING only (arms whose solution style differs from the dev references
                                       # raise dev NLL while accuracy improves: VI/EN Y+ s0, 2026-11-08)
    "dev_blowup_x": 2.0,               # ABORT: dev loss > 2 x its own step-0 value (a real blow-up)
    "quick_collapse_frac": 0.5,        # ABORT: quick-eval accuracy <= 50 % of its own step-0 value ...
    "quick_collapse_checks": 2,        # ... at this many consecutive mid-run checks (lead 2026-11-08)
}


class GuardAbort(RuntimeError):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}")
        self.reason, self.detail = reason, detail


# ------------------------------------------------------------------------------------------------ pre-flight
def norm_q(text: str) -> str:
    t = unicodedata.normalize("NFC", str(text)).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t)).strip()


def qhash(text: str) -> str:
    return hashlib.sha1(norm_q(text).encode("utf-8")).hexdigest()


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_dataset(path, n_rows: int, g: dict) -> dict:
    sha = file_sha256(path)
    out = {"sha256": sha, "rows": n_rows}
    if g.get("expected_sha256") and sha != g["expected_sha256"]:
        raise GuardAbort("PREFLIGHT_DATA_SHA", f"{path}: {sha} != manifest {g['expected_sha256']}")
    if g.get("expected_rows") is not None and n_rows != g["expected_rows"]:
        raise GuardAbort("PREFLIGHT_DATA_ROWS", f"{path}: {n_rows} rows != manifest {g['expected_rows']}")
    return out


def overlap_with_eval(questions: list[str], eval_hashes: set[str]) -> dict:
    hits = [q for q in questions if qhash(q) in eval_hashes]
    out = {"n_checked": len(questions), "n_eval_hashes": len(eval_hashes), "n_overlap": len(hits),
           "examples": [h[:120] for h in hits[:3]]}
    if hits:
        raise GuardAbort("PREFLIGHT_EVAL_OVERLAP", f"{len(hits)} training questions equal eval questions: {out}")
    return out


def preflight_items(items: list[dict], decode, *, prompt_fn, completion_fn, eos_id: int, pad_id: int | None,
                    g: dict) -> dict:
    """``items``: encoded items with the source fields kept (``q``, ``sol``, ``ans``).  ``decode(ids) -> str``."""
    lo, hi = g["label_frac_range"]
    fracs = []
    for it in items:
        n = len(it["ids"])
        fracs.append(sum(1 for x in it["labels"] if x != -100) / max(1, n))
    bad = [round(f, 3) for f in fracs if not lo <= f <= hi]
    if len(bad) > 0.02 * len(items):
        raise GuardAbort("PREFLIGHT_LABEL_FRAC", f"{len(bad)} items outside {lo}-{hi}: {bad[:5]}")
    ex = []
    for it in items[:g["n_decode"]]:
        lab = [x for x in it["labels"] if x != -100]
        ids = it["ids"]
        if not lab or lab[-1] != eos_id:
            raise GuardAbort("PREFLIGHT_EOS", f"{it['id']}: last trained token {lab[-1:]} != eos {eos_id}")
        if ids[-1] != eos_id:
            raise GuardAbort("PREFLIGHT_EOS", f"{it['id']}: sequence does not end in EOS")
        first = next(i for i, x in enumerate(it["labels"]) if x != -100)
        if any(x == -100 for x in it["labels"][first:]):
            raise GuardAbort("PREFLIGHT_LABEL_GAP", f"{it['id']}: masked token inside the completion")
        if pad_id is not None and pad_id != eos_id and pad_id in lab:
            raise GuardAbort("PREFLIGHT_LABEL_PAD", f"{it['id']}: padding token trained")
        want_c = completion_fn(it["sol"], it["ans"])
        got_c = decode(lab[:-1])
        if got_c.strip() != want_c.strip():
            raise GuardAbort("PREFLIGHT_LABEL_TEXT", f"{it['id']}: trained text != completion\n"
                                                     f"got : {got_c[:300]!r}\nwant: {want_c[:300]!r}")
        if it["ans"] is not None and "\\boxed{" + str(it["ans"]).strip() + "}" not in got_c:
            raise GuardAbort("PREFLIGHT_ANSWER", f"{it['id']}: answer {it['ans']!r} not in the trained text")
        want_p = prompt_fn(it["q"])
        got_p = decode(ids[:first])
        if not got_p.endswith(want_p) or len(got_p) - len(want_p) > 40:  # only a BOS may precede the prompt
            raise GuardAbort("PREFLIGHT_TEMPLATE", f"{it['id']}: prompt round-trip {got_p[-200:]!r}")
        ex.append({"id": it["id"], "prompt_tail": got_p[-120:], "trained_text": got_c, "n_trained": len(lab),
                   "n_total": len(ids), "eos_trained": True})
    return {"label_frac_min": round(min(fracs), 4), "label_frac_mean": round(sum(fracs) / len(fracs), 4),
            "label_frac_max": round(max(fracs), 4), "decoded": ex}


def check_step0(loss: float, g: dict) -> None:
    band = g.get("step0_loss_band")
    if band and not band[0] <= loss <= band[1]:
        raise GuardAbort("PREFLIGHT_STEP0_LOSS", f"step-0 loss {loss:.4f} outside {band}")


# ------------------------------------------------------------------------------------------------ in-loop
class LossGuard:
    def __init__(self, g: dict, total_steps: int, fp16_scaler: bool):
        self.g, self.total, self.scaler = g, total_steps, fp16_scaler
        self.losses: list[float] = []
        self.tok: list[float] = []

    def update(self, step: int, loss: float, gnorm, skipped: bool, tok_per_s: float | None = None) -> None:
        g = self.g
        if not math.isfinite(loss):
            raise GuardAbort("NONFINITE_LOSS", f"step {step}: loss {loss}")
        gn = float(gnorm) if not isinstance(gnorm, str) else float("inf")
        if not math.isfinite(gn) and not (self.scaler and skipped):  # fp16: inf gnorm on a skipped step is normal
            raise GuardAbort("NONFINITE_GNORM", f"step {step}: gnorm {gnorm}")
        self.losses.append(loss)
        n = len(self.losses)
        if n > g["spike_after"]:
            med = statistics.median(self.losses[-51:-1])
            if loss > g["spike_x"] * med:
                raise GuardAbort("LOSS_SPIKE", f"step {step}: loss {loss:.4f} > {g['spike_x']} x median {med:.4f}")
        if n <= max(1, int(0.25 * self.total)) and n >= 5 and loss < g["collapse_below"]:
            raise GuardAbort("LOSS_COLLAPSE", f"step {step}: loss {loss:.4f} < {g['collapse_below']} early "
                                              f"(label leak or duplicated data?)")
        if n == 100:
            a, b = sum(self.losses[1:21]) / 20, sum(self.losses[50:100]) / 50
            if b > a * (1 - g["min_progress"]):
                raise GuardAbort("NO_PROGRESS", f"mean loss steps 50-100 {b:.4f} vs 1-20 {a:.4f}")
        if tok_per_s is not None and g.get("expected_tok_per_s"):
            self.tok.append(tok_per_s)
            w = g["tok_window"]
            if len(self.tok) >= w + 5:  # skip the first steps (compile, probe)
                rate = statistics.median(self.tok[-w:])
                if rate < g["min_tok_frac"] * g["expected_tok_per_s"]:
                    raise GuardAbort("THROUGHPUT", f"step {step}: {rate:.0f} tok/s < {g['min_tok_frac']} x "
                                                   f"{g['expected_tok_per_s']}")

    def recent_mean(self, k: int = 20) -> float | None:
        return sum(self.losses[-k:]) / len(self.losses[-k:]) if self.losses else None


# ------------------------------------------------------------------------------------------------ mid-run
def quick_collapse(history: list[dict], base: dict | None, g: dict) -> str | None:
    """Accuracy-based arm monitor (dev NLL on seed-style references is not comparable across solution styles):
    abort when quick-eval accuracy is <= quick_collapse_frac x its step-0 value at quick_collapse_checks consecutive
    checks."""
    k = g.get("quick_collapse_checks", 2)
    if not base or not base.get("correct") or len(history) < k:
        return None
    lim = g.get("quick_collapse_frac", 0.5) * base["correct"]
    last = history[-k:]
    if all(h is not None and h["correct"] <= lim for h in last):
        return (f"QUICK_ACC_COLLAPSE accuracy {[h['correct'] for h in last]} <= {g.get('quick_collapse_frac', 0.5)} "
                f"x step-0 {base['correct']} at {k} consecutive checks")
    return None


def quick_stats(rows: list[dict]) -> dict:
    """rows: {correct, no_answer, repetition, truncated} per greedy item."""
    n = max(1, len(rows))
    return {k: round(sum(bool(r[k]) for r in rows) / n, 4) for k in ("correct", "no_answer", "repetition",
                                                                     "truncated")} | {"n": len(rows)}


def midrun_verdict(dev_loss: float | None, train_recent: float | None, quick: dict | None, base_quick: dict | None,
                   g: dict, dev0: float | None = None) -> list[str]:
    """Abort reasons (empty = continue).  Warnings: :func:`midrun_warnings`."""
    return _midrun(dev_loss, train_recent, quick, base_quick, g, dev0)[0]


def midrun_warnings(dev_loss, train_recent, quick, base_quick, g: dict, dev0=None) -> list[str]:
    return _midrun(dev_loss, train_recent, quick, base_quick, g, dev0)[1]


def _midrun(dev_loss, train_recent, quick, base_quick, g: dict, dev0=None) -> tuple[list[str], list[str]]:
    """dev_v1's references are GSM8K/MATH-style solutions: an arm trained on another solution style (Y+'s
    "**Bước 1:**" steps) raises their NLL while its accuracy improves (VI Y+ s0: dev 0.79 -> 0.90 at step 42, quick acc
    0.36 -> 0.52; EN Y+ s0: 0.78 -> 0.97, 0.61 -> 0.63).  So dev vs TRAIN loss is a warning; only a blow-up of dev
    loss against its own step-0 value (or non-finite) aborts."""
    out, warn = [], []
    if dev_loss is not None:
        if not math.isfinite(dev_loss):
            out.append("DEV_NONFINITE")
        elif dev0 and dev_loss > g.get("dev_blowup_x", 2.0) * dev0:
            out.append(f"DEV_BLOWUP dev {dev_loss:.4f} > {g.get('dev_blowup_x', 2.0)} x step-0 {dev0:.4f}")
        elif train_recent and dev_loss > g["max_dev_train_ratio"] * train_recent:
            warn.append(f"dev {dev_loss:.4f} > {g['max_dev_train_ratio']} x train {train_recent:.4f} (style shift?)")
    if quick:
        if quick["repetition"] > g["abort_repetition"]:
            out.append(f"FORMAT_REPETITION {quick['repetition']:.2f} > {g['abort_repetition']}")
        elif quick["repetition"] > g["max_repetition"]:
            warn.append(f"repetition {quick['repetition']:.2f} > {g['max_repetition']}")
        if quick["no_answer"] > g["abort_no_answer"]:
            out.append(f"FORMAT_NO_ANSWER {quick['no_answer']:.2f} > {g['abort_no_answer']}")
        if quick["correct"] == 0 and quick["n"] > 0:
            out.append(f"FORMAT_ZERO_ACC 0/{quick['n']}")
        if base_quick and quick["correct"] < base_quick["correct"] - g["max_acc_drop"] - 1e-9:
            warn.append(f"quick_acc {quick['correct']:.3f} < step-0 {base_quick['correct']:.3f} - {g['max_acc_drop']}")
    return out, warn
