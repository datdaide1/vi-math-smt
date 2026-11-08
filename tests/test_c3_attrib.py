"""D5 / C3: loaders reproduce the pilot reports, every absurd item is cause-coded, the identity control holds.

Reads data/pilot/ (G1-G4 outputs, read-only); skipped when the pilot data is absent."""
from pathlib import Path

import numpy as np
import pytest

from vi_math_verified_aug.analysis import c3_data as D
from vi_math_verified_aug.analysis.c3_absurd import CODING, FACTOR, summarize
from vi_math_verified_aug.analysis.c3_attrib import operator_fix, paired

pytestmark = pytest.mark.skipif(not (D.PILOT / "symbolic_v2" / "judge_opus.jsonl").exists(), reason="no pilot data")


@pytest.fixture(scope="module")
def A():
    return D.load_all()


def _by(items, v):
    return [i for i in items if i.variant == v]


def test_g4_counts_match_report_tables(A):
    # data/pilot/symbolic_v2/report_tables.md, table C2 (tag-passing) and C (N all parsed)
    g4 = A["G4"]
    assert {v: len(_by(g4, v)) for v in ("E", "R1", "R2", "N")} == {"E": 90, "R1": 90, "R2": 87, "N": 91}
    absurd = {v: sum(i.y["absurd"] for i in _by(g4, v)) for v in ("E", "R1", "R2", "N")}
    assert absurd == {"E": 10, "R1": 24, "R2": 22, "N": 5}
    teacher = {v: sum(1 - i.y["not_teacher"] for i in _by(g4, v)) for v in ("E", "R1", "R2", "N")}
    assert teacher == {"E": 33, "R1": 48, "R2": 43, "N": 73}


def test_g3_counts_match_g3_report(A):
    g3 = A["G3"]                                   # G3 §3: 139 tag-passing, 66 survivors, teacher 23/66
    assert len(g3) == 139
    surv = [i for i in g3 if i.survivor]
    assert len(surv) == 66
    assert sum(1 - i.y["not_teacher"] for i in surv) == 23
    assert sum(i.y["absurd"] for i in surv) == 8


def test_seed_baseline_matches_g3(A):
    s = D.seed_labels()
    assert len(s) == 60
    assert sum(1 - v["y"]["not_teacher"] for v in s.values()) == 10      # 17% [9-28]
    assert sum(v["y"]["fluency"] for v in s.values()) == 53              # 88%


def test_lab_v2_identical_to_g3_labeller():
    judge = pytest.importorskip("vi_math_verified_aug.pilot_symbolic.judge")
    rows = D.load_jsonl(D.PILOT / "symbolic" / "run_b" / "judge_opus.jsonl")
    for r in rows[:60]:
        if r.get("verdict"):
            assert D.lab_v2(r["verdict"]) == judge.lab_v2(r["verdict"])


def test_every_absurd_item_is_cause_coded(A):
    for v, items in (("G3", A["G3"]), *((v, _by(A["G4"], v)) for v in ("E", "R1", "R2", "N"))):
        s = summarize({v: items})[v]
        assert s["missing_coding"] == [], (v, s["missing_coding"])
    r1b = D.load_r1b()
    if r1b:
        assert summarize({"R1b": r1b})["R1b"]["missing_coding"] == []
    absurd_keys = {f"{i.variant}:{i.slot}" for i in A["G3"] + A["G4"] + r1b if i.y["absurd"]}
    assert set(CODING) == absurd_keys                                   # no stale codes either
    assert all(c in FACTOR for cs in CODING.values() for c in cs)


def test_operator_fix_identity_control(A):
    """Slots where G4 kept G3's program reuse G3's text and judge row -> labels must be identical, and the
    unchanged operators (M1/M2/M5) must show a zero per-slot difference."""
    r = operator_fix(A["G3"], _by(A["G4"], "E"), D.g3_mutations(), D.g4_mutations())
    assert r["all"]["same_program_label_agreement"] == 1.0
    un = r["unchanged ops M1/M2/M5"]
    assert abs(un["absurd"]["d"]) < 1e-9 and abs(un["tail"]["d"]) < 1e-9


def test_paired_uses_only_common_slots():
    mk = lambda slot, absurd, op="M2": D.Item("t", "x", "Y", "r", "m", "p", op, slot.split(":")[0], slot, "opus_v2",  # noqa: E731
                                            {c: 0 for c in D.ALL_OUTCOMES} | {"absurd": absurd})
    a = [mk("s1:1", 0), mk("s2:1", 0), mk("s3:1", 1)]
    b = [mk("s1:1", 1), mk("s2:1", 1), mk("s4:1", 1)]
    r = paired(a, b)
    assert r["n_pairs"] == 2
    assert r["absurd"]["d"] == pytest.approx(1.0) and r["absurd"]["c"] == 2
