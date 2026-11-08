"""G4 arm-Y v2 helpers: IR localization (USD -> đồng, exact rescale or fallback), R2 IR view (never the answer
or a derived value), story-role filters, answer-outside-tags gate. Pure CPU, no API."""
import random
from fractions import Fraction

from vi_math_verified_aug.pilot_symbolic_v2 import roles as RL
from vi_math_verified_aug.pilot_symbolic_v2.localize import ir_view, keep_units, localize, money_map
from vi_math_verified_aug.pilot_symbolic_v2.pipeline import answer_untagged
from vi_math_verified_aug.symbolic import execute, mutate, tagged_text
from vi_math_verified_aug.symbolic.fixtures import GSM8K_IDS, all_fixtures
from vi_math_verified_aug.symbolic.units import QType

FIX = all_fixtures()


def test_localize_rescales_usd_exactly_and_answer_scales():
    P = FIX["gsm8k-1"]                          # 12 đô la/giờ, 50 phút -> 10 đô la
    PL, info = localize(P)
    assert info["rescaled"]
    assert execute(PL)[PL.query] == 10_000
    assert all(str(q.unit.lemma) != "đô la" for q in PL.quantities.values())
    assert "12.000" in money_map(PL, info)
    assert "phút" in keep_units(PL) or "giờ" in keep_units(PL)    # 60 phút/giờ conversion is kept


def test_localize_leaves_non_money_programs_alone():
    P = FIX["gsm8k-0"]
    PL, info = localize(P)
    assert PL is P and not info["rescaled"]


def test_localize_every_fixture_is_exact_or_falls_back():
    for sid in GSM8K_IDS:
        P = FIX[sid]
        PL, info = localize(P)
        a0, a1 = execute(P)[P.query], execute(PL)[PL.query]
        if not info["rescaled"]:
            assert PL is P
        else:
            q = PL.quantities[PL.query]
            assert a1 in (a0, a0 * 1000, a0 * 10, a0 / 1000), sid


def test_ir_view_shows_givens_but_never_derived_values_or_the_answer():
    for sid in GSM8K_IDS:
        P = FIX[sid]
        PL, _ = localize(P)
        v = ir_view(PL)
        for q in PL.givens():
            assert f'<q n="{q.qid}">' in v["facts"]
        ans = execute(PL)[PL.query]
        rel_q = v["relations"] + v["question"]
        # derived lines are formulas over [ids]; no "= <number>" for a derived node
        for line in v["relations"].splitlines():
            rhs = line.split(" = ", 1)[-1]
            assert "[" in rhs or line.startswith("- CHƯA_BIẾT") or line.startswith("- PHƯƠNG_TRÌNH")
        import re
        assert str(ans) not in re.sub(r"\[\d+\]", "", v["question"])
        assert "đáp" not in rel_q.lower()


def test_m3_role_filter_rejects_count_templates_on_rates_and_fees_on_unit_prices():
    from vi_math_verified_aug.symbolic.operators import TEMPLATES
    tpl = {t.verb_vi: t for ts in TEMPLATES.values() for t in ts}
    P = FIX["gsm8k-1"]
    rate = next(q for q in P.givens() if q.qtype in (QType.RATE, QType.MONEY))
    ok, why = RL.m3_template_ok(rate, tpl["mua thêm"])
    assert not ok
    from dataclasses import replace
    price = replace(rate, label_vi="giá mỗi quả trứng", qtype=QType.MONEY)
    assert not RL.m3_template_ok(price, tpl["trả thêm phí"])[0]
    assert RL.m3_template_ok(price, tpl["được giảm giá"])[0]
    wage = replace(rate, label_vi="số tiền kiếm được", qtype=QType.MONEY)
    assert not RL.m3_template_ok(wage, tpl["phải trả thêm thuế"])[0]


def test_role_templates_never_widen_x4_library():
    for sid in GSM8K_IDS:
        for q in FIX[sid].quantities.values():
            assert set(t.verb_vi for t in RL.role_templates(q)) <= set(t.verb_vi for t in RL.ORIG_TEMPLATES(q))


def test_det_check_and_describe_run_on_real_mutations():
    for sid in GSM8K_IDS:
        P = FIX[sid]
        for op in ("M3", "M4", "M5"):
            r = mutate(op, P, random.Random(7))
            if not r.ok:
                continue
            ok, why = RL.det_check(op, P, r.program, r.edits)
            assert isinstance(ok, bool)
            d = RL.describe(op, P, r.program, r.edits)
            assert d.strip()


def test_answer_untagged_ignores_tagged_givens_and_day_names():
    t = 'Lan có <q n="1">4</q> thỏi son. Thứ Ba Lan mua thêm <q n="2">5</q> cái. Hỏi Lan còn bao nhiêu?'
    assert not answer_untagged(t, Fraction(4))
    assert not answer_untagged(t, Fraction(3))
    assert answer_untagged(t + " Biết rằng 9 là số cần tìm.", Fraction(9))
