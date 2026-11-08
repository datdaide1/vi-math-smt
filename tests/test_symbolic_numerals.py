"""Vietnamese numeral normalizer (T1 §2.1 rule (ii))."""
from fractions import Fraction as F

import pytest

from vi_math_verified_aug.symbolic.numerals import (contains_value, fmt_like, fmt_vi, int_to_words,
                                                    numbers_in_text, parse_literal, parse_number)

PARSE_CASES = [
    ("48", {F(48)}),
    ("1.000", {F(1), F(1000)}),            # ambiguous: VN grouping or EN decimal -> both readings
    ("1.234.567", {F(1234567)}),
    ("2,5", {F(5, 2)}),                    # VN decimal comma
    ("1.234,5", {F(2469, 2)}),
    ("0,125", {F(1, 8)}),
    ("0.10", {F(1, 10)}),
    ("$1.50", {F(3, 2)}),
    ("2/5", {F(2, 5)}),
    ("3 phần 4", {F(3, 4)}),
    ("20%", {F(20)}),
    ("20 phần trăm", {F(20)}),
    ("hai", {F(2)}),
    ("Mười", {F(10)}),
    ("mười lăm", {F(15)}),
    ("hai mươi mốt", {F(21)}),
    ("hai mươi tư", {F(24)}),
    ("một trăm linh năm", {F(105)}),
    ("ba nghìn", {F(3000)}),
    ("15 nghìn", {F(15000)}),
    ("1,5 triệu", {F(1500000)}),
    ("nửa", {F(1, 2)}),
    ("một nửa", {F(1, 2)}),
    ("hai rưỡi", {F(5, 2)}),
    ("trăm rưỡi", {F(150)}),
    ("một phần ba", {F(1, 3)}),
    ("ba phần tư", {F(3, 4)}),
    ("gấp đôi", {F(2)}),
    ("gấp 3 lần", {F(3)}),
    ("tá", {F(12)}),
    ("nửa tá", {F(6)}),
    ("hai chục", {F(20)}),
    ("\\dfrac{5}{6}", {F(5, 6)}),
    ("\\frac34", {F(3, 4)}),
    ("-7", {F(-7)}),
    ("con mèo", set()),
]


@pytest.mark.parametrize("lit,expected", PARSE_CASES)
def test_parse_literal(lit, expected):
    assert parse_literal(lit) == expected


def test_parse_number_prefers_grouping_for_three_digit_groups():
    assert parse_number("1.500") == 1500
    assert parse_number("1,200") == 1200
    assert parse_number("2,5") == F(5, 2)


def test_int_to_words_round_trip():
    for n in list(range(0, 1200)) + [2024, 10005, 999999, 1500000]:
        assert F(n) in parse_literal(int_to_words(n)), (n, int_to_words(n))


def test_int_to_words_standard_forms():
    assert int_to_words(21) == "hai mươi mốt"
    assert int_to_words(15) == "mười lăm"
    assert int_to_words(105) == "một trăm linh năm"
    assert int_to_words(24) == "hai mươi tư"


@pytest.mark.parametrize("v,orig,out", [
    (3, "hai", "ba"), (F(1, 3), "một nửa", "một phần ba"), (3, "gấp đôi", "gấp ba"),
    (F(9, 4), "1.50", "2.25"), (F(3, 20), "0.10", "0.15"), (F(5, 2), "2,5", "2,5"),
    (30, "20%", "30%"), (4, "gấp 3 lần", "gấp 4 lần"), (14, "Mười", "Mười bốn"),
])
def test_fmt_like_keeps_surface_style(v, orig, out):
    s = fmt_like(F(v), orig)
    assert s == out
    assert F(v) in parse_literal(s)


def test_fmt_vi_vietnamese_conventions():
    assert fmt_vi(F(2500)) == "2500"              # school texts: no grouping below 5 digits
    assert fmt_vi(F(12500)) == "12.500"
    assert fmt_vi(F(1250000)) == "1.250.000"
    assert fmt_vi(F(5, 2)) == "2,5"
    assert fmt_vi(F(1, 3)) == "1/3"


def test_numbers_in_text_digits_and_words():
    t = "Lan có 48 cái kẹo, hai mươi quả táo và một nửa số bánh. Giá là 5."
    ms = numbers_in_text(t, words=True)
    lits = [m.literal for m in ms]
    assert lits == ["48", "hai mươi", "một nửa", "5"]
    assert t[ms[-1].start:ms[-1].end] == "5"


def test_contains_value_all_surface_forms():
    assert contains_value("Còn lại 1.500 đồng", 1500)
    assert contains_value("được 2,5 kg", F(5, 2))
    assert contains_value("có hai mươi bạn", 20)
    assert not contains_value("tháng 4", 5)
