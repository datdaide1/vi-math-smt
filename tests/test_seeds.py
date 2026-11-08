"""S3 seed-pool unit tests (deterministic parts only; no API, no data files)."""
from vi_math_verified_aug.seeds.checks import deterministic_repair, g0, g1, number_diff
from vi_math_verified_aug.seeds.common import is_dev, seed_id
from vi_math_verified_aug.seeds.mta import mask, unmask


def test_seed_id_matches_pilot_definition():
    # pilot_gen/seeds.py: sha1(f"vi_gsm8k_train||{index}")[:12]
    import hashlib
    assert seed_id("vi_gsm8k_train", 5) == hashlib.sha1(b"vi_gsm8k_train||5").hexdigest()[:12]
    assert is_dev("0" * 12) and not is_dev("000000000001")


def test_escape_damage_repaired():
    r = {"dataset": "math", "problem_en": r"Find $\left(x\right)$ and $a\times b$.",
         "solution_en": r"$\boxed{1}$", "problem_vi": "Tìm $\\left(x\right)$ và $a\times b$.",
         "solution_vi": r"$\boxed{1}$", "gold_raw_vi": "1", "gold_raw_en": "1"}
    out, ch = deterministic_repair(r)
    assert out["problem_vi"] == r"Tìm $\left(x\right)$ và $a\times b$."
    assert {c["kind"] for c in ch} == {"escape_damage"}


def test_gsm8k_final_line_vn_thousands():
    r = {"dataset": "gsm8k", "problem_en": "x", "solution_en": "a\n#### 9000", "problem_vi": "x",
         "solution_vi": "a\n#### 9.000", "gold_raw_vi": "9000", "gold_raw_en": "9000"}
    out, ch = deterministic_repair(r)
    assert out["solution_vi"].endswith("#### 9000") and ch[0]["kind"] == "final_line_format"


def test_cr3_template_not_flagged_but_calque_is():
    en = "James has 7 more than 4 times the number of Oreos Jordan has. How many does James have?"
    good = "James có nhiều hơn 4 lần số bánh Oreo của Jordan là 7 chiếc. Hỏi James có bao nhiêu chiếc?"
    bad = "James có nhiều hơn 7 lần so với 4 lần số bánh Oreo của Jordan. Hỏi James có bao nhiêu chiếc?"
    sol_en, sol_vi = "#### 43", "#### 43"
    assert not [f for f in g1(good, sol_vi, en, sol_en, "43", "gsm8k") if "CR-3" in f]
    assert [f for f in g1(bad, sol_vi, en, sol_en, "43", "gsm8k") if "CR-3" in f]


def test_cr2_of_the_remaining():
    en = "There are 25 layers. Four of the remaining are quartz. How many are left?"
    bad = "Có 25 lớp. Bốn lớp còn lại là thạch anh. Còn bao nhiêu lớp?"
    good = "Có 25 lớp. Bốn trong số các lớp còn lại là thạch anh. Còn bao nhiêu lớp?"
    assert [f for f in g1(bad, "#### 1", en, "#### 1", "1", "gsm8k") if "CR-2" in f]
    assert not [f for f in g1(good, "#### 1", en, "#### 1", "1", "gsm8k") if "CR-2" in f]


def test_number_diff_coordinates_and_coins():
    assert number_diff("Find the distance between (0,4) and (3,0).", "Tìm khoảng cách giữa (0,4) và (3,0).",
                       "math") == ([], [])
    assert number_diff("Karen has 32 quarters.", "Karen có 32 đồng xu 25 xu.", "gsm8k") == ([], [])
    miss, add = number_diff("Salem is 15 times the size of Leesburg.", "Salem lớn gấp 8 lần Leesburg.", "gsm8k")
    assert miss == [15.0] and add == [8.0]


def test_g0_swallowed_dollar_amount():
    en = r"Ramon sells two tacos for $\$$2.50 and three for $\$$2.70."
    vi = "Ramon bán hai chiếc taco với giá $2.50 đô la và ba chiếc với giá $2.70 đô la."
    assert [f for f in g0(vi, "x", en, "x", "math") if "delimits prose" in f]


def test_mask_roundtrip():
    spans = {}
    t = r"Let $x=2$ and \[y=3\]. Then <<2*3=6>> \boxed{6}"
    m = mask(t, "math", spans)
    assert "$" not in m and "<<" not in m and unmask(m, spans) == t
