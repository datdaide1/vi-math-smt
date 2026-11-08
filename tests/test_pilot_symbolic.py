"""G3 arm-Y pilot helpers: editor view (never shows the answer), tag check, gates. Pure CPU, no API."""
import json
import random
from fractions import Fraction

from vi_math_verified_aug.pilot_symbolic.editor import check_output, editor_view, final_text
from vi_math_verified_aug.pilot_symbolic.gates import Deduper, item_gates, jaccard, shingles
from vi_math_verified_aug.symbolic import execute, mutate, tagged_text
from vi_math_verified_aug.symbolic.fixtures import GSM8K_IDS, all_fixtures

FIX = all_fixtures()
PARTNERS = [FIX[k] for k in GSM8K_IDS]


def _first(op, P):
    for s in range(30):
        r = mutate(op, P, random.Random(s), **({"partners": PARTNERS} if op == "M6" else {}))
        if r.ok:
            return r
    return None


def test_editor_view_numbers_sentences_and_hides_the_answer():
    for sid in GSM8K_IDS:
        P = FIX[sid]
        for op in ("M2", "M3", "M4", "M5"):
            r = _first(op, P)
            if r is None:
                continue
            v = editor_view(P, r.program, r.edits, op)
            assert v["text_tagged"] == tagged_text(P)
            assert v["sentences"].startswith("[0] ")
            givens = {q.qid: q.value for q in r.program.givens()}
            for e in json.loads(v["edits"]):
                assert "answer" not in e
                if e.get("value") is not None and e["type"] in ("STATE_FACT", "INSERT_STEP"):
                    assert e["q"] in givens          # every value shown to the editor belongs to a given


def test_m6_view_marks_the_reference_and_tag_check_accepts_partner_numerals():
    for sid in GSM8K_IDS:
        P = FIX[sid]
        r = _first("M6", P)
        if r is None:
            continue
        v = editor_view(P, r.program, r.edits, "M6")
        assert v["text_tagged"].count("<ref>") == 1
        assert "first_text" not in v["edits"]
        partner = next(x for x in PARTNERS if x.seed_id == r.program.meta["partner"])
        # an output that keeps the <ref> tag fails; removing the ref literal and keeping all tags passes
        assert "leftover <ref> tag" in check_output(v["text_tagged"], P, r.program, "M6", partner)
        import re
        ok = re.sub(r"<ref>.*?</ref>", "số đó", v["text_tagged"])
        assert check_output(ok, P, r.program, "M6", partner) == []
        assert "<q" not in final_text(ok)
        return
    raise AssertionError("no M6 on fixtures")


def test_gates_and_dedup():
    seed = ("Lan có 3 quả táo và mẹ cho Lan thêm một rổ cam rất to vào buổi sáng hôm qua ở chợ quê. "
            "Hỏi buổi chiều hôm đó Lan có tất cả bao nhiêu quả táo trong nhà?")
    g = item_gates(seed.replace("3 quả", "5 quả"), seed, Fraction(5), "M2")
    assert g["answer_in_text"] and g["leak_ok"] and not g["chrf_ok"]
    assert item_gates("đáp án là 7", "x", Fraction(9), "M1")["leak_regex"]
    d = Deduper()
    t = "Một cửa hàng có 12 cái bánh, bán 5 cái. Hỏi còn bao nhiêu cái bánh?"
    assert d.check_and_add("a:1", "a", t)[0]
    assert not d.check_and_add("b:1", "b", t)[0]
    assert jaccard(shingles(t), shingles(t)) == 1.0
