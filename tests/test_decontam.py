"""X5: decontamination tooling (normalisation, n-gram index, MinHash LSH, gate). numpy-only; no torch."""
import numpy as np
import pytest

from vi_math_verified_aug.eval import decontam as dc

Q1 = ("Vịt của Janet đẻ 16 quả trứng mỗi ngày. Cô ấy ăn ba quả vào bữa sáng mỗi sáng và làm bánh nướng xốp cho bạn "
      "bè mỗi ngày với bốn quả. Cô ấy bán số trứng còn lại tại chợ nông sản hàng ngày với giá 2 đô la cho mỗi quả "
      "trứng vịt tươi. Hỏi mỗi ngày cô ấy kiếm được bao nhiêu đô la tại chợ nông sản?")
Q2 = ("Một chiếc áo choàng cần 2 cuộn sợi màu xanh và một nửa số đó sợi màu trắng. Hỏi tổng cộng cần bao nhiêu "
      "cuộn sợi để may chiếc áo choàng?")
Q3 = r"Giải phương trình \(x^2 - 5x + 6 = 0\) và tính tổng các nghiệm của phương trình đã cho trong tập số thực."


# ---------------------------------------------------------------- normalisation
def test_nfc_and_lowercase():
    nfd = "Đường thẳng".encode("utf-8").decode("utf-8")
    import unicodedata
    assert dc.tokenize(unicodedata.normalize("NFD", nfd)) == dc.tokenize(nfd) == ["đường", "thẳng"]


def test_strip_diacritics():
    assert dc.strip_diacritics("Đường thẳng đi qua điểm") == "Duong thang di qua diem"
    assert dc.tokenize("Đường thẳng", diacritics=False) == ["duong", "thang"]


@pytest.mark.parametrize("raw,canon", [
    ("2,5", "2.5"), ("2{,}5", "2.5"), ("1.000.000", "1000000"), ("10 000", "10000"),
    ("1.234,5", "1234.5"), ("0.125", "0.125"), ("3.14", "3.14"),
])
def test_number_canon(raw, canon):
    assert dc.tokenize(raw) == [canon]


def test_latex_canon_equivalences():
    a = dc.normalize(r"Tính $\dfrac{1}{2} \cdot x \le 3$")
    b = dc.normalize(r"Tính \(\frac{1}{2}\times x\leq 3\)")
    c = dc.normalize("Tính frac 1 2 × x ≤ 3")
    assert a == b == c


def test_html_entities_and_tags():
    assert dc.tokenize("<p>x &gt; 0</p>") == ["x", ">", "0"]


def test_template_variant_masks_numbers():
    t = dc.tokenize("Có 16 quả trứng, giá 2,5 đô", template=True)
    assert t == ["có", "#", "quả", "trứng", "giá", "#", "đô"]


def test_variant_tokens_consistent():
    vt = dc.variant_tokens(Q1)
    assert vt["exact"] == dc.tokenize(Q1)
    assert vt["nodiac"] == dc.tokenize(Q1, diacritics=False)
    assert vt["template"] == dc.tokenize(Q1, template=True)


# ---------------------------------------------------------------- n-gram hashing / index
def _th(text, variant="exact"):
    return dc.token_hashes(dc.variant_tokens(text)[variant])


def test_ngram_hashes_deterministic_and_length_mixed():
    th = _th(Q1)
    assert np.array_equal(dc.ngram_hashes(th, 13), dc.ngram_hashes(th.copy(), 13))
    assert len(dc.ngram_hashes(th, 13)) == len(th) - 12
    assert len(dc.ngram_hashes(th[:5], 13)) == 0
    # the same leading tokens hashed as a 5-gram vs a 6-gram must differ
    assert dc.ngram_hashes(th, 5)[0] != dc.ngram_hashes(th, 6)[0]


def _index(docs, n=13, variant="exact", **kw):
    ix = dc.NgramIndex(n, **kw)
    for i, d in enumerate(docs):
        ix.add(i, _th(d, variant))
    return ix.finalize()


def test_exact_containment_full_coverage():
    ix = _index([Q1, Q2, Q3])
    corpus_doc = "Bài 5. " + Q2 + " Lời giải: cần 3 cuộn."
    res = ix.query(_th(corpus_doc))
    assert set(res) == {1}
    assert res[1] == ix.n_grams[1]          # every 13-gram of Q2 found → coverage 1.0


def test_no_false_hit_on_unrelated():
    ix = _index([Q1, Q2])
    assert ix.query(_th(Q3)) == {}


def test_partial_overlap_coverage():
    ix = _index([Q1])
    half = " ".join(Q1.split()[: len(Q1.split()) // 2])
    res = ix.query(_th(half + " rồi đến một đoạn hoàn toàn khác không liên quan gì cả"))
    cov = res[0] / ix.n_grams[0]
    assert 0.2 < cov < 0.8


def test_diacritic_stripped_copy_caught_only_by_nodiac():
    nodiac_copy = dc.strip_diacritics(Q1)
    assert _index([Q1]).query(_th(nodiac_copy)) == {}
    ix = _index([Q1], variant="nodiac")
    res = ix.query(_th(nodiac_copy, "nodiac"))
    assert res[0] == ix.n_grams[0]


def test_value_perturbed_copy_caught_only_by_template():
    pert = dc.perturb_numbers(Q1, seed=3)
    assert pert != Q1
    ex = _index([Q1])
    tm = _index([Q1], variant="template")
    r_ex = ex.query(_th(pert))
    r_tm = tm.query(_th(pert, "template"))
    assert r_tm[0] == tm.n_grams[0]
    assert r_ex.get(0, 0) < ex.n_grams[0]


def test_short_item_whole_sequence_match():
    short = "Tính giá trị của biểu thức 2 + 3 * 4"      # 10 tokens < 13
    ix = _index([short])
    assert ix.n_grams[0] == 1 and ix.short_lens == {len(dc.tokenize(short))}
    assert ix.query(_th("Câu 1: " + short + " bằng bao nhiêu?")) == {0: 1}
    assert ix.query(_th("Tính giá trị của biểu thức 2 + 3 * 5")) == {}


def test_sharded_disk_index_matches_memory(tmp_path):
    docs = [Q1, Q2, Q3] * 3
    mem = _index(docs)
    disk = _index(docs, shard_grams=40, workdir=tmp_path / "w")
    assert len(disk.shards) > 1
    q = _th(Q2 + " " + Q3)
    assert mem.query(q) == disk.query(q)
    disk.save(tmp_path / "saved")
    loaded = dc.NgramIndex.load(tmp_path / "saved")
    assert loaded.query(q) == mem.query(q)
    assert isinstance(loaded.shards[0].keys, np.memmap)


def test_contains_any():
    ix = _index([Q1, Q2])
    assert ix.contains_any(_th("x " + Q2))
    assert not ix.contains_any(_th(Q3))


# ---------------------------------------------------------------- MinHash
def test_minhash_estimate_close_to_exact():
    mh = dc.MinHasher(num_perm=256)
    t1 = dc.tokenize(Q1)
    t2 = t1[:40] + dc.tokenize("thêm một vài từ mới ở giữa câu này") + t1[40:]
    est = mh.jaccard(mh.signature(dc.token_hashes(t1)), mh.signature(dc.token_hashes(t2)))
    ex = dc.exact_jaccard(t1, t2)
    assert abs(est - ex) < 0.1


def test_minhash_identical_and_disjoint():
    mh = dc.MinHasher()
    s1 = mh.signature(_th(Q1))
    assert mh.jaccard(s1, mh.signature(_th(Q1))) == 1.0
    assert mh.jaccard(s1, mh.signature(_th(Q3))) < 0.1


def test_lsh_finds_near_duplicate_not_unrelated():
    mh = dc.MinHasher()
    lsh = dc.MinHashLSH(bands=32)
    for i, d in enumerate([Q1, Q2, Q3]):
        lsh.insert(i, mh.signature(_th(d)))
    near = Q2.replace("chiếc áo choàng", "cái áo choàng")
    res = lsh.query(mh.signature(_th(near)), threshold=0.5)
    assert res and res[0][0] == 1
    assert lsh.query(mh.signature(_th("Một câu hoàn toàn khác về hình học không gian và thể tích khối chóp")), 0.5) == []


# ---------------------------------------------------------------- scanner + gate (no encoder)
def test_scanner_end_to_end():
    targets = {"t": [dc.Doc("a", Q1), dc.Doc("b", Q2), dc.Doc("c", Q3)]}
    sc = dc.Scanner(targets)
    corpus = [dc.Doc("x0", "không liên quan " * 10), dc.Doc("x1", "Đề: " + Q2 + " Đáp án 3"),
              dc.Doc("x2", dc.perturb_numbers(Q1, 7)), dc.Doc("x3", dc.strip_diacritics(Q3))]
    sc.feed(corpus, progress_every=0)
    r = {x["id"]: x for x in sc.results()}
    assert r["b"]["ng13_exact"]["cov"] == 1.0 and r["b"]["ng13_exact"]["doc"] == "x1"
    assert r["a"]["ng13_template"]["cov"] == 1.0 and r["a"]["ng13_template"]["doc"] == "x2"
    assert r["c"]["ng13_nodiac"]["cov"] == 1.0 and r["c"]["ng13_exact"]["cov"] == 0.0
    f = dc.hit_flags(r["b"], dc.Thresholds())
    assert f["ng13_any"] and f["ng13_cov50"] and f["minhash50"]
    s = dc.summarize(sc.results(), dc.Thresholds())
    assert s["t"]["n"] == 3 and s["t"]["ng13_cov50"] >= 1 and s["t"]["ng13_template_cov50"] >= 2


def test_gate_g4_g5():
    g = dc.Gate([dc.Doc("t1", Q2)])
    leak = g.check("Bài toán: " + Q2)
    assert not leak["pass"] and leak["g4_ngram"]["test_id"] == "t1"
    leak2 = g.check(dc.strip_diacritics(Q2))
    assert not leak2["pass"] and leak2["g4_ngram"]["variant"] == "nodiac"
    ok = g.check(Q1)
    assert ok["pass"]
    g.accept(0, ok)
    # one-word edit at the end of a 70-token item: Jaccard(5-shingles) ≈ 0.93 ≥ 0.8 → G5 duplicate
    dup = g.check(Q1.replace("tại chợ nông sản?", "tại chợ nông sản đó?"))
    assert not dup["pass"] and dup["g5_dup"]["accepted_id"] == 0
    # a different item with J < 0.8 is accepted
    assert g.check(Q3)["pass"]


# ---------------------------------------------------------------- page extraction
def test_extract_exam_statement_loigiaihay_layout():
    page = ('<div id="box-content"><div class="box-question" id="sub-question-1"><p><strong>Đề bài</strong></p>'
            '<p><strong>Câu 1 (2 điểm):</strong></p><p>a) Rút gọn \\(A = \\sqrt{28}\\)</p>'
            '<p><strong>Câu 2:</strong></p><p>Giải hệ phương trình gồm hai phương trình bậc nhất hai ẩn sau</p>'
            '</div><div id="sub-question-2"><p>Lời giải</p><p>Câu 1: A = 2</p></div></div>')
    lines = dc.extract_exam_statement(page)
    assert any("Rút gọn" in ln for ln in lines) and not any("A = 2" in ln for ln in lines)
    units = dc.segment_questions(lines)
    assert len(units) == 2 and units[0].startswith("Câu 1")


def test_variant_hashes_equal_token_hashes_of_variants():
    for text in (Q1, Q3, "Đường thẳng 2,5 và 1.000 đồng", ""):
        vh = dc.variant_hashes(text)
        vt = dc.variant_tokens(text)
        for v in dc.VARIANTS:
            assert np.array_equal(vh[v], dc.token_hashes(vt[v])), v


def test_bloom_prefilter_does_not_lose_hits():
    docs = [Q1, Q2, Q3]
    ix = _index(docs)
    assert all(ix._bloom_test(dc.ngram_hashes(_th(d), 13)).all() for d in docs)
    for d in docs:
        assert np.array_equal(ix.query_grams(_th(d)), ix.query_grams(_th(d), prefilter=False))


def test_minhash_bound_skips_long_docs_but_keeps_near_dups():
    sc = dc.Scanner({"t": [dc.Doc("b", Q2)]})
    assert sc._maybe_signature(_th(Q2)) is not None
    assert sc._maybe_signature(_th(Q1 + " " + Q3 + " " + Q2)) is None     # |A| >> |B| → J < 0.3 impossible


def test_literal_newline_unescaped_but_latex_kept():
    assert dc.unescape_newlines("a\\nGiải \\neq b \\na) c") == "a\nGiải \\neq b \na) c"
    assert dc.tokenize("Bài 1\\nGiải phương trình") == ["bài", "1", "giải", "phương", "trình"]
    assert "≠" in dc.tokenize(r"$x \neq 1$")


def test_problem_segments():
    page = ("xin giới thiệu lời giải chi tiết bài 7 trang 61 SGK Toán 10.\\nChúng tôi cung cấp lời giải.\\n\\n"
            "Đề bài\\nGiải các bất phương trình sau:\\na) \\(2{x^2} + 3x + 1 \\ge 0\\)\\n"
            "Phương pháp giải - Xem chi tiết\\nGiải bất phương trình bậc hai.\\n"
            "Bài 2\\nTìm tập nghiệm của phương trình x^2 - 4 = 0 trên tập số thực.\\nLời giải\\nx = 2 hoặc x = -2")
    segs = dc.problem_segments(page)
    assert segs[0].startswith("Giải các bất phương trình sau") and "Phương pháp" not in segs[0]
    assert any(x.startswith("Bài 2 Tìm tập nghiệm") for x in segs)
    assert not any("x = 2 hoặc" in x for x in segs)


@pytest.mark.parametrize("raw,body", [
    ("Câu 3 (1,5 điểm): Cho phương trình x^2 = 4", "Cho phương trình x^2 = 4"),
    ("Câu 2. (0,5 điểm) Vẽ đồ thị", "Vẽ đồ thị"),
    ("Bài 1: Rút gọn biểu thức", "Rút gọn biểu thức"),
    ("Câu 21:  Vẽ đồ thị của hàm số", "Vẽ đồ thị của hàm số"),
    ("Cho tam giác ABC", "Cho tam giác ABC"),
])
def test_strip_question_header(raw, body):
    assert dc.strip_question_header(raw) == body


def test_gate_catches_value_perturbed_test_item():
    g = dc.Gate([dc.Doc("t1", Q1)])
    pert = dc.perturb_numbers(Q1, seed=11)
    r = g.check(pert)
    assert not r["pass"] and r["g4_ngram"]["test_id"] == "t1"
    assert dc.Gate([dc.Doc("t1", Q1)], template_cov=None).check(Q3)["pass"]
