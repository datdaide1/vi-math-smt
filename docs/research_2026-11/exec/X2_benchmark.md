# X2: Vi-ExamMath collection pipeline and exam collection (2026-11-07)

Workstream X2 of `EXEC_BRIEF2.md`. Contribution C1 of `01_TECHNICAL_DESIGN.md`: native Vietnamese grade-10 entrance
exams, keys checked by a solver, a "symbolically certifiable or not" tag, and a temporal split.

**Bottom line.** 270 official/school Toán vào 10 exams (58 provinces, exam years 2017–2026) → **4,209 items** →
**1,584 Core candidates** (target ≥ 500: **met 3×**). 1,062 Core candidates are tagged certifiable = yes.
Post-cutoff Core: **112 items from 2026 exams** (103 official + 9 minh hoạ) and **333 from 2025+**.
Agreement where two keys can be compared: SymPy vs the page-solution key 96.6 % (396/410); SymPy vs blind Claude
97.5 % (157/161); page-solution vs blind Claude 86.4 % (273/316, plus 58 partial). Claude spend: **$2.497 (Batch)**.

**Update (Addendum A, same day).** Every Core item now has a blind Claude key (1,585/1,585; extra $6.34, total
$8.84).
- Key status: 3-agree 385 · 2-agree 569 · conflict 325 · fewer than 2 keys 306. The QA queue is re-frozen with
  conflicts first.
- Addendum B lists candidate sources for the 2024 / HSG / chuyên gaps.
- §5–§6 below are the first-pass numbers.

The pipeline is new code written from 04 §3, T5 §3 and C1. The stopped agent's files and the `_scratch` prototype were
read only to learn their failure modes. They are archived, not deleted, in `data/vi_exam/_superseded/`. The only thing reused is the cached raw HTML, as data.

---

## 1. Pipeline (all under `vi_math_verified_aug/vi_exam/`)

| Step | Module / command | What it does |
|---|---|---|
| fetch layer | `crawl.py` | `Fetcher`: robots.txt checked per host and fail-closed. ≥ 2.0 s between requests (raised to any robots Crawl-delay). Generic UA `Mozilla/5.0 (compatible; ViExamMath-research-crawler/1.0; non-commercial academic research)` with no personal data. Cache `data/vi_exam/raw/<host>/<sha1(url)>.html` + `.json` (url, final_url, status, `fetched_at` UTC, body sha1). `fetch_log.jsonl` appends every network fetch. Backs off on 429/503. |
| discover | `python -m vi_math_verified_aug.vi_exam.discover` | Two routes, merged: (a) the hub `/de-thi-vao-10-mon-toan-c949.html` → 62 province/school listings → `-a<id>` articles; (b) **all 119 monthly article sitemaps 2017-01…2026-11** (`sitemap_articles.xml` → `monthly-detail-m<M>-y<Y>.xml`), filtered by slug. Mock/compilation listings ("đề tham khảo", "tổng hợp 50 đề") are recorded but not collected. Output: `exam_index.jsonl` (274 candidates; 268 found by both routes, 4 only by sitemaps, 2 "đề số 1/2"). |
| collect | `python -m vi_math_verified_aug.vi_exam.collect all` | Exam pages (274/274 available; 4 new network fetches). Wayback CDX earliest capture → `first_seen.jsonl`. |
| build (offline) | `python -m vi_math_verified_aug.vi_exam.build` | page → DOM blocks (`page.py`) → question tree + solution alignment (`segment.py`) → tags (`tagging.py`) → provisional key (`answers.py`) → SymPy key + key comparison (`symkey.py`) → `items.jsonl`, `items_private.jsonl`, `qa_queue.jsonl`, `exams.jsonl`, `coverage.csv`, `summary.json`, `official_sources.jsonl`. Streams one page at a time and appends JSONL (fits the ~0.9 GB free-RAM limit). About 95 s. |
| Claude key | `claude_key.py submit/collect` (run with `E:\venv_tools`) | Blind solve by Batch API, `claude-opus-5-5`, effort `low`; merged by re-running build. |
| test | `.venv/Scripts/python.exe -m pytest tests/test_vi_exam_segment.py -q` | **5 passed** (§3). |

## 2. Crawl policy evidence

- `loigiaihay.com/robots.txt` (cached): `User-agent: * / Allow: /`. Disallowed are only rating/feedback AJAX paths and one
  article. 0 URLs were blocked by robots.
- This session made **255 network requests**, all HTTP 200: loigiaihay 124 (sitemaps 120 + 4 exam pages),
  web.archive.org CDX 127, hanoi.edu.vn 4. The busiest minute had **30 requests**, i.e. 1 per 2 s, and only one
  crawler process ran at a time. The four pairs of identical second-resolution timestamps in `fetch_log.jsonl` are
  completion times of back-to-back CDX calls. Request starts are always ≥ 2 s apart (`Fetcher._sleep_turn`).
- The 270 exam pages were cached earlier the same day by the stopped agent, all with `fetched_at`. Its UA string had
  no personal email ("contact: academic research project vi-math-smt").
- **Hà Nội Sở GD-ĐT**: robots disallows `/tai/` (the download path). 4 requests (robots, homepage, the exam-office
  section `https://hanoi.edu.vn/phong-quan-ly-thi-va-kdcl`, site search) found no static links to the Toán vào 10
  text or đáp án. As instructed, nothing was downloaded. `official_sources.jsonl` records, for each of the 9 Hà Nội
  exams, the loigiaihay URL plus the official site/section URLs with `status: manual_download_required`.

## 3. Segmentation (fixes R6 X-SEG-COLLISION)

- Boundaries come from **DOM leaf blocks** (`p`, `li`, `td`, tables; `<br>` splits lines), in document order.
- Questions are kept as a **list**. An item's identity is its position: `<exam>.s<section>q<index>[.<sub>[.<subsub>]]`.
  The printed "Câu N" is only metadata, so numbering that restarts in PHẦN I/II/III cannot overwrite anything.
- Sub-questions `a)`, `a.`, `1)`, `1.1` (two levels), including several on one line (`a.\(…\) b.\(…\) c.\(…\)`),
  become **separate items with `parent_id`**. Dependent parts (`ở câu a`) get the earlier statement but not its answer
  (T5). MCQ options are attached to their question, including options written inline in one paragraph.
- Solutions are aligned to questions by a **monotonic DP over text similarity** (+0.25 when the numbers match), not by
  label. "Bài IV" vs "Câu 4" and restarted numbering cannot mis-join. 2,469 of 2,690 questions aligned (91.8 %).
  The worked part (after "Cách giải") is then split by the problem's own sub-labels.
- **Test** `tests/test_vi_exam_segment.py`:
  - `test_legacy_dict_loses_repeated_numbering` reproduces the prototype's dict logic verbatim on a synthetic
    2026-format exam: 7 questions → 3 kept, and PHẦN II Câu 1 is overwritten.
  - `test_dom_segmentation_keeps_every_question`: 3/2/2 questions, 13 leaves, T/F statements linked to their parent.
  - `test_solution_alignment_with_restarted_numbers`: PHẦN III solutions join PHẦN III, not PHẦN I.
  - `test_real_page_lao_cai_2026_keeps_22_questions` uses the cached live page. The legacy logic keeps **12**, the
    figure R6 reported. The new segmenter keeps **12 + 4 + 6 = 22** questions and 16 T/F sub-statements.

## 4. Item schema (`items.jsonl`, one JSON per line)

`id, parent_id, exam_id, question_label, sub_label, section{ordinal,type,title}, position, points`.
`problem` (context stem + this part + options, LaTeX kept verbatim as MathJax `\( \)`), `problem_own`, `context`,
`options`.
`has_figure, n_images, has_table`.
`answer_type` ∈ {numeric, expression, solution_set, parameter_condition, mcq, true_false, proof, mixed_proof,
construction, open}.
`topic`, `topics`, `symbolically_certifiable` ∈ {yes, maybe, no}.
`core_candidate`, `core_exclusion`.
`provisional_answer{value, value_final, method, confidence}`.
`sympy_key{key_type, values, …}`, `claude_blind_key{answer, model, batch_id}`, `key_agreements{…}`, `key_status`.
`provenance{site, source_url, solution_url, page_title, province, school, exam_year, exam_kind, issuer_type,
official_mark, header_issuer, exam_date, exam_date_note, fetched_at, first_seen, first_seen_evidence{wayback_first,
sitemap_lastmod, og_image_date}, hanoi_official_source}`.
`license_basis`, `release_eligible`, `temporal{exam_2026, exam_2025_plus, first_seen_2026_06_plus}`, `dup_of`, `canary`
(GUID), `qa{status, flags, keys{provisional_solution, sympy, claude_blind, official}}`.

- **Third-party solution text appears only in `items_private.jsonl`** (`qa_private.solution_text`, `conclusion`,
  `align_sim`). Do not release it (04 §3.3). The public file holds only the extracted provisional answer value.
- `exam_year` comes from the paper's printed header ("NĂM HỌC 2026 – 2027"), falling back to the title. Three
  title/header mismatches are flagged. Example: the article titled "Đồng Tháp năm 2021" is the 2026 paper (header +
  June-2026 sitemap).
- `exam_date`: only one page prints it; otherwise null with a note.
- `first_seen` = min(Wayback earliest 200 capture, sitemap `lastmod`, og:image upload date). Each of these is an
  upper bound on when the text became public.
- `license_basis`: state issuer → Điều 15(2) + NĐ 17/2023 Điều 8(2) reading (releasable with provenance). University
  schools (PTNK, KHTN) → research use only, not releasable.
- `symbolically_certifiable`:
  - yes = equation / system / quadratic / Vi-ét / inequation / function or expression value, gradable, no geometry or
    proof;
  - maybe = word problems, probability/statistics, extremum, number theory;
  - no = proof, synthetic geometry, construction, figure, open.

## 5. Counts

| | Items | Figure-free | Core candidates | Certifiable yes / maybe / no (Core) |
|---|---|---|---|---|
| All exams (270) | **4,209** | **3,732** | **1,584** | **1,062 / 411 / 111** |
| Exam year 2026 | 482 | 359 | **112** (official 103, minh hoạ 9) | 59 / 45 / 8 |
| Exam year ≥ 2025 | 1,183 | 896 | **333** | 188 / 125 / 20 |
| first_seen ≥ 2026-06-01 | — | — | **86** | 47 / 34 / 5 |

- All items by certifiable: yes 1,089 / maybe 1,016 / no 2,104.
- All items by answer type: MCQ 1,170, numeric 1,106, proof 724, solution_set 318, open 299,
  parameter_condition 205, expression 179, construction 136, true_false 72.
- Why non-Core items are excluded: MCQ 939, proof 715, figure 477, open 288, construction 120, T/F 48,
  exact duplicate 23, near-duplicate (5-gram Jaccard ≥ 0.8) 6, multi-target (two asked values in one part; to be split
  in QA) 9.
- Core by year: 2017 13 · 2018 131 · 2019 270 · 2020 241 · 2021 273 · 2022 35 · 2023 288 · **2024 0** · 2025 221 ·
  2026 112.
- Core by topic: expression 341, quadratic 317, word problem 185, plane geometry (text-only "Tính…") 181, Vi-ét 172,
  system 149, probability/statistics 43, function 37, extremum 36, solid geometry 30, other 49.
- Core issuer: state 1,573 (release-eligible) and university 11. Kind: official 1,529, minh hoạ 44, school 11.
- 2026 papers are MCQ/T-F heavy (172 MCQ + 40 T/F of 482 items) and have more figures (123), which is why they yield
  fewer Core items.

## 6. Answer keys and agreement

**Key 1, provisional answer** (from the page's worked solution, private):
- Methods over Core: `vay_math` 1,082 · `vay_text` 243 · `last_math` 184 · `short_answer_line` 3 · none 72.
- 1,512 / 1,584 Core items have one.
- MCQ letters come from key tables or "Chọn X" (850 of 939 figure-free MCQ).
- Manual read of a random 20 Core items (seed 7): **segmentation and problem text correct 20/20**, provisional answer
  correct about 16/20. Errors: partial chains, inequality answers, and an equality case taken instead of the min value.
  The provisional key is therefore never trusted alone.

**Key 2, SymPy** (`symkey.py`; every root is verified by substitution and the stated domain is respected):
- 438 Core items have a key: solve_system 117, eval_expression 114, solve_equation 110,
  simplify_equivalence 84, solve_for_value 9, vieta_eval 4.
- The LaTeX → SymPy translator is strict and uses Vietnamese conventions: decimal comma, `.` = multiplication,
  real odd roots, `x_{1}` symbols.

**Key 3, Claude blind solve** (no solution, no provisional, no program):
- Batch `msgbatch_01QthrjRiHX9wNDmqBmtfyBw` (pilot, 20) and `msgbatch_01Ue7YybFZADW1vjrRKyD8yp` (main, 454).
- **474 items, all 333 post-2025 Core items included**. 474/474 succeeded, 473 with an `ĐÁP ÁN:` line.
- 182,250 input + 213,241 output tokens. **$2.497** at batch prices ($2 / $10 per MTok); per item $0.0053.
- Logs: `logs/api/X2/*_requests.jsonl`, `*_results.jsonl`, `cost_ledger.jsonl`. No NIM or Gemini calls.
- The server-side refusal-fallback parameter was not set. All 474 returned `end_turn`, so no refusals occurred.

| Pair | agree | disagree | partial | unparsed / no key | **agreement rate** (agree / (agree + disagree)) |
|---|---|---|---|---|---|
| provisional vs SymPy | 396 | 14 | — | 28 / 1,146 | **96.6 %** |
| SymPy vs Claude | 157 | 4 | — | 5 / 308 | **97.5 %** |
| provisional vs Claude | 273 | 43 | 58 | 100 | **86.4 %** |

- Who was wrong:
  - Reading the disagreements shows the provisional extractor is usually the wrong key: it captures an equality
    case, part of a two-quantity answer, or the wrong sub-part.
  - The 4 SymPy-vs-Claude conflicts are source-text issues. In 2 of them the LaTeX literally says
    `\sqrt {x + 2}` where both humans' solution and Claude read `\sqrt x + 2`. The other 2 are a multi-part ask
    without markers and one SymPy definition mis-pick, now guarded.
- **Key status over Core** (`key_status`): consensus_3 **124**, consensus_2 **442**, conflict **51**, unverified 967.
- **QA queue** (`qa_queue.jsonl`, 1,584 rows, T5 §3.1 stages):
  - stage 4 human adjudication: 51 (conflicts);
  - stage 3 LLM critique: 566 (consensus);
  - stage 2 more keys: 967;
  - the 150-item random audit (stage 5) is to be sampled from the stage-3 set.
- Not available: an official key (Sở PDFs, manual), and the Claude program-writing key (T5 key ii). The brief allowed
  only the blind solve.

## 7. Coverage (cell = exams / Core items; `·` = no exam on loigiaihay)

| Province | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | Exams | Core |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| An Giang | · | 1/1 | 1/3 | 1/3 | 1/3 | · | 1/6 | · | 1/1 | · | 6 | 17 |
| Bình Dương | · | 1/8 | 1/5 | 1/5 | 1/4 | · | 1/4 | · | 1/2 | · | 6 | 28 |
| Bình Phước | · | · | 1/8 | 1/8 | 1/8 | 1/8 | 1/9 | · | 1/5 | · | 6 | 46 |
| Bình Thuận | · | 1/4 | 1/5 | 1/3 | 1/4 | · | 1/2 | · | · | · | 5 | 18 |
| Bình Định | · | 1/6 | 1/8 | 1/6 | 1/6 | · | 1/8 | · | 1/8 | · | 6 | 42 |
| Bạc Liêu | · | · | 1/6 | 1/6 | 1/5 | · | 1/8 | · | 1/7 | · | 5 | 32 |
| Bắc Giang | · | 1/4 | 1/7 | · | 1/5 | · | 1/8 | · | · | · | 4 | 24 |
| Bắc Kạn | · | · | · | · | · | · | · | · | 1/8 | · | 1 | 8 |
| Bắc Ninh | · | 1/7 | 1/5 | · | 1/4 | · | · | · | 1/7 | 1/8 | 5 | 31 |
| Bến Tre | · | · | 1/7 | 1/7 | 1/6 | 1/5 | 1/6 | · | 1/1 | · | 6 | 32 |
| Cao Bằng | · | · | · | · | · | · | · | · | 1/8 | · | 1 | 8 |
| Cà Mau | · | · | 1/8 | · | 1/8 | · | · | · | · | · | 2 | 16 |
| Cần Thơ | · | 1/3 | 1/3 | · | 1/3 | · | 1/3 | · | 1/5 | · | 5 | 17 |
| Huế | · | 1/10 | 1/8 | 1/8 | 1/8 | · | 1/8 | · | 1/8 | 1/9 | 7 | 59 |
| Hà Nam | · | · | 1/6 | · | · | · | 1/9 | · | · | · | 2 | 15 |
| Hà Nội | 1/6 | 1/7 | 1/8 | 1/8 | 1/6 | · | 1/6 | · | 3/16 | 1/7 | 10 | 64 |
| Hà Tĩnh | · | · | 1/5 | 1/5 | 1/6 | · | 1/6 | · | · | · | 4 | 22 |
| Hòa Bình | · | · | 1/6 | 1/7 | 1/4 | · | 1/10 | · | · | · | 4 | 27 |
| Hưng Yên | · | 1/6 | 1/6 | 1/0 | · | · | 1/0 | · | · | · | 4 | 12 |
| Hải Dương | · | 1/6 | 1/6 | 1/5 | 1/7 | · | 1/7 | · | · | · | 5 | 31 |
| Hải Phòng | · | 1/8 | 1/7 | 1/8 | 1/9 | · | 1/0 | · | 1/3 | 1/5 | 7 | 40 |
| Hậu Giang | · | · | 1/3 | 1/9 | 1/7 | · | 1/9 | · | · | · | 4 | 28 |
| Khánh Hòa | · | · | 1/6 | 1/6 | 1/8 | · | 1/4 | · | · | 1/5 | 5 | 29 |
| Kiên Giang | · | 1/6 | 1/6 | · | 1/5 | · | 1/6 | · | · | · | 4 | 23 |
| Kon Tum | · | · | · | · | · | · | · | · | 1/7 | · | 1 | 7 |
| Lai Châu | · | · | · | · | · | · | · | · | 1/6 | 1/4 | 2 | 10 |
| Long An | · | · | · | · | · | · | 1/8 | · | 2/16 | · | 3 | 24 |
| Lào Cai | · | · | 1/8 | 1/6 | 1/9 | 1/10 | 1/10 | · | 1/10 | 1/3 | 7 | 56 |
| Lâm Đồng | · | · | · | · | 1/10 | · | · | · | · | · | 1 | 10 |
| Lạng Sơn | · | · | · | · | · | · | 1/6 | · | · | 1/8 | 2 | 14 |
| Nam Định | · | · | 1/5 | 1/5 | · | · | 1/6 | · | · | · | 3 | 16 |
| Nghệ An | · | 1/5 | · | 1/7 | 1/7 | · | 1/8 | · | 1/7 | · | 5 | 34 |
| Ninh Bình | · | · | · | 1/9 | 1/7 | · | 1/8 | · | 1/8 | 1/7 | 5 | 39 |
| Ninh Thuận | · | · | 1/4 | 1/6 | 1/2 | · | 1/6 | · | · | · | 4 | 18 |
| Phú Thọ | · | 1/4 | 1/4 | 1/6 | 1/7 | · | 1/7 | · | 1/5 | 1/6 | 7 | 39 |
| Phú Yên | · | 1/4 | 1/4 | 1/3 | 1/3 | · | 1/3 | · | 1/2 | · | 6 | 19 |
| Quảng Bình | · | · | 1/7 | · | · | · | 1/6 | · | · | · | 2 | 13 |
| Quảng Nam | · | · | · | · | · | · | · | · | 1/5 | · | 1 | 5 |
| Quảng Ngãi | · | · | 1/5 | 1/5 | 1/7 | · | 1/6 | · | 1/6 | · | 5 | 29 |
| Quảng Ninh | · | · | 1/8 | 1/9 | 1/7 | · | 1/8 | · | 1/4 | 1/5 | 6 | 41 |
| Quảng Trị | · | · | 1/5 | · | 1/4 | 1/5 | 1/5 | · | · | 1/3 | 5 | 22 |
| Sóc Trăng | · | · | 1/5 | 1/4 | 1/4 | · | 1/4 | · | 1/6 | · | 5 | 23 |
| Sơn La | · | · | · | 1/6 | 1/6 | 1/7 | 1/4 | · | 1/6 | · | 5 | 29 |
| TP Hồ Chí Minh | 1/7 | · | 1/7 | 1/8 | · | · | 1/7 | · | 3/15 | 3/7 | 10 | 51 |
| Thanh Hóa | · | 1/3 | 1/6 | 1/7 | 1/7 | · | 1/7 | · | · | 1/5 | 6 | 35 |
| Thái Bình | · | · | 1/6 | 1/10 | 1/6 | · | · | · | 1/6 | · | 4 | 28 |
| Thái Nguyên | · | 1/8 | 1/5 | 1/9 | 1/8 | · | 1/9 | · | · | 1/5 | 6 | 44 |
| Tiền Giang | · | · | 1/6 | 1/7 | 1/6 | · | 1/6 | · | 1/4 | · | 5 | 29 |
| Trà Vinh | · | 1/4 | · | · | 1/7 | · | 1/6 | · | · | · | 3 | 17 |
| Tuyên Quang | · | · | 1/5 | · | 1/1 | · | 1/0 | · | 1/0 | · | 4 | 6 |
| Tây Ninh | · | · | 1/8 | · | 1/8 | · | 1/8 | · | 1/5 | 3/8 | 7 | 37 |
| Vĩnh Long | · | 1/7 | 1/4 | 1/4 | 1/5 | · | · | · | · | · | 4 | 20 |
| Vĩnh Phúc | · | 1/6 | 1/5 | · | 1/4 | · | 1/7 | · | 1/5 | · | 5 | 27 |
| Yên Bái | · | · | · | · | · | 1/0 | 1/0 | · | 1/3 | · | 3 | 3 |
| Đà Nẵng | · | 1/5 | 1/7 | 1/7 | 1/5 | · | 1/6 | · | · | 1/7 | 6 | 37 |
| Đắk Lắk | · | · | 1/8 | 1/8 | 1/7 | · | 1/10 | · | 1/11 | · | 5 | 44 |
| Đắk Nông | · | · | · | 1/6 | 1/5 | · | · | · | · | · | 2 | 11 |
| Đồng Nai | · | · | 1/9 | 1/9 | 1/8 | · | 1/8 | · | 1/5 | 1/10 | 6 | 49 |
| Đồng Tháp | · | 1/9 | 1/7 | 1/6 | 1/7 | · | · | · | · | 1/0 | 5 | 29 |
| **Total** | **2/13** | **23/131** | **45/270** | **38/241** | **46/273** | **6/35** | **47/288** | **0/0** | **40/221** | **23/112** | **270** | **1,584** |

(The PTNK and KHTN 2025 school exams are counted under TP Hồ Chí Minh and Hà Nội; `school` holds the school name.)

- Exams with 0 Core items are all-MCQ papers (Hưng Yên 2020/2023, Yên Bái 2022/2023, Tuyên Quang 2023/2025,
  Đồng Tháp 2026), or duplicate re-posts of one paper: Tây Ninh 2026 has 3 article ids, of which 2 are duplicates; HCM
  2026 likewise.
- Dedup is exact match plus 5-gram Jaccard ≥ 0.8.

## 8. Gaps and risks

1. **No 2024 exams on loigiaihay.** The mid-2024 monthly sitemaps contain no 2024 Toán vào 10 article; May 2024 only
   re-posts 2023 papers. **2022 is thin (6 exams), and 2017 has 2.** Filling these needs another source.
   toanmath.com's robots disallows its PDF paths, and VnExpress has only images (OCR).
2. **HSG Toán 9: none.** No such listing in the event sitemap (23,752 events) or in any article slug for 2017–2026.
3. **2026 coverage is 23 papers / 23 provinces** (the 2025 merger to 34 provinces changed exam issuers).
   - 2026 papers yield few Core items: the new format is MCQ/T-F/short answer, with more figures.
   - Option: convert the **939 figure-free MCQs** (850 with a key) to free response when the options are numeric. This
     would add post-cutoff Core items.
4. **first_seen is weak before 2022.** The site (re)published 2017–2021 papers in 2022 (earliest Wayback 2022-02).
   For the temporal DiD, use `exam_year` (exam held in May–June of that year) as the primary date and `first_seen` as a
   secondary one. Wayback had captures for only 98 of 274 pages.
5. **Keys.**
   - 967 Core items still have fewer than 2 comparable keys. The Claude key covered 474 items (all of 2025+); about
     $5.9 more at the measured $0.0053/item would cover the remaining 1,110 (needs a budget decision).
   - The SymPy key covers only 28 % of Core: equation/system/evaluation/simplify. Word problems, extremum,
     probability and geometry need T5's Claude-written program (key ii), not done here.
   - The provisional (page) key is about 80 % precise. It is a QA aid, not a gold label.
6. **Source text defects.** Some formulas are missing on the page itself (R6) and some LaTeX is literally wrong
   (`\sqrt {x + 2}` vs √x + 2). Key conflicts surface them, so the 51 conflicts go to the human queue first.
7. **Official texts.** The Hà Nội official text/keys are manual-download only. Text fidelity against Sở PDFs is not yet
   checked for any province.
8. **Rule-based tags.** answer_type, topic and certifiable come from regexes. Expect some mislabels (e.g. text-only
   geometry "Tính" items are Core but certifiable = no). Stage 3 critique should re-check them.

## 9. Files

- Data (`data/vi_exam/`):
  - `exam_index.jsonl`, `listings.jsonl`, `exams.jsonl`;
  - `items.jsonl` (public fields), `items_private.jsonl` (**not for release**), `qa_queue.jsonl`;
  - `claude_keys.jsonl`, `first_seen.jsonl`, `official_sources.jsonl`;
  - `coverage.csv`, `summary.json`, `fetch_log.jsonl`;
  - `raw/` (cache with timestamps);
  - `_superseded/` (stopped agent's code and outputs; safe to delete).
- Code: `vi_math_verified_aug/vi_exam/{crawl,discover,collect,page,segment,tagging,answers,symkey,build,claude_key}.py`.
  `scrape.py` and `verify.py` are now pointers to these modules.
- Test: `tests/test_vi_exam_segment.py`.
- Logs: `logs/X2/*.log`, `logs/api/X2/`.
- Nothing was committed (the lead commits).

---

## Addendum A: full Claude keying, recomputed key status, frozen QA queue (approved cap $6.5)

**Run.** Batch `msgbatch_01LLkD6u8QMGeGQqupH5Hv14`, `claude-opus-5-5`, effort low, 1,111 requests (all Core items not
yet keyed).
- 1,111/1,111 succeeded, all `end_turn`.
- 422,855 input + 549,517 output tokens = **$6.341** (cap $6.5). Per item $0.0057.
- **Total Claude spend for X2: $8.84** (pilot $0.099 + main $2.398 + rest $6.341). Ledger:
  `logs/api/X2/cost_ledger.jsonl`.
- Core items with a Claude key: **1,585/1,585**.

**Fixes made before submitting:**
1. **Repeated sub-labels inside one question** (two independent `a)/b)` groups, 7 items) produced duplicate item IDs,
   and the Batch API rejected the duplicate `custom_id`. This is the same failure class as the "Câu N" collision, one
   level down: the tree kept both nodes, but the ID was label-based.
   - IDs now get a positional suffix (`…a~2`), and the items carry the flag `repeated_sub_label`.
   - Their provisional key is withheld because the label-based solution split is ambiguous.
   - Core is now **1,585** (+1).
2. **Content-addressed Claude keys.** Each key row stores `problem_sha1` of the exact text sent, and build attaches a
   key only if the hash matches the item's current problem.
   - Existing 474 rows were backfilled from the request logs: 472 matched; 2 had been produced for the `~2` sibling
     and were re-pointed (backup `data/vi_exam/claude_keys.before_sha_backfill.jsonl`).
   - Batch `custom_id` = id with `.`→`_` and `~`→`-`.
3. `partial` (one key's numbers a strict subset of the other's) now counts as a **conflict** for key status.
4. The LaTeX converter follows TeX: an undelimited argument is one token, and malformed input is a parse failure, not
   a crash.

**Agreement on all 1,585 Core items** (rate = agree / (agree + disagree)):

| Pair | agree | disagree | partial | unparsed | no key | rate |
|---|---|---|---|---|---|---|
| provisional (page solution) vs SymPy | 396 | 14 | — | 28 | 1,147 | **96.6 %** |
| SymPy vs Claude | 418 | 7 | — | 13 | 1,147 | **98.4 %** |
| provisional vs Claude | 924 | 137 | 184 | 340 | — | **87.1 %** (74.3 % if partial counts as disagree) |

**Key status (Core):**

| Status | All Core | certifiable yes | exam year 2025+ | exam year 2026 |
|---|---|---|---|---|
| 3 keys agree (`consensus_3`) | **385** | 385 | 85 | 28 |
| 2 keys agree (`consensus_2`) | **569** | 382 | 123 | 41 |
| conflict | **325** (146 strict disagreements + 179 partial-only) | 109 | 72 | 20 |
| fewer than 2 comparable keys (`unverified`) | **306** | 187 | 53 | 23 |

- 303 of the 306 `unverified` items have only the provisional and Claude keys, and the two could not be compared
  automatically. Most are worded or multi-quantity answers and inequality conditions.
- A human can grade them directly from the two stored keys.

**Frozen QA queue.** `data/vi_exam/qa_queue.jsonl` has 1,585 rows with `queue_rank`, ordered conflict → unverified →
consensus_2 → consensus_3, newer `exam_year` first within each group.
- Ranks 1–325 = conflicts (2026 conflicts first) → stage 4, two human annotators.
- Ranks 326–631 = unverified → stage 2, more keys (the Claude-written program key, T5 key ii).
- The rest → stage 3, LLM critique. The 150-item stage-5 random audit samples from these.
- Freeze record `data/vi_exam/qa_queue.freeze.json`: frozen_at 2026-11-07T11:07:43Z, sha256 `0f58ee1c…e36aeaaf`,
  n = 1,585.

## Addendum B: candidate sources for the 2024 / HSG Toán 9 / chuyên gaps (probed, NOT crawled)

**Method.**
- Web search found one sample page per site.
- `vi_math_verified_aug/vi_exam/source_probe.py` fetched only that site's robots.txt and that single sample page
  through the same polite fetcher: 29 requests in total, cached under `data/vi_exam/raw/<host>/`.
- It then measured whether the problem text is HTML text with LaTeX/MathML, or images/PDF.
- Raw measurements: `data/vi_exam/source_probe.jsonl`.

| Source | Gap it could fill | robots.txt | Sample probed | Content format (measured) | Verdict |
|---|---|---|---|---|---|
| **thi.tuyensinh247.com** | **2024 Toán vào 10, about all provinces** (hub `dap-an-de-thi-vao-lop-10-nam-2024-tat-ca-cac-tinh-c29a81338`; per-province pages for Đà Nẵng, Bắc Giang, Lai Châu, Khánh Hòa, Hải Dương, Thái Nguyên, Bình Dương 2024, …) | 200, `User-agent: *`; disallows only AJAX/comment/captcha paths and a list of old articles; no Crawl-delay | Đà Nẵng 2024 (`c29a81475`) | **Images**: 0 MathJax, 0 MathML; exam and solution are `…/2024/0603/de-thi-vao-10-mon-toan-2024-da-nang-result.jpg` + `dap-an-…-1..3.png` | **Best 2024 coverage, but needs math OCR.** Claude vision is allowed for test items (Claude-only rule) and gives text + LaTeX; then human-verify. Third-party solution images stay QA-private. |
| hoatieu.vn | 2024/2026 đáp án by province; HSG Toán 9 cấp tỉnh (Quảng Nam, Bắc Giang, Nghệ An, Huế, …) | 200, no Disallow lines | HSG Toán 9 Quảng Nam (`244959`) | **PDF viewer** (`f.hoatieu.vn/pdf/viewer.html?file=…pdf`) with a "Hoatieu Pro 99.000đ" paywall; 0 LaTeX | Low (PDF + paywall). Discovery index only. |
| vietjack.com | HSG Toán 9 by Sở, 2025–2026 (e.g. Hưng Yên) | robots.txt 404 (= no restriction, RFC 9309) | HSG Toán 9 Hưng Yên 2025-2026 | **Images** (`…hung-yen-nam-2025-2026-1..6.PNG`); the solution is paywalled per the page text | Low–medium (images need OCR; solutions paywalled). |
| vndoc.com | HSG Toán 9 collections (cấp huyện/tỉnh), 2025-26 sets of 22 provinces | 200, no Disallow lines | "50 đề HSG lớp 9 Toán" (`223609`) | PDF→HTML render (`bg1.png` backgrounds + text layer). **The text layer is garbled**: formulas are split per glyph ("Cho bi / u th / c / 2 / 2 / x …"). Pro paywall | Low (unusable text without OCR). |
| thcs.toanmath.com | **HSG Toán 9 cấp tỉnh 2023–2026** (Hà Tĩnh 25-26, Tuyên Quang 24-25, Quảng Bình 24-25, Phú Thọ 23-24, Gia Lai …) | 200; **Disallow `/thcs-pdf/`, `/thcs-word/`, `/data-thcs/`** | Tuyên Quang HSG 2024-2025 | The article HTML has only a short quoted excerpt; the paper is an embedded PDF at **`/thcs-pdf/…pdf` (disallowed)** | **Discovery index only** (as in 04 §3.1): it lists which Sở set which HSG paper; the PDF must come from the Sở. |
| hoc247.net | chuyên Toán (Sở HCM chuyên 2019, PTNK 2018-19, …); thi thử | 200; disallows only quiz/account paths | chuyên Toán Sở HCM 2019 (`doc5519`) | **HTML text + MathJax** (12 inline formulas, 8 Câu/Bài headers) plus 4 PDF/iframe embeds; the full document is behind "xem online / đăng nhập" | **Medium for chuyên/older years.** The only probed site with text/LaTeX like loigiaihay, but only a preview is public and no 2024 sample was found. Next step: probe 2–3 listing pages. |
| 2048.vn | Ôn vào 10 Toán (incl. Hà Nội 2023-2024) | 200; disallows dashboard/quiz/join paths | listing `/on-vao-10/toan?page=12` | Listing only: 24 images, 0 LaTeX | Unknown (quiz-style platform). Low priority. |
| znews.vn (newspaper) | Hà Nội 2024 đề + gợi ý đáp án (by Tuyensinh247 teachers) | 200; disallows preview/tag/query patterns | HN 2024 Toán (`post1479826`) | **Images** (`photo.znews.vn/…/2024_06_07/1..6.jpg`) | Timeline evidence (publication time) + OCR fallback only. |
| tienphong.vn (newspaper) | Hà Nội 2024 **official** đề + đáp án release | 200; disallows api/search/tag | `post-1645927.tpo` | 84 characters of static text (JS-rendered or image) | Timeline evidence; points to the Sở release. |
| dspace.ctu.edu.vn | Toán học & Tuổi trẻ HSG lời giải (Bình Định, Bắc Ninh, Hưng Yên 2017–19) | **robots.txt unreachable → fail-closed, not fetched** | — | Scanned journal PDFs (per the search snippet) | Excluded unless robots becomes readable; old years only. |
| **Official Sở sites** (hanoi.edu.vn, hcm.edu.vn; others, incl. Sở "đề tham khảo" PDFs mirrored on cdn.thuvienphapluat.vn / lawnet) | 2024 official texts + keys; HSG cấp tỉnh | hanoi.edu.vn: Disallow `/tai/` (download path), `/Config/`. hcm.edu.vn: Disallow `/Config/`, `/qi.aspx`, `/Timkiem` | hanoi.edu.vn homepage + exam-office section + search (§2) | Announcements in HTML; papers are PDF attachments (on Hà Nội the download path is disallowed) | **Authoritative but manual**: download by hand, then OCR/re-type. Highest legal value (state issuer, releasable). |

**Conclusion for the gaps.**
- No probed site offers 2024 or HSG Toán 9 papers as HTML text/LaTeX the way loigiaihay does.
- The realistic paths are:
  - **(a) 2024:** Claude-vision OCR of the tuyensinh247 image pages (robots-allowed; about 60 provinces × 1–2 exam
    images), human-verified against Sở PDFs where available.
  - **(b) HSG Toán 9:** use thcs.toanmath.com / vndoc as an index, manually download the official Sở PDFs, then OCR.
  - **(c) chuyên:** hoc247 (text/LaTeX preview), after a listing probe.
- Third-party worked solutions stay QA-private (04 §3.3).
- Nothing beyond the probe was crawled. Each path needs a go-ahead and an OCR budget.

**Search sources used:**
- [tuyensinh247 2024 hub](https://thi.tuyensinh247.com/dap-an-de-thi-vao-lop-10-nam-2024-tat-ca-cac-tinh-c29a81338.html)
- [tuyensinh247 Đà Nẵng 2024](https://thi.tuyensinh247.com/dap-an-de-thi-vao-lop-10-mon-toan-da-nang-2024-c29a81475.html)
- [hoatieu HSG Toán 9 Quảng Nam](https://hoatieu.vn/de-thi-hoc-sinh-gioi-toan-9-cap-tinh-quang-nam-244959)
- [vietjack HSG Toán 9 Hưng Yên 2025-2026](https://vietjack.com/de-thi/de-thi-hoc-sinh-gioi-toan-9-so-gddt-hung-yen-nam-2025-2026.jsp)
- [vndoc 50 đề HSG Toán 9](https://vndoc.com/50-de-thi-hoc-sinh-gioi-lop-9-mon-toan-cap-huyen-tinh-223609)
- [thcs.toanmath HSG Tuyên Quang 2024-2025](https://thcs.toanmath.com/2025/03/de-chon-hoc-sinh-gioi-tinh-toan-9-nam-2024-2025-so-gddt-tuyen-quang.html)
- [hoc247 chuyên Toán HCM 2019](https://hoc247.net/tu-lieu/de-thi-tuyen-sinh-lop-10-mon-toan-chuyen-nam-2019-so-gd-dt-tp-hcm-co-dap-an-doc5519.html)
- [2048.vn ôn vào 10](https://2048.vn/on-vao-10/toan?page=12)
- [znews HN 2024](https://znews.vn/de-thi-va-goi-y-dap-an-mon-toan-tuyen-sinh-lop-10-o-ha-noi-post1479826.html)
- [Tiền Phong HN 2024 official](https://tienphong.vn/post-1645927.tpo)
- [CTU dspace THTT](https://dspace.ctu.edu.vn/jspui/handle/123456789/57527)
