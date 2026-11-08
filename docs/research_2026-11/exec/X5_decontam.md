# X5 — Decontamination and contamination-scan tooling (2026-11-07)

Owner: X5. Scope: `vi_math_verified_aug/eval/decontam.py` (rewritten from scratch, replacing the stub),
`decontam_modal.py` (GPU embedding stage), `decontam_report.py` (tables), `tests/test_decontam.py`, and `data/contam/`.
Design sources: T5 §3.4 (C2), 04 §5 and H7, T1 §2.3 G4/G5. Nothing is committed (the lead commits).

## 0. Headline results

| # | Finding | Evidence |
|---|---|---|
| 1 | **Lexical overlap misses almost all translated copies.** `vinhnq29/vietnamese-elementary-math` ships its own Vietnamese translation of the GSM8K **test** split, aligned row by row with ours. 13-gram coverage ≥ 0.5 flags only **38/1,319** of those items (any shared 13-gram: 535). bge-m3 cosine ≥ 0.85 flags **1,280/1,319**, and its top-1 hit is the aligned item for 1,318/1,319 | §4.1 table, `embed_modal/smoke/embed_vinhnq29.jsonl` |
| 2 | **MetaMathQA-vi inherits MATH train/test near-twins.** For Vi-MATH test: 39 items have 13-gram coverage ≥ 0.5, 202 have templated (numbers → #) coverage ≥ 0.5, and 635 / 234 / 54 have cos ≥ 0.85 / 0.90 / 0.95 against the 15,013 seed questions. Inspected pairs are the same problem or the same problem with changed values/operands, e.g. `algebra/320` and `algebra/1038` copied verbatim, `intermediate_algebra/704` with 2/a+1/b vs 1/a+2/b. Vi-GSM8K test vs MetaMathQA-vi: 0 lexical, 16 at cos ≥ 0.85 (GSM8K train/test template twins) | §4.2 |
| 3 | **Vi-ExamMath is mostly clean, but the 2025 papers reuse textbook items.** 33 of 3,024 question units meet the review rule (§5.3). 31 of them come from LuminarAI, which is a montoan.com.vn textbook-solution crawl. By exam year that is **15/450 (3.3 %) in 2025**, versus 0–1.4 % for the other years. In the inspected cos ≥ 0.90 pairs, 5 of 15 are the same item, e.g. the Long An 2025 sample paper's cone-roof problem and cement-factory problem, which come from the textbook. Another 7 of 15 are the same template with other values or options | §4.3 |
| 4 | **The `daominhwysi/toanmath.com-alpha` labels are not text.** All 513,271 `label` values are document slugs (11,219 distinct documents), and the page content exists only as images. No document is a grade-10 entrance exam ("tuyển sinh lớp 10": 0; lớp 6–9: 0). Its coverage is THPT (grades 10–12) and university entrance. Title-level overlap with Vi-ExamMath: **0/274 exams**. H7's toanmath concern therefore does not apply to Vi-ExamMath, and a text scan of this corpus would need OCR | §4.4 |
| 5 | **The G4 threshold "cos < 0.92 (bge-m3)" is too lax.** At 0.92 the gate misses 15 % of independently re-translated Vietnamese copies and 12 % of number-perturbed copies. **Recommendation: bge-m3 cos < 0.85 for Vi↔Vi and < 0.80 for Vi↔EN.** Also add a templated-13-gram rule (coverage ≥ 0.5). Keep G5 (MinHash J < 0.8) | §3, §5 |
| 6 | **Resources.** Local scans peaked at ≤ 474 MB RSS; the 395k-document disk index peaked at 936 MB. No torch was imported locally. Modal spend for X5 was **$0.155** against a $0.50 cap | §2.4 |

## 1. Tool design (`vi_math_verified_aug/eval/decontam.py`)

Parts 1–3 below need only numpy and run in the project `.venv`. Encoders import torch lazily. 39 tests in
`tests/test_decontam.py` all pass: `.venv/Scripts/python -m pytest tests/test_decontam.py -q` gives 39 passed.

**1. Normalisation** (`tokenize`, `normalize`, `variant_tokens`, `variant_hashes`). The pipeline runs these steps:
- HTML unescape and tag strip;
- literal `\n` unescape, except where it starts a LaTeX command such as `\neq` or `\nu` (217k of 306k LuminarAI rows contain escaped `\n`);
- NFC and lowercase;
- LaTeX canonicalisation: delimiters dropped, `\dfrac/\tfrac`→`frac`, `\le/\leqslant`→`≤`, `\cdot/\times`→`*`, `\left/\right` dropped, `\text{…}` unwrapped, Unicode math mapped;
- Vietnamese number canonicalisation: `2,5`/`2{,}5`→`2.5`, `1.000.000` and `10 000`→digits, `1.234,5`→`1234.5`.

Tokens are numbers, letter runs and operators. There are three **variants**: `exact`, `nodiac` (diacritics stripped,
`đ`→`d`) and `template` (every number → `#`, which catches value-perturbed copies).

**2. n-gram index** (`NgramIndex`).
- Token hashes are 64-bit blake2b with a cache. n-gram hashes use a polynomial rolling hash with a splitmix64 finaliser, salted by n. One sweep gives every length (`multi_ngram_hashes`).
- Each index holds sorted `uint64` keys and `uint32` doc-id postings, in shards. With a `workdir`, each shard is `np.save`d and reopened memory-mapped.
- A one-hash bit filter (64 bits/key, capped at 2³¹ bits) sits in front of `searchsorted`. Profiling showed `searchsorted` plus `unique` took 75 % of scan time before the filter; after it, a 10 kB doc takes ~10 ms instead of 47 ms.
- Docs shorter than n (≥ 6 tokens) are indexed as one whole-sequence gram. Queries hash those lengths too, so a short item contained verbatim is still found.
- `query()` returns {doc: #shared unique grams}. Coverage = shared / n_grams(indexed doc).
- Scans **index the small side** (the 9,343 test items) and **stream the corpora**, so RAM is O(test set).
- `bench-index` covers the opposite case. It indexed all 395,000 MetaMathQA-vi rows (63.8 M 13-grams, 4 shards, **730 MB on disk**, 256 MB filter) in **138 s**, with **peak RSS 936 MB** and **0.89 ms per query**. Linear extrapolation to 1 M docs is ~1.9 GB on disk and ~6 min, with RAM bounded by the shard size (20 M grams ≈ 0.5–0.9 GB peak during sort).
- Cross-check: the index direction found shared 13-grams for 749 Vi-MATH and 10 Vi-GSM8K items. These are exactly the streaming-direction counts.

**3. MinHash LSH** (`MinHasher`, `MinHashLSH`). This is my own numpy code (datasketch is installed but not used): word 5-gram shingles, 128 permutations, 32 bands × 4 rows (threshold ≈ 0.42), with incremental insert and query.
- Permutation family: h_i(x) = splitmix64(x XOR s_i).
- The first version used (a·x+b) mod (2⁶¹−1) with 31-bit a, b. That map is almost monotone when a·x < 4P, so every permutation picked the same shingles. Its estimate was 0.04 for a true Jaccard of 0.69; the test caught it.
- After the fix, over 30 random edits the estimate minus the exact Jaccard has mean −0.006 and SD 0.041 (SE at 128 permutations ≈ 0.04).
- In scans, MinHash is skipped when J ≥ 0.3 is impossible. The bound is |A ∩ targets| / |A|, computed with a 5-gram-shingle bit filter. This also drops long pages compared against short items, where J ≤ |B|/|A|.

**4. Embeddings** (`Encoder`, `EmbeddingIndex`, `calibrate_pairs`).
- Uses HF `transformers`: bge-m3 with CLS pooling, e5 with mean pooling and the `query: ` prefix. Precision is bf16 or fp16. Embeddings are L2-normalised and the scorer keeps the top-3 matches per target.
- The corpus-scale run is `decontam_modal.py`, which sends the package source to a Modal GPU (§2.3).

**5. CLI** (`python -m vi_math_verified_aug.eval.decontam …`):
- `scan --corpus SRC… --target SRC… --out DIR [--embed-model M --embed-threshold τ]` writes `items.jsonl` (one row per target item). Each row has every detector's best corpus match: coverage, shared count, #corpus docs, doc id, MinHash J, and top-3 cosines. It also writes `summary.json` (counts, runtime, peak RSS), `examples.md` (matched windows) and `corpus_snippets.jsonl`.
- `gate --target … --candidates FILE` applies G4/G5 to generated items using `Gate` (§5).
- `calibrate`, `bench-index`, `toanmath-titles` and `resummarize` are also available.
- Sources are registered names: `vi_gsm8k_test`, `vi_math_test`, `vi_exam_raw`, `luminarai`, `luminarai_segments`, `hllj_crawl`, `hllj_mcq`, `vinhnq29:<split>`, `metamathqa_vi`, `toanmath_labels`, `gsm8k_en_test`, `math_en_test`. A generic form `jsonl:<path>:<field>+<field>` also works.

**Units scanned.**
- Vi-GSM8K test: 1,319 items (`data/clean/vi_gsm8k_test/vi_gsm8k_test_full_v1.1.jsonl`, `question`).
- Vi-MATH test: 5,000 items (`vi_math_test_full_v1.1.jsonl`, `problem`).
- Vi-ExamMath: 3,024 question units, extracted read-only from X2's cache `data/vi_exam/raw`.
  - X2's `items.jsonl` was moved to `_superseded` while this work ran, so I parse the pages myself. For loigiaihay, I take the `sub-question-1` "Đề bài" block up to the solution and split it at "Câu/Bài N".
  - Non-exam pages yield nothing: sitemaps, category pages and archive CDX JSON. Of the 274 exam articles, 273 yield units; one is image-only.
  - The "Câu 3 (1,5 điểm):" headers are stripped.
  - Re-run against X2's final `items.jsonl` with `--target jsonl:data/vi_exam/items.jsonl:context+text` once it exists.

## 2. Runs, environment, cost

### 2.1 Environment
- Local package code and tests run in the project `.venv`.
- Corpus I/O runs in a new documented env **`E:\venv_dc`** (Python 3.11.7): numpy 2.4.6, pyarrow 26.0.1, huggingface_hub 2.4.0, transformers 5.22.2, torch 2.14.1+cu126.
  - torch is installed but was **never imported locally**, per the coordinator's RAM instruction.
- Small HF parquets are cached in `data/contam/_cache/` (LuminarAI 52 MB, hllj 5 MB, vinhnq29 4.5 MB).
- MetaMathQA-vi is streamed line by line from the local 824 MB file.
- toanmath-alpha: only the `label` and `path` column chunks are read, via HTTP range requests from the 315 × ~1 GB parquets (pyarrow + HfFileSystem). That took 747 s and produced 84 MB of JSONL. No image bytes were fetched.

### 2.2 Lexical scans (local, sequential)
Command for each corpus:
`PYTHONPATH=. E:/venv_dc/Scripts/python -m vi_math_verified_aug.eval.decontam scan --corpus <C> --target vi_gsm8k_test --target vi_math_test --target vi_exam_raw --out data/contam/scans/lex_<C>`

| Corpus | Docs streamed | Wall time | Peak RSS | Corpus docs sharing ≥ 1 exact 13-gram with a test item |
|---|---|---|---|---|
| hllj/vi_math_problem_crawl | 10,246 | 20 s | 279 MB | 8 |
| hllj/vi_grade_school_math_mcq (per problem) | 13,615 | 18 s | 272 MB | 11 |
| vinhnq29 (4 splits) | 17,003 | 23 s | 275 MB | 544 |
| LuminarAI (question+answer, whole pages, median 9.2 kB) | 306,406 | 316 s | 474 MB | 890 |
| MetaMathQA-vi (query_vi+response_vi) | 395,000 | 418 s | 249 MB | 10,856 |

The machine was shared with 5 other workstreams during these runs, so the wall times are pessimistic.

### 2.3 Embedding stage (Modal)
- Code: `vi_math_verified_aug/eval/decontam_modal.py`. Each run prints an ENVCHECK line, has a 1,800 s hard timeout and a per-stage deadline, and writes every stage to the `x5-embed` volume, which I pulled back with `modal volume get`.
- Package versions are pinned in the Modal image: torch 2.14.1, transformers 5.22.2, numpy 2.4.6, pyarrow 26.0.1.
- **smoke** (`--corpora vinhnq29 --tag smoke`): L4, 123 s. It covered calibration of bge-m3 and e5-small, the target embeddings and vinhnq29. Cost **$0.044**.
- **main** (`--corpora hllj_mcq,hllj_crawl,metamathqa_vi,luminarai --no-do-calib --batch 512 --tag main`): it waited 12 min for an L4, so I stopped the app (no cost while queued) and relaunched with `gpu=["L4","A10G"]`. It ran on an **A10G** in 295 s: 13.6k + 10.2k + 15.0k + 75.1k documents at 310–1,050 docs/s, including the in-container parquet downloads. Cost **$0.111**.
- **X5 total: $0.155** against the $0.50 cap. `modal billing report --for today --show-resources` before and after is saved in `data/contam/logs/modal_billing_{before,after}.txt`.
- Embedding units:
  - MetaMathQA-vi: the **15,013 distinct `original_question_vi`**, which are the GSM8K/MATH-train seeds behind all 395k rephrasings.
  - LuminarAI: the **75,072 distinct problem-statement segments** from `problem_segments()`. Whole rows are 9 kB pages whose first 256 tokens are marketing text.
  - All other corpora: the question field.
  - Embedding targets keep the "Câu N" headers (they were prepared before the header-strip fix). Cosine is barely affected.
- I did not embed toanmath-alpha, because its labels are slugs (§4.4).

### 2.4 Memory
Every local process stayed under the 1.5 GB cap: scans ≤ 474 MB, toanmath label fetch 73 MB, disk-index benchmark 936 MB. Peak RSS is read with `GetProcessMemoryInfo` and recorded in every `summary.json`. The embedding run's peak GPU allocation was 2.6 GB.

## 3. Calibration (bge-m3 vs multilingual-e5-small, inside the Modal job)

Positives are known pairs of the same item. Two negative sets are used: **random** pairs (20k), and **hard** negatives, which are each item's most similar *other* item in the paired set. Hard negatives are the realistic FPR at scan time.
Source: `data/contam/embed_modal/smoke/summary.json`.

**bge-m3** (dense, CLS, max_len 256)

| Pair set (n) | top-1 retrieval | pos median / 1st pct | hard-neg 99th pct | τ @ 1 % FPR random | recall / hard FPR @0.80 | @0.85 | @0.88 | @0.90 | @0.92 |
|---|---|---|---|---|---|---|---|---|---|
| Vi-GSM8K ↔ EN GSM8K (1,319) | 1.000 | 0.873 / 0.697 | 0.772 | 0.554 | 0.839 / 0.5 % | 0.643 / 0.08 % | 0.450 / 0 | 0.321 / 0 | 0.194 / 0 |
| Vi-MATH ↔ EN MATH (1,500) | 0.998 | 0.876 / 0.680 | 0.864* | 0.586 | 0.820 / 4.6 %* | 0.619 / 1.9 %* | 0.477 / 0.6 % | 0.372 / 0.4 % | 0.278 / 0.3 % |
| Vi-GSM8K ↔ vinhnq29 re-translation (1,319) | 1.000 | 0.970 / 0.812 | 0.823 | 0.581 | 0.996 / 1.4 % | **0.970 / 0.76 %** | 0.935 / 0.53 % | 0.908 / 0.53 % | 0.851 / 0.15 % |
| Vi-GSM8K ↔ number-perturbed copy (1,319) | 1.000 | 0.965 / 0.842 | 0.809 | 0.582 | 0.998 / 1.5 % | **0.988 / 0.45 %** | 0.968 / 0.45 % | 0.938 / 0.38 % | 0.879 / 0.30 % |

\* The MATH test set contains real within-set near-twins, so its "hard negatives" include true duplicates.

**multilingual-e5-small** has a compressed scale: random pairs have median 0.80–0.84.
- Vi↔Vi re-translation: recall 0.997 at 0.92 but hard FPR 14.8 %; at **0.95**, recall 0.983 with hard FPR 1.0 %.
- Vi↔EN GSM8K: at 0.90, recall 0.765 with hard FPR 0.4 %.
- Vi↔EN MATH: at 0.92, recall 0.50 with hard FPR 5.1 %.

**Reading.**
- bge-m3 separates translation-level copies cleanly: hard negatives reach 0.82 at the 99th percentile, while re-translation positives have median 0.97.
- Cross-lingual pairs sit lower (median 0.87). A single threshold that serves both directions would be 0.80, at a hard FPR of 0.5 % (GSM8K).
- e5-small is usable only with language-specific thresholds (0.95 Vi↔Vi), and it is clearly worse cross-lingually.
- **Choice: bge-m3 everywhere** (1.1 GB in bf16; 545 docs/s on L4 for short questions).

## 4. Scan results

Full per-corpus × target table, from `python -m vi_math_verified_aug.eval.decontam_report`; see also `data/contam/logs/report_tables.txt`. Column notes:
- Coverage is "fraction of the test item's 13-grams found in one corpus doc".
- `ng13_any` counts items ≥ 13 tokens that share at least one 13-gram (the GPT-3 rule).
- `review_rule` is defined in §5.3.

| corpus | target | n | ng13_any | ng13 cov≥.5 | ng8 cov≥.5 | tmpl13 cov≥.5 | MinHash≥.5 | cos≥.85 | cos≥.90 | cos≥.85 & no lexical | review_rule |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hllj_crawl | Vi-GSM8K | 1319 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| hllj_crawl | Vi-MATH | 5000 | 1 | 2 | 2 | 11 | 0 | 0 | 0 | 0 | 3 |
| hllj_crawl | Vi-ExamMath | 3024 | 2 | 0 | 0 | 1 | 0 | 2 | 0 | 2 | 1 |
| hllj_mcq | Vi-GSM8K | 1319 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| hllj_mcq | Vi-MATH | 5000 | 1 | 1 | 1 | 11 | 0 | 4 | 0 | 4 | 3 |
| hllj_mcq | Vi-ExamMath | 3024 | 2 | 0 | 0 | 2 | 0 | 21 | 0 | 21 | 2 |
| vinhnq29 | **Vi-GSM8K** | 1319 | 535 | 38 | 81 | 37 | 3 | **1280** | **1197** | 1242 | 1197 |
| vinhnq29 | Vi-MATH | 5000 | 1 | 0 | 0 | 8 | 0 | 7 | 1 | 7 | 1 |
| vinhnq29 | Vi-ExamMath | 3024 | 3 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 2 |
| LuminarAI | Vi-GSM8K | 1319 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| LuminarAI | Vi-MATH | 5000 | 43 | 1 | 2 | 21 | 0 | 4 | 0 | 4 | 2 |
| LuminarAI | **Vi-ExamMath** | 3024 | 244 | 4 | 22 | 30 | 0 | 221 | 15 | 209 | **31** |
| MetaMathQA-vi | Vi-GSM8K | 1319 | 10 | 0 | 0 | 0 | 0 | 16 | 1 | 16 | 1 |
| MetaMathQA-vi | **Vi-MATH** | 5000 | 749 | 39 | 102 | 202 | 3 | 635 | 234 | 568 | **336** |
| MetaMathQA-vi | Vi-ExamMath | 3024 | 41 | 0 | 9 | 10 | 0 | 1 | 0 | 1 | 0 |

### 4.1 (a) External Vietnamese corpora vs Vi-GSM8K test
- The only real source is **vinhnq29**: its `vigsm8k_test` split *is* the GSM8K test set in another translation, aligned 1:1 with ours. Any training mix containing vinhnq29 must drop that split, and its `vigsm8k_train` is GSM8K train. Training on it contaminates the Vi-GSM8K test set completely. This corpus is also our best measure of how blind lexical detection is to translated copies.
- hllj, LuminarAI and toanmath contain no GSM8K items. The highest cosines (0.80–0.83) are generic "Một lớp học có 24 học sinh…" items.

### 4.2 (b) MetaMathQA-vi (5CD-AI) vs the Vi test sets
- **Vi-MATH test.** 39 items have 13-gram coverage ≥ 0.5 and 336 meet the review rule. Examples:
  - verbatim: `algebra/320` "Có bao nhiêu số nguyên trong tập nghiệm của |x−2| ≤ 5.6"; `algebra/1038` (the parabola through (2,3),(4,3)); `intermediate_algebra/5` (the AMC ⌊(3¹⁰⁰+2¹⁰⁰)/(3⁹⁶+2⁹⁶)⌋ item);
  - changed values: `intermediate_algebra/704` (min 2/a+1/b vs 1/a+2/b), `precalculus/166`, `intermediate_algebra/199` (+2018 vs −2018), `counting_and_probability/37` (50 vs 20 houses).

  These are MATH train/test near-twins carried into MetaMathQA. The 5CD-AI translation is Google MT, as is our Vi-MATH, so many survive verbatim in Vietnamese.
- **Vi-GSM8K test.** 0 lexical hits; 16 items at cos ≥ 0.85, all GSM8K-train template twins (`test/632` Max's stamps vs train "Bella mua tem…").
- **Implication for S5 (MetaMathQA-vi arm).** Remove the hits with the G4 rule (§5) before training. The matched seeds number about 340 (~2.3 % of 15,013). Many more rows are affected, since each seed has ~26 rephrasings, and value-changed rephrasings are caught by the templated rule.

### 4.3 (a) External corpora vs Vi-ExamMath (cached pages)
- **LuminarAI** (montoan.com.vn crawl of grade 6–12 textbook solutions; 04 called it "loigiaihay format", but rows are whole pages with misaligned question/answer fields):
  - 31 items meet the review rule; 221 have cos ≥ 0.85 and 15 have cos ≥ 0.90.
  - Hand inspection of the 15 pairs at ≥ 0.90:
    - **5 are the same problem**: Long An 2025 sample paper q4 (cone roof, d = 12 m, l = 8.5 m, 250,000 đ/m²) and q7 (cement factory, 15,300 tonnes); Quảng Nam 2025 sample paper q7 (die rolled 50 times); Tây Ninh 2023 q2 (x²−5x+6=0); Đồng Tháp 2021 q50 (cone R = 2, l = 5).
    - **7 are the same template with other values or options**, e.g. Bến Tre 2022 VAT problem (4.35 vs 21.7 million), "Hàm số nào là hàm số bậc nhất?".
    - **3 share only the concept.**
  - From 0.85 to 0.88 the matches are almost all the same question *type* (standard MCQ stems). That is why 0.85 is right for a generation gate but too loose for flagging benchmark items.
  - Several real copies have 13-gram coverage of only 0.35–0.41 because of light edits ("8,5 m" vs "8,5 mét", "Hình 26" vs "Hình minh họa"). The lexical and embedding signals are complementary.
- **By exam year (review rule):** 2017 0/5, 2018 2/192, 2019 3/411, 2020 1/292, 2021 6/447, 2022 1/111, 2023 3/545, **2025 15/450**, 2026 2/293, unknown year 0/297.
  - The 2025 papers, the first under the new curriculum and including many *minh họa* (sample) papers, reuse textbook exercises.
  - **This matters for C1/T5 §3.4 C1 (temporal DiD).** A 2025 item can be pre-cutoff content, because the textbooks were published before the models. Treat review-flagged items as pre-cutoff, or exclude them from the post-cutoff slice. The list is in `data/contam/scans/review_flags.json` (hand-off to X2).
- **hllj (both) and vinhnq29:** ≤ 2 Vi-ExamMath items each meet the rule (elementary-level generic problems).
- **MetaMathQA-vi:** 0 Vi-ExamMath items.

### 4.4 toanmath.com-alpha
- `python -m vi_math_verified_aug.eval.decontam toanmath-titles` gives the overlap file `data/contam/scans/toanmath_titles/toanmath_title_overlap.json`.
- **Contents:** 513,271 pages (one row per page image) in 11,219 documents.
  - `label` equals the document slug: 34–86 characters, identical to `path.split('/')[1]` for 100 % of rows in chunk 0.
  - THPT-level only: 2,523 "toán 12" slugs, 1,481 THPT-QG/TN slugs, 1,907 "toán 10" slugs; 0 slugs mention lớp 6–9 or vào 10.
  - The only 4 "tuyển sinh" slugs are university-entrance papers.
- **Matching Vi-ExamMath:** 274 exams matched on province slug + exam year (a "2021-2022" school year counts as the 2021 exam), excluding chuyên papers. **0 found.**
- **Conclusion:** text contamination from this corpus can only be measured after OCR, and it is very unlikely to touch grade-10 entrance exams. H7's "the contamination argument cannot rest on obscure source" still holds for loigiaihay/montoan content (§4.3), not for toanmath-alpha.

## 5. Recommended thresholds

### 5.1 G4: generated training item vs the test sets
Test sets: Vi-ExamMath, Vi-GSM8K, Vi-MATH, PolyMath-vi, DeepSEA-vi. Implemented in `decontam.Gate`. Reject the item if **any** of these holds:
1. it shares ≥ 1 exact **or** diacritic-stripped 13-gram with a test item. This keeps T1's "13-gram = 0"; a false reject is cheap at generation time;
2. **(new)** ≥ 50 % of a test item's *number-templated* 13-grams appear in it. This catches value-perturbed copies, which rules 1 and 3 miss;
3. bge-m3 cosine ≥ **0.85** to any Vietnamese test item, instead of T1's 0.92. At 0.85 the rule catches 97.0 % of re-translations and 98.8 % of number-perturbed copies, with 0.5–0.8 % hard-negative FPR. At 0.92 it would miss 15 % and 12 %;
4. bge-m3 cosine ≥ **0.80** to the English originals (GSM8K/MATH test). That catches 84 % of cross-lingual pairs at 0.5 % hard FPR. A higher cross-lingual threshold barely helps: 0.85 → 64 % recall.

If e5-small must be used, use **0.95** for Vi↔Vi.

### 5.2 G5: within-arm deduplication
Keep **MinHash (5-gram shingles) Jaccard < 0.8**, with LSH at 16 bands × 8 rows (threshold ≈ 0.71), as implemented in `Gate`. MinHash at 0.8 only catches near-verbatim duplicates; one changed number breaks 5 shingles. Value-perturbed siblings are by design in the symbolic arm, so G5 must not use the templated variant. The "≤ 3 accepted per seed per arm" cap in R5 handles siblings.

### 5.3 Benchmark-side review rule (C2 report, Vi-ExamMath QA)
Flag an item for human review if **any** of these holds:
1. 13-gram coverage ≥ 0.5;
2. for items ≥ 20 tokens, 13-gram coverage ≥ 0.3 or templated coverage ≥ 0.5;
3. bge-m3 cosine ≥ **0.90**.

Cosine 0.85 is too loose for short standard MCQ stems, and template rules on items < 20 tokens fire on every "Giải phương trình ax²+bx+c=0". On the current corpora this flags 33/3,024 Vi-ExamMath units, 338/5,000 Vi-MATH items and 1,197/1,319 Vi-GSM8K items; the last group is entirely the vinhnq29 re-translation.

## 6. Limits, open items, deviations
- The embedding stage covered problem segments (LuminarAI) and seed questions (MetaMathQA-vi), not every row or rephrasing. Lexical scans covered every row in full.
- The hllj-crawl embedding uses the first 1,500 characters of each lesson.
- Vi-ExamMath targets come from my own page parser, not X2's (in-flux) `items.jsonl`. Re-run with the `jsonl:` source when X2 publishes. There are 297 units with no parsed exam year (non-`-aNNN` URLs).
- Not done: PolyMath-vi / DeepSEA-vi as targets (not in the repo); the English-seed cross-lingual scan (Vi test vs GSM8K/MATH train); the `query_en` side of MetaMathQA. The tool supports these through `jsonl:` sources.
- The `Gate` CLI has not run on real generated data, because `data/pilot/` belongs to G1. That is the obvious next step for G1/X5.
- Deviations from the brief:
  - The embedding stage ran on a Modal **A10G** (fallback list `["L4","A10G"]`) because no L4 was available for 12 min.
  - Extra files: `decontam_modal.py` and `decontam_report.py` in `eval/`; env `E:\venv_dc`.
  - My scratchpad was shared with another agent's patch scripts; I used `x5_`-prefixed names after noticing.

## 7. Files
- Code: `vi_math_verified_aug/eval/decontam.py`, `decontam_modal.py`, `decontam_report.py`; tests: `tests/test_decontam.py` (39 tests).
- Outputs (`data/contam/`, gitignored):
  - `scans/lex_<corpus>/{items.jsonl,summary.json,examples.md,corpus_snippets.jsonl}`;
  - `scans/final_table.json`, `scans/review_flags.json`, `scans/bench_index.json`, `scans/toanmath_titles/`;
  - `embed_modal/{smoke,main}/` (per-target top-3 cosines and doc ids, snippets, `target_emb.npy` (9,343 × 1,024 fp16), `summary.json` with calibration);
  - `embed_inputs/inputs.json`; `logs/` (all run logs, Modal billing before/after).

## 8. Addendum: re-scan against X2's final `data/vi_exam/items.jsonl` (4,209 items, 1,585 Core)

**What was run** (X2's files were only read, never edited):
- Lexical scans used the new source `vi_exam_items`, with text = `problem` (context + item text + options). One scan per corpus:
  `decontam scan --corpus <C> --target vi_exam_items --out data/contam/scans/vex_<C>`. Each took 9–339 s with peak RSS ≤ 361 MB.
- The bge-m3 Modal run (`--inputs data/contam/embed_inputs/inputs_vex.json --tag vex`) covered the 5 Vietnamese corpora as before, plus **`metamathqa_en`**: the 13,929 distinct English MetaMathQA seed questions (GSM8K/MATH train). That is the Vi↔EN check at τ = 0.80.
  - It ran on an A10G (L4/A10G fallback list) in 325 s and cost **$0.119**, against the $0.30 cap.
  - X5's total Modal spend is now **$0.274**. Logs: `data/contam/logs/modal_billing_*vex.txt`, `modal_vex.log`.
- Tables: `python -m vi_math_verified_aug.eval.decontam_report vex` (output in `data/contam/logs/vex_subsets.txt`, `data/contam/scans/vi_exam_subsets.json`).

**Per-item flag file: `data/contam/vi_exam_flags.jsonl`.** It has 846 rows covering 682 of 4,209 items, one row per (item, corpus, signal). Fields:
- `item_id`, `corpus`, `n_tokens`, `core`;
- `signal`, which fires under the recommended G4 thresholds and is one of `ng13` / `ng13_nodiac` (13-gram coverage ≥ 0.5), `tmpl13` (number-templated 13-gram coverage ≥ 0.5), `minhash` (J ≥ 0.5), `cos_vi` (bge-m3 ≥ 0.85 vs a Vietnamese corpus) or `cos_en` (≥ 0.80 vs the English seeds);
- `similarity`;
- `matched_id`, a corpus doc or segment id. Its text is in `data/contam/embed_modal/vex/snippets_<corpus>.jsonl` (embedding hits) or `data/contam/scans/vex_<corpus>/corpus_snippets.jsonl` (lexical hits);
- `review_rule`: whether the item also meets the stricter §5.3 rule.

**Core items per (corpus, signal):**

| corpus | signal | Core items |
|---|---|---|
| LuminarAI | cos_vi | 258 |
| LuminarAI | tmpl13 | 70 |
| LuminarAI | ng13 | 4 |
| LuminarAI | ng13_nodiac | 4 |
| MetaMathQA-vi | tmpl13 | 65 |
| MetaMathQA-vi | ng13 | 5 |
| MetaMathQA-vi | ng13_nodiac | 5 |
| MetaMathQA-vi | cos_vi | 4 |
| MetaMathQA-en | cos_en | 11 |
| vinhnq29 | cos_vi | 10 |
| vinhnq29 | tmpl13 | 1 |
| hllj_mcq | cos_vi | 8 |
| hllj_crawl | cos_vi | 2 |

MinHash ≥ 0.5 never fires.

**Post-cutoff Core subsets after removing flagged items.** The subsets use X2's `temporal` flags.

| subset | n | flagged, G4 thresholds | **remaining** | flagged, G4 without `tmpl13` on items < 20 tokens | remaining | flagged, review rule (§5.3) | **remaining** |
|---|---|---|---|---|---|---|---|
| Core, all years | 1,585 | 329 | 1,256 | 282 | 1,303 | 39 | 1,546 |
| **Core, exam 2026** | 112 | 24 | **88** | 22 | 90 | 2 | **110** |
| **Core, exam 2025+** | 333 | 88 | **245** | 79 | 254 | 15 | **318** |
| **Core, first_seen ≥ 2026-06** | 86 | 15 | **71** | 15 | 71 | 1 | **85** |

**How to read the two columns.**
- The G4 thresholds were tuned for a *generation gate*, where a false reject is cheap. On benchmark items they mostly flag shared *question types*, not shared items.
- I sampled flags drawn at random (seed 0). In 8 `cos_vi` flags from 0.85–0.90, 6 `tmpl13` flags and 5 `cos_en` flags, **none was the same item**. Typical matches:
  - "Giải hệ phương trình {x+y=2; 3x−2y=11}" vs a textbook system with other coefficients;
  - a 13-token "Giải phương trình x²−4x+3=0" whose number template matches any quadratic;
  - "Giải phương trình 4x−3=2−x" vs an English "Solve for x".
- The review-rule flags (39 Core) are where real copies sit. Examples:
  - Long An 2025 sample paper q7 (cement factory, cos 0.988, 13-gram coverage 0.60) and q3c (charity concert, 0.937 / 0.60);
  - Ninh Bình 2025 sample paper q1.4.2 (60 numbered cards, 0.923);
  - Bạc Liêu 2025 "(2x−3)(x+5)=0";
  - Thanh Hóa 2026 "2x²+3x−5=0", the only first_seen ≥ 2026-06 item flagged. It is a textbook-standard quadratic that also appears in MetaMathQA-vi.

**Recommendation for C1/T5 (§3.4 C1):**
- Primary post-cutoff slices: **remove the review-rule items** (2026: 110; 2025+: 318; first_seen ≥ 2026-06: 85).
- Robustness check: the **G4-cleaned slices** (88 / 245 / 71), reported as a sensitivity analysis ("no item resembling a public textbook/corpus problem"). Because their false positives are type-level, the G4 slices are also skewed away from standard question types.
- The main concentration of copies is the 2025 papers: 15 of 333 Core 2025+ items meet the review rule. 13 come from LuminarAI textbook pages and 2 from MetaMathQA-vi (short standard quadratics).
