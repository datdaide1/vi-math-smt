# 04 — Data & Evaluation re-validation (2026-11-07)

**Scope.** Re-check the Sept-2026 data and evaluation assumptions in `DATA_SOURCES.md`, `RESEARCH_PLAN.md` §4–§5 and §6.4, and
memory `vi-math-dataset-landscape.md` against live sources. The aim is a **Q2-level** paper: C1 is the Vi-ExamMath benchmark and C2 is the
controlled augmentation comparison. The hand-off is internal, but the benchmark must be **releasable**.

**Evidence convention.** `[E:x]` marks evidence collected in this session (a fetched URL, an HF API or datasets-server response, or command
output). The scratch artefacts are in
`C:\Users\tranh\AppData\Local\Temp\claude\E--MATH-REASONING-THESIS-vi-math-smt\884a7b4b-700c-4c10-ad97-d56477e30ce1\scratchpad\data\`
(file names are given inline). `[UNVERIFIED]` marks a claim carried over from memory or prior notes that was not re-checked today.
Nothing in this document is legal advice.

---

## 0. Executive summary — what changed versus the Sept-2026 hypotheses

| # | Sept-2026 hypothesis | Nov-2026 finding | Consequence |
|---|---|---|---|
| H1 | "MGSM-vi" is a usable independent Vietnamese eval | The original MGSM (`juletxara/mgsm`) has **no `vi`** config. Every `vi` MGSM variant (BenchMAX_Math, CohereLabs/global-mgsm, PolyMath-low) is the **same 250 GSM8K-test items** ("Janet's ducks" is row 0) `[E: eval_splits.txt, samples.txt]` | These variants are **not independent** of Vi-GSM8K. Use them only as an **alternative-translation probe**, not as an extra benchmark |
| H2 | Vi-GSM8K (1,319) is a clean co-primary eval | The golds are identical to the original GSM8K. However, GSM8K-Platinum **removed 110** flawed items and **revised 10** `[E: gsm_check.py output]` | Report on **Vi-GSM8K-Platinum (1,209)** as the headline number and keep 1,319 for comparability |
| H3 | Vi-MATH translations are fine | Verified clean: algebra and prealgebra are index-aligned with `EleutherAI/hendrycks_math` test (level match 1187/1187 and 871/871). Boxed golds are identical in 1180/1187 and 860/871, and every difference is a translated `\text{}` unit `[E: q5.out]` | Keep. Add a **Vi-MATH-500** subset (206 algebra+prealgebra items in MATH-500) `[E: math500.json]` |
| H4 | The local `metaMathQA-395K_json_vi` is a "known-good" S5 corpus | Its schema matches **5CD-AI** exactly (HF card has **no license** tag). On a 5,000-row sample, **12.1 % of LaTeX-bearing rows lose ≥1 LaTeX command**, for example `\frac`→`\ sự cố`, `\begin`→`started`, `\pi`→`số Pi` `[E: q4.out, q3.py]` | S5 needs a **LaTeX-parity + answer-parity filter** before use, which also becomes a C3 data-quality datum |
| H5 | VnExpress publishes đề + lời giải as HTML+LaTeX | The 2026 Quảng Ninh grade-10 article shows the exam as **images** (no MathJax) `[E: WebFetch vnexpress …5089346.html]` | VnExpress drops to an OCR-only / timeline-evidence source |
| H6 | toanmath.com Word/PDF can be crawled and converted with pandoc | `robots.txt` **disallows `/toanmath-word/`, `/toanmath-pdf/`, `/thcs-pdf/`, `/thcs-word/`**. The 2026 exam page offers only an embedded PDF from `/thcs-pdf/` `[E: robots/, pages/tm_cantho.html]` | Do **not** bulk-crawl toanmath files. Manual per-item reference only |
| H7 | Vi-ExamMath sources are not yet in open training corpora | `daominhwysi/toanmath.com-alpha` (Apr 2026) holds **513,271 page images + labels, about 312 GB**, scraped from toanmath. `LuminarAI/vietnamese-ms-hs-textbook-math-300K` (306k rows) is loigiaihay-formatted text `[E: eval_meta.tsv, samples.txt, datasets-server /info]` | The contamination argument cannot rest on "obscure source". It must rest on **post-release dates + n-gram scans against these corpora** (§5) |
| H8 | Vi-Elementary-Bench has 1,010 items; V-Math releases data | arXiv 2604.17794 lists **no size and no data link**, and its HTML version returns 404. V-Math's code repo "will be made public upon the reviewers' request", with no dataset link. THPT-Ladder (632 items / 21 exams / 11 subjects) has no data link `[E: arXiv abs/html fetches]` | Still no **released** native Vietnamese free-response math benchmark with solutions. The C1 gap holds. "1,010 items" is **[UNVERIFIED]** |
| H9 | Our `eval/math_verify.py` is ready | All 47 tests pass, but probing found **false positives**: `"125"≡"0.125"`, and a partial answer `"8"≡"x=7; y=8"`. It also has **misses**: `1{,}5`, set braces, nested `\frac{\sqrt{}}{}`. HF `math-verify` 0.9.0 fixes the structural cases but **misreads Vietnamese numbers**: `\boxed{2,5}`→set {2,5}, `1.000.000`→1.0, `885.000 đồng`→885 `[E: probe_mv.out, probe_hf.out]` | Keep our module as the **Vietnamese normalisation layer** in front of HF math-verify. Fix the six bugs listed in §4.3 |
| H10 | Redistributing official exam text is "usable with attribution" | Luật SHTT **Điều 15(2)** excludes "văn bản hành chính" from copyright. **NĐ 17/2023 Điều 8(2)** defines that term as "văn bản của cơ quan nhà nước, …". Third-party solutions are clearly protected. The new **Điều 7(5)** (Luật 131/2025/QH15, in force 2026-04-01) covers *use* for research and AI training, **not redistribution** `[E: caselaw.vn, cov.gov.vn, luatvietan.vn]` | Release official exam text, our own solutions and source URLs only (§3.4) |

---

## 1. Training data candidates (Task 1)

All rows come from `https://huggingface.co/api/datasets/<id>` plus `datasets-server /size|/info` `[E: train_meta.tsv, eval_meta.tsv]`.
"Verif." describes the answer field that makes an item machine-checkable.

### 1.1 English math corpora with chain-of-thought

| Dataset | Rows (verified) | License (HF tag) | Verif. field | CoT style / generator | Notes for this project |
|---|---|---|---|---|---|
| `openai/gsm8k` | 7,473 train / 1,319 test | MIT | `#### n` | human | **Seed (S0).** Same family as Vi-GSM8K test |
| `EleutherAI/hendrycks_math` | 12,500 (7.5k/5k) | MIT | `\boxed{}` | human | **Seed** = algebra (1,744 train) + prealgebra (1,205 train) = 2,949 `[E: ura-hcmut info]` |
| `meta-math/MetaMathQA` | 395,000 | MIT | `The answer is:` | GPT-3.5 rephrase/FOBAR/SV of GSM8K/MATH **train** | Source of S5 (EN) and the RQ-E English control. Built from the *same seeds* as S0, so arms stay comparable |
| `microsoft/orca-math-word-problems-200k` | 200,035 | MIT | free text | GPT-4-Turbo | GSM-style. No structured answer, so it needs extraction |
| `TIGER-Lab/MathInstruct` | 262,039 | MIT (mixed upstream) | partial | CoT+PoT | Mixed upstream licenses. Avoid |
| `hkust-nlp/dart-math-hard` | 585,392 | MIT | yes | DeepSeekMath-7B-RL rejection-sampled | Difficulty-aware RS, a strong prior-art baseline to cite |
| `nvidia/OpenMathInstruct-1` | 6.88 M | **other** (NVIDIA) | `expected_answer` | Mixtral code-interpreter | The 5CD-AI 50k vi slice derives from this, so it inherits the "other" license |
| `nvidia/OpenMathInstruct-2` | 13.97 M (+1M/2M/5M slices) | CC-BY-4.0 | `expected_answer` | Llama-3.1-405B | Best-licensed large GSM/MATH-augmented corpus. Candidate **S5b** if we translate it ourselves |
| `AI-MO/NuminaMath-CoT` | 859,494 | Apache-2.0 | partial | GPT-4o | Includes gsm8k 7,345 / cn_k12 276k (README) |
| `AI-MO/NuminaMath-1.5` | 896,215 | Apache-2.0 | `answer` (+ `notfound`/proof flags) | GPT-4o | 631,522 "word" items. cn_k12 268,819. synthetic_amc removed (README table) |
| `open-r1/OpenR1-Math-220k` | 93,733 default / 225,129 all | Apache-2.0 | `answer`, `correctness_math_verify` | DeepSeek-R1 long CoT | Already Math-Verify-labelled |
| `zwhe99/DeepMath-103K` | 103,022 | MIT | `final_answer`, `difficulty` | R1 ×3 | Decontaminated against many benchmarks (arXiv 2504.11456). Mostly hard |
| `nvidia/OpenMathReasoning` | 5.68 M (cot 3.2M / tir 1.7M / genselect 0.57M) | CC-BY-4.0 | `expected_answer` | R1/QwQ long CoT. AoPS-sourced | Competition level. Lengths far exceed our 512–640-token budget |
| `nvidia/Nemotron-Math-v2` | 7.09 M | CC-BY-4.0 / CC-BY-SA-4.0 (per row) | `expected_answer` | multiple | Has a per-row `license` column, so filtering by row is mandatory |
| `open-thoughts/OpenThoughts3-1.2M` | 1.2 M | Apache-2.0 | none structured | QwQ long CoT, mixed domains | Out of scope |
| `a-m-team/AM-Thinking-v1-Distilled` | rows=0 via API (jsonl) | no tag. Card: "**research purposes only**, commercial use … strictly prohibited" | Math-Verify graded | long CoT | **Exclude** from anything released |
| `a-m-team/AM-DeepSeek-R1-Distilled-1.4M` | – | CC-BY-NC-4.0 | – | – | Exclude (NC) |
| `SynthLabsAI/Big-Math-RL-Verified` | gated (401) | Apache-2.0 | yes | RL-filtered | Gated. Not needed |
| `GAIR/LIMO` 817 · `simplescaling/s1K-1.1` 1,000 | | Apache-2.0 · MIT | yes | long CoT | Vi-S1K is a translation of s1K (paper abstract), so it is irrelevant for ≤640-token SFT |
| `madrylab/gsm8k-platinum` | 1,209 | MIT | `#### n` | – | **Use for evaluation** (cleaned GSM8K test) |

**Takeaways.**
1. The long-CoT corpora of 2025–26 (OpenMathReasoning, OpenR1, OpenThoughts3, AM-*, Nemotron-Math) are built for distilling R1-style traces of
   several thousand tokens. They do **not** fit the project's ≤1.7B, `max_new_tokens` 512–640, SFT-only regime (§6.3–6.4). Cite them as
   context and do not use them as arms.
2. For a *controlled* comparison, the seed must be shared across arms. GSM8K-train + MATH-algebra/prealgebra-train (MIT) remains correct,
   and MetaMathQA (MIT) is an augmentation of exactly these seeds, so S5 is a fair "existing best practice" arm.
3. If a second translate-train source is wanted (S5b), **OpenMathInstruct-2 (CC-BY-4.0)** is preferable to the 5CD-AI
   OpenMathInstruct-1 slice, which is licensed "other" upstream with no license on the vi card.

### 1.2 Vietnamese math training sets

| Dataset | Rows | License tag | Native or translated | Quality / licence concerns |
|---|---|---|---|---|
| `5CD-AI/Vietnamese-395k-meta-math-MetaMathQA-gg-translated` (= local `data/test/metaMATH/…_vi.jsonl`, same 7 fields) | 395,000 | **none** | Google-Translate of MIT data | **12.1 % LaTeX corruption** in LaTeX rows. Number-multiset mismatch in 444/5,000 queries (mostly benign rewording). Final-answer disagreement 39/5,000 `[E: q3/q4]` |
| `5CD-AI/…OpenMathInstruct-1-50k-gg-translated` | 49,999 | none (upstream "other") | MT | Code-interpreter traces. Licence chain is weak |
| `5CD-AI/…orca-math-…-200k-gg-translated` | 200,035 | none | MT | No structured answer |
| `ura-hcmut/Vietnamese-MATH` | 12,500 (7.5k/5k, 7 configs, EN+VI columns) | **MIT** | MT (parallel) | Useful for an **independent second translation** of MATH (test-set translation variance) and as a Vi-MATH-train fallback |
| `stair-lab/MATH_vi` | 25,000 (two MT engines: `azr`, `gcp`) | CC-BY-NC-SA-4.0 | MT | NC. Translation-variance probe only |
| `danganhdat/lighteval-math-vi-translated` | 12,500 | none | MT | – |
| `hllj/vi_grade_school_math_mcq` | 2,733 | "MIT" | native (vietjack crawl, per memory) | Licence is the uploader's claim over crawled content, so it is **not trustworthy** |
| `hllj/vi_math_problem_crawl` | 10,246 (`title, contents, url`) | "MIT" | native, raw tech12h-style text, uncleaned `[E: samples.txt]` | Same issue |
| `vinhnq29/vietnamese-elementary-math` | 17,003 | none | native? | Unknown provenance |
| `LuminarAI/vietnamese-ms-hs-textbook-math-300K` | 306,406 | none (card text says apache-2.0) | native, **loigiaihay-format** ("Phương pháp giải – Xem chi tiết") | **Contamination vector** for Vi-ExamMath, so it must be scanned (§5) |
| `daominhwysi/toanmath.com-alpha` | 513,271 page images (`image,label,path`), about 312 GB parquet | "apache-2.0" (self-declared) | native, scraped toanmath PDFs | **Contamination vector** (OCR-style training data). Too large to download, so scan its `label` text in streaming mode if feasible |
| `schneiderkamplab/dfm14-vi-synthetic-math-code-r1` | 423 | other | synthetic | – |
| `KalvinPhan/VietMathInstruct-mini` | 250 | MIT | ? | Tiny |
| Vi-S1K (arXiv 2604.17794) | ? | ? | Gemini-2.5-Flash-Lite translation of s1K | **No public link found** `[E]` |
| Zalo AI 2023 Elementary Math | ~1k | ? | native | **[UNVERIFIED]**, not re-checked today |

HF search over 17 queries found no new **native, licensed, answer-verified** Vietnamese math training set in 2026 `[E: hf_search.txt]`.

### 1.3 Recommendation: seed and training-data strategy

| Option | Evidence for | Evidence against | Verdict |
|---|---|---|---|
| **Translated seeds (current: Vi-GSM8K-train + Vi-MATH-alg/prealg-train)** | MIT upstream. Vi-MATH translation verified gold-preserving (H3). Matches the Vi-GSM8K/Vi-MATH eval domain. All arms share it | Translationese: US units and dollars, mixed decimal conventions (Vi-GSM8K test has 212 EN-style `d.dd` vs 12 VN-style `d,dd` decimals in its questions `[E]`) | **Keep as the C2 seed.** It is the only choice that keeps C2 controlled and licence-clean |
| Native Vietnamese seeds (hllj, LuminarAI, scraped) | Authentic language | No usable licence, uncleaned, and they **overlap the Vi-ExamMath source sites**, which would leak the benchmark into training | **Reject for training.** Use only as contamination-scan corpora |
| Synthesized (S1/S2) | This *is* the research question | – | Arms, not seeds |
| Translate-train (S5) from 5CD-AI MetaMathQA-vi | Already on disk. Same seed lineage | No licence tag. 12 % LaTeX corruption | **Use with filters**: (a) LaTeX command multiset parity EN↔VI, (b) final-answer parity, (c) number-multiset parity, (d) type ∈ {GSM_*, MATH algebra/prealgebra}. Report the filter's rejection rate as a C3 data-quality datum. For any *released* S5 subset, re-translate the filtered EN rows ourselves (MIT upstream → clean chain) |

To make the translation-artifact argument concrete, add a **small native-style robustness slice** to the C2 evaluation: a "localised"
Vi-GSM8K subset of about 150 items where units and currency are converted to Vietnamese conventions (đồng, km, decimal comma). This
measures whether arms overfit to translationese. It costs little and is a reviewer-proof add-on.

---

## 2. Evaluation sets (Task 2)

### 2.1 External Vietnamese / multilingual math benchmarks

| Benchmark | Vi size (verified) | Native or translated | Format | License | Availability | Contamination / independence |
|---|---|---|---|---|---|---|
| **BenchMAX_Math** `vi` (`LLaMAX/BenchMAX_Math`) | 250 test | translated MGSM, i.e. **GSM8K test** | numeric | CC-BY-4.0 | HF | **Same items as Vi-GSM8K** → alternative-translation probe |
| **CohereLabs/global-mgsm** `vi` | 250 | its `vi` is taken from BenchMAX (card "Source Attributions") | numeric + VN instruction "Đáp án:" | per source | HF | Duplicate of the above |
| **Qwen/PolyMath** `vi` | 4 × 125 = 500 (low/medium/high/top) | expert-calibrated translation. `low` = MGSM items | numeric/expr | **no license tag** | HF | `low` overlaps GSM8K. `medium`/`high`/`top` are harder and useful as a ceiling slice |
| **amphora/MCLM** `MT-MATH100` | 99 items, one `vi` column | translated MATH | expr | MIT | HF | MATH-test derived, so it overlaps Vi-MATH |
| **DeepSEA** (Google, arXiv 2610.26155, 2026-10-19) | MGSM in 13 langs incl. vi | professional translation | numeric | Apache-2.0 / CC-BY-4.0 (MGSM part) | GitHub, **password-zipped** to resist crawling | GSM8K-test derived. Its anti-crawl packaging is a good model for our release |
| **M3Exam-vi** (via `SeaLLMs/SeaExam`, config `m3exam-vietnamese`) | 1,745 total, **276 math MCQ** (202 high / 50 mid / 24 low) `[E]` | **native** official exam MCQ | MCQ (4 options) | Apache-2.0 (SeaExam tag). Original M3Exam licence **[UNVERIFIED]** | HF. Also a lighteval task `m3exams` | Pre-2023 public exams, so **likely contaminated**. Leaderboard supplement only |
| **VNHSGE** (`github.com/Xdao85/VNHSGE`) | math 2,500 MCQ (50 exams × 50), 2019–2023, with explanation field `E` | native | MCQ | **no license** (only "please cite") | GitHub (Word/JSON) | Pre-2024 and widely mirrored (HF `Elfsong/vnhsge-v`, `studyforptd/VNHSGE-processed-dpo`), so contaminated |
| **VMLU** (vmlu.ai) | Vi-MQA v1.5, 58 subjects of about 200 Q. 5 math subjects (elementary, middle, high school, discrete, statistics & probability) | native | MCQ | **not stated** on site | site/GitHub. Mirrors on HF (`tridm/VMLU` 10,880 rows) | MCQ only. Licence unclear, so cite and do not redistribute |
| **Global-MMLU** `vi` / **MMLU-ProX** `vi` | math subjects inside | translated | MCQ | Apache-2.0 / MIT | HF. Tasks exist in lm-eval (`global_mmlu`, `mmlu_prox/vi`) | Translated MMLU, so contaminated. Optional |
| **V-Math / NHSGMEs** (arXiv 2509.12251) | about 500 exam sets in 2025 format (MCQ + T/F + short answer) | native | mixed | none stated | **not released** ("upon reviewers' request") | – |
| **THPT-Ladder** (arXiv 2608.18336, 2026-08-18) | 632 items, 21 exams, 11 subjects | native | T/F with partial credit | none | **no data link** | – |
| **Vi-Elementary-Bench** (arXiv 2604.17794, 2026-04-20) | **size not stated** | ? | LLM-as-judge | paper CC-BY-4.0 | **no data link** | Closest scoop risk. Keep monitoring |
| **VMMU** (arXiv 2508.13680) | 2,548 MCQ, 7 domains incl. math | native, multimodal | MCQ + image | – | – | Multimodal, so out of scope |

**arXiv sweep 2025-06 → 2026-11** (`abs:vietnamese AND {mathematical, benchmark, reasoning, exam}`) found **no new Vietnamese math benchmark**
beyond those above `[E: arxiv_recent.txt]`. The C1 gap still holds: no **released**, native, free-response, solver-verified Vietnamese
grade-10 / HSG math benchmark exists.

### 2.2 The project's own test sets (inspected)

| File | Rows | Schema | Check | Result |
|---|---|---|---|---|
| `data/test/gsm8k/gsm8k_test_vietnamese.jsonl` | 1,319 | `question, answer(with <<>> and ####), ground_truth` | gold vs `openai/gsm8k` (index-aligned) | **0 mismatches** `[E]` |
| | | | vs GSM8K-Platinum | 110 removed + 10 revised (statuses: consensus 1,100 / verified 99 / revised 10) `[E]` |
| | | | number-multiset EN↔VI | 36/1,319 flagged. Spot check of 8: all benign (repetition dropped, "1/2"→"một nửa", coin values made explicit) `[E]` |
| | | | conventions | Mixed decimal and thousands separators. 534 `$` signs in questions. All golds are integers |
| `data/test/math/*_vi.jsonl` (7 files) | 5,000 (= full MATH test, not MATH-500) | `problem, level, type, solution, subject` | aligned to `EleutherAI/hendrycks_math` test (alg/prealg) | level-aligned 100 %. Boxed golds identical 1180/1187 and 860/871. 48 boxed answers contain translated `\text{}` units `[E]` |
| | | | LaTeX hygiene | 15 problems with unbalanced braces, 3 with an odd number of `$` (all 5,000) |
| `data/test/metaMATH/…_vi.jsonl` | 395,000 | 5CD-AI schema | see §1.2 | 12.1 % LaTeX loss among LaTeX rows |

### 2.3 Recommended evaluation suite

| Role | Set | N | Why |
|---|---|---|---|
| **C2 co-primary #1** | **Vi-GSM8K-Platinum** | 1,209 | Removes the 8.3 % of flawed items. Re-translate the 10 revised questions. Also report full 1,319 for literature comparability |
| **C2 co-primary #2** | **Vi-ExamMath-Core** | 300–400 | Native, dated, contamination-controlled |
| C2 secondary | **Vi-MATH in-scope** (algebra + prealgebra test) | 2,058 | Matches the seed scope. Add the **Vi-MATH-500 alg+prealg** slice (206) for comparability with MATH-500 numbers |
| Robustness | GSM-Plus-style VN perturbations + Vi-ExamMath `template` subset | ~150–300 | As planned |
| **Translation-variance probe (new)** | BenchMAX_Math `vi` (250) **and** our Vi-GSM8K rows for the same 250 items | 250 | Same problems, two translations. The accuracy gap bounds the translation artifact |
| Leaderboard only | PolyMath-vi medium/high/top (375), M3Exam-vi math (276 MCQ), Vi-ExamMath-Hard | – | Ceiling and native MCQ context. Never a C2 statistic |

Statistics at Q2 level: report a paired bootstrap (10k) 95 % CI per arm, a **McNemar / paired permutation test** for arm pairs on the same
items, ≥3 seeds, and the item-level prediction files.

---

## 3. Benchmark construction: sources and releasability (Task 3)

### 3.1 Source reachability (fetched 2026-11-07)

| Source | Reachable | robots.txt (relevant lines) | Content format seen | ToS / copyright notice | Role |
|---|---|---|---|---|---|
| **loigiaihay.com** | 200. Prior-prototype URL still live (443 KB, 148 MathJax `\(`) | `Allow: /` (only rating/comment AJAX disallowed) | HTML + MathJax LaTeX, sub-question blocks | meta `Copyright (C) 2013 Loigiaihay.com`. `/policy/vi.html` has **no IP/scraping clause** (operator: Công ty CP công nghệ giáo dục Thành Phát) | **Primary scrape source** (problem text + pointer to solution) |
| **thcs.toanmath.com / toanmath.com** | 200. 2026 listing shows 2026-27 exams (Cần Thơ, Cà Mau, Gia Lai, …) | **Disallow `/thcs-pdf/`, `/thcs-word/`, `/data-thcs/`, `/toanmath-pdf/`, `/toanmath-word/`** | Exam = embedded **PDF** at a disallowed path | "Copyright © 2026 TOANMATH.com". No ToS | **Index / discovery only**. Do not bulk-download files |
| **tak12.com** | 200 | `Disallow: /tai-lieu/` | React SPA (needs JS rendering) | none found | Low priority |
| **vnexpress.net** | 200. `/giao-duc/tuyen-sinh/lop-10` lists 2026 exam articles | `Allow: /` (AI user-agents explicitly allowed) | 2026 exam as **images** | none in article body | **Timeline evidence** (publication timestamp) + OCR fallback |
| vietnamnet.vn / tuoitre.vn | 200 | `Allow: /` | not inspected | – | Timeline evidence |
| **hanoi.edu.vn** (Sở GD&ĐT Hà Nội) | 200 | `Disallow: /tai/` (download path), `/Config/` | Announcements found (tuyển sinh 10 2026-27) | state body | **Authoritative source** for official text and answer keys (manual download) |
| diendantoanhoc.org | 403, Cloudflare managed challenge | – | – | – | **Not automatable** (challenge bypass is off-limits). Drop |
| vietjack.com | robots 404 | – | – | – | Not needed |
| tuyensinh247.com | 200 | long disallow list | – | – | Optional |

Implication: the measured yield of the prototype (13 exams → 58 regex-answer items from one loigiaihay listing) remains the realistic
basis. Volume for Core-400 must come from **loigiaihay multi-province + multi-year** (its robots allow it), with official PDFs from the
Sở websites used manually for **verification of text fidelity**, not as a crawl target.

### 3.2 Legal status — what the statutes say

| Provision | Text (verified) | Bearing on Vi-ExamMath |
|---|---|---|
| **Luật SHTT Điều 15(2)** | Not protected: "Văn bản quy phạm pháp luật, văn bản hành chính, văn bản khác thuộc lĩnh vực tư pháp và bản dịch chính thức của văn bản đó" `[E: caselaw.vn Luật 50/2005]` | Exams issued by a state body may fall under "văn bản hành chính" |
| **NĐ 17/2023/NĐ-CP Điều 8(2)** | "Văn bản hành chính quy định tại khoản 2 Điều 15 … bao gồm văn bản của cơ quan nhà nước, tổ chức chính trị, …" `[E: caselaw.vn NĐ 17/2023]` | The definition is by **issuer** ("documents of state agencies"), not by genre. An official đề + đáp án/hướng dẫn chấm issued by **Bộ GD&ĐT or a Sở GD&ĐT** therefore has a credible claim to be excluded from protection |
| Luật SHTT Điều 15 as amended 2025 | Adds a clause excluding standalone ideas, slogans and titles `[E: cov.gov.vn]`. Clause 2 unchanged per that summary | No change to the analysis |
| **Điều 7(5)** (Luật 131/2025/QH15, in force **2026-04-01**) | Permits using lawfully published texts and data "để phục vụ mục đích nghiên cứu khoa học, thử nghiệm, huấn luyện hệ thống trí tuệ nhân tạo" if it does not unreasonably harm rightsholders. The Government will set conditions for copyrighted material `[E: luatvietan.vn, cov.gov.vn]` | Covers **our scraping and use** for research and evaluation. It does **not** authorise **redistributing** third-party works |
| NĐ 17/2023 Điều 25, 28 | Private copying for research is capped (≤10 % of a printed work). Quotation must serve commentary and cite the source `[E]` | Wholesale republication of third-party solutions is not "trích dẫn hợp lý" |

**Caveats.** (i) No court decision or Copyright Office guidance was found that classifies **exam papers** specifically under Điều 15(2), so
this is a reasoned reading, not settled law. (ii) "Cơ quan nhà nước" fits Bộ GD&ĐT and the Sở GD&ĐT (provincial specialised agencies)
**[UNVERIFIED characterisation; standard administrative-law reading]**. It fits **less clearly** for chuyên schools run by universities
(e.g. Chuyên KHTN under VNU, a public service unit rather than a state agency) and **not at all** for private schools or teacher-made
"đề minh họa / đề thi thử". (iii) Lời giải chi tiết on loigiaihay, VnExpress and toanmath are original works of their authors or sites
and **are protected**.

### 3.3 Is "official text + own solutions + URLs" releasable?

Yes. This is the defensible design, and it is what a Q2 benchmark paper needs (the benchmark itself can be released).

| Field | Release? | Basis |
|---|---|---|
| Problem statement of an **official** Bộ/Sở exam (verbatim, re-typed or normalised by us) | **Yes**, with issuer, year, province, source URL | Điều 15(2) + NĐ 17/2023 Điều 8(2). Normalisation is ours |
| Official final answer / đáp án key | **Yes** | Same issuer, so the same exclusion applies. A bare numeric answer is also a fact |
| **Our own** reference solution (written by us or by our LLM and then human-checked), solver certificate, metadata | **Yes**, under CC-BY-4.0 | Our work |
| Third-party worked solution (loigiaihay, VnExpress, toanmath) | **No.** Release `solution_url` only, plus an optional "rehydrate" script for users | Protected work. Điều 7(5) covers our *use*, not redistribution |
| Items from non-state issuers (private schools, thi thử, university-run chuyên) | **Exclude from the public release**, or release only with permission | No Điều 15(2) cover |
| Figures/images from third-party pages | **No** (Core is figure-free anyway) | – |

Release mechanics, borrowed from DeepSEA: an HF dataset with **gated auto-accept** plus a **canary GUID** in every file, a takedown
contact, and a per-item `issuer_type ∈ {state, university, private}` and `license_basis` field. This makes provenance complete for the
internal hand-off and enables a public release that can be filtered by issuer type.

---

## 4. Evaluation tooling (Task 4)

### 4.1 Versions (PyPI JSON, fetched 2026-11-07) `[E: pypi_versions.txt]` and GitHub API

| Tool | Version / date | License | Vietnamese-relevant facts | Verdict |
|---|---|---|---|---|
| **math-verify** (HF) | **0.9.0** (2026-01-10). Repo last push 2026-01-10 | Apache-2.0 | Good at sets, intervals, `\frac{\sqrt 3}{2}`, `%`, units, `x=2,y=3`↔`(2;3)`. **Wrong on VN numbers**: `\boxed{2,5}`→{2,5}, `1{,}5`→{1,5}, `1.000.000`→1.0, `885.000 đồng`→885. `x=2 hoặc x=3`→keeps only `x=3`. On Windows its timeouts spawn processes and crash, so set `parsing_timeout=None, timeout_seconds=None` or run on Linux `[E: probe_hf.out]` | **Adopt as the equivalence backend** behind a VN pre-normaliser |
| latex2sympy2-extended | 1.11.0 (2026-01-10) | MIT | math-verify dependency | – |
| **lm-evaluation-harness** | **0.4.13** (2026-08-31) | MIT | Tasks: `mgsm` (**no vi**), `global_mmlu`, `mmlu_prox/vi`, `gsm8k_platinum`, `gsm_plus`. `minerva_math` scores with `math_verify` `[E: GitHub API]` | Use for MCQ externals only |
| **lighteval** | **0.13.0** (2025-11-24, about a year old) | MIT | Multilingual tasks: `mgsm` (no vi), `m3exams` (**vi**), `global_mmlu` | Optional for M3Exam-vi |
| **OpenCompass** | **0.5.5** (2026-11-06) | Apache-2.0 | Heavy. Hosts CompassVerifier (arXiv 2508.03686) | Not needed |
| **evalchemy** | **not on PyPI**. GitHub last push 2026-02-24, no license field | – | Wraps lm-eval | Skip |
| **Qwen2.5-Math evaluation** | GitHub last push **2025-01-11**, no license field | – | Stale. English/Chinese parsing | Skip (cite only) |
| **vLLM** | 0.33.0 (2026-11-02) | Apache-2.0 | Python ≥3.11,<3.15 | Inference engine (as planned) |
| xVerify (arXiv 2504.10481) / CompassVerifier | GitHub active 2026-10-28 / 2025-08 | NOASSERTION / none | Trained judges. Vietnamese performance **[UNVERIFIED]** | Optional second opinion only |

### 4.2 The project's `vi_math_verified_aug/eval/math_verify.py`

`pytest tests/test_math_verify.py` passes (47 checks). The probe in `probe_mv.out` found:

| # | Bug | Example (pred, gold → result) | Severity |
|---|---|---|---|
| B1 | Thousands-collapse also applied to **gold decimals with exactly 3 digits** | `"125"` vs `"0.125"` → **True**. `"3500"` vs `"3.500"` → True | **False positive** |
| B2 | Gold side also runs `extract_answer` (last-number fallback), so a **partial answer to a multi-answer gold is accepted** | `"8"` vs `"x=7; y=8"` → **True**. `"3"`-only vs `"2; 3"` → True | **False positive.** Critical for Vi-ExamMath (systems, two-root equations) |
| B3 | `x = 2 hoặc x = 3` → keeps the last `=` RHS | normalises to `3` | Miss |
| B4 | LaTeX decimal comma `1{,}5` not handled | False vs 1.5 | Miss (common in VN Word→LaTeX) |
| B5 | Nested braces in `\frac{\sqrt{3}}{2}`, set braces `\{-1;2\}` | False | Miss |
| B6 | `"1,500"` read as 1500 regardless of locale. `extract_answer` returns the cue's whole tail (`Vậy x=4 thì A=2` → 2) | – | Locale-ambiguous / miss |

**Recommendation: keep the module and re-architect it.** Do not replace it with a stock harness, because no harness handles Vietnamese
number conventions.

1. **Gold is structured, never extracted.** Store `answer_canonical` (LaTeX) + `answer_type ∈ {int, rational, decimal, radical, tuple,
   set, interval, expr}` + `locale`. Only predictions go through extraction. This fixes B1 and B2.
2. **VN pre-normaliser** (our code): `{,}`→`.`. A locale-aware separator rule (Vi-ExamMath = `vi`, where comma is decimal and dot is
   thousands; Vi-GSM8K/Vi-MATH = mixed, so decide per item by the gold format). Map `hoặc` / `và` / `;` to sets for `set` types. Strip
   unit words.
3. **Equivalence**: `math_verify.parse/verify` on the normalised strings, then fall back to our SymPy/numeric path. Tuples are ordered and
   sets are unordered according to `answer_type`.
4. Grow the test file to ≥150 hand-labelled cases, including **≥50 real model outputs** from Qwen3-1.7B on Vi-ExamMath. Report the
   scorer's **false-positive and false-negative rates against a 300-item human-adjudicated sample** in the paper. TinyV (arXiv
   2505.14625) reports that rule-based verification leaves many false negatives, with >38 % of responses affected in their Big-Math
   setting, so reviewers will ask for this.

### 4.3 LLM-as-judge

Use it only as an **adjudicator** for items where the rule scorer fails to parse or where rule and judge disagree. Never use it as the
primary metric. Protocol: `deepseek-v4-flash` on NIM [model availability per memory, **UNVERIFIED** today], reference-guided binary
judgement. Validate on 300 stratified items against 2 humans and report Cohen's κ and judge FP/FN. General LLM-judge biases (position,
verbosity, self-preference) are documented in Zheng et al., arXiv 2306.05685. The judge model must differ from every generator used in
S2/S3 (self-preference).

### 4.4 Harness decision

Use a **thin custom runner**: vLLM + fixed prompt templates + the scorer above + JSONL logs (prompt, raw output, parsed answer, verdict,
scorer version). Use lm-eval 0.4.13 only for Global-MMLU/MMLU-ProX `vi` MCQ if those rows are reported. Reason: none of the harnesses
ships a Vietnamese free-response math task, and §6.4 already requires a single scoring function across all arms.

---

## 5. Contamination detection (Task 5)

| Method | Needs | Credibility in 2026 | Use here |
|---|---|---|---|
| **Temporal split** (Roberts et al., arXiv 2310.10628) | Item dates vs model release | **High** and cheap | Model dates verified on HF: Qwen3-0.6B/1.7B-Base **2025-04-28**, DeepSeek-R1-Distill-Qwen-1.5B 2025-01-20, Qwen2.5(-Math)-1.5B 2024-09, Sailor2-1B 2024-12-03 `[E]`. **June-2025 and June-2026 grade-10 exams post-date all baselines.** Report pre/post-cutoff accuracy gaps per model as the headline contamination evidence |
| **Exact / n-gram overlap** (8–13-gram, GPT-3/Llama style) on normalised text | Access to corpora | **High for corpora you can see**, blind to closed pretraining | Run against every training arm, 5CD-AI sets, `LuminarAI` (52 MB parquet), `hllj`, `vinhnq29`, and the `label` text of `toanmath.com-alpha` (stream if feasible). Implement in `eval/decontam.py`, currently a **stub** |
| **Embedding near-duplicate** (multilingual encoder) | Same | Medium. Catches paraphrases and translations | With the n-gram scan. Threshold calibrated on known translated pairs (Vi-GSM8K↔BenchMAX) |
| **Perturbation / rephrase tests** (GSM-Symbolic 2410.05229, GSM1k 2405.00332) | Templates or new items | **High.** The accepted way to show memorisation effects | Vi-ExamMath `template` subset + numeric-perturbation slice. Report the accuracy drop |
| **Alternative-translation probe** (new) | BenchMAX-vi vs our Vi-GSM8K on the same 250 items | Medium | Separates translation artifact from contamination |
| TS-Guessing / completion probes (2311.09783) | Base-model logprobs | Medium-low, qualitative | Optional appendix |
| **Min-K%++** (2404.02936) and other MIAs | Base-model logprobs | **Low as standalone evidence**. MIAs perform near chance on IID member/non-member splits (Duan et al., 2402.07841), and blind baselines beat them (Das et al., 2406.16201) | Only as an *auxiliary* signal on pre- vs post-cutoff Vi-ExamMath items. Never claim "not contaminated" from it |
| Canary GUID + gated release (DeepSEA practice) | Release step | Prevents *future* contamination | Do it |

Bottom line: the contamination-control claim of C1 = **dated items (post-2025-04) + verified provenance + n-gram/embedding scans against
the identified Vietnamese scraped corpora + perturbation deltas**. That combination is credible in 2026. MIA scores alone are not.

---

## 6. Action list (ordered)

1. `eval/math_verify.py`: fix B1–B6 via a structured gold + VN pre-normaliser + HF `math-verify==0.9.0` backend. Pin `latex2sympy2_extended==1.11.0`. Run on Linux, or disable math-verify timeouts on Windows.
2. Build `Vi-GSM8K-Platinum` (1,209) from `madrylab/gsm8k-platinum`. Re-translate the 10 revised items. Build the Vi-MATH-500 and alg+prealg slices (MATH-500 `unique_id` ↔ problem match).
3. S5 filter for MetaMathQA-vi (LaTeX parity, answer parity, number parity, type scope). Log the rejection rate.
4. Implement `eval/decontam.py` (n-gram + embedding) and run it against the corpora in §5.
5. Vi-ExamMath scraper: loigiaihay as primary. Store `source_url`, `solution_url`, `issuer`, `issuer_type`, `exam_date`, `published_at` (first seen), `license_basis`. Do not store third-party solution text in the release split.
6. Add the translation-variance probe (BenchMAX-vi) and the PolyMath-vi / M3Exam-vi leaderboard rows.

## 7. Evidence index

- HF metadata: `train_meta.tsv`, `eval_meta.tsv`, `eval_splits.txt`, `samples.txt`, `cards.txt`, `hf_search.txt`
- Quality checks: `gsm_check.py` (+`gsm_platinum_diff.json`), `quality.out`, `q3.py`, `q4.out`, `q5.out`
- Scorer probes: `probe_mv.out` (project scorer), `probe_hf.out` (HF math-verify 0.9.0)
- Tooling versions: `pypi_versions.txt`. GitHub API calls are inline in §4.1
- Sites: `robots/*.txt`, `robots_summary.txt`, `pages/`
- Papers (arXiv API verified titles and dates): 2604.17794, 2509.12251, 2608.18336, 2610.26155, 2404.02936, 2402.07841, 2406.16201, 2410.05229, 2405.00332, 2310.10628, 2311.09783, 2504.11456, 2505.14625, 2504.10481, 2508.03686, 2306.05685
- Law: caselaw.vn (Luật 50/2005 Điều 15; NĐ 17/2023 Điều 8, 25, 28), cov.gov.vn and luatvietan.vn (Luật 131/2025/QH15, Điều 7(5), effective 2026-04-01)
